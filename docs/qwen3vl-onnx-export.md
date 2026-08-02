# Qwen3-VL ONNX 导出问题记录

> 记录 2026-08-02 将 Qwen3-VL-8B-Instruct 导出为 ONNX 全过程中遇到的问题、
> 根因与解决方案，供后续复现与排查参考。

## 一、最终结论

Qwen3-VL 的 ONNX 采用**两段式导出**（社区 llm-export 等工具的同款思路）：

| 图 | 输入 | 输出 | 体积 |
|----|------|------|------|
| `qwen3vl_vision.onnx`（视觉塔） | `pixel_values` | `image_embeds` + `deepstack_embeds` | ~1.1 GB |
| `qwen3vl_decoder.onnx`（文本解码器） | `input_ids` + `attention_mask`(4D) + `position_ids` + `image_embeds` + `deepstack_embeds` | `logits` | ~16 GB |

两者均为**静态 shape**（固定 `seq_len=1024` 与固定图像预处理尺寸），`dtype=float16`。

## 二、环境

- GPU：RTX 5090 32GB（CUDA 13.0）
- torch 2.9.1+cu128、transformers 4.57.3、onnx 1.19.1、numpy 1.26.4、scipy 1.13.1
- 模型：本地 HF safetensors 权重 `models/Qwen3-VL-8B-Instruct`（17GB，4 个分片）

## 三、遇到的问题与解决

### 1. transformers 导入崩溃：scipy / numpy ABI 不匹配

- 现象：`ValueError: All ufuncs must have type numpy.ufunc`（`scipy.special` 加载失败）
- 根因：scipy 1.17.1 按 numpy 2.x ABI 编译，但环境是 numpy 1.26.4。且 `tensorrt-llm 1.2.1` 声明 `numpy<2`，**不能升级 numpy**。
- 解决：
  ```bash
  pip install numpy==1.26.4 scipy==1.13.1
  ```

### 2. 离线环境下去 HF 下载 processor 配置

- 现象：`HTTPSConnectionPool(host='huggingface.co', ...): Network is unreachable`
- 根因：config 里模型 `source: huggingface`，指向仓库 ID；机器无外网。
- 解决：改为 `source: local` + 本地路径 `../models/Qwen3-VL-8B-Instruct`（权重已在本地）。

### 3. Qwen3-VL 的 fast image processor 只支持 PyTorch 张量

- 现象：`ValueError: Only returning PyTorch tensors is currently supported.`
- 解决：`processor(..., return_tensors="pt")` 后再手动 `.numpy()` 转 numpy。

### 4. 文本里没有图像占位符

- 现象：`input_ids` 只有纯文本 token，找不到视觉 token（151652/151655/151653），`attention_mask.sum()` 只有 6。
- 根因：transformers 4.57 的 Qwen3-VL processor **不再自动插入图像占位符**（由 chat template 负责）。
- 解决：prompt 前手动拼接
  ```
  <|vision_start|><|image_pad|><|vision_end|> + prompt
  ```

### 5. torch 2.9 默认 dynamo 导出器拒绝数据相关 Python 逻辑（整模型导出不可行）

- 现象：`GuardOnDataDependentSymNode`，报错点 `fast_pos_embed_interpolate` 的
  `torch.linspace(0, self.num_grid_per_side - 1, h)`（`h` 来自 `image_grid_thw` 的数值）。
- 根因：torch 2.9 的 `torch.onnx.export` 已默认走 `torch.export`（dynamo）导出器，
  它不允许图里出现依赖张量**数值**的 Python 调用（`.item()`、`.tolist()`、逐 token 循环）。
  Qwen3-VL 的视觉塔、mRoPE position ids、占位符合并全部中招。
- 解决：放弃整模型导出，改为**两段式**：视觉塔与文本解码器分开导出，把无法进图的部分
  （position ids、4D 掩码）放到图外由 Python/numpy 计算。

### 6. 旧版 TorchScript 导出器（dynamo=False）输出是错的

- 现象：视觉塔 ONNX 输出误差 0.12（输出幅度才 0.08），解码器误差 0.85、32/32 位置 argmax 全错。
- 根因：`torch.jit.trace` 本身输出正确（与 eager 完全一致），但**ONNX 算子转换环节有 bug**；
  两个导出器（legacy 与 dynamo）的转换结果都错，说明不是某单个算子的问题而是整条转换链不可信。
- 解决：全部改用 dynamo 导出器，并建立「ONNX 输出 vs HF 参考」数值对拍（误差必须到 fp16 精度级别，argmax 必须一致）。

### 7. transformers 4.57 的因果掩码用 torch.vmap 构造，无法追踪

- 现象：TorchScript 追踪时在 `create_causal_mask -> eager_mask -> sdpa_mask -> _vmap_for_bhqkv`
  处崩溃（functorch vmap 递归，`RuntimeError: unordered_map::at`）。
- 解决：把**因果 + padding 掩码构造成 4D 浮点输入**（0 参与 / -inf 屏蔽，shape `(1,1,seq,seq)`）。
  HF 的 `_preprocess_mask_arguments` 对 4D 掩码会 early-exit **原样返回**，从而完全绕开 vmap 路径。

### 8. DeepStack 动态 gather 破坏 ONNX shape inference

- 现象：ONNX Runtime 加载解码器失败：
  `Node (/language_model/Add_1) Op (Add) [ShapeInferenceError] Incompatible dimensions`
- 根因：HF 的 `_deepstack_process` 用 `hidden_states[mask]` 动态 gather，ONNX 推断不出维度。
- 解决：把紧凑的 deepstack 特征 `(L, n_vis, hidden)` 展开为**全序列形式**
  `(L, seq, hidden)`（图像位置填特征、其余 0），注入方式改为静态 shape 的 `Add`。

### 9. 视觉塔剩余的数据相关逻辑

- 视觉塔还有 `torch.linspace`、`torch.split(lengths.tolist())`、`repeat_interleave` 等
  依赖 grid 数值的逻辑，dynamo 同样过不去。
- 解决：导出前把视觉塔替换为**常量驱动版本**：
  - 用 numpy 预计算 position embedding 的索引/权重，注册为 buffer（图中常量）；
  - `visual.forward` 用 Python 常量（h/w/merge 尺寸）重写；
  - 注意力改为单图单分块直调（跳过 `torch.split(lengths.tolist())`）。

### 10. 导出脚本自身的三个 bug

| Bug | 现象 | 修复 |
|-----|------|------|
| 4D 掩码误传给 `get_rope_index` | `IndexError: too many indices for tensor of dimension 1`（rope 需要 2D 掩码） | 拆成 `attention_mask_2d`（算 position ids）与 `attention_mask_4d`（喂图） |
| 解码器残留 `dynamo=False` | 导出"成功"但数值全错 | 改为 `dynamo=True` |
| `vision_onnx_file` 未加入路径解析 | 产物写到相对路径、位置不受控 | 加入 `resolve_task_config` 的解析键 |

### 11. fp32 导出体积过大

- fp32 整模型约 32GB，磁盘（当时剩 19GB）放不下。
- 解决：`dtype: float16` 导出（decoder ~16GB），并在 GPU 上追踪加速。

### 12. 验证测试的坑：跨进程对比必须使用相同随机种子

- 现象：一度以为导出全错——视觉塔 diff 0.12、解码器 0.85、argmax 全不一致。
- 根因：**导出脚本用 `seed=0`、验证脚本用 `seed=1`**，两个随机模型权重不同，
  跨进程对比毫无意义。
- 解决：统一 seed，并在同一进程内「导出 + 运行 + 对拍」。

## 四、最终验证结果

用全 fp16 的微型随机 Qwen3-VL 模型（结构同真实模型）：

- 两个图均通过 `torch.onnx.export(dynamo=True)` + `onnx.checker.check_model`；
- ONNX Runtime 推理 vs HF 参考：视觉塔最大误差 6.1e-5，解码器最大误差 7.3e-4（fp16 精度级别）；
- `argmax` 完全一致。

## 五、设计要点与已知限制

- **静态 shape**：`seq_len=1024`（图片约 468 个视觉 token + prompt + 生成），
  图片必须与导出时使用同一预处理尺寸（默认都是 `onnx/0000000109.png`），换图需重新导出。
- **单图、无视频**：视觉塔常量版只支持 t=1 的单帧图片。
- **无 KV cache**：解码每步重跑整图（mask 前移一位），功能正确但偏慢；
  与项目原有 Qwen3 文本链路（`run_qwen3` / `bench_qwen3`）风格一致。
- **磁盘**：decoder.onnx 约 16GB，转 OpenVINO/TensorRT 前需清理空间
  （可删 5.4GB 的 GGUF 或转换完成后删掉 decoder.onnx 中间产物）。

## 六、使用命令

```bash
# 导出（生成 qwen3vl_vision.onnx + qwen3vl_decoder.onnx）
(cd onnx && python src/export_onnx.py --config config.yaml --mode qwen3vl)

# 推理 / 基准（需先 pip install onnxruntime 或 onnxruntime-gpu）
(cd onnx && python src/inference.py --config config.yaml --mode qwen3vl)
(cd onnx && python src/benchmark.py --config config.yaml --mode qwen3vl)
```
