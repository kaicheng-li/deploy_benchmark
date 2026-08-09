# Qwen3-VL OpenVINO 转换与脚本调整记录

> 记录 2026-08-02 把两段式 Qwen3-VL ONNX 转为 OpenVINO IR，以及为支持
> qwen3vl 模式对 OpenVINO 侧脚本所做的调整。前置步骤见
> [qwen3vl-onnx-export.md](./qwen3vl-onnx-export.md)（ONNX 导出问题记录）。

## 一、目标与结论

把已导出的两份 ONNX 图转成 OpenVINO IR，并在项目内打通 OpenVINO 的
qwen3vl 多模态推理与基准链路：

| ONNX（输入） | OpenVINO IR（输出） |
|--------------|---------------------|
| `onnx/onnx_models/qwen3vl_vision.onnx` | `openvino/openvino_models/qwen3vl_vision.xml` + `.bin`（~1.15GB） |
| `onnx/onnx_models/qwen3vl_decoder.onnx` | `openvino/openvino_models/qwen3vl_decoder.xml` + `.bin`（~16.4GB） |

转换与编译验证均已通过。

## 二、原理

1. **转换是纯 CPU 图转换**：`ov.convert_model(onnx_file)` 只做 ONNX 图解析与
   格式翻译（`.xml` 结构 + `.bin` 权重），不需要 GPU、不需要跑模型前向。
   这也是为什么 OpenVINO 转换比 ONNX 导出轻得多（解码器 16GB 约 90 秒）。
2. **两段式对应两份 IR**：视觉塔和解码器各转各的，保持 ONNX 链路的图边界。
3. **推理逻辑与 ONNX 链路完全一致**：视觉 IR 每请求跑一次得到
   `image_embeds + deepstack_embeds`；解码器 IR 逐 token 跑。`position_ids`
   与 4D 因果掩码仍在 Python/numpy 预计算后作为显式输入——这两部分含
   逐 token 的 Python 控制流，进不了静态图。
4. **设备选择**：`device: "CPU"`。OpenVINO 的 GPU 只服务 Intel 显卡/NPU，
   当前机器是 NVIDIA（RTX 5090），CPU 是正确的选择。

## 三、脚本调整点

### 1. `openvino/config.yaml`

- 新增模型 `qwen3_vl_8b_instruct`：`source: local`，指向
  `../models/Qwen3-VL-8B-Instruct`（本地 HF 权重）。
- 新增任务 `qwen3vl`：
  - `onnx_file` / `vision_onnx_file`：两份 ONNX 输入路径；
  - `ir_file` / `vision_ir_file`：两份 IR 输出路径；
  - `image` / `prompt` / `max_new_tokens` / `seq_len: 1024` / `device: "CPU"`。

### 2. `openvino/src/convert_model.py`

- `--mode` 选项扩展为 `vision | qwen3 | qwen3vl`；
- `resolve_task_config` 的本地路径解析键加入 `vision_onnx_file`、`vision_ir_file`；
- 抽出 `convert_one(onnx_path, ir_path)` 通用转换函数（`ov.convert_model` +
  `ov.save_model(..., compress_to_fp16=False)`，保留 fp16 精度）；
- `qwen3vl` 模式依次转换视觉塔和解码器两份图。

### 3. `openvino/src/inference.py` / `benchmark.py`

- 新增 `_OVSession` 适配类：把 OpenVINO `compiled model` 包装成与
  ONNX Runtime session 相同的 `run(output_names, feed) -> list[ndarray]`
  接口（输出按名字匹配，失败时按顺序回退）；
- 新增 `run_qwen3vl` / `bench_qwen3vl`：复用 `onnx/src/qwen3vl_utils.py`
  里的 `prepare_inputs`（图像占位符 + padding）、`compute_rope_index`
  （numpy 移植的 mRoPE）、`build_causal_mask`（4D 因果掩码）和 `generate`
  （贪心解码循环），只需把 ORT session 换成 `_OVSession`；
- `main()` 的 mode 分支与报错信息同步更新。

> 复用而不是复制解码逻辑，保证 OpenVINO 与 ONNX 两条链路行为严格一致，
> 后续修复只改 `qwen3vl_utils.py` 一处。

## 四、验证结果

```bash
python src/convert_model.py --config config.yaml --mode qwen3vl
```

- 视觉塔转换 ~3s，解码器转换 ~90s；
- `ov.Core().read_model` 解析两份 IR 成功；
- `compile_model("CPU")` 编译两份 IR 成功；
- 输入输出与 ONNX 设计一致：
  - 视觉塔：`pixel_values (1872,1536)` → `image_embeds` + `deepstack_embeds`
  - 解码器：`input_ids (1,1024)` + `attention_mask (1,1,1024,1024)` +
    `position_ids (3,1,1024)` + `image_embeds (468,4096)` +
    `deepstack_embeds (3,468,4096)` → `logits`

## 五、使用命令

```bash
# 转换（两份 ONNX -> 两份 IR）
(cd openvino && python src/convert_model.py --config config.yaml --mode qwen3vl)

# 推理 / 基准
(cd openvino && python src/inference.py --config config.yaml --mode qwen3vl)
(cd openvino && python src/benchmark.py --config config.yaml --mode qwen3vl)
```

## 六、已知限制与注意事项

- **静态 shape**：`seq_len=1024`、图片预处理尺寸固定（与 ONNX 导出一致），
  换图需重新导出 ONNX 再转 IR；
- **单图、无视频、无 KV cache**：推理每步重跑整张解码器图，CPU 上较慢
  （16GB 模型，64 token 可能需要数分钟），功能正确但性能一般；
- **NVIDIA 机器上 OpenVINO 只能走 CPU**：要 GPU 加速请走 TensorRT-LLM 链路；
- 磁盘占用：两份 IR 合计约 17.5GB，另有 ONNX 中间产物约 17.5GB，
  磁盘紧张时可删除 ONNX 产物（IR 已生成后不再需要）。
