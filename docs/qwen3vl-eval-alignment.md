# Qwen3-VL ONNX / OpenVINO 推理与指标评估对齐说明

> 目标：让 Qwen3-VL 在 ONNX 与 OpenVINO 两个后端上的推理行为和指标评估
> 完全可比，Python 与 C++ 两套实现遵循同一口径。状态截至 2026-08-02，
> 本轮只做代码对齐，未运行推理/基准。

## 一、统一评估口径

### 任务定义

- 单张图片 + 文本 prompt → 文本生成（与项目里其他文本后端的 text-generation 口径一致）。
- 默认输入：`onnx/0000000109.png` + `"Describe this image in detail."`，
  两个后端共用同一份数据。

### 推理行为

- 两段式：视觉塔（`pixel_values → image_embeds + deepstack_embeds`）+
  文本解码器（5 输入 → `logits`）。
- 静态 shape：`seq_len=1024`、固定图片预处理尺寸。
- 解码方式：无 KV cache 的贪心解码，每步重跑整张解码器图；
  `position_ids`（numpy 移植的 mRoPE）与 4D 因果掩码在 Python 侧预计算后
  作为显式输入。
- ONNX 与 OpenVINO 共用同一份 `onnx/src/qwen3vl_utils.py`，推理行为天然一致。

### 指标口径

| 指标 | 定义 |
|------|------|
| TTFT | 请求开始到**首个生成 token** 的耗时（含视觉塔前向 + 第一步解码） |
| TPOT | 逐 token 解码平均耗时（**不含**视觉塔） |
| E2E | 完整请求耗时 |
| P50/P95/P99 | 各延迟指标的分位值（`common.metrics` 线性插值口径） |
| tok/s | 总生成 token / 总耗时 |
| req/s | 请求数 / 总耗时 |

### 报告输出

- Python：统一走 `common.metrics.BenchmarkMetrics` +
  `common.reporter.BenchmarkReporter.save_all()`，输出
  `results/onnx_qwen3vl_benchmark_*.{md,json,csv}` 与
  `results/openvino_qwen3vl_benchmark_*.{md,json,csv}`，字段一一对应可对比。

## 二、Python 侧调整（本轮已完成）

### 共享工具 `onnx/src/qwen3vl_utils.py`

- 新增 `generate_timed()`：与 `generate()` 解码行为完全一致，额外返回
  `(text, input_tokens, output_tokens, ttft_ms, tpot_ms, e2e_ms)`，
  供基准计时使用。

### `onnx/src/benchmark.py` / `openvino/src/benchmark.py`

- `bench_qwen3vl` 从「直接打印 avg/p50/p95/p99」改为：
  每次请求生成一个 `TimingResult(ttft, tpot, e2e, input_tokens, output_tokens)`
  → `BenchmarkMetrics.from_timings()` → `BenchmarkReporter.save_all()`，
  与 llama.cpp / vLLM / TensorRT 文本链路的报告结构对齐。
- 修复潜在 bug：原实现直接复用 `feeds`，解码会就地修改
  `input_ids / attention_mask`，导致后一轮基准从已生成的序列开始；
  现在每一轮先 `feeds` 拷贝再解码。
- framework 标识：`"ONNX Runtime"` / `"OpenVINO"`；
  device：ONNX 按 provider（CUDA→cuda，否则 cpu），OpenVINO 按 `device` 字段。

### `onnx/config.yaml` / `openvino/config.yaml`

- 两个 qwen3vl 任务都增加 `output_dir: "../results"`，prompt 与 seq_len 已一致。

## 三、C++ 侧现状与差距

### 现状

- `cpp_common/benchmark_utils.{h,cpp}`：`BenchResult` 已含
  avg/min/max/p50/p95/p99、吞吐(tok/s)、RSS 内存，百分位用线性插值
  （与 Python `common.metrics` 同口径）。
- `onnx/src/benchmark.cpp`、`openvino/src/benchmark.cpp` + `inference.cpp`：
  目前只有 `vision`（RF-DETR）和 `qwen3`（纯文本，**随机 token、单次前向、
  无解码循环**）两个模式，**没有 qwen3vl 模式**。

### 差距

要让 C++ 与 Python 的 Qwen3-VL 评估对齐，还缺：

1. **Tokenizer**：C++ 侧目前没有 tokenizer 依赖（qwen3 模式用随机 token）。
   Qwen3-VL 需要按 `tokenizer.json` 编码（含 `<|vision_start|>...` 占位符），
   需要引入 HuggingFace tokenizers C++ 库。
2. **图像预处理/切块**：对齐 `Qwen3VLImageProcessor`（保持长宽比缩放、
   min/max pixels、patchify → `(1872, 1536)`、grid 计算）。
3. **mRoPE position ids 的 C++ 移植**：对应
   `qwen3vl_utils.compute_rope_index`。
4. **4D 因果掩码**构造（简单）。
5. **双模型解码循环**：视觉塔一次 + 解码器逐 token，argmax + EOS 停止。
6. **BenchResult 扩展**：增加 `ttft_avg/p50/p95/p99`、`tpot_avg/...` 字段，
   与 Python `BenchmarkMetrics` 对齐；命令行增加 `qwen3vl` 模式及配套参数。

### 建议实施顺序（后续）

1. 扩展 `BenchResult`（TTFT/TPOT 字段 + 打印）——低风险；
2. 公共部分（rope index、4D mask、解码循环骨架）放到 `cpp_common`；
3. ONNX 与 OpenVINO 各自的 `run_qwen3vl` 实现（复用骨架，差异只在推理接口）；
4. tokenizer 与图像预处理接入（引入依赖，需编译验证）。

## 四、验证方式（后续运行时）

- 同一 seed/prompt/图片下：ONNX 与 OpenVINO 输出 token 一致（argmax 相同），
  数值差异在 fp16 精度范围；
- 两个后端的 `results/*.md` 字段一一对应，可直接横向对比；
- Python 与 C++ 报告字段名/口径一致（TTFT/TPOT/E2E/P50/P95/P99/tok/s/内存）。
