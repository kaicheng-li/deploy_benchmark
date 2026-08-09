# Qwen3-VL TensorRT-LLM 多模态模型部署报告

## 1. 部署结论

本次在 NVIDIA GeForce RTX 5090 上完成了 Qwen3-VL-8B-Instruct 的 TensorRT-LLM 多模态部署和图片问答验证。

部署状态：**成功**

- 模型：Qwen3-VL-8B-Instruct
- 模型来源：本地 Hugging Face checkpoint
- 推理后端：TensorRT-LLM PyTorch backend
- TensorRT-LLM：1.2.1
- TensorRT：10.14.1.48
- PyTorch：2.9.1+cu128
- GPU：RTX 5090 32GB
- 服务接口：OpenAI-compatible `/v1/chat/completions`
- 实测请求：HTTP 200，能够返回正确的中文图片描述

## 2. 为什么 Qwen3-VL 不转换成传统 `.engine`

这里需要区分两种 TensorRT-LLM 运行方式。

### 2.1 传统 Engine backend

传统流程通常是：

```text
Hugging Face 权重
    -> convert_checkpoint
TensorRT-LLM checkpoint
    -> trtllm-build
序列化 .engine
    -> Runtime 加载
```

这种方式适合结构固定、输入输出边界明确的模型，例如纯文本 decoder-only LLM 或普通视觉网络。模型结构、输入 shape、KV Cache 和插件在构建阶段固化，启动时只需要反序列化 engine。

### 2.2 Qwen3-VL 的 PyTorch backend

Qwen3-VL 是一个视觉语言模型，不是单一的文本网络。一次请求包含：

```text
图片
  -> 图像预处理 / patch 切分
  -> Vision Encoder
  -> 图像特征投影与 image token 对齐
  -> 与文本 token 融合
  -> Qwen 语言模型自回归生成
  -> KV Cache 持续解码
```

其中视觉 token 数量会受到图片尺寸和切块结果影响，文本长度、图片数量、mRoPE position id、图像占位符和生成长度也可能变化。对于当前 TensorRT-LLM 1.2.1，Qwen3-VL 的推荐可用路径是 PyTorch backend：

```text
本地 Hugging Face 权重
    -> trtllm-serve
    -> TensorRT-LLM PyTorch backend
    -> 优化 CUDA/TensorRT-LLM kernels + CUDA Graph + KV Cache
```

因此没有单独的 `.engine` 文件，但仍然是 TensorRT-LLM 部署。模型权重在服务启动时加载到 GPU，TensorRT-LLM 运行时完成算子选择、CUDA Graph warmup、KV Cache 分配和自回归执行。

这不是“模型不能部署”或“因为配置没有写转换命令”，而是当前多模态后端的执行形态不同：Qwen3-VL 的视觉编码和语言生成需要在请求运行期协同处理，直接加载 HF 结构可以保留这套动态逻辑。

### 2.3 为什么 RF-DETR 又可以转 Engine

RF-DETR-Seg 是固定输入图片到检测/分割输出的视觉模型：

```text
pixel_values -> backbone/transformer -> logits、boxes、masks
```

它不包含文本 tokenizer、自回归循环、视觉 token 与文本融合或 KV Cache。模型图可以先导出 ONNX，再由 TensorRT 构建固定 engine，因此项目对 RF-DETR 使用：

```text
PyTorch -> ONNX -> TensorRT .engine
```

两者差异来自模型结构和运行时能力，不是因为一个模型在本地、另一个模型不在本地。

### 2.4 面试时的标准解释

可以这样回答：

> 我部署的是 TensorRT-LLM 的 PyTorch backend，不是传统的 TensorRT engine backend。Qwen3-VL 是视觉编码器和语言模型组成的多模态模型，图片经过预处理后会产生动态数量的视觉 token，还要和文本 token、mRoPE position id、KV Cache 一起参与自回归生成。当前 TensorRT-LLM 1.2.1 对这条链路提供了 `trtllm-serve` 直接加载 Hugging Face checkpoint 的运行方式，因此不强行走纯文本模型的 `convert_checkpoint -> trtllm-build` 流程。TensorRT-LLM 仍然负责 kernel、CUDA Graph、KV Cache 和调度优化，只是优化产物由运行时管理，不以单独 `.engine` 文件交付。RF-DETR 是固定图的视觉模型，所以可以先导出 ONNX 再构建传统 TensorRT engine。

如果面试官追问“为什么不一定要 engine”，可以补充：

> `.engine` 是 TensorRT 传统构建链路的序列化产物，不是使用 TensorRT-LLM 的唯一证明。是否生成 engine 取决于目标模型是否有稳定的 engine converter，以及模型输入和控制流能否在构建阶段固定。对当前 Qwen3-VL 版本，PyTorch backend 是更直接、可复现的多模态部署路径；如果交付要求必须有 engine，则需要另行确认对应版本的 Qwen3-VL engine conversion 支持，不能直接套用 Qwen3-8B 纯文本的转换命令。

## 3. 环境与配置文件

本项目已补齐两类复现文件：

- `tensorrt/requirements.txt`：固定本次验证使用的 TensorRT、TensorRT-LLM、PyTorch、Transformers、NumPy、SciPy、PyYAML、Requests 和 Pillow 版本；
- `tensorrt/environment.yml`：用于新建 Conda 环境，Python 3.12，并从 NVIDIA Python index 安装 TensorRT 和 TensorRT-LLM。

新环境创建：

```bash
cd deploy_benchmark
conda env create -f tensorrt/environment.yml
conda activate qwen3vl-tensorrt-llm
```

现有环境补装/校验：

```bash
pip install -r tensorrt/requirements.txt
python - <<'PY'
import torch, tensorrt, tensorrt_llm
print(torch.__version__, torch.cuda.is_available())
print(tensorrt.__version__)
print(getattr(tensorrt_llm, "__version__", "unknown"))
PY
```

`tensorrt/config.yaml` 已将默认模式设为 `qwen3vl`，模型来源设为本地路径，默认 prompt 与压测文件的第一个 prompt 一致。这样从项目目录启动时不需要修改路径，也不会因为默认模式仍是 `vision` 而误启动 RF-DETR。

## 4. 本次部署流程

### 4.1 本地模型检查

模型目录包含 4 个 safetensors 分片以及 `config.json`、`tokenizer.json`、`preprocessor_config.json` 和 `chat_template.json`，满足离线加载要求，不需要从 Hugging Face 下载。

### 4.2 启动服务

```bash
cd deploy_benchmark/tensorrt
python src/serve.py \
  --config config.yaml \
  --mode qwen3vl \
  --host 127.0.0.1 \
  --port 8001
```

服务启动阶段实际完成了：

1. 读取 Qwen3-VL 配置和本地权重；
2. 初始化视觉编码器和语言模型；
3. 分配 KV Cache；
4. 执行 TensorRT-LLM autotuner 和 CUDA Graph warmup；
5. 启动 OpenAI 兼容 HTTP 服务。

日志记录的关键资源信息：

- 模型权重占用约 16.33GB；
- 推理初始化峰值约 27.31GiB；
- RTX 5090 总显存约 32GB；
- 实测稳态显存约 30,799MiB。

### 4.3 单次多模态推理

```bash
python src/inference.py \
  --config config.yaml \
  --mode qwen3vl \
  --host 127.0.0.1 \
  --port 8001 \
  --image ../onnx/0000000109.png \
  --prompt "请用中文描述这张图片中的主要内容。" \
  --max-tokens 32
```

客户端将本地图片编码为 Base64 data URL，构造 OpenAI 多模态消息：

```json
{
  "messages": [{
    "role": "user",
    "content": [
      {"type": "text", "text": "请用中文描述这张图片中的主要内容。"},
      {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
    ]
  }]
}
```

## 5. 实测指标

测试条件：同一张图片、同一个中文 prompt、`temperature=0`、`max_tokens=32`，每个并发级别测量 3 个请求；请求为非流式 HTTP 请求。

| 并发数 | 单请求平均 E2E | 单请求 P50 | 单请求最大值 | 墙钟耗时 | 请求吞吐 | 输出吞吐 |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 9.58 s | 9.59 s | 9.65 s | 28.74 s | 0.104 req/s | 3.34 tok/s |
| 2 | 15.61 s | 18.69 s | 18.69 s | 28.15 s | 0.107 req/s | 3.41 tok/s |
| 4 | 27.84 s | 27.84 s | 27.84 s | 27.85 s | 0.108 req/s | 3.45 tok/s |

单次请求 token 统计：

- 输入 token：487；其中包含图片对应的视觉 token；
- 输出 token：32；本轮达到 `max_tokens` 上限；
- HTTP 状态码：200；
- 输出内容能够正确识别为黑白城市道路街景。

### 指标解释

- **E2E**：从 HTTP 请求发出到完整回答返回的时间；包含图片处理、视觉编码和文本生成。
- **TTFT**：真实首 token 时间需要流式接口。本次非流式请求无法得到真实 TTFT，不能把 E2E 当成严格 TTFT。
- **TPOT**：本次只用完整响应近似计算，不能替代逐 token 计时。
- **请求吞吐**：完成请求数除以本轮墙钟耗时。
- **输出吞吐**：输出 token 总数除以本轮墙钟耗时。

完整原始结果：`results/tensorrt_qwen3vl_deployment_report.json`。

## 6. 部署方式选择建议

### 选择当前 PyTorch backend

适合：

- 目标是快速完成 Qwen3-VL 图片问答服务；
- 需要支持动态图片尺寸、动态视觉 token 和 OpenAI 多模态接口；
- 接受启动时加载权重和 CUDA Graph warmup；
- 不要求交付独立 `.engine` 文件。

### 选择传统 Engine backend

适合：

- TensorRT-LLM 对目标模型提供完整、稳定的 checkpoint converter；
- 输入 shape、视觉 token 数和生成参数可以固定；
- 需要可序列化 engine、固定构建产物和更严格的构建期优化；
- 愿意维护模型转换、插件、shape profile 和 engine 版本兼容性。

对于当前 Qwen3-VL + TensorRT-LLM 1.2.1 组合，直接使用 PyTorch backend 是可运行的多模态部署方案；强行套用纯文本 LLM 的 `convert_checkpoint -> trtllm-build` 流程并不能保证视觉编码、图像 token 融合和生成逻辑正确。

## 7. 风险与后续工作

1. 当前基准是非流式请求，需增加流式 token 时间戳才能得到真实 TTFT 和 TPOT。
2. 32GB 显存下稳态占用约 30.8GB，并发继续提高可能触发显存不足。
3. 当前测量每个并发只有 3 个请求，适合部署验证，不足以作为正式容量规划结论。
4. 如果交付要求明确规定必须有 `.engine` 文件，应先确认目标 TensorRT-LLM 版本是否提供 Qwen3-VL 的完整 engine conversion 支持，再单独设计固定 shape 和多模态输入转换链路。

## 8. 最终结论

本次已经完成 Qwen3-VL 的 TensorRT-LLM 多模态部署：本地 HF 权重成功加载，服务成功启动，图片问答返回 HTTP 200。当前实现不生成传统 `.engine`，原因是采用了适配 Qwen3-VL 动态视觉语言结构的 PyTorch backend；RF-DETR 则是固定图结构视觉网络，适合 ONNX 到 TensorRT engine 的传统转换流程。
