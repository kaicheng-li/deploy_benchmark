# deploy_benchmark

模型部署评测项目 — 覆盖主流推理框架的端到端部署、性能基准测试与对比分析。

## 框架能力矩阵

| 维度 | vLLM | TensorRT-LLM | ONNX Runtime | llama.cpp | OpenVINO |
|------|------|-------------|-------------|-----------|----------|
| **核心场景** | LLM 高吞吐服务 | LLM + 视觉, GPU 极致优化 | 通用跨平台 | LLM 量化推理 | Intel 平台加速 |
| **模型类型** | 文本 LLM | LLM + CV + 语音 | 全类型 | 文本 LLM | LLM + CV |
| **主要硬件** | NVIDIA GPU | NVIDIA GPU | CPU / GPU / NPU | CPU + GPU offload | Intel CPU/GPU/NPU |
| **推理范式** | 服务端 (API) | 引擎 (Engine) | 运行时 (Session) | 库 (Library) | 编译模型 |
| **部署场景** | ☁️ 云端 GPU 服务器 | ☁️ 云端 GPU 服务器 | 🌐 云/边/端 全场景 | 💻 桌面 / 📱 边缘 | 🏭 工业边缘 / PC |
| **核心优化** | PagedAttention, Continuous Batching | Kernel Fusion, FP8/INT4 | 图优化, 多后端 | GGUF 量化, mmap | 图优化, 异构调度 |
| **量化支持** | FP16/BF16, INT8 KV, AWQ/GPTQ | FP16, INT8, INT4, FP8 | FP16, INT8, INT4 | Q2-Q8, K-quant, I-quant | FP16, INT8, INT4 |
| **编程语言** | Python | Python / C++ | Python / C / C++ / C# | C / C++ / Python | Python / C++ |
| **学习曲线** | ⭐ 低 (pip install) | ⭐⭐⭐ 高 (模型转换多步) | ⭐⭐ 中 | ⭐⭐ 中 | ⭐⭐ 中 |
| **社区生态** | 活跃, 更新快 | NVIDIA 官方, 文档全 | 微软维护, 最广泛 | 极活跃, GGUF 标准 | Intel 官方 |

### 一句话总结

| 框架 | 适合场景 |
|------|---------|
| **vLLM** | 你有 NVIDIA GPU，要部署 LLM API 服务，追求高并发吞吐 |
| **TensorRT-LLM** | 你有 NVIDIA GPU，要榨干硬件性能，能接受模型转换流程 |
| **ONNX Runtime** | 你要跨平台/跨硬件，模型格式不确定，通用性优先 |
| **llama.cpp** | 你要在 CPU 或笔记本上跑 LLM，内存受限，量化优先 |
| **OpenVINO** | 你部署在 Intel 芯片上（Xeon/Arc/NPU），做边缘 AI |

---

## 测试条件 & 硬件要求

### 各框架运行前提

| 框架 | 最低硬件 | 推荐硬件 | 必须安装 |
|------|---------|---------|---------|
| **vLLM** | NVIDIA GPU 8GB+ | A100/L40S 40GB+ | CUDA 12.1+, `pip install vllm` |
| **TensorRT-LLM** | NVIDIA GPU 8GB+ | A100/H100 | CUDA 12.4+, `tensorrt_llm` pip wheel |
| **ONNX Runtime** | CPU 即可 | 任意 GPU | `pip install onnxruntime` (CPU) 或 `onnxruntime-gpu` |
| **llama.cpp** | CPU 4核 8GB | CPU 16核 + 任意 GPU | CMake + C++ 编译器 或 `pip install llama-cpp-python` |
| **OpenVINO** | Intel CPU (任何) | Intel Xeon / Arc GPU | `pip install openvino` |

### 测试矩阵推荐

```
                     NVIDIA GPU          CPU Only         Intel GPU/NPU
                    ┌──────────┐    ┌──────────┐    ┌──────────┐
vLLM                │    ✅    │    │    ❌    │    │    ❌    │
TensorRT-LLM        │    ✅    │    │    ❌    │    │    ❌    │
ONNX Runtime        │    ✅    │    │    ✅    │    │    ✅    │
llama.cpp           │    ✅    │    │    ✅    │    │    ❌    │
OpenVINO            │    ⚠️    │    │    ✅    │    │    ✅    │
                    └──────────┘    └──────────┘    └──────────┘
```

- ✅ 主力场景，性能最优
- ⚠️ 可以跑但不是该框架的优势场景
- ❌ 不支持或不推荐

### 你现在可以测试什么？

| 你的硬件条件 | 可以测试的框架 | 推荐测试模型 |
|-------------|--------------|------------|
| **有 NVIDIA GPU** (8G+) | 全部 5 个框架 | Qwen2-7B, LLaMA-3-8B |
| **仅 CPU** (x86) | ONNX Runtime, llama.cpp, OpenVINO | Qwen2-1.5B, TinyLLaMA |
| **MacBook (Apple Silicon)** | llama.cpp, ONNX Runtime | Qwen2-7B-Q4 (GGUF) |
| **Intel Arc GPU** | OpenVINO, ONNX Runtime | 分类/检测模型 + LLM |

---

## 对比维度设计

### 维度 1：延迟 (Latency)

| 子指标 | 含义 | 适用框架 |
|--------|------|---------|
| **TTFT** (Time To First Token) | 用户感知的首字响应速度 | vLLM, TensorRT-LLM, llama.cpp |
| **TPOT** (Time Per Output Token) | 生成速度 | 同上 |
| **E2E Latency** | 请求→完整响应的总时间 | 全部 |
| **P50 / P95 / P99** | 长尾延迟分布 | 全部 |

### 维度 2：吞吐 (Throughput)

| 子指标 | 含义 | 适用框架 |
|--------|------|---------|
| **tokens/s** | 单请求生成速度 | 全部 |
| **requests/s** | 并发处理能力 | vLLM (>其他), TensorRT-LLM |
| **QPS vs 并发曲线** | 并发增长时吞吐变化 | vLLM, TensorRT-LLM |

### 维度 3：资源效率

| 子指标 | 含义 |
|--------|------|
| **GPU 显存占用** | 模型加载后剩余显存 |
| **CPU 内存占用** | 推理进程 RSS |
| **模型体积** | FP32 vs FP16 vs INT4 磁盘占用 |
| **首包冷启动时间** | 模型加载 + 首次推理耗时 |

### 维度 4：精度

| 子指标 | 含义 |
|--------|------|
| **输出一致性** | 同一 prompt，FP32 vs INT8 输出差异 |
| **Perplexity** | 量化后困惑度损失 |
| **下游任务分** | MMLU / C-Eval 等评测变化 |

### 维度 5：工程化

| 子指标 | 含义 |
|--------|------|
| **部署复杂度** | 从零到跑通的行数/步骤 |
| **模型转换成本** | HF → 目标格式的时间与工具链 |
| **文档质量** | 上手难度 |
| **Docker 化难度** | 镜像大小、构建复杂度 |
| **并发模型** | 是否原生支持 Continuous Batching |

---

## 项目结构

```
deploy_benchmark/
├── scripts/                # 一键部署 & 评测脚本
│   ├── deploy_all.sh       # 全框架部署
│   └── deploy_<framework>.sh  # 单框架部署
├── vllm/                   # vLLM 推理 (纯 Python)
├── tensorrt/               # TensorRT 推理
│   ├── src/                # C++ 源码
│   └── CMakeLists.txt      # C++ 构建
├── onnx/                   # ONNX Runtime 推理 (Python)
├── llamacpp/               # llama.cpp 推理
│   ├── src/                # C++ 源码
│   └── python/             # Python 封装
├── openvino/               # OpenVINO 推理
│   ├── src/                # C++ 源码
│   └── CMakeLists.txt      # C++ 构建
├── common/                 # Python 公共模块（指标、报告、数据加载）
├── cpp_common/             # C++ 公共模块
├── docker/                 # Docker 镜像
├── models/                 # 模型存放目录
└── results/                # 基准测试结果输出
```

## 快速开始

```bash
# 一键部署全部框架
bash scripts/deploy_all.sh

# 或按需部署单个框架
bash scripts/deploy_vllm.sh
bash scripts/deploy_tensorrt.sh
bash scripts/deploy_onnx.sh
bash scripts/deploy_llamacpp.sh
bash scripts/deploy_openvino.sh
```

## 环境要求

- Python 3.10+
- CUDA 12.1+ (TensorRT-LLM) / CUDA 11.8+ (vLLM)
- CMake 3.20+ (llama.cpp C++ 构建)
- Docker (可选)

---

## 从零到跑通的典型流程

以 **Qwen2-7B-Instruct** 为例：

```bash
# 1. 准备环境
python -m venv venv && source venv/bin/activate

# 2. 下载模型
huggingface-cli download Qwen/Qwen2-7B-Instruct --local-dir models/Qwen2-7B-Instruct

# 3. 部署全部框架
bash scripts/deploy_all.sh

# 4. 分别转换/构建模型
cd onnx    && python export_onnx.py --config config.yaml
cd openvino && python convert_model.py --config config.yaml
cd tensorrt && python engine_builder.py --config config.yaml
# llama.cpp: 需要先用 convert-hf-to-gguf.py 转成 GGUF

# 5. 分别跑基准测试
cd vllm     && python server.py --model ../models/Qwen2-7B-Instruct &
               python benchmark.py --config config.yaml
cd tensorrt && python benchmark.py --config config.yaml
cd onnx     && python benchmark.py --config config.yaml
cd llamacpp/python && python benchmark.py --config ../config.yaml
cd openvino && python benchmark.py --config config.yaml

# 6. 汇总对比报告
# 各框架的 results/ 目录下会生成 Markdown + JSON + CSV 报告
```

---

## Python vs C++：什么时候用什么语言

| 框架 | 原型/验证 | 生产部署 | 原因 |
|------|-----------|----------|------|
| **vLLM** | Python | Python | 纯 Python 框架，无 C++ API |
| **TensorRT** | Python | **C++** | Python 快速跑通，C++ 可绕过 GIL 直驱 engine，延迟更低 |
| **ONNX Runtime** | Python | **C++** | Python 快速跑通，C++ 榨干延迟（生产推荐 C++） |
| **llama.cpp** | Python 封装 | **C++ 原生** | 性能核心在 C++，Python 绑定隔了一层 |
| **OpenVINO** | Python | **C++ 原生** | 边缘设备上 C++ 是首选，延迟和资源都更优 |

> **结论**：本项目 Python 脚本覆盖所有框架（快速验证），C++ 实现聚焦 TensorRT、llama.cpp、OpenVINO（生产级延迟测试），ONNX Runtime 保留双语言能力。

---

## 租用 GPU 服务器测试指南

### 适合你的场景

当前你只有一台 GPU 云服务器，这个配置能跑什么：

| 可测 | 不能充分发挥 |
|------|-------------|
| ✅ vLLM — GPU 大显存机型直接测 | ❌ OpenVINO 的 Intel 优化（没有 Intel CPU/GPU） |
| ✅ TensorRT-LLM — 同 GPU | ❌ llama.cpp 纯 CPU 场景（GPU 服务器 CPU 也强，但测不出 CPU 极限） |
| ✅ ONNX Runtime — GPU 和 CPU 都能测 | ❌ OpenVINO NPU 场景 |
| ✅ llama.cpp — CUDA offload 模式 | |
| ⚠️ OpenVINO — CPU device 能跑，但不是它的最优场景 | |

### 推荐测试顺序

```bash
# 第1步：选一个中小模型，在 GPU 上按框架逐个跑通
MODEL="Qwen/Qwen2-1.5B-Instruct"   # 1.5B 适合 8GB 显存
# MODEL="Qwen/Qwen2-7B-Instruct"   # 7B 需要 16GB+

# 第2步：跑 LLM 对比（vLLM / TensorRT-LLM / ONNX / llama.cpp / OpenVINO）
# 第3步：跑 CV 对比（ONNX / OpenVINO / TensorRT）用 ResNet50 或 ViT
# 第4步：汇总 results/ 下的报告，生成最终对比表格
```

### GPU 机型选择建议

| 预算 | GPU | 适合测什么 |
|------|-----|-----------|
| 低（几毛/小时） | T4 16GB | 7B 模型全框架，vLLM+TensorRT 对比 |
| 中（几块/小时） | A10/L40S 24-48GB | 7B-13B，加 CV 模型 |
| 高（十几块/小时） | A100 40-80GB | 70B 模型，多并发压测 |

> **简历贴士**：简历上注明测试环境（GPU 型号 + 显存 + CUDA 版本），让数据有上下文。比如 "在 A100-40G 上，Qwen2-7B 的 vLLM(TTFT=12ms, 吞吐=3200tok/s) vs TensorRT-LLM(TTFT=8ms, 吞吐=4100tok/s)"。```
