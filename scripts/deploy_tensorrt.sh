#!/usr/bin/env bash
# ============================================================================
# TensorRT-LLM 部署脚本
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRT_DIR="$(dirname "$SCRIPT_DIR")/tensorrt"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
log_info() { echo -e "${GREEN}[TensorRT]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[TensorRT]${NC} $*"; }
log_err()  { echo -e "${RED}[TensorRT]${NC} $*"; }

PYTHON="${PYTHON:-python3}"

log_info "开始部署 TensorRT-LLM..."

# ── 检查 CUDA ─────────────────────────────────────────────────
if ! command -v nvidia-smi &>/dev/null; then
    log_err "TensorRT 需要 NVIDIA GPU 和 CUDA，部署终止"
    exit 1
fi

# ── Python 依赖 ───────────────────────────────────────────────
cd "$TRT_DIR"

log_info "安装 Python 依赖..."
$PYTHON -m pip install --quiet torch nvidia-cuda-runtime-cu12 2>/dev/null || true
$PYTHON -m pip install --quiet tensorrt_llm 2>/dev/null || {
    log_warn "tensorrt_llm pip 安装失败"
    log_warn "请手动安装: pip install tensorrt_llm --extra-index-url https://pypi.nvidia.com"
}

$PYTHON -m pip install --quiet -r requirements.txt 2>/dev/null || true

# ── 创建 engine 目录 ──────────────────────────────────────────
mkdir -p trt_engines

# ── C++ 编译 (可选，需要 TensorRT SDK) ─────────────────────────
log_info "检查 TensorRT C++ SDK..."
TRT_SDK_FOUND=false

# 尝试常见路径
for path in /usr/src/tensorrt /usr/local/tensorrt /opt/tensorrt "$TENSORRT_DIR"; do
    if [ -f "$path/include/NvInfer.h" ]; then
        log_info "找到 TensorRT SDK: $path"
        TRT_SDK_FOUND=true
        break
    fi
done

if $TRT_SDK_FOUND && command -v cmake &>/dev/null; then
    log_info "编译 TensorRT C++ benchmark..."
    mkdir -p build && cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release 2>/dev/null && \
        cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4) 2>/dev/null && \
        log_info "C++ benchmark 编译成功: ./build/trt_benchmark" || \
        log_warn "C++ 编译失败，仅 Python 可用"
    cd ..
elif ! $TRT_SDK_FOUND; then
    log_warn "未找到 TensorRT C++ SDK，跳过 C++ 编译"
    log_warn "如需 C++ 版本:"
    log_warn "  1. 下载: https://developer.nvidia.com/tensorrt/download"
    log_warn "  2. 解压后: export TENSORRT_DIR=/path/to/TensorRT-10.x.x"
    log_warn "  3. 重新运行此脚本"
    log_warn "  或者: apt install tensorrt-dev (如果系统包管理器支持)"
fi

log_info "TensorRT 部署完成 ✓"
echo ""
echo "  使用方式:"
echo "    构建视觉引擎: (cd tensorrt && python src/engine_builder.py --config config.yaml)"
echo "    视觉推理:     (cd tensorrt && python src/inference.py --config config.yaml)"
echo "    Qwen3-VL 服务: (cd tensorrt && python src/serve.py --config config.yaml --mode qwen3vl)  # TRT-LLM >= 1.2.0"
echo "    Qwen3-VL 推理: (cd tensorrt && python src/inference.py --config config.yaml --mode qwen3vl --image ../onnx/0000000109.png)"
echo "    Qwen3-VL 压测: (cd tensorrt && python src/benchmark.py --config config.yaml --mode qwen3vl)"
echo "    C++ 基准:  cd tensorrt/build && ./trt_benchmark --engine ./trt_engines/model.engine"
