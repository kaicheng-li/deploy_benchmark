#!/usr/bin/env bash
# ============================================================================
# llama.cpp 部署脚本 (C++ 编译 + Python 封装)
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_DIR="$(dirname "$SCRIPT_DIR")/llamacpp"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info() { echo -e "${GREEN}[llama.cpp]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[llama.cpp]${NC} $*"; }

PYTHON="${PYTHON:-python3}"

log_info "开始部署 llama.cpp..."

cd "$LLAMA_DIR"

# ── Clone llama.cpp ────────────────────────────────────────────
if [ ! -d "llama.cpp" ]; then
    log_info "Clone llama.cpp 源码..."
    git clone --depth 1 https://github.com/ggerganov/llama.cpp.git
else
    log_info "llama.cpp 源码已存在，跳过 clone"
fi

# ── C++ 编译 ──────────────────────────────────────────────────
log_info "编译 llama.cpp C++ 程序..."
cd llama.cpp

HAS_CUDA=false
if command -v nvidia-smi &>/dev/null; then
    HAS_CUDA=true
fi

CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=OFF"
if $HAS_CUDA; then
    log_info "启用 CUDA 加速..."
    CMAKE_FLAGS="-DCMAKE_BUILD_TYPE=Release -DGGML_CUDA=ON"
fi

mkdir -p build && cd build
cmake .. $CMAKE_FLAGS
cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4)
cd ../..

# ── 编译本项目 C++ benchmark ──────────────────────────────────
log_info "编译 benchmark 程序..."
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4) || log_warn "benchmark 编译失败，仅 Python 可用"
cd ..

# ── Python 封装 ───────────────────────────────────────────────
log_info "安装 llama.cpp Python 封装..."
$PYTHON -m pip install --quiet llama-cpp-python 2>/dev/null || {
    log_warn "llama-cpp-python 安装失败"
    log_warn "请手动安装: pip install llama-cpp-python"
}

log_info "llama.cpp 部署完成 ✓"
echo ""
echo "  使用方式:"
echo "    C++ 推理:     cd llamacpp/build && ./llama_benchmark --model ../../models/model.gguf --prompt 'Hello'"
echo "    Python 推理:  cd llamacpp/python && python inference.py --model ../../models/model.gguf --prompt 'Hello'"
echo "    Python 基准:  cd llamacpp/python && python benchmark.py --model ../../models/model.gguf"
