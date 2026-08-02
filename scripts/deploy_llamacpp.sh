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

# ── Python binding (the project no longer vendors llama.cpp source)
log_info "安装 llama.cpp Python 封装..."
$PYTHON -m pip install --quiet -r requirements.txt 2>/dev/null || {
    log_warn "llama-cpp-python 安装失败"
    log_warn '请手动安装: pip install "llama-cpp-python>=0.3.34"'
}

log_info "llama.cpp 部署完成 ✓"
echo ""
echo "  使用方式:"
echo "    Python 推理:  (cd llamacpp && python src/inference.py --config config.yaml)"
echo "    Python 基准:  (cd llamacpp && python src/benchmark.py --config config.yaml)"
