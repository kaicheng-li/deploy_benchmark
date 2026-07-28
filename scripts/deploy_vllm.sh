#!/usr/bin/env bash
# ============================================================================
# vLLM 部署脚本
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_DIR="$(dirname "$SCRIPT_DIR")/vllm"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info() { echo -e "${GREEN}[vLLM]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[vLLM]${NC} $*"; }

log_info "开始部署 vLLM..."

cd "$VLLM_DIR"

# ── Python 依赖 ───────────────────────────────────────────────
PYTHON="${PYTHON:-python3}"
log_info "安装 Python 依赖..."
$PYTHON -m pip install --quiet -r requirements.txt 2>/dev/null || {
    log_warn "部分依赖安装失败，尝试逐个安装..."
    $PYTHON -m pip install vllm torch transformers
}

log_info "vLLM 部署完成 ✓"
echo ""
echo "  使用方式:"
echo "    启动服务:  cd vllm && python server.py --config config.yaml"
echo "    基准测试:  cd vllm && python benchmark.py --config config.yaml"
