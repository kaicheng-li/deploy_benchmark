#!/usr/bin/env bash
# ============================================================================
# ONNX Runtime 部署脚本
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ONNX_DIR="$(dirname "$SCRIPT_DIR")/onnx"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info() { echo -e "${GREEN}[ONNX]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[ONNX]${NC} $*"; }

PYTHON="${PYTHON:-python3}"

log_info "开始部署 ONNX Runtime..."

cd "$ONNX_DIR"

log_info "安装 Python 依赖..."
$PYTHON -m pip install --quiet onnx onnxruntime 2>/dev/null || true

# 如果有 CUDA，安装 GPU 版本
if command -v nvidia-smi &>/dev/null; then
    log_info "检测到 CUDA，安装 onnxruntime-gpu..."
    $PYTHON -m pip install --quiet onnxruntime-gpu 2>/dev/null || true
fi

$PYTHON -m pip install --quiet -r requirements.txt 2>/dev/null || true

mkdir -p onnx_models

log_info "ONNX Runtime 部署完成 ✓"
echo ""
echo "  使用方式:"
echo "    导出 RF-DETR: cd onnx && python src/export_onnx.py --config config.yaml --mode vision"
echo "    导出 Qwen3:  cd onnx && python src/export_onnx.py --config config.yaml --mode qwen3"
echo "    推理:      cd onnx && python src/inference.py --config config.yaml --mode <vision|qwen3>"
echo "    基准测试:  cd onnx && python src/benchmark.py --config config.yaml --mode <vision|qwen3>"
