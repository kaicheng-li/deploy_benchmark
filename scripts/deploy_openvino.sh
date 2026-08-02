#!/usr/bin/env bash
# ============================================================================
# OpenVINO 部署脚本
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OV_DIR="$(dirname "$SCRIPT_DIR")/openvino"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
log_info() { echo -e "${GREEN}[OpenVINO]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[OpenVINO]${NC} $*"; }

PYTHON="${PYTHON:-python3}"

log_info "开始部署 OpenVINO..."

cd "$OV_DIR"

log_info "安装 Python 依赖..."
$PYTHON -m pip install --quiet openvino 2>/dev/null || log_warn "openvino 安装失败"
$PYTHON -m pip install --quiet optimum[openvino] 2>/dev/null || log_warn "optimum[openvino] 安装失败"
$PYTHON -m pip install --quiet -r requirements.txt 2>/dev/null || true

mkdir -p openvino_models

# ── C++ 编译 (可选) ──────────────────────────────────────────
if command -v cmake &>/dev/null; then
    log_info "尝试编译 OpenVINO C++ benchmark..."
    # 检测 OpenVINO C++ SDK
    if [ -n "${INTEL_OPENVINO_DIR:-}" ] || [ -d "/opt/intel/openvino" ]; then
        mkdir -p build && cd build
        cmake .. -DCMAKE_BUILD_TYPE=Release 2>/dev/null && \
            cmake --build . --config Release -j$(nproc 2>/dev/null || echo 4) 2>/dev/null && \
            log_info "C++ benchmark 编译成功: ./build/ov_benchmark" || \
            log_warn "C++ 编译失败，仅 Python 可用"
        cd ..
    else
        log_warn "未检测到 OpenVINO C++ SDK，跳过 C++ 编译"
        log_warn "如需 C++ 版本，请安装: apt install openvino-dev 或从 Intel 官网下载"
    fi
fi

log_info "OpenVINO 部署完成 ✓"
echo ""
echo "  使用方式:"
echo "    转换 RF-DETR: cd openvino && python src/convert_model.py --config config.yaml --mode vision"
echo "    转换 Qwen3:  cd openvino && python src/convert_model.py --config config.yaml --mode qwen3"
echo "    推理:      cd openvino && python src/inference.py --config config.yaml --mode <vision|qwen3>"
echo "    基准测试:  cd openvino && python src/benchmark.py --config config.yaml --mode <vision|qwen3>"
echo "    C++ 基准:  cd openvino/build && ./ov_benchmark --model ../openvino_models/vision.xml --mode vision"
