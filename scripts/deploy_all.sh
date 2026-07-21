#!/usr/bin/env bash
# ============================================================================
# deploy_benchmark — 一键部署全部推理框架
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
cd "$ROOT_DIR"

GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC}  $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC}  $*"; }
log_step()  { echo -e "${CYAN}[STEP]${NC}  $*"; }

echo "============================================================================"
echo "  deploy_benchmark — 全框架一键部署"
echo "============================================================================"
echo ""

# ── 环境检测 ──────────────────────────────────────────────────
detect_python() {
    if command -v python3 &>/dev/null; then
        echo "python3"
    elif command -v python &>/dev/null; then
        echo "python"
    else
        echo ""
    fi
}

PYTHON=$(detect_python)
if [ -z "$PYTHON" ]; then
    log_warn "未检测到 Python, 仅部署 C++ 相关组件"
fi

# 检查 CUDA
HAS_CUDA=false
if command -v nvidia-smi &>/dev/null; then
    HAS_CUDA=true
    log_info "检测到 CUDA GPU"
else
    log_warn "未检测到 CUDA, GPU 框架将使用 CPU 模式"
fi

# ── 创建必要目录 ──────────────────────────────────────────────
log_step "创建目录结构..."
mkdir -p data models results

# ── 生成示例测试数据 ──────────────────────────────────────────
log_step "生成示例测试数据..."
cat > data/prompts.txt << 'EOF'
Explain the attention mechanism in transformer models.
请用中文介绍深度学习的基本原理和主要应用场景。
Write a Python function to find the k-th largest element in an array.
Describe the differences between CNN and Transformer architectures.
What are the key innovations in the LLaMA model architecture?
Explain how beam search works in sequence generation.
请用代码实现一个简单的二分查找算法。
EOF
log_info "示例数据已生成: data/prompts.txt"

# ── 安装公共 Python 依赖 ──────────────────────────────────────
if [ -n "$PYTHON" ]; then
    log_step "安装公共 Python 依赖..."
    $PYTHON -m pip install --quiet pyyaml requests numpy 2>/dev/null || true
fi

# ── 逐框架部署 ────────────────────────────────────────────────
deploy_framework() {
    local name="$1"
    local script="$2"
    log_step "部署 $name ..."
    if [ -f "$SCRIPT_DIR/$script" ]; then
        bash "$SCRIPT_DIR/$script"
        log_info "$name 部署完成 ✓"
    else
        log_warn "$script 不存在，跳过 $name"
    fi
}

# 按需部署每个框架（跳过失败的继续）
set +e

deploy_framework "vLLM"        "deploy_vllm.sh"
deploy_framework "TensorRT"    "deploy_tensorrt.sh"
deploy_framework "ONNX Runtime" "deploy_onnx.sh"
deploy_framework "llama.cpp"   "deploy_llamacpp.sh"
deploy_framework "OpenVINO"    "deploy_openvino.sh"

set -e

# ── 完成 ──────────────────────────────────────────────────────
echo ""
echo "============================================================================"
log_info "全部部署完成！"
echo ""
echo "  下一步:"
echo "    1. 下载模型到 models/ 目录"
echo "    2. 使用各框架目录下的脚本进行推理和基准测试"
echo "    3. 运行跨框架对比报告:"
echo "       cd vllm  && python benchmark.py --config config.yaml"
echo "       cd tensorrt && python benchmark.py --config config.yaml"
echo "       cd onnx && python benchmark.py --config config.yaml"
echo "       cd llamacpp/python && python benchmark.py --model ../../models/your_model.gguf"
echo "       cd openvino && python benchmark.py --config config.yaml"
echo "============================================================================"
