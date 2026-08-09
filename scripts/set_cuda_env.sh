#!/usr/bin/env bash
# ============================================================================
# 为 onnxruntime-gpu (CUDA 12 build) 设置 CUDA 运行库路径。
# 使用方式:  source scripts/set_cuda_env.sh   （GPU 运行前 source 一次）
# ============================================================================

PYTHON="${PYTHON:-python3}"
SP="$($PYTHON -c 'import site; print(site.getsitepackages()[0])' 2>/dev/null)"
if [ -z "$SP" ]; then
    echo "[set_cuda_env] 无法定位 site-packages" >&2
    return 1 2>/dev/null || exit 1
fi

NV="$SP/nvidia"
export LD_LIBRARY_PATH="$NV/cuda_runtime/lib:$NV/cublas/lib:$NV/cudnn/lib:$NV/cufft/lib:$NV/cusparse/lib:$NV/cusolver/lib:$NV/curand/lib:${LD_LIBRARY_PATH:-}"
echo "[set_cuda_env] LD_LIBRARY_PATH 已设置（onnxruntime-gpu CUDA EP 使用）"
