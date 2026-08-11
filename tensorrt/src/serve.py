"""Launch an OpenAI-compatible server via ``trtllm-serve``.

Qwen3-VL（多模态）在 TensorRT-LLM >= 1.2.0 中由 PyTorch backend 原生支持，
不需要 convert_checkpoint / trtllm-build 生成 engine，直接加载 Hugging Face
checkpoint 即可。本脚本是 ``trtllm-serve`` 的薄封装，与 vllm/server.py 风格一致。

使用:
    python src/serve.py --config config.yaml --mode qwen3vl
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_config, resolve_task_config
from common.logger import setup_logger

logger = setup_logger("tensorrt_serve")


def build_command(cfg: dict, host_override: str | None, port_override: int | None,
                  tp_override: int | None) -> list[str]:
    serve_cfg = cfg.get("serve", {})
    cmd = [
        "trtllm-serve",
        cfg["model_path"],
        "--host", host_override or serve_cfg.get("host", "0.0.0.0"),
        "--port", str(port_override or serve_cfg.get("port", 8001)),
        "--tp_size", str(tp_override or serve_cfg.get("tp_size", 1)),
    ]
    max_model_len = serve_cfg.get("max_model_len")
    if max_model_len:
        # TensorRT-LLM 1.2.1 exposes this limit as --max_seq_len.
        cmd.extend(["--max_seq_len", str(max_model_len)])
    for option in ("max_batch_size", "max_num_tokens", "free_gpu_memory_fraction"):
        value = serve_cfg.get(option)
        if value:
            cmd.extend([f"--{option}", str(value)])
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch trtllm-serve (PyTorch backend)")
    parser.add_argument("--config", default=str(BACKEND_DIR / "config.yaml"))
    parser.add_argument("--mode", choices=("qwen3vl",), help="Override config.yaml mode")
    parser.add_argument("--host", help="Override serve.host")
    parser.add_argument("--port", type=int, help="Override serve.port")
    parser.add_argument("--tp-size", type=int, help="Override serve.tp_size")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    mode, cfg = resolve_task_config(
        config,
        config_path,
        ("image",) if config["mode"] == "qwen3vl" else (),
    )
    if mode != "qwen3vl":
        parser.error("serve.py 目前只支持 mode: qwen3vl（多模态 PyTorch backend 服务）")
    if shutil.which("trtllm-serve") is None:
        raise RuntimeError(
            "trtllm-serve 未找到。请安装 TensorRT-LLM >= 1.2.0 "
            "（PyTorch backend 支持 Qwen3-VL，且需要 NVIDIA GPU + CUDA）"
        )

    cmd = build_command(cfg, args.host, args.port, args.tp_size)
    print(f"[TensorRT] 启动命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
