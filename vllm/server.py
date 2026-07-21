"""vLLM OpenAI-compatible 推理服务启动脚本。

使用方式:
    python server.py --config config.yaml
    python server.py --model Qwen/Qwen2-7B-Instruct --port 8000
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_command(config: dict, model_override: str | None, port_override: int | None) -> list[str]:
    """构建 vLLM 启动命令。"""
    server_cfg = config.get("server", {})

    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_override or server_cfg["model"],
        "--host", server_cfg.get("host", "0.0.0.0"),
        "--port", str(port_override or server_cfg.get("port", 8000)),
        "--dtype", server_cfg.get("dtype", "auto"),
        "--max-model-len", str(server_cfg.get("max_model_len", 4096)),
        "--gpu-memory-utilization", str(server_cfg.get("gpu_memory_utilization", 0.90)),
        "--tensor-parallel-size", str(server_cfg.get("tensor_parallel_size", 1)),
        "--max-num-seqs", str(server_cfg.get("max_num_seqs", 256)),
    ]

    if server_cfg.get("trust_remote_code", False):
        cmd.append("--trust-remote-code")

    return cmd


def main():
    parser = argparse.ArgumentParser(description="启动 vLLM OpenAI 推理服务")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    parser.add_argument("--model", type=str, default=None, help="模型名称或路径 (覆盖配置文件)")
    parser.add_argument("--port", type=int, default=None, help="服务端口 (覆盖配置文件)")
    args = parser.parse_args()

    config = load_config(args.config)
    cmd = build_command(config, args.model, args.port)

    print(f"[vLLM] 启动命令: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
