"""Build the TensorRT engine selected by config.yaml mode."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from common.config import load_config, resolve_path, resolve_task_config
from common.logger import setup_logger

logger = setup_logger("tensorrt_builder")


def step1_convert_checkpoint(cfg: dict) -> None:
    build_cfg = cfg["build"]
    engine_dir = resolve_path(cfg["_config_path"], build_cfg["engine_dir"])
    checkpoint_dir = engine_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "tensorrt_llm.commands.convert_checkpoint",
        "--model_dir",
        cfg["model_path"],
        "--output_dir",
        str(checkpoint_dir),
        "--dtype",
        build_cfg["dtype"],
        "--tp_size",
        "1",
        "--pp_size",
        "1",
    ]
    if build_cfg.get("quantization") == "int8_kv_cache":
        cmd.extend(["--use_weight_only", "--weight_only_precision", "int8"])
    logger.info("Converting checkpoint -> %s", checkpoint_dir)
    subprocess.run(cmd, check=True)


def step2_build_engine(cfg: dict) -> None:
    build_cfg = cfg["build"]
    engine_dir = resolve_path(cfg["_config_path"], build_cfg["engine_dir"])
    cmd = [
        "trtllm-build",
        "--checkpoint_dir",
        str(engine_dir / "checkpoint"),
        "--output_dir",
        str(engine_dir),
        "--gemm_plugin",
        str(build_cfg.get("use_gemm_plugin", "float16")).lower(),
        "--gpt_attention_plugin",
        str(build_cfg.get("use_gpt_attention_plugin", "float16")).lower(),
        "--max_batch_size",
        str(build_cfg["max_batch_size"]),
        "--max_input_len",
        str(build_cfg["max_input_len"]),
        "--max_output_len",
        str(build_cfg["max_output_len"]),
        "--max_beam_width",
        str(build_cfg.get("max_beam_width", 1)),
    ]
    logger.info("Building TensorRT-LLM engine -> %s", engine_dir)
    subprocess.run(cmd, check=True)


def build_all(config_path: str, mode_override: str | None = None) -> None:
    config, config_file = load_config(config_path)
    if mode_override:
        config["mode"] = mode_override
    mode_key = config["mode"]
    mode, cfg = resolve_task_config(
        config,
        config_file,
        ("onnx_file", "engine_file") if mode_key == "vision" else (
            ("image",) if mode_key == "qwen3vl" else ()
        ),
    )
    if mode == "vision":
        from src.vision_engine_builder import build_engine

        print(build_engine(cfg))
        return
    if mode == "qwen3vl":
        logger.info("Qwen3-VL 由 TensorRT-LLM PyTorch backend 原生支持（需要 >= 1.2.0）")
        logger.info("该模式不需要 convert_checkpoint / trtllm-build 生成 engine")
        logger.info("直接启动 OpenAI 兼容服务: python src/serve.py --config config.yaml --mode qwen3vl")
        logger.info("或手动执行: trtllm-serve %s --tp_size %s", cfg["model_path"], cfg.get("serve", {}).get("tp_size", 1))
        return
    if mode != "llm":
        raise ValueError("config.yaml mode must be 'vision', 'llm' or 'qwen3vl'")

    cfg["_config_path"] = config_file
    logger.info("TensorRT-LLM engine build for %s", cfg["model_id"])
    step1_convert_checkpoint(cfg)
    step2_build_engine(cfg)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the configured TensorRT task")
    parser.add_argument("--config", default=str(BACKEND_DIR / "config.yaml"))
    parser.add_argument("--mode", choices=("vision", "llm", "qwen3vl"),
                        help="Override config.yaml mode")
    args = parser.parse_args()
    build_all(args.config, args.mode)


if __name__ == "__main__":
    main()
