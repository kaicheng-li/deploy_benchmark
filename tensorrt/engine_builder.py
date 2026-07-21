"""TensorRT-LLM Engine 构建脚本。

将 HuggingFace 模型转换为 TensorRT-LLM engine。

使用方式:
    python engine_builder.py --config config.yaml

注意：需要安装 tensorrt_llm 包。
    pip install tensorrt_llm
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

# 添加公共模块路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logger import setup_logger

logger = setup_logger("tensorrt_builder")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def step1_convert_checkpoint(config: dict) -> None:
    """Step 1: 将 HF 模型转成 TensorRT-LLM checkpoint。"""
    build_cfg = config["build"]

    model_path = build_cfg["model_path"]
    engine_dir = Path(build_cfg["engine_dir"])
    checkpoint_dir = engine_dir / "checkpoint"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Step 1: 转换 checkpoint -> {checkpoint_dir}")

    cmd = [
        sys.executable, "-m", "tensorrt_llm.commands.convert_checkpoint",
        "--model_dir", model_path,
        "--output_dir", str(checkpoint_dir),
        "--dtype", build_cfg["dtype"],
        "--tp_size", "1",
        "--pp_size", "1",
    ]

    if build_cfg.get("quantization") == "int8_kv_cache":
        cmd.extend(["--use_weight_only", "--weight_only_precision", "int8"])

    logger.info(f"  执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("checkpoint 转换完成")


def step2_build_engine(config: dict) -> None:
    """Step 2: 从 checkpoint 构建 TensorRT engine。"""
    build_cfg = config["build"]
    engine_dir = Path(build_cfg["engine_dir"])

    logger.info(f"Step 2: 构建 TensorRT engine -> {engine_dir}")

    cmd = [
        "trtllm-build",
        "--checkpoint_dir", str(engine_dir / "checkpoint"),
        "--output_dir", str(engine_dir),
        "--gemm_plugin", build_cfg.get("use_gemm_plugin", "float16"),
        "--gpt_attention_plugin", str(build_cfg.get("use_gpt_attention_plugin", "float16")).lower(),
        "--max_batch_size", str(build_cfg["max_batch_size"]),
        "--max_input_len", str(build_cfg["max_input_len"]),
        "--max_output_len", str(build_cfg["max_output_len"]),
        "--max_beam_width", str(build_cfg.get("max_beam_width", 1)),
    ]

    logger.info(f"  执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info("Engine 构建完成")


def build_all(config_path: str) -> None:
    """完整构建流程。"""
    config = load_config(config_path)
    logger.info("=" * 60)
    logger.info("  TensorRT-LLM Engine 构建")
    logger.info(f"  模型: {config['build']['model_path']}")
    logger.info(f"  精度: {config['build']['dtype']}")
    logger.info("=" * 60)

    try:
        step1_convert_checkpoint(config)
        step2_build_engine(config)
        logger.info("\nTensorRT engine 构建成功！")
    except subprocess.CalledProcessError as e:
        logger.error(f"构建失败: {e}")
        raise


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TensorRT-LLM Engine 构建")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件路径")
    args = parser.parse_args()

    build_all(args.config)
