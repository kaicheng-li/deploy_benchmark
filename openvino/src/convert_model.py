"""ONNX → OpenVINO IR (.xml/.bin) via openvino.convert_model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_config as load_yaml_config, resolve_task_config
from common.logger import setup_logger

logger = setup_logger("openvino_convert")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(BACKEND_DIR / "config.yaml"))
    parser.add_argument(
        "--mode", choices=("vision", "qwen3", "qwen3vl"), help="Override config.yaml mode"
    )
    args = parser.parse_args()
    config, config_path = load_yaml_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    mode, cfg = resolve_task_config(
        config,
        config_path,
        ("onnx_file", "vision_onnx_file", "ir_dir", "ir_file", "vision_ir_file", "image"),
    )

    import openvino as ov

    def convert_one(onnx_path: str | Path, ir_path: str | Path) -> None:
        onnx_path = Path(onnx_path)
        ir_path = Path(ir_path)
        ir_path.parent.mkdir(parents=True, exist_ok=True)
        logger.info(f"Converting ONNX → IR: onnx={onnx_path}")
        # 同目录 <name>.onnx.data 外部权重会被自动加载
        model = ov.convert_model(str(onnx_path))
        ov.save_model(model, str(ir_path), compress_to_fp16=False)
        logger.info(f"IR saved: {ir_path}")

    if mode == "qwen3vl":
        # 两段式：视觉塔 + 文本解码器各转一份 IR
        convert_one(cfg["vision_onnx_file"], cfg["vision_ir_file"])
        convert_one(cfg["onnx_file"], cfg["ir_file"])
    else:
        convert_one(cfg["onnx_file"], cfg["ir_file"])


if __name__ == "__main__":
    main()
