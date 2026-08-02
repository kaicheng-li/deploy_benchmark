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
    parser.add_argument("--mode", choices=("vision", "qwen3"), help="Override config.yaml mode")
    args = parser.parse_args()
    config, config_path = load_yaml_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    mode, cfg = resolve_task_config(config, config_path, ("onnx_file", "ir_dir", "ir_file", "image"))

    onnx_file = Path(cfg["onnx_file"])
    ir_file = Path(cfg["ir_file"])
    ir_dir = ir_file.parent
    model_name = ir_file.stem

    ir_dir.mkdir(parents=True, exist_ok=True)

    import openvino as ov

    logger.info(f"Converting ONNX → IR: mode={mode}, onnx={onnx_file}")
    # 同目录 <name>.onnx.data 外部权重会被自动加载
    model = ov.convert_model(str(onnx_file))
    ov.save_model(model, str(ir_file), compress_to_fp16=False)
    logger.info(f"IR saved: {ir_file}")


if __name__ == "__main__":
    main()