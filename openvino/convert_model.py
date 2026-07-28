"""ONNX → OpenVINO IR (.xml/.bin) via openvino.convert_model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import openvino as ov

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import default_config_path, load_config as load_yaml_config, resolve_task_config
from common.logger import setup_logger

logger = setup_logger("openvino_convert")



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path(__file__)))
    args = parser.parse_args()
    config, config_path = load_yaml_config(args.config)
    mode, cfg = resolve_task_config(config, config_path, ("onnx_file", "ir_dir", "ir_file", "image"))

    onnx_file = Path(cfg["onnx_file"])
    ir_file = Path(cfg["ir_file"])
    ir_dir = ir_file.parent
    model_name = ir_file.stem

    ir_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Converting ONNX → IR: mode={mode}, onnx={onnx_file}")
    # 同目录 <name>.onnx.data 外部权重会被自动加载
    model = ov.convert_model(str(onnx_file))
    ov.save_model(model, str(ir_file), compress_to_fp16=False)
    logger.info(f"IR saved: {ir_file}")


if __name__ == "__main__":
    main()