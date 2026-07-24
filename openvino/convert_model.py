"""ONNX → OpenVINO IR (.xml/.bin) via openvino.convert_model."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import openvino as ov
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logger import setup_logger

logger = setup_logger("openvino_convert")


def load_config() -> dict[str, Any]:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    config = load_config()
    mode = config["mode"]
    cfg = config[mode]

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