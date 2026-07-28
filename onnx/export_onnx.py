"""Export this project's ONNX model.

This script intentionally only knows about the two models in config.yaml:
RF-DETR Seg and Qwen.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logger import setup_logger


logger = setup_logger("onnx_export")


def force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def load_config() -> dict[str, Any]:
    with open("config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def selected_config(config: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    mode = config["mode"]
    if mode not in {"vision", "qwen"}:
        raise ValueError("config.yaml mode must be 'vision' or 'qwen'")
    return mode, config[mode]


def export_vision(cfg: dict[str, Any]) -> None:
    import torch
    from torch import nn
    from transformers import AutoImageProcessor, RfDetrForInstanceSegmentation

    class Wrapper(nn.Module):
        def __init__(self, model: RfDetrForInstanceSegmentation) -> None:
            super().__init__()
            self.model = model

        def forward(
            self,
            pixel_values: torch.Tensor,
            pixel_mask: torch.Tensor,
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            outputs = self.model(pixel_values=pixel_values, pixel_mask=pixel_mask)
            return outputs.logits, outputs.pred_boxes, outputs.pred_masks

    model_path = cfg["model_path"]
    onnx_file = Path(cfg["onnx_file"])
    onnx_file.parent.mkdir(parents=True, exist_ok=True)

    processor = AutoImageProcessor.from_pretrained(model_path)
    size = getattr(processor, "size", {}) or {}
    height = int(size.get("height", 432))
    width = int(size.get("width", 432))
    opset = max(int(cfg.get("opset_version", 18)), 18)

    logger.info(f"Export vision model: {model_path}")
    logger.info(f"Output: {onnx_file}")

    model = RfDetrForInstanceSegmentation.from_pretrained(model_path).eval()
    wrapper = Wrapper(model).eval()
    pixel_values = torch.randn(1, 3, height, width, dtype=torch.float32)
    pixel_mask = torch.ones(1, height, width, dtype=torch.long)

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (pixel_values, pixel_mask),
            str(onnx_file),
            input_names=["pixel_values", "pixel_mask"],
            output_names=["logits", "pred_boxes", "pred_masks"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=True,
        )

    import onnx

    onnx.checker.check_model(str(onnx_file))
    logger.info("Vision export finished.")


def export_qwen(cfg: dict[str, Any]) -> None:
    import torch
    from torch import nn
    from transformers import AutoModelForCausalLM, AutoTokenizer

    class Wrapper(nn.Module):
        def __init__(self, model: AutoModelForCausalLM) -> None:
            super().__init__()
            self.model = model

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
        ) -> torch.Tensor:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
            )
            return outputs.logits

    model_path = cfg["model_path"]
    onnx_file = Path(cfg["onnx_file"])
    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    seq_len = int(cfg.get("seq_len", 32))
    opset = max(int(cfg.get("opset_version", 18)), 18)

    logger.info(f"Export Qwen model: {model_path}")
    logger.info(f"Output: {onnx_file}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True).eval()
    wrapper = Wrapper(model).eval()

    encoded = tokenizer("hello", return_tensors="pt")
    input_ids = encoded["input_ids"][:, :seq_len]
    attention_mask = encoded["attention_mask"][:, :seq_len]
    if input_ids.shape[1] < seq_len:
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        pad_len = seq_len - input_ids.shape[1]
        input_ids = torch.cat(
            [input_ids, torch.full((1, pad_len), pad_id, dtype=torch.long)],
            dim=1,
        )
        attention_mask = torch.cat(
            [attention_mask, torch.zeros((1, pad_len), dtype=torch.long)],
            dim=1,
        )

    with torch.no_grad():
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask),
            str(onnx_file),
            input_names=["input_ids", "attention_mask"],
            output_names=["logits"],
            opset_version=opset,
            dynamic_axes={
                "input_ids": {0: "batch", 1: "sequence"},
                "attention_mask": {0: "batch", 1: "sequence"},
                "logits": {0: "batch", 1: "sequence"},
            },
            do_constant_folding=True,
            dynamo=False,
        )

    import onnx

    onnx.checker.check_model(str(onnx_file))
    logger.info("Qwen export finished.")


def main() -> None:
    force_utf8_stdio()
    if os.environ.get("OMP_NUM_THREADS") == "0":
        os.environ["OMP_NUM_THREADS"] = "1"
    if os.environ.get("MKL_NUM_THREADS") == "0":
        os.environ["MKL_NUM_THREADS"] = "1"
    mode, cfg = selected_config(load_config())
    if mode == "vision":
        export_vision(cfg)
    elif mode == "qwen":
        export_qwen(cfg)
    else:
        raise ValueError("config.yaml mode must be 'vision' or 'qwen'")


if __name__ == "__main__":
    main()
