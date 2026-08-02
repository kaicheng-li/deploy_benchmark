"""Export this project's ONNX model.

This script intentionally only knows about the two models in config.yaml:
RF-DETR Seg and Qwen3.
"""

from __future__ import annotations

import os
import argparse
import sys
from pathlib import Path
from typing import Any


BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_config as load_yaml_config, resolve_task_config
from common.logger import setup_logger


logger = setup_logger("onnx_export")


def force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def selected_config(
    config: dict[str, Any], config_path: Path
) -> tuple[str, dict[str, Any]]:
    return resolve_task_config(config, config_path, ("onnx_file", "image"))


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


def export_qwen3(cfg: dict[str, Any]) -> None:
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

    logger.info(f"Export Qwen3 model: {model_path}")
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
    logger.info("Qwen3 export finished.")


def export_qwen3vl(cfg: dict[str, Any]) -> None:
    """Export Qwen3-VL (multimodal) as a single ONNX graph.

    原理：
    - Qwen3-VL 内部图像 embedding 的合并（masked_scatter / torch.split）依赖
      token 数值，无法用动态 shape 导出，因此导出图采用静态 shape：
      固定 seq_len（文本 padding 到该长度）与固定图像尺寸。
    - mRoPE position_ids 的求解是逐 token 的 Python 控制流，无法进图，因此
      把 position_ids 作为显式输入，运行时由 qwen3vl_utils.compute_rope_index
      在 numpy 中计算。
    - 强制 eager attention，避免 SDPA/FlashAttention 无法导出。
    """
    import torch
    from torch import nn
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    from qwen3vl_utils import Qwen3VLConstants, prepare_inputs

    model_path = cfg["model_path"]
    onnx_file = Path(cfg["onnx_file"])
    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    seq_len = int(cfg.get("seq_len", 1024))
    opset = max(int(cfg.get("opset_version", 18)), 18)

    logger.info(f"Export Qwen3-VL model: {model_path}")
    logger.info(f"Output: {onnx_file} (seq_len={seq_len}, opset={opset})")

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    ).eval()
    # 视觉塔与文本塔都走 eager attention（from_pretrained 只设置顶层 config）
    model.config._attn_implementation = "eager"
    model.visual.config._attn_implementation = "eager"
    model.language_model.config._attn_implementation = "eager"

    class Qwen3VLWrapper(nn.Module):
        def __init__(self, qwen3vl: Qwen3VLForConditionalGeneration) -> None:
            super().__init__()
            self.model = qwen3vl

        def forward(
            self,
            input_ids: torch.Tensor,
            attention_mask: torch.Tensor,
            pixel_values: torch.Tensor,
            image_grid_thw: torch.Tensor,
            position_ids: torch.Tensor,
        ) -> torch.Tensor:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                pixel_values=pixel_values,
                image_grid_thw=image_grid_thw,
                position_ids=position_ids,
                use_cache=False,
            )
            return outputs.logits

    consts = Qwen3VLConstants.from_config(model.config)
    feeds = prepare_inputs(
        processor,
        tokenizer,
        cfg["image"],
        cfg.get("prompt", "Describe this image in detail."),
        seq_len,
    )

    def to_tensor(array):
        return torch.from_numpy(array)

    input_ids = to_tensor(feeds["input_ids"])
    attention_mask = to_tensor(feeds["attention_mask"])
    pixel_values = to_tensor(feeds["pixel_values"])
    image_grid_thw = to_tensor(feeds["image_grid_thw"])

    with torch.no_grad():
        position_ids, _ = model.model.get_rope_index(
            input_ids,
            image_grid_thw,
            None,
            attention_mask=attention_mask,
        )

        wrapper = Qwen3VLWrapper(model).eval()
        torch.onnx.export(
            wrapper,
            (input_ids, attention_mask, pixel_values, image_grid_thw, position_ids),
            str(onnx_file),
            input_names=[
                "input_ids",
                "attention_mask",
                "pixel_values",
                "image_grid_thw",
                "position_ids",
            ],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
        )

    import onnx

    onnx.checker.check_model(str(onnx_file))
    logger.info(
        f"Qwen3-VL export finished: {onnx_file} "
        f"(image tokens={int(image_grid_thw.prod(-1) // consts.spatial_merge_size ** 2)})"
    )


def main() -> None:
    force_utf8_stdio()
    if os.environ.get("OMP_NUM_THREADS") == "0":
        os.environ["OMP_NUM_THREADS"] = "1"
    if os.environ.get("MKL_NUM_THREADS") == "0":
        os.environ["MKL_NUM_THREADS"] = "1"
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(BACKEND_DIR / "config.yaml"))
    parser.add_argument(
        "--mode", choices=("vision", "qwen3", "qwen3vl"), help="Override config.yaml mode"
    )
    args = parser.parse_args()
    config, config_path = load_yaml_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    mode, cfg = selected_config(config, config_path)
    if mode == "vision":
        export_vision(cfg)
    elif mode == "qwen3":
        export_qwen3(cfg)
    elif mode == "qwen3vl":
        export_qwen3vl(cfg)
    else:
        raise ValueError("config.yaml mode must be 'vision', 'qwen3' or 'qwen3vl'")


if __name__ == "__main__":
    main()
