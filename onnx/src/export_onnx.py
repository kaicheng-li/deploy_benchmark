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
    return resolve_task_config(config, config_path, ("onnx_file", "vision_onnx_file", "image"))


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
    """Export Qwen3-VL (multimodal) as two ONNX graphs: vision tower + text decoder.

    两段式导出（详见 qwen3vl_utils.py 模块 docstring）：
    - 视觉塔：pixel_values -> image_embeds + deepstack；image_grid_thw 固化为
      Python 常量，绕开 torch.export 对数据相关数值（linspace 长度等）的限制。
    - 文本解码器：input_ids + position_ids + 视觉特征 -> logits；position_ids
      由调用方在 numpy 中计算（逐 token 控制流无法进图）。
    两者都是静态 shape（固定 seq_len 与固定图像尺寸）。
    """
    import torch
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLForConditionalGeneration

    from qwen3vl_utils import (
        Qwen3VLConstants,
        Qwen3VLDecoderWrapper,
        Qwen3VLVisionWrapper,
        build_causal_mask,
        prepare_inputs,
    )

    model_path = cfg["model_path"]
    onnx_file = Path(cfg["onnx_file"])
    vision_onnx_file = Path(cfg["vision_onnx_file"])
    onnx_file.parent.mkdir(parents=True, exist_ok=True)
    seq_len = int(cfg.get("seq_len", 1024))
    opset = max(int(cfg.get("opset_version", 18)), 18)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "fp32": torch.float32,
        "fp16": torch.float16,
    }
    dtype_name = str(cfg.get("dtype", "float32")).lower()
    if dtype_name not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_name} (use float32/fp32 or float16/fp16)")
    torch_dtype = dtype_map[dtype_name]
    device_name = str(cfg.get("device", "cpu")).lower()
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("device=cuda 但当前环境没有可用 CUDA，请改成 device: cpu")
    device = torch.device("cuda" if device_name == "cuda" else "cpu")

    logger.info(f"Export Qwen3-VL model: {model_path}")
    logger.info(
        f"Outputs: {vision_onnx_file} + {onnx_file} "
        f"(seq_len={seq_len}, opset={opset}, "
        f"dtype={dtype_name}, device={device_name})"
    )

    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_path,
        trust_remote_code=True,
        torch_dtype=torch_dtype,
        attn_implementation="eager",
    ).eval()
    model = model.to(device)
    # 视觉塔与文本塔都走 eager attention（from_pretrained 只设置顶层 config）
    model.config._attn_implementation = "eager"
    model.visual.config._attn_implementation = "eager"
    model.language_model.config._attn_implementation = "eager"

    consts = Qwen3VLConstants.from_config(model.config)
    feeds = prepare_inputs(
        processor,
        tokenizer,
        cfg["image"],
        cfg.get("prompt", "Describe this image in detail."),
        seq_len,
    )

    def to_tensor(array):
        return torch.from_numpy(array).to(device)

    input_ids = to_tensor(feeds["input_ids"])
    attention_mask_2d = to_tensor(feeds["attention_mask"])
    attention_mask_4d = to_tensor(build_causal_mask(feeds["attention_mask"]))
    pixel_values = to_tensor(feeds["pixel_values"]).to(torch_dtype)

    with torch.no_grad():
        # 视觉塔导出（grid 固化为常量）
        vision_wrapper = Qwen3VLVisionWrapper(model.visual, feeds["image_grid_thw"]).eval()
        torch.onnx.export(
            vision_wrapper,
            (pixel_values,),
            str(vision_onnx_file),
            input_names=["pixel_values"],
            output_names=["image_embeds", "deepstack_embeds"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=True,
        )
        logger.info(f"Vision tower exported: {vision_onnx_file}")

        # 示例视觉特征（形状与真实一致即可，数值不影响 decoder 图结构）
        vision_out = vision_wrapper(pixel_values)
        image_embeds = torch.zeros_like(vision_out[0])
        deepstack_embeds = torch.zeros_like(vision_out[1])

        # 文本解码器导出
        position_ids, _ = model.model.get_rope_index(
            input_ids,
            to_tensor(feeds["image_grid_thw"]),
            None,
            attention_mask=attention_mask_2d,
        )
        decoder_wrapper = Qwen3VLDecoderWrapper(model).eval()
        torch.onnx.export(
            decoder_wrapper,
            (
                input_ids,
                attention_mask_4d,
                position_ids,
                image_embeds,
                deepstack_embeds,
            ),
            str(onnx_file),
            input_names=[
                "input_ids",
                "attention_mask",
                "position_ids",
                "image_embeds",
                "deepstack_embeds",
            ],
            output_names=["logits"],
            opset_version=opset,
            do_constant_folding=True,
            dynamo=True,
        )
        logger.info(f"Text decoder exported: {onnx_file}")

    import onnx

    onnx.checker.check_model(str(onnx_file))
    onnx.checker.check_model(str(vision_onnx_file))
    visual_tokens = int(feeds["image_grid_thw"].prod(-1) // consts.spatial_merge_size**2)
    logger.info(
        f"Qwen3-VL export finished: {vision_onnx_file} + {onnx_file} "
        f"(image tokens={visual_tokens})"
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
