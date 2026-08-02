"""Run this project's OpenVINO model.

Only two modes: vision (RF-DETR Seg) or qwen3 (Qwen3).
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_config as load_yaml_config, resolve_task_config
from common.logger import setup_logger

logger = setup_logger("openvino_inference")


def force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")



def load_labels(model_path: str | Path) -> dict[int, str]:
    with open(Path(model_path) / "config.json", "r", encoding="utf-8") as f:
        raw = json.load(f).get("id2label", {})
    return {int(key): str(value) for key, value in raw.items()}


# ── vision ────────────────────────────────────────────────────────

def run_vision(cfg: dict[str, Any]) -> None:
    import openvino as ov
    import torch
    from PIL import Image
    from transformers import AutoImageProcessor

    core = ov.Core()
    compiled = core.compile_model(cfg["ir_file"], cfg.get("device", "CPU"))

    image = Image.open(cfg["image"]).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(cfg["model_path"])
    encoded = dict(processor(images=image, return_tensors="np"))

    feed = {}
    for key, value in encoded.items():
        if key in {"pixel_values"}:
            feed[key] = value.astype(np.float32)
        else:
            feed[key] = value.astype(np.int64)

    result = compiled(feed)
    outputs = SimpleNamespace(
        logits=torch.from_numpy(result["logits"]),
        pred_boxes=torch.from_numpy(result["pred_boxes"]),
        pred_masks=torch.from_numpy(result["pred_masks"]),
    )
    out = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
        threshold=float(cfg.get("threshold", 0.5)),
        mask_threshold=0.0,
    )[0]

    labels = load_labels(cfg["model_path"])
    segments = out["segments_info"]
    print(f"Image: {cfg['image']}")
    print(f"Segments: {len(segments)}")
    for item in segments[:20]:
        label_id = int(item["label_id"])
        label = labels.get(label_id, f"class_{label_id}")
        print(f"  id={item['id']} label={label} score={float(item['score']):.4f}")


# ── qwen3 ──────────────────────────────────────────────────────────

def run_qwen3(cfg: dict[str, Any]) -> None:
    import openvino as ov
    from transformers import AutoTokenizer

    core = ov.Core()
    compiled = core.compile_model(cfg["ir_file"], cfg.get("device", "CPU"))

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    encoded = tokenizer(cfg["prompt"], return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    for _ in range(int(cfg.get("max_new_tokens", 32))):
        logits = compiled({"input_ids": input_ids, "attention_mask": attention_mask})["logits"]
        next_id = int(np.argmax(logits[0, -1]))
        input_ids = np.concatenate([input_ids, np.array([[next_id]], dtype=np.int64)], axis=1)
        attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))


# ── entry ─────────────────────────────────────────────────────────

def main() -> None:
    force_utf8_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(BACKEND_DIR / "config.yaml"))
    parser.add_argument("--mode", choices=("vision", "qwen3"), help="Override config.yaml mode")
    args = parser.parse_args()
    config, config_path = load_yaml_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    mode, cfg = resolve_task_config(config, config_path, ("onnx_file", "ir_dir", "ir_file", "image"))

    if mode == "vision":
        run_vision(cfg)
    elif mode == "qwen3":
        run_qwen3(cfg)
    else:
        raise ValueError("config.yaml mode must be 'vision' or 'qwen3'")


if __name__ == "__main__":
    main()
