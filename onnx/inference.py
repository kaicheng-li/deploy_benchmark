"""Run this project's ONNX model.

This script intentionally only knows about the two models in config.yaml:
RF-DETR Seg and Qwen.
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.config import default_config_path, load_config as load_yaml_config, resolve_task_config
from common.logger import setup_logger


logger = setup_logger("onnx_inference")


def force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")


def selected_config(
    config: dict[str, Any], config_path: Path
) -> tuple[str, dict[str, Any]]:
    return resolve_task_config(config, config_path, ("onnx_file", "image"))


def create_session(cfg: dict[str, Any]) -> ort.InferenceSession:
    options = ort.SessionOptions()
    options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    options.intra_op_num_threads = int(cfg.get("threads", 4))

    provider = cfg.get("provider", "CPUExecutionProvider")
    providers = [provider]
    if provider == "CUDAExecutionProvider":
        providers.append("CPUExecutionProvider")

    logger.info(f"Load ONNX model: {cfg['onnx_file']}")
    return ort.InferenceSession(cfg["onnx_file"], options, providers=providers)


def load_labels(model_path: str | Path) -> dict[int, str]:
    with open(Path(model_path) / "config.json", "r", encoding="utf-8") as f:
        raw = json.load(f).get("id2label", {})
    return {int(key): str(value) for key, value in raw.items()}


def run_vision(cfg: dict[str, Any]) -> None:
    import torch
    from transformers import AutoImageProcessor

    session = create_session(cfg)
    image = Image.open(cfg["image"]).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(cfg["model_path"])
    encoded = dict(processor(images=image, return_tensors="np"))

    feed = {}
    for item in session.get_inputs():
        value = encoded[item.name]
        if item.type == "tensor(float)":
            value = value.astype(np.float32)
        elif item.type == "tensor(int64)":
            value = value.astype(np.int64)
        feed[item.name] = value

    output_names = [item.name for item in session.get_outputs()]
    output_map = dict(zip(output_names, session.run(output_names, feed)))
    outputs = SimpleNamespace(
        logits=torch.from_numpy(output_map["logits"]),
        pred_boxes=torch.from_numpy(output_map["pred_boxes"]),
        pred_masks=torch.from_numpy(output_map["pred_masks"]),
    )
    result = processor.post_process_instance_segmentation(
        outputs,
        target_sizes=[image.size[::-1]],
        threshold=float(cfg.get("threshold", 0.5)),
        mask_threshold=0.0,
    )[0]

    labels = load_labels(cfg["model_path"])
    segments = result["segments_info"]
    print(f"Image: {cfg['image']}")
    print(f"Segments: {len(segments)}")
    for item in segments[:20]:
        label_id = int(item["label_id"])
        label = labels.get(label_id, f"class_{label_id}")
        print(f"  id={item['id']} label={label} score={float(item['score']):.4f}")


def run_qwen(cfg: dict[str, Any]) -> None:
    from transformers import AutoTokenizer

    session = create_session(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    encoded = tokenizer(cfg["prompt"], return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    for _ in range(int(cfg.get("max_new_tokens", 32))):
        feed = {"input_ids": input_ids, "attention_mask": attention_mask}
        logits = session.run(["logits"], feed)[0]
        next_id = int(np.argmax(logits[0, -1]))
        input_ids = np.concatenate([input_ids, np.array([[next_id]], dtype=np.int64)], axis=1)
        attention_mask = np.concatenate([attention_mask, np.ones((1, 1), dtype=np.int64)], axis=1)
        if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
            break

    print(tokenizer.decode(input_ids[0], skip_special_tokens=True))


def main() -> None:
    force_utf8_stdio()
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(default_config_path(__file__)))
    args = parser.parse_args()
    config, config_path = load_yaml_config(args.config)
    mode, cfg = selected_config(config, config_path)
    if mode == "vision":
        run_vision(cfg)
    elif mode == "qwen":
        run_qwen(cfg)
    else:
        raise ValueError("config.yaml mode must be 'vision' or 'qwen'")


if __name__ == "__main__":
    main()
