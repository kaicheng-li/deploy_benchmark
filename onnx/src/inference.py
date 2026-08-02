"""Run this project's ONNX model.

This script intentionally only knows about the two models in config.yaml:
RF-DETR Seg and Qwen3.
"""

from __future__ import annotations

import json
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
from PIL import Image

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_config as load_yaml_config, resolve_task_config
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


def create_session(cfg: dict[str, Any]):
    import onnxruntime as ort
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


def run_qwen3(cfg: dict[str, Any]) -> None:
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


def run_qwen3vl(cfg: dict[str, Any]) -> None:
    """Qwen3-VL 多模态推理：图片 + 文本 -> 文本。

    运行前需要先执行 export_onnx.py --mode qwen3vl 生成 qwen3vl.onnx。
    ONNX 图是静态 shape，推理时图片必须与导出时使用同一预处理尺寸
    （默认都是 onnx/0000000109.png，或同一 max_pixels/min_pixels 配置）。
    """
    from transformers import AutoProcessor, AutoTokenizer, Qwen3VLConfig

    from qwen3vl_utils import Qwen3VLConstants, generate, prepare_inputs

    session = create_session(cfg)
    processor = AutoProcessor.from_pretrained(cfg["model_path"], trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    consts = Qwen3VLConstants.from_config(Qwen3VLConfig.from_pretrained(cfg["model_path"]))

    feeds = prepare_inputs(
        processor,
        tokenizer,
        cfg["image"],
        cfg.get("prompt", "Describe this image in detail."),
        int(cfg.get("seq_len", 1024)),
    )
    text, input_tokens, output_tokens = generate(
        session,
        feeds,
        tokenizer,
        consts,
        max_new_tokens=int(cfg.get("max_new_tokens", 64)),
    )

    print(f"\n[Image] {cfg['image']}")
    print(f"[Prompt] {cfg.get('prompt', '')}")
    print(f"[Output] {text}")
    print(f"[Tokens] input={input_tokens}, output={output_tokens}")


def main() -> None:
    force_utf8_stdio()
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
        run_vision(cfg)
    elif mode == "qwen3":
        run_qwen3(cfg)
    elif mode == "qwen3vl":
        run_qwen3vl(cfg)
    else:
        raise ValueError("config.yaml mode must be 'vision', 'qwen3' or 'qwen3vl'")


if __name__ == "__main__":
    main()
