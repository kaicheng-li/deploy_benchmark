"""OpenVINO 推理脚本 — 支持 LLM 和 CV。

使用方式:
    # LLM
    python inference.py --config config.yaml --prompt "Hello world"
    # CV
    python inference.py --config config.yaml --task image-classification --image cat.jpg
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logger import setup_logger

logger = setup_logger("openvino_inference")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ov_model(config: dict):
    """加载 OpenVINO 编译模型和 tokenizer。"""
    import openvino as ov
    from transformers import AutoTokenizer

    runtime_cfg = config["runtime"]
    model_dir = Path(runtime_cfg["model_dir"])

    tokenizer = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=True)
    core = ov.Core()
    compiled = core.compile_model(
        str(model_dir / "openvino_model.xml"),
        runtime_cfg.get("device", "CPU"),
    )
    logger.info(f"设备: {runtime_cfg.get('device', 'CPU')}")
    return compiled, tokenizer


# ── LLM ───────────────────────────────────────────────────────

def generate_text(compiled, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
    generated = list(input_ids[0])

    for _ in range(max_new_tokens):
        model_input = np.array([generated[-1024:]], dtype=np.int64)
        outputs = compiled(model_input)
        logits = list(outputs.values())[0] if isinstance(outputs, dict) else outputs[0]
        next_token = int(np.argmax(logits[0, -1, :]))
        generated.append(next_token)
        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(generated, skip_special_tokens=True)


# ── CV ────────────────────────────────────────────────────────

def preprocess_image(image_path: str, size: tuple = (224, 224)) -> np.ndarray:
    from PIL import Image
    img = Image.open(image_path).convert("RGB").resize(size, Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def classify_image(compiled, image_path: str) -> list[tuple[str, float]]:
    tensor = preprocess_image(image_path)
    outputs = compiled(tensor)
    logits = list(outputs.values())[0] if isinstance(outputs, dict) else outputs[0]
    top5_idx = np.argsort(logits[0])[-5:][::-1]

    from onnx.inference import load_imagenet_labels
    labels = load_imagenet_labels()
    return [(labels.get(int(i), f"class_{i}"), float(logits[0][i])) for i in top5_idx]


# ── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenVINO 推理")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str, default=None, choices=["text-generation", "image-classification"])
    parser.add_argument("--prompt", type=str, default="Hello world")
    parser.add_argument("--image", type=str, default=None)
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    config = load_config(args.config)
    task = args.task or config.get("task", "text-generation")

    compiled, tokenizer = load_ov_model(config)

    if task == "text-generation":
        output = generate_text(compiled, tokenizer, args.prompt, args.max_tokens)
        print(f"\n[输入] {args.prompt}")
        print(f"[输出] {output}")
    else:
        if not args.image:
            logger.error("CV 任务需要 --image 参数")
            sys.exit(1)
        results = classify_image(compiled, args.image)
        print(f"\n[图像] {args.image}")
        for label, score in results:
            print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main()
