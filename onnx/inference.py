"""ONNX Runtime 推理脚本 — 支持 LLM 和 CV。

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

logger = setup_logger("onnx_inference")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_session(config: dict) -> "ort.InferenceSession":
    import onnxruntime as ort

    runtime_cfg = config["runtime"]
    onnx_dir = Path(runtime_cfg["onnx_dir"])
    model_path = str(onnx_dir / "model.onnx")

    sess_opt = ort.SessionOptions()
    opt_cfg = runtime_cfg.get("session_options", {})
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt.intra_op_num_threads = opt_cfg.get("intra_op_num_threads", 4)

    provider = runtime_cfg.get("provider", "CPUExecutionProvider")
    providers = [provider]
    if provider == "CUDAExecutionProvider":
        providers.append("CPUExecutionProvider")

    session = ort.InferenceSession(model_path, sess_opt, providers=providers)
    logger.info(f"Provider: {session.get_providers()}")
    return session


# ── LLM ───────────────────────────────────────────────────────

def run_text(session, config: dict, prompt: str) -> dict:
    from transformers import AutoTokenizer
    model_name = config["export"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)

    inputs = tokenizer(prompt, return_tensors="np")
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    feed_dict = {n: inputs[n] for n in input_names if n in inputs}

    outputs = session.run(output_names, feed_dict)
    result = dict(zip(output_names, outputs))
    return result


# ── CV ────────────────────────────────────────────────────────

def preprocess_image(image_path: str, input_shape: tuple = (3, 224, 224)) -> np.ndarray:
    """基本图像预处理：resize → normalize → (1, C, H, W)。"""
    try:
        from PIL import Image
    except ImportError:
        logger.error("需要 Pillow: pip install Pillow")
        raise

    img = Image.open(image_path).convert("RGB")
    img = img.resize(input_shape[1:], Image.BILINEAR)

    arr = np.array(img, dtype=np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    arr = (arr - mean) / std
    arr = np.transpose(arr, (2, 0, 1))       # HWC → CHW
    arr = np.expand_dims(arr, axis=0)         # → (1, C, H, W)
    return arr


IMAGENET_LABELS_CACHE = None


def load_imagenet_labels() -> dict[int, str]:
    global IMAGENET_LABELS_CACHE
    if IMAGENET_LABELS_CACHE is not None:
        return IMAGENET_LABELS_CACHE
    try:
        import requests, json
        url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
        resp = requests.get(url, timeout=10)
        IMAGENET_LABELS_CACHE = {i: name for i, name in enumerate(resp.json())}
    except Exception:
        IMAGENET_LABELS_CACHE = {}
    return IMAGENET_LABELS_CACHE


def run_image(session, config: dict, image_path: str) -> dict:
    """CV 单张图推理，返回 top-5 分类结果。"""
    export_cfg = config["export"]
    input_shape = tuple(export_cfg.get("input_shape", [1, 3, 224, 224]))

    tensor = preprocess_image(image_path, input_shape)
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name

    outputs = session.run([output_name], {input_name: tensor.astype(np.float32)})
    logits = outputs[0][0]

    top5_idx = np.argsort(logits)[-5:][::-1]
    labels = load_imagenet_labels()
    top5 = [(labels.get(int(i), f"class_{i}"), float(logits[i])) for i in top5_idx]

    return {"image": image_path, "top5": top5}


# ── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime 推理")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str, default=None, choices=["text-generation", "image-classification"])
    parser.add_argument("--prompt", type=str, default="Hello, how are you?")
    parser.add_argument("--image", type=str, default=None, help="图像路径")
    parser.add_argument("--max_tokens", type=int, default=512)
    args = parser.parse_args()

    config = load_config(args.config)
    task = args.task or config.get("task", "text-generation")

    session = create_session(config)

    if task == "text-generation":
        result = run_text(session, config, args.prompt)
        print(f"\n[输入] {args.prompt}")
        print(f"[输出 Shape] {[v.shape for v in result.values()]}")
    else:
        image_path = args.image
        if not image_path:
            logger.error("CV 任务需要 --image 参数")
            sys.exit(1)
        result = run_image(session, config, image_path)
        print(f"\n[图像] {result['image']}")
        for label, score in result["top5"]:
            print(f"  {label}: {score:.4f}")


if __name__ == "__main__":
    main()
