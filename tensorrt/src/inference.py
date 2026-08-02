"""Run the TensorRT task selected by config.yaml mode."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from common.config import load_config, resolve_path, resolve_task_config
from common.logger import setup_logger

logger = setup_logger("tensorrt_inference")


def run_llm_inference(cfg: dict, prompt: str, max_new_tokens: int = 512) -> str:
    try:
        from tensorrt_llm.runtime import ModelRunnerCpp
    except ImportError as error:
        raise RuntimeError("TensorRT-LLM is required for mode: llm") from error

    engine_dir = resolve_path(cfg["_config_path"], cfg["runtime"]["engine_dir"])
    runner = ModelRunnerCpp.from_dir(engine_dir=str(engine_dir), rank=0)
    input_ids = [runner.tokenizer.encode(prompt, add_special_tokens=True)]
    with runner.session as session:
        output_ids = session.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            end_id=runner.tokenizer.eos_token_id,
            pad_id=runner.tokenizer.pad_token_id or runner.tokenizer.eos_token_id,
        )
    return runner.tokenizer.decode(output_ids[0][0], skip_special_tokens=True)


def image_to_data_url(image_path: str) -> str:
    import base64
    import mimetypes
    from pathlib import Path

    path = Path(image_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image file type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def run_qwen3vl_inference(cfg: dict, prompt: str, image_path: str, max_new_tokens: int = 128) -> str:
    """Qwen3-VL 多模态推理：请求 trtllm-serve 的 OpenAI chat/completions 接口。"""
    import requests

    serve_cfg = cfg.get("serve", {})
    host = serve_cfg.get("host", "0.0.0.0")
    port = serve_cfg.get("port", 8001)
    url = f"http://{host}:{port}/v1/chat/completions"
    payload = {
        "model": cfg.get("model_id", "default"),
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_to_data_url(image_path)}},
                ],
            }
        ],
        "max_tokens": max_new_tokens,
        "temperature": 0.0,
    }
    response = requests.post(url, json=payload, timeout=600)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the configured TensorRT task")
    parser.add_argument("--config", default=str(BACKEND_DIR / "config.yaml"))
    parser.add_argument("--mode", choices=("vision", "llm", "qwen3vl"),
                        help="Override config.yaml mode")
    parser.add_argument("--prompt")
    parser.add_argument("--image", help="Override tasks.vision.image")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    config, config_file = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    mode, cfg = resolve_task_config(
        config,
        config_file,
        ("engine_file", "image") if config["mode"] == "vision" else (
            ("image",) if config["mode"] == "qwen3vl" else ()
        ),
    )
    if mode == "vision":
        from src.vision_inference import run_vision

        run_vision(cfg, args.image)
        return
    if mode == "qwen3vl":
        prompt = args.prompt or cfg.get("prompt", "Describe this image in detail.")
        image = args.image or cfg["image"]
        print(run_qwen3vl_inference(cfg, prompt, image, args.max_tokens))
        return
    if mode != "llm":
        raise ValueError("config.yaml mode must be 'vision', 'llm' or 'qwen3vl'")
    if not args.prompt:
        parser.error("--prompt is required when mode: llm")

    cfg["_config_path"] = config_file
    print(run_llm_inference(cfg, args.prompt, args.max_tokens))


if __name__ == "__main__":
    main()
