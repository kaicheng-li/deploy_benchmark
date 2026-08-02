"""llama.cpp Python inference script."""

from __future__ import annotations

import argparse
import base64
import mimetypes
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.config import load_config as load_yaml_config, resolve_model_path as resolve_config_model_path, resolve_path
from common.logger import setup_logger

logger = setup_logger("llamacpp_inference")


def resolve_gguf_model_path(model_path: str) -> str:
    path = Path(model_path).expanduser()
    if path.is_file() and path.suffix.lower() == ".gguf":
        return str(path)
    if path.is_dir():
        candidates = [
            item for item in sorted(path.glob("*.gguf"))
            if not item.name.lower().startswith("mmproj-")
        ]
        if len(candidates) == 1:
            return str(candidates[0])
        if not candidates:
            raise ValueError(f"No model GGUF found in: {path}")
        raise ValueError(f"Multiple model GGUF files found in {path}; pass one file path")
    raise ValueError(f"Model path must be a GGUF file or directory: {path}")


def load_model(
    model_path: str,
    n_ctx: int = 4096,
    n_threads: int = 8,
    n_gpu_layers: int = -1,
    verbose: bool = False,
):
    model_path = resolve_gguf_model_path(model_path)
    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    logger.info("Model loaded: %s", model_path)
    return llm


def resolve_mmproj_path(mmproj_path: str | None, model_path: str) -> str:
    if mmproj_path is not None:
        path = Path(mmproj_path).expanduser()
    else:
        model_file = Path(resolve_gguf_model_path(model_path))
        candidates = sorted(model_file.parent.glob("mmproj-*.gguf"))
        if len(candidates) != 1:
            raise ValueError(
                "Pass --mmproj when the model directory does not contain exactly one mmproj-*.gguf file"
            )
        path = candidates[0]
    if not path.is_file() or path.suffix.lower() != ".gguf":
        raise ValueError(f"Vision projector must be a GGUF file: {path}")
    return str(path)


def image_to_data_url(image_path: str) -> str:
    path = Path(image_path).expanduser()
    if not path.is_file():
        raise ValueError(f"Image file does not exist: {path}")
    mime_type, _ = mimetypes.guess_type(path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"Unsupported image file type: {path}")
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def augment_prompt(prompt: str, data: str | None) -> str:
    if data is None:
        return prompt
    path = Path(data).expanduser()
    value = path.read_text(encoding="utf-8") if path.is_file() else data
    return f"{prompt}\n\nData:\n{value}"


def generate(
    llm,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    image_paths: list[str] | None = None,
) -> dict:
    content: str | list[dict[str, object]] = prompt
    if image_paths:
        content = [{"type": "text", "text": prompt}]
        content.extend(
            {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
            for path in image_paths
        )
    output = llm.create_chat_completion(
        messages=[{"role": "user", "content": content}],
        max_tokens=max_tokens,
        temperature=temperature,
    )
    choice = output["choices"][0]["message"]
    usage = output["usage"]
    return {
        "text": choice.get("content") or "",
        "input_tokens": usage["prompt_tokens"],
        "output_tokens": usage["completion_tokens"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="llama.cpp inference")
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "config.yaml"))
    parser.add_argument("--model", help="Override YAML GGUF model path")
    parser.add_argument("--prompt", help="Override YAML prompt")
    parser.add_argument("--data", help="Append local UTF-8 data file or literal text")
    parser.add_argument("--image", action="append", default=[], help="Local image path; can be repeated")
    parser.add_argument("--mmproj", help="Vision projector GGUF path; defaults to YAML when images are used")
    parser.add_argument("--n-ctx", type=int, default=None, help="Override runtime.n_ctx")
    parser.add_argument("--n-threads", type=int, default=None, help="Override runtime.n_threads")
    parser.add_argument("--n-gpu-layers", type=int, default=None, help="Override runtime.n_gpu_layers")
    parser.add_argument("--max-tokens", type=int, default=None, help="Override request.max_tokens")
    parser.add_argument("--temperature", type=float, default=None)
    args = parser.parse_args()

    config, config_path = load_yaml_config(args.config)
    model_cfg = config["model"]
    runtime_cfg = config["runtime"]
    request_cfg = config["request"]
    model_path = args.model or resolve_config_model_path(model_cfg, config_path)
    prompt_value = args.prompt or request_cfg["prompt"]
    image_paths = args.image or request_cfg.get("images", [])
    data_value = args.data if args.data is not None else request_cfg.get("data")
    mmproj_path = args.mmproj or (
        str(resolve_path(config_path, model_cfg["mmproj_path"])) if image_paths else None
    )
    n_ctx = args.n_ctx if args.n_ctx is not None else runtime_cfg["n_ctx"]
    n_threads = args.n_threads if args.n_threads is not None else runtime_cfg["n_threads"]
    n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None else runtime_cfg["n_gpu_layers"]
    max_tokens = args.max_tokens if args.max_tokens is not None else request_cfg["max_tokens"]
    temperature = args.temperature if args.temperature is not None else request_cfg["temperature"]
    if os.environ.get("OMP_NUM_THREADS") == "0":
        os.environ["OMP_NUM_THREADS"] = "1"

    llm = load_model(
        model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )

    if image_paths:
        from llama_cpp.llama_chat_format import MTMDChatHandler

        llm.chat_handler = MTMDChatHandler(
            clip_model_path=resolve_mmproj_path(mmproj_path, model_path),
            verbose=False,
        )
    prompt = augment_prompt(prompt_value, data_value)
    result = generate(llm, prompt, max_tokens, temperature, image_paths)

    print(f"\n[Input] {prompt}")
    print(f"[Output] {result['text']}")
    print(f"[Tokens] input={result['input_tokens']}, output={result['output_tokens']}")


if __name__ == "__main__":
    main()
