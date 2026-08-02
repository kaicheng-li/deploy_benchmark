"""Benchmark the TensorRT task selected by config.yaml mode."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))
sys.path.insert(0, str(BACKEND_DIR))

from common.config import load_config, resolve_path, resolve_task_config
from common.data_loader import DataLoader
from common.logger import setup_logger
from common.metrics import BenchmarkMetrics, TimingResult
from common.reporter import BenchmarkReporter

logger = setup_logger("tensorrt_benchmark")


def load_runner(config: dict):
    """Load TensorRT-LLM runner."""
    from tensorrt_llm.runtime import ModelRunnerCpp

    runtime_cfg = config["runtime"]
    runner = ModelRunnerCpp.from_dir(
        engine_dir=str(resolve_path(config["_config_path"], runtime_cfg["engine_dir"])),
        rank=0,
    )
    return runner


def run_benchmark(
    runner,
    prompts: list[str],
    max_new_tokens: int = 512,
    warmup: int = 10,
) -> list[TimingResult]:
    """Run TensorRT-LLM benchmark."""
    timings: list[TimingResult] = []
    tokenizer = runner.tokenizer

    logger.info("Warmup: %s requests", warmup)
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    for i in range(min(warmup, len(prompts))):
        input_ids = [tokenizer.encode(prompts[i], add_special_tokens=True)]
        with runner.session as session:
            session.generate(input_ids, max_new_tokens=32, end_id=eos_id, pad_id=pad_id)

    logger.info("Benchmarking %s prompts", len(prompts))
    start_time = time.perf_counter()

    for i, prompt in enumerate(prompts):
        input_ids = [tokenizer.encode(prompt, add_special_tokens=True)]
        input_len = len(input_ids[0])

        t0 = time.perf_counter()
        with runner.session as session:
            output_ids = session.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                end_id=eos_id,
                pad_id=pad_id,
            )
        e2e_ms = (time.perf_counter() - t0) * 1000
        output_len = len(output_ids[0][0]) - input_len if output_ids and output_ids[0] else 0

        timings.append(
            TimingResult(
                ttft=e2e_ms,
                tpot=e2e_ms / max(output_len, 1),
                e2e_latency=e2e_ms,
                input_tokens=input_len,
                output_tokens=max(output_len, 0),
            )
        )

        if (i + 1) % 10 == 0:
            logger.info("Progress: %s/%s", i + 1, len(prompts))

    elapsed = time.perf_counter() - start_time
    logger.info("Completed: %s requests, %.1fs", len(timings), elapsed)

    return timings


def send_chat_request(
    api_url: str,
    prompt: str,
    image_data_url: str | None,
    model_name: str,
    max_tokens: int = 128,
    timeout: int = 600,
) -> TimingResult | None:
    """向 trtllm-serve 发送一条多模态 chat 请求（非流式）。"""
    import requests

    content: list[dict] = [{"type": "text", "text": prompt}]
    if image_data_url:
        content.append({"type": "image_url", "image_url": {"url": image_data_url}})
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False,
    }
    try:
        start = time.perf_counter()
        response = requests.post(f"{api_url}/v1/chat/completions", json=payload, timeout=timeout)
        response.raise_for_status()
        e2e_ms = (time.perf_counter() - start) * 1000

        data = response.json()
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)
        return TimingResult(
            ttft=e2e_ms,  # 非流式, TTFT 近似 e2e
            tpot=e2e_ms / max(output_tokens, 1),
            e2e_latency=e2e_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("Request failed: %s", exc)
        return None


def run_qwen3vl_benchmark(
    api_url: str,
    prompts: list[str],
    image_data_url: str | None,
    model_name: str,
    max_tokens: int = 128,
    max_concurrency: int = 1,
    warmup: int = 3,
) -> list[TimingResult]:
    """并发多模态基准（复用 vllm/benchmark.py 的线程池模式）。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    results: list[TimingResult] = []
    for prompt in prompts[:warmup]:
        send_chat_request(api_url, prompt, image_data_url, model_name, max_tokens)

    logger.info("Benchmarking %s prompts, concurrency=%s", len(prompts), max_concurrency)
    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = [
            executor.submit(
                send_chat_request, api_url, prompt, image_data_url, model_name, max_tokens
            )
            for prompt in prompts
        ]
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)
    return results


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


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorRT benchmark")
    parser.add_argument("--config", default=str(BACKEND_DIR / "config.yaml"))
    parser.add_argument("--mode", choices=("vision", "llm", "qwen3vl"),
                        help="Override config.yaml mode")
    parser.add_argument("--data", help="Test data. Defaults to tasks.llm.benchmark.prompt_file")
    parser.add_argument("--image", help="Override tasks.vision.image")
    parser.add_argument("--host", help="Override qwen3vl serve host")
    parser.add_argument("--port", type=int, help="Override qwen3vl serve port")
    parser.add_argument("--concurrency", type=int, help="Override qwen3vl concurrency")
    parser.add_argument("--output", default="../results", help="Output directory")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    mode, cfg = resolve_task_config(
        config,
        config_path,
        ("engine_file", "image") if config["mode"] == "vision" else (
            ("image",) if config["mode"] == "qwen3vl" else ()
        ),
    )
    if mode == "vision":
        from src.vision_inference import run_vision

        run_vision(cfg, args.image)
        return
    if mode == "qwen3vl":
        bench_cfg = cfg.get("benchmark", {})
        serve_cfg = cfg.get("serve", {})
        api_url = (
            f"http://{args.host or serve_cfg.get('host', '0.0.0.0')}:"
            f"{args.port or serve_cfg.get('port', 8001)}"
        )
        image_path = args.image or cfg.get("image")
        image_data_url = image_to_data_url(image_path) if image_path else None

        data_value = args.data or bench_cfg.get("prompt_file")
        data_path = resolve_path(config_path, data_value) if data_value else Path()
        if not data_path.exists():
            logger.warning("Data file does not exist: %s. Using built-in prompts.", data_path)
            prompts = [
                "Describe this image in detail.",
                "What objects are visible in this image?",
                "请用中文描述这张图片的内容。",
            ] * 20
        else:
            prompts = DataLoader(data_path).load_prompts(
                max_samples=int(bench_cfg.get("num_requests", 20))
            )

        max_tokens = int(bench_cfg.get("max_output_len", 128))
        concurrency_levels = bench_cfg.get("max_concurrency", [args.concurrency or 1])
        reporter = BenchmarkReporter(output_dir=args.output)
        for concurrency in concurrency_levels:
            logger.info("\n%s\n  并发级别: %s\n%s", "=" * 50, concurrency, "=" * 50)
            timings = run_qwen3vl_benchmark(
                api_url,
                prompts,
                image_data_url,
                cfg.get("model_id", "default"),
                max_tokens=max_tokens,
                max_concurrency=int(concurrency),
                warmup=int(bench_cfg.get("warmup_requests", 3)),
            )
            if timings:
                metrics = BenchmarkMetrics.from_timings(
                    timings,
                    framework="TensorRT-LLM",
                    model_name=cfg["model_id"],
                    device="cuda",
                )
                reporter.add_result(metrics)
                logger.info(metrics.summary())
        saved = reporter.save_all(prefix="tensorrt_qwen3vl_benchmark")
        logger.info("Report saved: %s", saved)
        return
    if mode != "llm":
        raise ValueError("config.yaml mode must be 'vision', 'llm' or 'qwen3vl'")

    cfg["_config_path"] = config_path
    bench_cfg = cfg.get("benchmark", {})

    data_value = args.data or bench_cfg.get("prompt_file")
    data_path = resolve_path(config_path, data_value) if data_value else Path()
    if not data_path.exists():
        logger.warning("Data file does not exist: %s. Using built-in prompts.", data_path)
        prompts = [
            "Explain the attention mechanism in transformers.",
            "请用中文介绍深度学习的基本原理。",
            "Write a Python function to implement binary search.",
        ] * 20
    else:
        prompts = DataLoader(data_path).load_prompts()

    runner = load_runner(cfg)
    reporter = BenchmarkReporter(output_dir=args.output)

    timings = run_benchmark(
        runner,
        prompts,
        max_new_tokens=bench_cfg.get("max_output_len", 512),
        warmup=bench_cfg.get("warmup_requests", 10),
    )

    if timings:
        try:
            import torch

            gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        except Exception:
            gpu_mem = 0

        metrics = BenchmarkMetrics.from_timings(
            timings,
            framework="TensorRT-LLM",
            model_name=cfg["model_id"],
            device="cuda",
            gpu_memory_mb=gpu_mem,
        )
        reporter.add_result(metrics)
        reporter.print_comparison()

    saved = reporter.save_all(prefix="tensorrt_benchmark")
    logger.info("Report saved: %s", saved)


if __name__ == "__main__":
    main()
