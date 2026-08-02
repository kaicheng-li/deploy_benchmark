"""Benchmark this project's ONNX models.

Only two modes: vision (RF-DETR Seg) or qwen3 (Qwen3).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_config as load_yaml_config, resolve_task_config
from common.logger import setup_logger

logger = setup_logger("onnx_benchmark")


def force_utf8_stdio() -> None:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")



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


# ── vision benchmark ──────────────────────────────────────────────

def bench_vision(cfg: dict[str, Any]) -> None:
    import torch
    from transformers import AutoImageProcessor
    from PIL import Image

    session = create_session(cfg)
    processor = AutoImageProcessor.from_pretrained(cfg["model_path"])
    image = Image.open(cfg["image"]).convert("RGB")
    encoded = dict(processor(images=image, return_tensors="np"))

    feed = {}
    for item in session.get_inputs():
        value = encoded[item.name]
        feed[item.name] = value.astype(np.float32) if item.type == "tensor(float)" else value.astype(np.int64)

    warmup = int(cfg.get("warmup", 5))
    iters = int(cfg.get("iterations", 100))

    logger.info(f"Warmup {warmup} rounds...")
    for _ in range(warmup):
        session.run(None, feed)

    logger.info(f"Benchmark {iters} iterations...")
    latencies = []
    for _ in range(iters):
        t0 = time.perf_counter()
        session.run(None, feed)
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    n = len(latencies)
    print(f"\nVision benchmark ({iters} iters, {warmup} warmup)")
    print(f"  avg={np.mean(latencies):.2f}ms  p50={latencies[n//2]:.2f}ms  "
          f"p95={latencies[int(n*0.95)]:.2f}ms  p99={latencies[int(n*0.99)]:.2f}ms")
    print(f"  throughput={iters / (sum(latencies) / 1000):.1f} img/s")


# ── qwen3 benchmark ────────────────────────────────────────────────

def bench_qwen3(cfg: dict[str, Any]) -> None:
    from transformers import AutoTokenizer

    session = create_session(cfg)
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_path"], trust_remote_code=True)
    prompt = cfg.get("prompt", "Hello")
    encoded = tokenizer(prompt, return_tensors="np")
    input_ids = encoded["input_ids"].astype(np.int64)
    attention_mask = encoded["attention_mask"].astype(np.int64)

    warmup = int(cfg.get("warmup", 5))
    iters = int(cfg.get("iterations", 20))
    max_tokens = int(cfg.get("max_new_tokens", 32))

    logger.info(f"Warmup {warmup} rounds...")
    for _ in range(warmup):
        session.run(None, {"input_ids": input_ids, "attention_mask": attention_mask})

    logger.info(f"Benchmark {iters} iterations...")
    latencies = []
    for _ in range(iters):
        ids = input_ids.copy()
        attn = attention_mask.copy()
        t0 = time.perf_counter()
        for _ in range(max_tokens):
            logits = session.run(["logits"], {"input_ids": ids, "attention_mask": attn})[0]
            next_id = int(np.argmax(logits[0, -1]))
            ids = np.concatenate([ids, np.array([[next_id]], dtype=np.int64)], axis=1)
            attn = np.concatenate([attn, np.ones((1, 1), dtype=np.int64)], axis=1)
            if tokenizer.eos_token_id is not None and next_id == tokenizer.eos_token_id:
                break
        latencies.append((time.perf_counter() - t0) * 1000)

    latencies.sort()
    n = len(latencies)
    print(f"\nQwen3 benchmark ({iters} iters, {warmup} warmup, max_tokens={max_tokens})")
    print(f"  avg={np.mean(latencies):.2f}ms  p50={latencies[n//2]:.2f}ms  "
          f"p95={latencies[int(n*0.95)]:.2f}ms  p99={latencies[int(n*0.99)]:.2f}ms")
    print(f"  throughput={iters / (sum(latencies) / 1000):.2f} req/s")


# ── qwen3vl benchmark ─────────────────────────────────────────────

def bench_qwen3vl(cfg: dict[str, Any]) -> None:
    """Qwen3-VL 多模态基准：固定图片 + prompt，统计逐 token 解码的整请求延迟。"""
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
    max_tokens = int(cfg.get("max_new_tokens", 64))
    warmup = int(cfg.get("warmup", 3))
    iters = int(cfg.get("iterations", 10))

    logger.info(f"Warmup {warmup} rounds...")
    for _ in range(warmup):
        generate(session, feeds, tokenizer, consts, max_tokens)

    logger.info(f"Benchmark {iters} iterations...")
    latencies = []
    total_output_tokens = 0
    for _ in range(iters):
        t0 = time.perf_counter()
        _, _, output_tokens = generate(session, feeds, tokenizer, consts, max_tokens)
        latencies.append((time.perf_counter() - t0) * 1000)
        total_output_tokens += output_tokens

    latencies.sort()
    n = len(latencies)
    avg_ms = float(np.mean(latencies))
    print(f"\nQwen3-VL benchmark ({iters} iters, {warmup} warmup, max_tokens={max_tokens})")
    print(f"  avg={avg_ms:.2f}ms  p50={latencies[n // 2]:.2f}ms  "
          f"p95={latencies[int(n * 0.95)]:.2f}ms  p99={latencies[int(n * 0.99)]:.2f}ms")
    print(f"  throughput={iters / (sum(latencies) / 1000):.2f} req/s")
    print(f"  tokens/s={total_output_tokens / (sum(latencies) / 1000):.2f}")


# ── entry ─────────────────────────────────────────────────────────

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
    mode, cfg = resolve_task_config(config, config_path, ("onnx_file", "image"))

    if mode == "vision":
        bench_vision(cfg)
    elif mode == "qwen3":
        bench_qwen3(cfg)
    elif mode == "qwen3vl":
        bench_qwen3vl(cfg)
    else:
        raise ValueError("config.yaml mode must be 'vision', 'qwen3' or 'qwen3vl'")


if __name__ == "__main__":
    main()
