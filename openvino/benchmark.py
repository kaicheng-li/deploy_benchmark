"""OpenVINO 基准测试 — 支持 LLM 和 CV。

使用方式:
    # LLM
    python benchmark.py --config config.yaml --task text-generation --data ../data/prompts.txt
    # CV
    python benchmark.py --config config.yaml --task image-classification --data ../data/images
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import BenchmarkMetrics, TimingResult, TaskType
from common.reporter import BenchmarkReporter
from common.logger import setup_logger
from common.data_loader import DataLoader

logger = setup_logger("openvino_benchmark")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_ov_model(config: dict):
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
    return compiled, tokenizer


# ── LLM benchmark ──────────────────────────────────────────────

def run_text_benchmark(compiled, tokenizer, prompts: list[str],
                       warmup: int = 10) -> list[TimingResult]:
    timings = []

    logger.info(f"预热 {warmup} 轮...")
    for i in range(min(warmup, len(prompts))):
        input_ids = tokenizer(prompts[i][:1024], return_tensors="np")["input_ids"]
        compiled(input_ids)

    logger.info(f"开始评测: {len(prompts)} prompts")
    for i, prompt in enumerate(prompts):
        input_ids = tokenizer(prompt, return_tensors="np")["input_ids"]
        input_len = input_ids.shape[1]

        t0 = time.perf_counter()
        outputs = compiled(input_ids)
        e2e_ms = (time.perf_counter() - t0) * 1000

        result = list(outputs.values())[0] if isinstance(outputs, dict) else outputs[0]
        output_len = result.shape[1] if len(result.shape) > 1 else 1

        timings.append(TimingResult(
            ttft=e2e_ms, tpot=e2e_ms / max(output_len, 1),
            e2e_latency=e2e_ms, input_tokens=input_len, output_tokens=output_len,
        ))
        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(prompts)}")
    return timings


# ── CV benchmark ───────────────────────────────────────────────

def preprocess_batch(image_paths: list[Path], size: tuple = (224, 224)) -> np.ndarray:
    from PIL import Image
    batch = []
    for p in image_paths:
        img = Image.open(p).convert("RGB").resize(size, Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        arr = np.transpose(arr, (2, 0, 1))
        batch.append(arr)
    return np.stack(batch, axis=0)


def run_cv_benchmark(compiled, config: dict, images: list[Path],
                     warmup: int = 10, batch_size: int = 1) -> list[TimingResult]:
    input_shape = tuple(config["convert"].get("input_shape", [1, 3, 224, 224]))
    size = (input_shape[2], input_shape[1])  # H, W
    timings = []

    logger.info(f"预热 {warmup} 轮...")
    for i in range(min(warmup // max(batch_size, 1), len(images) // max(batch_size, 1))):
        batch_imgs = images[i * batch_size:(i + 1) * batch_size]
        tensor = preprocess_batch(batch_imgs, size)
        compiled(tensor)

    logger.info(f"开始评测: {len(images)} images, batch={batch_size}")
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        if len(batch_imgs) < batch_size and i + batch_size > len(images):
            break

        t_pre0 = time.perf_counter()
        tensor = preprocess_batch(batch_imgs, size)
        t_pre = (time.perf_counter() - t_pre0) * 1000

        t_inf0 = time.perf_counter()
        outputs = compiled(tensor)
        t_inf = (time.perf_counter() - t_inf0) * 1000

        e2e_ms = t_pre + t_inf
        result = list(outputs.values())[0] if isinstance(outputs, dict) else outputs[0]

        for j in range(len(batch_imgs)):
            timings.append(TimingResult(
                e2e_latency=e2e_ms / batch_size,
                preprocess_ms=t_pre / batch_size,
                inference_ms=t_inf / batch_size,
                postprocess_ms=0.0,
                output_shape=result.shape,
            ))

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  进度: {min(i+batch_size, len(images))}/{len(images)}")
    return timings


# ── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="OpenVINO 基准测试")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str, default=None,
                        choices=["text-generation", "image-classification", "object-detection"])
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", type=str, default="../results")
    args = parser.parse_args()

    config = load_config(args.config)
    task: TaskType = args.task or config.get("task", "text-generation")  # type: ignore[assignment]
    bench_cfg = config.get("benchmark", {})
    warmup = bench_cfg.get("warmup_requests", 10)
    device = config["runtime"].get("device", "CPU")

    reporter = BenchmarkReporter(output_dir=args.output)

    compiled, tokenizer = load_ov_model(config)
    model_name = Path(config["convert"]["model_path"]).name

    if task == "text-generation":
        data_path = Path(args.data or bench_cfg.get("prompt_file", "../data/prompts.txt"))
        prompts = _load_or_fallback_text(data_path)
        timings = run_text_benchmark(compiled, tokenizer, prompts, warmup=warmup)
    else:
        data_path = Path(args.data or bench_cfg.get("image_dir", "../data/images"))
        images = _load_or_fallback_images(data_path)
        batch_size = args.batch_size or bench_cfg.get("batch_size", 1)
        timings = run_cv_benchmark(compiled, config, images, warmup=warmup, batch_size=batch_size)

    if timings:
        metrics = BenchmarkMetrics.from_timings(
            timings, framework="OpenVINO", model_name=model_name,
            device=device.lower(), task_type=task,
        )
        reporter.add_result(metrics)
        reporter.print_comparison()

    saved = reporter.save_all(prefix="openvino_benchmark")
    logger.info(f"报告已保存: {saved}")


def _load_or_fallback_text(data_path: Path) -> list[str]:
    if data_path.exists():
        return DataLoader(data_path).load_prompts()
    logger.warning(f"数据文件不存在: {data_path}, 使用内置 prompts")
    return ["Explain the attention mechanism."] * 20


def _load_or_fallback_images(data_path: Path) -> list[Path]:
    if data_path.exists():
        return DataLoader(data_path).load_images()
    logger.error(f"图像数据不存在: {data_path}")
    sys.exit(1)


if __name__ == "__main__":
    main()
