"""llama.cpp Python 封装基准测试。

使用方式:
    python src/benchmark.py --config config.yaml
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.metrics import BenchmarkMetrics, TimingResult
from common.reporter import BenchmarkReporter
from common.config import load_config as load_yaml_config, resolve_model_path, resolve_path
from common.logger import setup_logger
from common.data_loader import DataLoader

logger = setup_logger("llamacpp_benchmark")


def load_model(
    model_path: str,
    n_ctx: int = 4096,
    n_threads: int = 8,
    n_gpu_layers: int = -1,
):
    from llama_cpp import Llama

    return Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=False,
    )


def run_benchmark(
    llm,
    prompts: list[str],
    max_tokens: int = 512,
    warmup: int = 5,
) -> list[TimingResult]:
    """执行基准测试。"""
    timings: list[TimingResult] = []

    # 预热
    logger.info(f"预热 {warmup} 轮...")
    for i in range(min(warmup, len(prompts))):
        llm(prompts[i][:1024], max_tokens=32, temperature=0.0)

    logger.info(f"开始评测: {len(prompts)} prompts")
    start_time = time.perf_counter()

    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        output = llm(prompt[:1024], max_tokens=max_tokens, temperature=0.0)
        e2e_ms = (time.perf_counter() - t0) * 1000

        usage = output["usage"]

        timings.append(TimingResult(
            ttft=e2e_ms,
            tpot=e2e_ms / max(usage["completion_tokens"], 1),
            e2e_latency=e2e_ms,
            input_tokens=usage["prompt_tokens"],
            output_tokens=usage["completion_tokens"],
        ))

        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(prompts)}")

    elapsed = time.perf_counter() - start_time
    logger.info(f"完成: {len(timings)} requests, 耗时 {elapsed:.1f}s")

    return timings


def main():
    parser = argparse.ArgumentParser(description="llama.cpp benchmark")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[1] / "config.yaml"),
        help="YAML configuration path",
    )
    parser.add_argument("--model", help="Override the configured GGUF model path")
    parser.add_argument("--data", help="Override benchmark.prompt_file")
    parser.add_argument("--n-ctx", type=int, help="Override runtime.n_ctx")
    parser.add_argument("--n-threads", type=int, help="Override runtime.n_threads")
    parser.add_argument("--n-gpu-layers", type=int, help="Override runtime.n_gpu_layers")
    parser.add_argument("--max-tokens", type=int, help="Override benchmark.max_output_len")
    parser.add_argument("--warmup", type=int, help="Override benchmark.warmup_requests")
    parser.add_argument("--output", help="Override benchmark.output_dir")
    args = parser.parse_args()

    config, config_path = load_yaml_config(args.config)
    model_cfg = config["model"]
    runtime_cfg = config["runtime"]
    benchmark_cfg = config["benchmark"]
    model_path = args.model or resolve_model_path(model_cfg, config_path)
    data_value = args.data or benchmark_cfg.get("prompt_file")
    data_path = resolve_path(config_path, data_value) if data_value else Path()
    n_ctx = args.n_ctx if args.n_ctx is not None else runtime_cfg["n_ctx"]
    n_threads = args.n_threads if args.n_threads is not None else runtime_cfg["n_threads"]
    n_gpu_layers = args.n_gpu_layers if args.n_gpu_layers is not None else runtime_cfg["n_gpu_layers"]
    max_tokens = args.max_tokens if args.max_tokens is not None else benchmark_cfg["max_output_len"]
    warmup = args.warmup if args.warmup is not None else benchmark_cfg["warmup_requests"]
    output_dir = args.output or str(resolve_path(config_path, benchmark_cfg["output_dir"]))
    max_samples = int(benchmark_cfg.get("num_requests", 50))

    if not data_path.exists():
        logger.warning(f"数据文件不存在: {data_path}, 使用内置 prompts")
        prompts = [
            "Explain the attention mechanism in transformers.",
            "请用中文介绍深度学习的基本原理。",
            "Write a Python function to implement binary search.",
        ] * max(1, max_samples // 3)
    else:
        prompts = DataLoader(data_path).load_prompts(max_samples=max_samples)

    llm = load_model(
        model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )

    logger.info(f"模型: {model_path}")
    reporter = BenchmarkReporter(output_dir=output_dir)
    timings = run_benchmark(llm, prompts, max_tokens=max_tokens, warmup=warmup)

    if timings:
        metrics = BenchmarkMetrics.from_timings(
            timings,
            framework="llama.cpp",
            model_name=Path(model_path).stem,
            device="cuda" if n_gpu_layers != 0 else "cpu",
        )
        reporter.add_result(metrics)
        reporter.print_comparison()

    saved = reporter.save_all(prefix="llamacpp_benchmark")
    logger.info(f"报告已保存: {saved}")


if __name__ == "__main__":
    main()
