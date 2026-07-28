"""llama.cpp Python 封装基准测试。

使用方式:
    python benchmark.py --model ../../models/qwen2-7b-instruct-q4_k_m.gguf --data ../../data/prompts.txt
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.metrics import BenchmarkMetrics, TimingResult
from common.reporter import BenchmarkReporter
from common.config import default_config_path, load_config as load_yaml_config, resolve_model_path, resolve_path
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
    parser = argparse.ArgumentParser(description="llama.cpp 基准测试")
    parser.add_argument("--model", type=str, required=True, help="GGUF 模型路径")
    parser.add_argument("--data", type=str, default="../../data/prompts.txt", help="测试数据")
    parser.add_argument("--n_ctx", type=int, default=4096)
    parser.add_argument("--n_threads", type=int, default=8)
    parser.add_argument("--n_gpu_layers", type=int, default=-1)
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--output", type=str, default="../../results")
    args = parser.parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        logger.warning(f"数据文件不存在: {data_path}, 使用内置 prompts")
        prompts = [
            "Explain the attention mechanism in transformers.",
            "请用中文介绍深度学习的基本原理。",
            "Write a Python function to implement binary search.",
        ] * 10
    else:
        prompts = DataLoader(data_path).load_prompts(max_samples=50)

    llm = load_model(
        args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )

    logger.info(f"模型: {args.model}")
    reporter = BenchmarkReporter(output_dir=args.output)

    timings = run_benchmark(llm, prompts, max_tokens=args.max_tokens, warmup=args.warmup)

    if timings:
        metrics = BenchmarkMetrics.from_timings(
            timings,
            framework="llama.cpp",
            model_name=Path(args.model).stem,
            device="cuda" if args.n_gpu_layers != 0 else "cpu",
        )
        reporter.add_result(metrics)
        reporter.print_comparison()

    saved = reporter.save_all(prefix="llamacpp_benchmark")
    logger.info(f"报告已保存: {saved}")


if __name__ == "__main__":
    main()
