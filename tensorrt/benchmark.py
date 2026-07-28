"""TensorRT-LLM 基准测试。

使用方式:
    python benchmark.py --config config.yaml --data ../data/prompts.txt
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import BenchmarkMetrics, TimingResult
from common.reporter import BenchmarkReporter
from common.config import default_config_path, load_config, resolve_model_path, resolve_path
from common.logger import setup_logger
from common.data_loader import DataLoader

logger = setup_logger("tensorrt_benchmark")



def load_runner(config: dict):
    """加载 TensorRT-LLM runner。"""
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
    """执行基准测试。"""
    timings: list[TimingResult] = []
    tokenizer = runner.tokenizer

    # 预热
    logger.info(f"预热 {warmup} 轮...")
    eos_id = tokenizer.eos_token_id
    pad_id = tokenizer.pad_token_id or eos_id
    for i in range(min(warmup, len(prompts))):
        input_ids = [tokenizer.encode(prompts[i], add_special_tokens=True)]
        with runner.session as session:
            session.generate(input_ids, max_new_tokens=32, end_id=eos_id, pad_id=pad_id)

    logger.info(f"开始评测: {len(prompts)} prompts")
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

        timings.append(TimingResult(
            ttft=e2e_ms,
            tpot=e2e_ms / max(output_len, 1),
            e2e_latency=e2e_ms,
            input_tokens=input_len,
            output_tokens=max(output_len, 0),
        ))

        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(prompts)}")

    elapsed = time.perf_counter() - start_time
    logger.info(f"完成: {len(timings)} requests, 耗时 {elapsed:.1f}s")

    return timings


def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM 基准测试")
    parser.add_argument("--config", type=str, default=str(default_config_path(__file__)), help="配置文件")
    parser.add_argument("--data", type=str, default="../data/prompts.txt", help="测试数据")
    parser.add_argument("--concurrency", type=int, default=1, help="并发数(单 Runner 仅支持 1)")
    parser.add_argument("--output", type=str, default="../results", help="输出目录")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    config["_config_path"] = config_path
    bench_cfg = config.get("benchmark", {})

    data_path = Path(args.data)
    if not data_path.exists():
        logger.warning(f"数据文件不存在: {data_path}, 使用内置 prompts")
        prompts = [
            "Explain the attention mechanism in transformers.",
            "请用中文介绍深度学习的基本原理。",
            "Write a Python function to implement binary search.",
        ] * 20
    else:
        prompts = DataLoader(data_path).load_prompts()

    runner = load_runner(config)
    reporter = BenchmarkReporter(output_dir=args.output)

    timings = run_benchmark(
        runner,
        prompts,
        max_new_tokens=bench_cfg.get("max_output_len", 512),
        warmup=bench_cfg.get("warmup_requests", 10),
    )

    if timings:
        # 尝试获取 GPU 显存
        try:
            import torch
            gpu_mem = torch.cuda.max_memory_allocated() / (1024 * 1024) if torch.cuda.is_available() else 0
        except Exception:
            gpu_mem = 0

        metrics = BenchmarkMetrics.from_timings(
            timings,
            framework="TensorRT-LLM",
            model_name=config["model"]["id"],
            device="cuda",
            gpu_memory_mb=gpu_mem,
        )
        reporter.add_result(metrics)
        reporter.print_comparison()

    saved = reporter.save_all(prefix="tensorrt_benchmark")
    logger.info(f"报告已保存: {saved}")


if __name__ == "__main__":
    main()
