"""vLLM 吞吐与延迟基准测试。

使用 vLLM 自带的 benchmark_serving.py 或直接通过 OpenAI API 进行评测。

使用方式:
    python src/benchmark.py --host 127.0.0.1 --port 8000 --config config.yaml
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import requests

# 添加项目根路径以导入 common 模块
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_config, resolve_path
from common.metrics import BenchmarkMetrics, TimingResult
from common.reporter import BenchmarkReporter
from common.logger import setup_logger
from common.data_loader import DataLoader

logger = setup_logger("vllm_benchmark")



def send_request(
    api_url: str,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 300,
) -> Optional[TimingResult]:
    """发送单次请求并返回计时结果。"""
    payload = {
        "model": "default",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False,
    }

    try:
        start = time.perf_counter()
        resp = requests.post(
            f"{api_url}/v1/completions",
            json=payload,
            timeout=timeout,
        )
        resp.raise_for_status()
        e2e_ms = (time.perf_counter() - start) * 1000

        data = resp.json()
        usage = data.get("usage", {})
        input_tokens = usage.get("prompt_tokens", 0)
        output_tokens = usage.get("completion_tokens", 0)

        result = TimingResult(
            ttft=e2e_ms,                   # 非流式，TTFT ≈ e2e
            tpot=e2e_ms / max(output_tokens, 1),
            e2e_latency=e2e_ms,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )
        return result

    except Exception as e:
        logger.error(f"请求失败: {e}")
        return None


def run_benchmark(
    api_url: str,
    prompts: list[str],
    max_tokens: int = 512,
    max_concurrency: int = 1,
    warmup: int = 5,
    timeout: int = 300,
) -> list[TimingResult]:
    """运行基准测试，返回所有计时结果。"""
    results: list[TimingResult] = []

    # 预热
    logger.info(f"预热 {warmup} 轮...")
    for i, prompt in enumerate(prompts[:warmup]):
        send_request(api_url, prompt, max_tokens, timeout=timeout)

    logger.info(f"开始评测: {len(prompts)} prompts, 并发={max_concurrency}")
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(send_request, api_url, p, max_tokens, 0.0, timeout): i
            for i, p in enumerate(prompts)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    elapsed = time.perf_counter() - start_time
    logger.info(f"完成: {len(results)}/{len(prompts)} 成功, 耗时 {elapsed:.1f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="vLLM 基准测试")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--config", type=str, default=str(BACKEND_DIR / "config.yaml"), help="配置文件")
    parser.add_argument("--data", type=str, default="../data/prompts.txt", help="测试数据文件")
    parser.add_argument("--concurrency", type=int, default=8, help="并发数")
    parser.add_argument("--output", type=str, default="../results", help="结果输出目录")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}"

    # 加载测试数据
    data_path = Path(args.data)
    if not data_path.exists():
        # 使用示例 prompts
        logger.warning(f"数据文件不存在: {data_path}, 使用内置 prompts")
        prompts = [
            "请用中文介绍深度学习的基本原理。",
            "Explain the transformer architecture in detail.",
            "Write a Python function to sort a list of dictionaries by a given key.",
        ] * 20
    else:
        prompts = DataLoader(data_path).load_prompts()

    # 加载配置
    config, config_path = load_config(args.config)
    config["_config_path"] = config_path
    bench_cfg = config.get("benchmark", {})
    max_tokens = bench_cfg.get("max_output_len", 512)

    reporter = BenchmarkReporter(output_dir=args.output)

    # 多并发测试
    concurrency_levels = bench_cfg.get("max_concurrency", [args.concurrency])
    for concurrency in concurrency_levels:
        logger.info(f"\n{'='*50}\n  并发级别: {concurrency}\n{'='*50}")

        timings = run_benchmark(
            api_url=api_url,
            prompts=prompts,
            max_tokens=max_tokens,
            max_concurrency=concurrency,
            warmup=bench_cfg.get("warmup_requests", 5),
        )

        if timings:
            metrics = BenchmarkMetrics.from_timings(
                timings,
                framework="vLLM",
                model_name=config["model"]["id"],
                device="cuda",
            )
            reporter.add_result(metrics)
            logger.info(metrics.summary())

    # 保存报告
    saved = reporter.save_all(prefix="vllm_benchmark")
    logger.info(f"报告已保存: {json.dumps(saved, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()
