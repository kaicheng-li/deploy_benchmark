"""vLLM 吞吐与延迟基准测试。

使用 vLLM 自带的 benchmark_serving.py 或直接通过 OpenAI API 进行评测。

使用方式:
    python src/benchmark.py --host 127.0.0.1 --port 8000 --config config.yaml
"""

import argparse
import base64
import json
import mimetypes
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Optional

import requests

# 添加项目根路径以导入 common 模块
BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_config, resolve_model_path, resolve_path
from common.metrics import BenchmarkMetrics, TimingResult
from common.reporter import BenchmarkReporter
from common.logger import setup_logger
from common.data_loader import DataLoader

logger = setup_logger("vllm_benchmark")



def send_request(
    api_url: str,
    prompt: str,
    image_url: str,
    model: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
    timeout: int = 300,
) -> Optional[TimingResult]:
    """发送一次图文流式请求，并记录 TTFT、TPOT 与完整时延。"""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        }],
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }

    try:
        start = time.perf_counter()
        first_token_at: Optional[float] = None
        usage: dict = {}
        resp = requests.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
            stream=True,
        )
        resp.raise_for_status()
        e2e_ms = (time.perf_counter() - start) * 1000

        for line in resp.iter_lines(decode_unicode=True):
            if not line or not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            chunk = json.loads(data)
            usage = chunk.get("usage") or usage
            choices = chunk.get("choices") or []
            if choices and choices[0].get("delta", {}).get("content") and first_token_at is None:
                first_token_at = time.perf_counter()

        e2e_ms = (time.perf_counter() - start) * 1000
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        ttft_ms = ((first_token_at or time.perf_counter()) - start) * 1000

        result = TimingResult(
            ttft=ttft_ms,
            tpot=(e2e_ms - ttft_ms) / max(output_tokens - 1, 1),
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
    prompt: str,
    image_url: str,
    model: str,
    num_requests: int,
    max_tokens: int = 512,
    max_concurrency: int = 1,
    warmup: int = 5,
    timeout: int = 300,
) -> tuple[list[TimingResult], float]:
    """运行基准测试，返回所有计时结果。"""
    results: list[TimingResult] = []

    # 预热
    logger.info(f"预热 {warmup} 轮...")
    for _ in range(warmup):
        send_request(api_url, prompt, image_url, model, max_tokens, timeout=timeout)

    logger.info(f"开始图文评测: {num_requests} requests, 并发={max_concurrency}")
    start_time = time.perf_counter()

    with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
        futures = {
            executor.submit(
                send_request, api_url, prompt, image_url, model, max_tokens, 0.0, timeout
            ): i
            for i in range(num_requests)
        }
        for future in as_completed(futures):
            result = future.result()
            if result is not None:
                results.append(result)

    elapsed = time.perf_counter() - start_time
    logger.info(f"完成: {len(results)}/{num_requests} 成功, 耗时 {elapsed:.1f}s")

    return results, elapsed


def image_to_data_url(image_path: Path) -> str:
    """Return a local image as an OpenAI-compatible data URL."""
    if not image_path.is_file():
        raise FileNotFoundError(f"测试图片不存在: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if mime_type is None or not mime_type.startswith("image/"):
        raise ValueError(f"不支持的图片格式: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def gpu_memory_mb() -> float:
    """Read current total VRAM use from nvidia-smi when it is available."""
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(output.splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return 0.0


def main():
    parser = argparse.ArgumentParser(description="vLLM 基准测试")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="服务地址")
    parser.add_argument("--port", type=int, default=8000, help="服务端口")
    parser.add_argument("--config", type=str, default=str(BACKEND_DIR / "config.yaml"), help="配置文件")
    parser.add_argument("--image", help="本地测试图片路径，覆盖配置")
    parser.add_argument("--prompt", help="图像提问，覆盖配置")
    parser.add_argument("--requests", type=int, help="每个并发级别的请求数，覆盖配置")
    parser.add_argument("--output", type=str, default="../results", help="结果输出目录")
    args = parser.parse_args()

    api_url = f"http://{args.host}:{args.port}"

    # 加载配置
    config, config_path = load_config(args.config)
    config["_config_path"] = config_path
    bench_cfg = config.get("benchmark", {})
    max_tokens = bench_cfg.get("max_output_len", 512)
    prompt = args.prompt or bench_cfg.get("prompt", "图片描述了什么？")
    image_path = resolve_path(config_path, args.image or bench_cfg["image"])
    image_url = image_to_data_url(image_path)
    num_requests = args.requests or bench_cfg.get("num_requests", 20)
    served_model = resolve_model_path(config["model"], config_path)
    model_name = config["model"]["id"]

    reporter = BenchmarkReporter(output_dir=args.output)

    # 多并发测试
    concurrency_levels = bench_cfg.get("max_concurrency", [1])
    for concurrency in concurrency_levels:
        logger.info(f"\n{'='*50}\n  并发级别: {concurrency}\n{'='*50}")

        timings, elapsed = run_benchmark(
            api_url=api_url,
            prompt=prompt,
            image_url=image_url,
            model=served_model,
            num_requests=num_requests,
            max_tokens=max_tokens,
            max_concurrency=concurrency,
            warmup=bench_cfg.get("warmup_requests", 5),
        )

        if timings:
            metrics = BenchmarkMetrics.from_timings(
                timings,
                framework="vLLM",
                model_name=f"{model_name} (concurrency={concurrency})",
                device="cuda",
                task_type="text-generation",
                gpu_memory_mb=gpu_memory_mb(),
            )
            metrics.elapsed_seconds = elapsed
            metrics.requests_per_second = len(timings) / elapsed if elapsed else 0.0
            metrics.tokens_per_second = metrics.total_output_tokens / elapsed if elapsed else 0.0
            reporter.add_result(metrics)
            logger.info(metrics.summary())

    # 保存报告
    saved = reporter.save_all(prefix="vllm_benchmark")
    logger.info(f"报告已保存: {json.dumps(saved, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    main()

