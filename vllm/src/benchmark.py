"""vLLM multimodal benchmark with request-level records and resource sampling."""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any, Optional

import requests

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_DIR = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_DIR))

from common.config import load_config, resolve_model_path, resolve_path
from common.logger import setup_logger

logger = setup_logger("vllm_benchmark")


@dataclass
class RequestRecord:
    phase: str
    request_id: int
    concurrency: int
    status_code: Optional[int]
    success: bool
    input_tokens: int = 0
    output_tokens: int = 0
    ttft_ms: Optional[float] = None
    tpot_ms: Optional[float] = None
    e2e_ms: Optional[float] = None
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase,
            "request_id": self.request_id,
            "concurrency": self.concurrency,
            "status_code": self.status_code,
            "success": self.success,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "ttft_ms": self.ttft_ms,
            "tpot_ms": self.tpot_ms,
            "e2e_ms": self.e2e_ms,
            "error": self.error,
        }


def image_to_data_url(image_path: Path) -> str:
    if not image_path.is_file():
        raise FileNotFoundError(f"测试图片不存在: {image_path}")
    mime_type, _ = mimetypes.guess_type(image_path.name)
    if not mime_type or not mime_type.startswith("image/"):
        raise ValueError(f"不支持的图片格式: {image_path}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def send_request(
    api_url: str,
    prompt: str,
    image_url: str,
    model: str,
    phase: str,
    request_id: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout: int = 300,
) -> RequestRecord:
    started = time.perf_counter()
    status_code: Optional[int] = None
    try:
        payload = {
            "model": model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            }],
            "stream": True,
            "stream_options": {"include_usage": True},
            "temperature": temperature,
            "top_p": top_p,
            "max_tokens": max_tokens,
        }
        first_token_at: Optional[float] = None
        last_token_at: Optional[float] = None
        usage: dict[str, Any] = {}
        with requests.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            timeout=timeout,
            stream=True,
        ) as response:
            status_code = response.status_code
            if status_code >= 400:
                elapsed = (time.perf_counter() - started) * 1000
                return RequestRecord(
                    phase, request_id, concurrency, status_code, False,
                    e2e_ms=elapsed, error=response.text[:500],
                )
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or usage
                choices = chunk.get("choices") or []
                delta = choices[0].get("delta", {}) if choices else {}
                if delta.get("content"):
                    now = time.perf_counter()
                    first_token_at = first_token_at or now
                    last_token_at = now

        ended = time.perf_counter()
        input_tokens = int(usage.get("prompt_tokens", 0))
        output_tokens = int(usage.get("completion_tokens", 0))
        ttft_ms = ((first_token_at or ended) - started) * 1000
        tpot_ms = None
        if output_tokens > 1 and first_token_at is not None and last_token_at is not None:
            tpot_ms = (last_token_at - first_token_at) * 1000 / (output_tokens - 1)
        return RequestRecord(
            phase, request_id, concurrency, status_code, True,
            input_tokens, output_tokens, ttft_ms, tpot_ms,
            (ended - started) * 1000,
        )
    except Exception as exc:
        elapsed = (time.perf_counter() - started) * 1000
        return RequestRecord(
            phase, request_id, concurrency, status_code, False,
            e2e_ms=elapsed, error=f"{type(exc).__name__}: {exc}",
        )


def gpu_memory_mb() -> float:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
        )
        return float(output.splitlines()[0])
    except (OSError, subprocess.CalledProcessError, ValueError, IndexError):
        return 0.0


def vllm_cpu_rss_mb() -> float:
    try:
        import psutil

        roots = []
        for process in psutil.process_iter(["pid", "cmdline", "memory_info"]):
            cmdline = " ".join(process.info.get("cmdline") or [])
            if "vllm.entrypoints.openai.api_server" in cmdline:
                roots.append(process)
        processes = {p.pid: p for p in roots}
        for root in roots:
            try:
                for child in root.children(recursive=True):
                    processes[child.pid] = child
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass
        return sum(p.memory_info().rss for p in processes.values()) / (1024 * 1024)
    except (ImportError, OSError):
        return 0.0


def resources() -> tuple[float, float]:
    return gpu_memory_mb(), vllm_cpu_rss_mb()


def sample_resources(stop: threading.Event, samples: list[tuple[float, float]]) -> None:
    while not stop.wait(0.2):
        samples.append(resources())


def wait_until_ready(api_url: str, timeout: int = 900) -> float:
    started = time.perf_counter()
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            response = requests.get(f"{api_url}/v1/models", timeout=3)
            if response.ok:
                return time.perf_counter() - started
        except requests.RequestException:
            pass
        time.sleep(0.5)
    raise TimeoutError(f"vLLM 服务在 {timeout}s 内未就绪: {api_url}")


def stop_process_tree(process: subprocess.Popen) -> None:
    try:
        import psutil

        root = psutil.Process(process.pid)
        children = root.children(recursive=True)
        for child in children:
            child.terminate()
        root.terminate()
        _, alive = psutil.wait_procs(children + [root], timeout=10)
        for item in alive:
            item.kill()
    except (ImportError, OSError, subprocess.SubprocessError):
        process.terminate()
        process.wait(timeout=10)


def percentile(values: list[float], ratio: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    index = (len(values) - 1) * ratio
    lower = int(index)
    upper = min(lower + 1, len(values) - 1)
    return values[lower] + (values[upper] - values[lower]) * (index - lower)


def latency_summary(records: list[RequestRecord], field: str) -> dict[str, float]:
    values = [getattr(record, field) for record in records if getattr(record, field) is not None]
    values = [float(value) for value in values]
    return {
        "avg": mean(values) if values else 0.0,
        "p50": percentile(values, 0.50),
        "p95": percentile(values, 0.95),
        "p99": percentile(values, 0.99),
    }


def run_phase(
    api_url: str,
    prompt: str,
    image_url: str,
    model: str,
    phase: str,
    count: int,
    concurrency: int,
    max_tokens: int,
    temperature: float,
    top_p: float,
) -> tuple[list[RequestRecord], float]:
    started = time.perf_counter()
    records: list[RequestRecord] = []
    if phase == "warmup":
        for request_id in range(count):
            records.append(send_request(
                api_url, prompt, image_url, model, phase, request_id, concurrency,
                max_tokens, temperature, top_p,
            ))
    else:
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            futures = [
                executor.submit(
                    send_request, api_url, prompt, image_url, model, phase, request_id,
                    concurrency, max_tokens, temperature, top_p,
                )
                for request_id in range(count)
            ]
            for future in as_completed(futures):
                records.append(future.result())
    return records, time.perf_counter() - started


def summarize_level(
    concurrency: int,
    warmup_records: list[RequestRecord],
    records: list[RequestRecord],
    warmup_seconds: float,
    elapsed_seconds: float,
    loaded_gpu_mb: float,
    loaded_cpu_rss_mb: float,
    peak_gpu_mb: float,
    peak_cpu_rss_mb: float,
) -> dict[str, Any]:
    successful = [record for record in records if record.success]
    status_counts: dict[str, int] = {}
    for record in records:
        key = str(record.status_code) if record.status_code is not None else "exception"
        status_counts[key] = status_counts.get(key, 0) + 1
    return {
        "concurrency": concurrency,
        "formal_requests": len(records),
        "success_count": len(successful),
        "success_rate": len(successful) / len(records) if records else 0.0,
        "status_counts": status_counts,
        "warmup_requests": len(warmup_records),
        "warmup_success_count": sum(record.success for record in warmup_records),
        "warmup_seconds": warmup_seconds,
        "steady_wall_seconds": elapsed_seconds,
        "ttft_ms": latency_summary(successful, "ttft_ms"),
        "tpot_ms": latency_summary(successful, "tpot_ms"),
        "e2e_ms": latency_summary(successful, "e2e_ms"),
        "requests_per_second": len(successful) / elapsed_seconds if elapsed_seconds else 0.0,
        "tokens_per_second": (
            sum(record.output_tokens for record in successful) / elapsed_seconds
            if elapsed_seconds else 0.0
        ),
        "input_tokens": sum(record.input_tokens for record in successful),
        "output_tokens": sum(record.output_tokens for record in successful),
        "model_load_gpu_mb": loaded_gpu_mb,
        "steady_peak_gpu_mb": peak_gpu_mb,
        "model_load_cpu_rss_mb": loaded_cpu_rss_mb,
        "steady_peak_cpu_rss_mb": peak_cpu_rss_mb,
        "requests": [record.to_dict() for record in records],
        "warmup_detail": [record.to_dict() for record in warmup_records],
    }


def save_reports(
    output_dir: Path,
    config: dict[str, Any],
    image_path: Path,
    cold_start_seconds: Optional[float],
    levels: list[dict[str, Any]],
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = f"vllm_multimodal_benchmark_{timestamp}"
    all_requests = []
    for level in levels:
        all_requests.extend(level["warmup_detail"])
        all_requests.extend(level["requests"])
    report = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "framework": "vLLM",
        "model": config["model"],
        "server": config.get("server", {}),
        "benchmark": config.get("benchmark", {}),
        "image": str(image_path),
        "cold_start_seconds": cold_start_seconds,
        "levels": levels,
        "requests": all_requests,
    }
    json_path = output_dir / f"{prefix}.json"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    csv_path = output_dir / f"{prefix}.csv"
    columns = [
        "phase", "request_id", "concurrency", "status_code", "success",
        "input_tokens", "output_tokens", "ttft_ms", "tpot_ms", "e2e_ms", "error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows({key: item.get(key) for key in columns} for item in all_requests)

    md_path = output_dir / f"{prefix}.md"
    lines = [
        "# vLLM 多模态图像描述 Benchmark",
        "",
        f"生成时间（UTC）: {report['generated_at']}",
        "",
        "## 固定条件",
        "",
        f"- 模型: {config['model']['id']}",
        f"- 图片: {image_path}（启动时编码一次并复用于全部请求）",
        f"- Prompt: {config['benchmark']['prompt']}",
        "- API: /v1/chat/completions",
        f"- 参数: stream=true, temperature={config['benchmark']['temperature']}, "
        f"top_p={config['benchmark']['top_p']}, max_tokens={config['benchmark']['max_tokens']}, "
        f"max_model_len={config['server']['max_model_len']}",
        "- 每档: 预热 10 次，正式 100 次；并发 1、2、4",
        "",
        "TTFT 为请求发出到首个输出 token；TPOT 为 "
        "(最后一个 token 时间 - 首 token 时间) / (输出 token 数 - 1)；"
        "E2E 为请求发出到完整流结束。",
        "",
        "## 启动与资源",
        "",
        f"- 冷启动时间: {f'{cold_start_seconds:.3f}s' if cold_start_seconds is not None else '未由 benchmark 启动服务，未测'}",
        "",
        "## 并发汇总",
        "",
        "| 并发 | 成功率 | TTFT avg/p50/p95/p99 ms | TPOT avg/p50/p95/p99 ms | E2E avg/p50/p95/p99 ms | req/s | tok/s | 加载显存/稳态峰值 MiB | 加载 RSS/稳态峰值 MiB | 预热 s | 稳态墙钟 s |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for level in levels:
        def fmt(field: str) -> str:
            item = level[field]
            return "/".join(f"{item[key]:.2f}" for key in ("avg", "p50", "p95", "p99"))
        lines.append(
            f"| {level['concurrency']} | {level['success_count']}/{level['formal_requests']} "
            f"({level['success_rate']:.1%}) | {fmt('ttft_ms')} | {fmt('tpot_ms')} | "
            f"{fmt('e2e_ms')} | {level['requests_per_second']:.3f} | "
            f"{level['tokens_per_second']:.2f} | "
            f"{level['model_load_gpu_mb']:.0f}/{level['steady_peak_gpu_mb']:.0f} | "
            f"{level['model_load_cpu_rss_mb']:.0f}/{level['steady_peak_cpu_rss_mb']:.0f} | "
            f"{level['warmup_seconds']:.2f} | {level['steady_wall_seconds']:.2f} |"
        )
    lines.extend([
        "",
        "## HTTP 状态",
        "",
        "| 并发 | 状态码分布 |",
        "|---:|:---|",
    ])
    for level in levels:
        lines.append(f"| {level['concurrency']} | {json.dumps(level['status_counts'], ensure_ascii=False)} |")
    lines.extend([
        "",
        "请求级明细见同名 JSON/CSV；warmup 明细保留但不进入稳态延迟和吞吐计算。",
        "",
    ])
    md_path.write_text("\n".join(lines), encoding="utf-8")
    return {"markdown": str(md_path), "json": str(json_path), "csv": str(csv_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description="vLLM 多模态图像描述基准")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--config", default=str(BACKEND_DIR / "config.yaml"))
    parser.add_argument("--image")
    parser.add_argument("--prompt")
    parser.add_argument("--requests", type=int)
    parser.add_argument("--start-server", action="store_true", help="由 benchmark 启动并关闭 vLLM 服务")
    parser.add_argument("--output", default=str(PROJECT_DIR / "results"))
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    bench_cfg = config.get("benchmark", {})
    image_path = resolve_path(config_path, args.image or bench_cfg["image"])
    image_url = image_to_data_url(image_path)
    prompt = args.prompt or bench_cfg["prompt"]
    max_tokens = int(bench_cfg["max_tokens"])
    temperature = float(bench_cfg["temperature"])
    top_p = float(bench_cfg["top_p"])
    request_count = args.requests or int(bench_cfg["num_requests"])
    api_url = f"http://{args.host}:{args.port}"
    served_model = resolve_model_path(config["model"], config_path)

    server_process: Optional[subprocess.Popen] = None
    cold_start_seconds: Optional[float] = None
    try:
        if args.start_server:
            server_started = time.perf_counter()
            server_process = subprocess.Popen(
                [sys.executable, str(BACKEND_DIR / "src" / "server.py"), "--config", str(config_path)],
                cwd=BACKEND_DIR,
            )
            wait_until_ready(api_url)
            cold_start_seconds = time.perf_counter() - server_started
        else:
            wait_until_ready(api_url)

        levels = []
        for concurrency in config["benchmark"]["max_concurrency"]:
            loaded_gpu, loaded_cpu = resources()
            warmup, warmup_seconds = run_phase(
                api_url, prompt, image_url, served_model, "warmup",
                int(bench_cfg["warmup_requests"]), concurrency, max_tokens, temperature, top_p,
            )
            samples: list[tuple[float, float]] = [resources()]
            stop = threading.Event()
            sampler = threading.Thread(target=sample_resources, args=(stop, samples), daemon=True)
            sampler.start()
            formal, elapsed = run_phase(
                api_url, prompt, image_url, served_model, "formal",
                request_count, concurrency, max_tokens, temperature, top_p,
            )
            stop.set()
            sampler.join()
            peak_gpu = max([sample[0] for sample in samples] + [loaded_gpu])
            peak_cpu = max([sample[1] for sample in samples] + [loaded_cpu])
            levels.append(summarize_level(
                concurrency, warmup, formal, warmup_seconds, elapsed,
                loaded_gpu, loaded_cpu, peak_gpu, peak_cpu,
            ))
            logger.info(
                "concurrency=%d success=%d/%d req/s=%.3f tok/s=%.2f",
                concurrency, levels[-1]["success_count"], request_count,
                levels[-1]["requests_per_second"], levels[-1]["tokens_per_second"],
            )
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = (BACKEND_DIR / output_path).resolve()
        saved = save_reports(output_path, config, image_path, cold_start_seconds, levels)
        logger.info("报告已保存: %s", json.dumps(saved, ensure_ascii=False))
    finally:
        if server_process is not None:
            stop_process_tree(server_process)


if __name__ == "__main__":
    main()
