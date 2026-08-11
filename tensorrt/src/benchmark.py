"""Benchmark the TensorRT task selected by config.yaml mode."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import signal
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean

import psutil
import requests

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


@dataclass
class StreamRequestMetric:
    """One streamed /v1/chat/completions request measured by the client."""

    status_code: int | None
    success: bool
    input_tokens: int | None
    output_tokens: int | None
    ttft_ms: float | None
    tpot_ms: float | None
    e2e_ms: float
    error: str | None = None


def _percentile(values: list[float], quantile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (position - lower)


def _distribution(values: list[float]) -> dict[str, float | None]:
    return {
        "avg": mean(values) if values else None,
        "p50": _percentile(values, 0.50),
        "p95": _percentile(values, 0.95),
        "p99": _percentile(values, 0.99),
    }


def _gpu_memory_mib() -> int | None:
    try:
        output = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
        return int(output.splitlines()[0])
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError, IndexError):
        return None


def _process_tree_rss_mib(pid: int | None) -> float | None:
    if pid is None:
        return None
    try:
        root = psutil.Process(pid)
        processes = [root, *root.children(recursive=True)]
        return sum(process.memory_info().rss for process in processes if process.is_running()) / (1024**2)
    except (psutil.Error, ProcessLookupError):
        return None


class ResourceSampler:
    """Samples the TensorRT-LLM process tree during measured requests only."""

    def __init__(self, server_pid: int | None, interval_seconds: float) -> None:
        self.server_pid = server_pid
        self.interval_seconds = interval_seconds
        self.gpu_samples: list[int] = []
        self.rss_samples: list[float] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._sample, daemon=True)

    def _sample(self) -> None:
        while not self._stop.is_set():
            gpu = _gpu_memory_mib()
            rss = _process_tree_rss_mib(self.server_pid)
            if gpu is not None:
                self.gpu_samples.append(gpu)
            if rss is not None:
                self.rss_samples.append(rss)
            self._stop.wait(self.interval_seconds)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        self._thread.join(timeout=max(1.0, self.interval_seconds * 2))

    def peak(self) -> dict[str, float | int | None]:
        return {
            "gpu_memory_peak_mib": max(self.gpu_samples) if self.gpu_samples else None,
            "cpu_rss_peak_mib": max(self.rss_samples) if self.rss_samples else None,
        }


def _stream_chat_request(
    api_url: str,
    prompt: str,
    image_data_url: str,
    model_name: str,
    max_tokens: int,
    temperature: float,
    top_p: float,
    timeout_seconds: int,
) -> StreamRequestMetric:
    """Measure TTFT, TPOT and E2E from an SSE chat-completions response."""
    payload = {
        "model": model_name,
        "messages": [{"role": "user", "content": [
            {"type": "text", "text": prompt},
            {"type": "image_url", "image_url": {"url": image_data_url}},
        ]}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": temperature,
        "top_p": top_p,
        "max_tokens": max_tokens,
    }
    started = time.perf_counter()
    status_code: int | None = None
    first_token_at: float | None = None
    last_token_at: float | None = None
    usage: dict = {}
    saw_content = False
    try:
        with requests.post(
            f"{api_url}/v1/chat/completions",
            json=payload,
            stream=True,
            timeout=(30, timeout_seconds),
        ) as response:
            status_code = response.status_code
            response.raise_for_status()
            for line in response.iter_lines(decode_unicode=True):
                if not line or not line.startswith("data:"):
                    continue
                payload_line = line[5:].strip()
                if payload_line == "[DONE]":
                    continue
                event = json.loads(payload_line)
                if isinstance(event.get("usage"), dict):
                    usage = event["usage"]
                choices = event.get("choices") or []
                if not choices:
                    continue
                content = (choices[0].get("delta") or {}).get("content")
                if not content:
                    continue
                now = time.perf_counter()
                first_token_at = first_token_at or now
                last_token_at = now
                saw_content = True

        e2e_ms = (time.perf_counter() - started) * 1000
        input_tokens = usage.get("prompt_tokens")
        output_tokens = usage.get("completion_tokens")
        input_tokens = input_tokens if isinstance(input_tokens, int) else None
        output_tokens = output_tokens if isinstance(output_tokens, int) else None
        ttft_ms = (first_token_at - started) * 1000 if first_token_at is not None else None
        tpot_ms = None
        if output_tokens and output_tokens > 1 and first_token_at and last_token_at:
            tpot_ms = (last_token_at - first_token_at) * 1000 / (output_tokens - 1)
        return StreamRequestMetric(
            status_code=status_code,
            success=saw_content,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            e2e_ms=e2e_ms,
            error=None if saw_content else "No streamed content was received",
        )
    except Exception as error:  # noqa: BLE001 - one failed request must not abort the batch
        return StreamRequestMetric(
            status_code=status_code,
            success=False,
            input_tokens=None,
            output_tokens=None,
            ttft_ms=None,
            tpot_ms=None,
            e2e_ms=(time.perf_counter() - started) * 1000,
            error=str(error),
        )


def _run_streamed_requests(
    count: int, concurrency: int, request_kwargs: dict
) -> tuple[list[StreamRequestMetric], float]:
    started = time.perf_counter()
    results: list[StreamRequestMetric] = []
    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [executor.submit(_stream_chat_request, **request_kwargs) for _ in range(count)]
        for index, future in enumerate(as_completed(futures), start=1):
            try:
                results.append(future.result())
            except Exception as error:  # noqa: BLE001
                logger.error("Unhandled request future error: %s", error)
                results.append(StreamRequestMetric(
                    status_code=None,
                    success=False,
                    input_tokens=None,
                    output_tokens=None,
                    ttft_ms=None,
                    tpot_ms=None,
                    e2e_ms=0.0,
                    error=str(error),
                ))
            if index % 10 == 0 or index == count:
                logger.info("Progress: %s/%s", index, count)
    return results, (time.perf_counter() - started) * 1000


def _wait_for_server(api_url: str, timeout_seconds: int, log_path: Path) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        try:
            if requests.get(f"{api_url}/v1/models", timeout=5).status_code == 200:
                return
        except requests.RequestException:
            pass
        time.sleep(1)
    detail = ""
    if log_path.is_file():
        detail = "\n".join(log_path.read_text(encoding="utf-8", errors="replace").splitlines()[-40:])
    raise RuntimeError(f"trtllm-serve did not become ready within {timeout_seconds}s\n{detail}")


def _start_server(cfg: dict, host: str, port: int, output_dir: Path) -> tuple[subprocess.Popen, float, Path]:
    from src.serve import build_command

    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_path = output_dir / f"tensorrt_qwen3vl_server_{timestamp}.log"
    log_file = log_path.open("w", encoding="utf-8")
    started = time.perf_counter()
    process = subprocess.Popen(
        build_command(cfg, host, port, None),
        stdout=log_file,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        text=True,
    )
    log_file.close()
    return process, started, log_path


def _stop_server(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    for sig in (signal.SIGTERM, signal.SIGKILL):
        try:
            os.killpg(process.pid, sig)
            process.wait(timeout=30)
            return
        except (ProcessLookupError, subprocess.TimeoutExpired):
            pass


def _write_stream_report(
    output_dir: Path, report: dict, prefix_name: str | None = None
) -> dict[str, str]:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    prefix = output_dir / (prefix_name or f"tensorrt_qwen3vl_stream_benchmark_{timestamp}")
    json_path = prefix.with_suffix(".json")
    csv_path = prefix.with_suffix(".csv")
    md_path = prefix.with_suffix(".md")
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    with csv_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "concurrency", "request_index", "status_code", "success", "input_tokens",
                "output_tokens", "ttft_ms", "tpot_ms", "e2e_ms", "error",
            ],
        )
        writer.writeheader()
        for level in report["levels"]:
            for index, metric in enumerate(level["requests"], start=1):
                writer.writerow({"concurrency": level["concurrency"], "request_index": index, **metric})

    lines = [
        "# TensorRT-LLM Qwen3-VL Streaming Benchmark",
        "",
        f"- Model: {report['model']}",
        f"- Image: {report['image']}",
        f"- Prompt: {report['prompt']}",
        "- API: `/v1/chat/completions`, `stream=true`",
        f"- Parameters: temperature={report['parameters']['temperature']}, top_p={report['parameters']['top_p']}, max_tokens={report['parameters']['max_tokens']}, max_model_len={report['parameters']['max_model_len']}",
        f"- Workload: {report['parameters']['warmup_requests']} warmups and {report['parameters']['num_requests']} measured requests per concurrency level",
        "",
        "## Startup And Resources",
        "",
        f"- Cold start: {report['cold_start']['duration_ms'] / 1000:.2f} s" if report["cold_start"]["duration_ms"] is not None else "- Cold start: not measured (existing server)",
        f"- Model-load GPU memory: {report['cold_start']['gpu_memory_after_ready_mib']} MiB",
        "",
        "## Steady-State Results",
        "",
        "| Concurrency | Success | TTFT avg/P50/P95/P99 (ms) | TPOT avg/P50/P95/P99 (ms) | E2E avg/P50/P95/P99 (ms) | req/s | tok/s | GPU peak MiB | CPU RSS peak MiB |",
        "|---:|---:|---|---|---|---:|---:|---:|---:|",
    ]
    for level in report["levels"]:
        measured = level["measured"]
        resources = level["resources"]
        format_dist = lambda values: "/".join(
            "-" if values[key] is None else f"{values[key]:.2f}" for key in ("avg", "p50", "p95", "p99")
        )
        tok_s = measured["tokens_per_second"]
        cpu_peak = resources["cpu_rss_peak_mib"]
        lines.append(
            f"| {level['concurrency']} | {measured['successes']}/{measured['requests']} ({measured['success_rate']:.2%}) | "
            f"{format_dist(measured['ttft_ms'])} | {format_dist(measured['tpot_ms'])} | {format_dist(measured['e2e_ms'])} | "
            f"{measured['requests_per_second']:.3f} | {'-' if tok_s is None else f'{tok_s:.3f}'} | "
            f"{resources['gpu_memory_peak_mib'] or '-'} | {'-' if cpu_peak is None else f'{cpu_peak:.1f}'} |"
        )
    lines.extend([
        "",
        "TTFT is measured from request start to the first non-empty streamed content chunk.",
        "TPOT is `(last token timestamp - first token timestamp) / (output tokens - 1)`.",
        "Cold start and warmup are excluded from steady-state latency and throughput.",
    ])
    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"json": str(json_path), "csv": str(csv_path), "markdown": str(md_path)}


def run_qwen3vl_stream_benchmark(
    cfg: dict,
    api_url: str,
    image_data_url: str,
    output_dir: Path,
    server_pid: int | None,
    cold_start: dict,
) -> dict[str, str]:
    """Run the fixed stream=true workload and save raw/summary reports."""
    bench_cfg = cfg["benchmark"]
    request_kwargs = {
        "api_url": api_url,
        "prompt": cfg["prompt"],
        "image_data_url": image_data_url,
        "model_name": cfg["model_id"],
        "max_tokens": int(bench_cfg["max_tokens"]),
        "temperature": float(bench_cfg["temperature"]),
        "top_p": float(bench_cfg["top_p"]),
        "timeout_seconds": int(bench_cfg["request_timeout_seconds"]),
    }
    levels: list[dict] = []
    for concurrency in bench_cfg["max_concurrency"]:
        logger.info("Concurrency %s warmup: %s requests", concurrency, bench_cfg["warmup_requests"])
        warmup, warmup_wall_ms = _run_streamed_requests(
            int(bench_cfg["warmup_requests"]), int(concurrency), request_kwargs
        )
        logger.info("Concurrency %s measured: %s requests", concurrency, bench_cfg["num_requests"])
        try:
            sampler = ResourceSampler(server_pid, float(bench_cfg["resource_sample_interval_seconds"]))
            sampler.start()
            try:
                measured, wall_ms = _run_streamed_requests(
                    int(bench_cfg["num_requests"]), int(concurrency), request_kwargs
                )
            finally:
                sampler.stop()
        except Exception as error:
            logger.exception("Concurrency %s failed", concurrency)
            partial = {
                "status": "failed",
                "error": str(error),
                "completed_concurrency": [level["concurrency"] for level in levels],
                "levels": levels,
            }
            _write_stream_report(output_dir, partial, "tensorrt_qwen3vl_stream_checkpoint")
            raise
        successes = [metric for metric in measured if metric.success]
        output_tokens = [metric.output_tokens for metric in successes if metric.output_tokens is not None]
        levels.append({
            "concurrency": int(concurrency),
            "warmup": {
                "requests": len(warmup),
                "successes": sum(metric.success for metric in warmup),
                "wall_ms": warmup_wall_ms,
            },
            "measured": {
                "requests": len(measured),
                "successes": len(successes),
                "success_rate": len(successes) / len(measured) if measured else 0.0,
                "wall_ms": wall_ms,
                "total_input_tokens": sum(metric.input_tokens for metric in successes if metric.input_tokens is not None),
                "total_output_tokens": sum(output_tokens),
                "ttft_ms": _distribution([metric.ttft_ms for metric in successes if metric.ttft_ms is not None]),
                "tpot_ms": _distribution([metric.tpot_ms for metric in successes if metric.tpot_ms is not None]),
                "e2e_ms": _distribution([metric.e2e_ms for metric in successes]),
                "requests_per_second": len(successes) / (wall_ms / 1000) if wall_ms else 0.0,
                "tokens_per_second": sum(output_tokens) / (wall_ms / 1000) if wall_ms and output_tokens else None,
            },
            "resources": sampler.peak(),
            "requests": [asdict(metric) for metric in measured],
        })
        checkpoint = {
            "status": "in_progress",
            "completed_concurrency": [level["concurrency"] for level in levels],
            "model": cfg["model_id"],
            "image": cfg["image"],
            "prompt": cfg["prompt"],
            "api_url": api_url,
            "stream": True,
            "parameters": {
                "temperature": request_kwargs["temperature"],
                "top_p": request_kwargs["top_p"],
                "max_tokens": request_kwargs["max_tokens"],
                "max_model_len": int(cfg["serve"]["max_model_len"]),
                "warmup_requests": int(bench_cfg["warmup_requests"]),
                "num_requests": int(bench_cfg["num_requests"]),
                "concurrency": bench_cfg["max_concurrency"],
            },
            "cold_start": cold_start,
            "levels": levels,
        }
        _write_stream_report(output_dir, checkpoint, "tensorrt_qwen3vl_stream_checkpoint")
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "framework": "TensorRT-LLM PyTorch backend",
        "model": cfg["model_id"],
        "image": cfg["image"],
        "prompt": cfg["prompt"],
        "api_url": api_url,
        "stream": True,
        "parameters": {
            "temperature": request_kwargs["temperature"],
            "top_p": request_kwargs["top_p"],
            "max_tokens": request_kwargs["max_tokens"],
            "max_model_len": int(cfg["serve"]["max_model_len"]),
            "warmup_requests": int(bench_cfg["warmup_requests"]),
            "num_requests": int(bench_cfg["num_requests"]),
            "concurrency": bench_cfg["max_concurrency"],
        },
        "cold_start": {
            **cold_start,
        },
        "levels": levels,
    }
    return _write_stream_report(output_dir, report)


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
    parser.add_argument(
        "--start-server",
        action="store_true",
        help="Start trtllm-serve and record cold-start time before the benchmark",
    )
    parser.add_argument("--server-pid", type=int, help="Existing trtllm-serve parent PID for CPU RSS")
    parser.add_argument("--keep-server", action="store_true", help="Keep a server started by this command")
    parser.add_argument("--output", default="../results", help="Output directory")
    args = parser.parse_args()

    config, config_path = load_config(args.config)
    if args.mode:
        config["mode"] = args.mode
    if args.concurrency:
        config["tasks"][config.get("mode", "vision")]["benchmark"] = dict(
            config["tasks"][config.get("mode", "vision")].get("benchmark", {})
        )
        config["tasks"][config.get("mode", "vision")]["benchmark"]["max_concurrency"] = [
            args.concurrency
        ]
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
        listen_host = args.host or serve_cfg.get("host", "0.0.0.0")
        request_host = "127.0.0.1" if listen_host == "0.0.0.0" else listen_host
        port = args.port or serve_cfg.get("port", 8001)
        api_url = f"http://{request_host}:{port}"
        image_path = args.image or cfg.get("image")
        if not image_path:
            raise ValueError("tasks.qwen3vl.image is required")
        image_data_url = image_to_data_url(image_path)
        output_dir = Path(args.output).expanduser().resolve()
        output_dir.mkdir(parents=True, exist_ok=True)
        server_process: subprocess.Popen | None = None
        server_pid = args.server_pid
        server_log: Path | None = None
        started_at: float | None = None
        try:
            if args.start_server:
                server_process, started_at, server_log = _start_server(cfg, listen_host, int(port), output_dir)
                server_pid = server_process.pid
                _wait_for_server(
                    api_url, int(bench_cfg.get("request_timeout_seconds", 600)), server_log
                )
            else:
                _wait_for_server(api_url, 20, Path())
            cold_start = {
                "duration_ms": (time.perf_counter() - started_at) * 1000 if started_at else None,
                "gpu_memory_after_ready_mib": _gpu_memory_mib(),
                "cpu_rss_after_ready_mib": _process_tree_rss_mib(server_pid),
                "server_log": str(server_log) if server_log else None,
            }
            saved = run_qwen3vl_stream_benchmark(
                cfg, api_url, image_data_url, output_dir, server_pid, cold_start
            )
            logger.info("Stream benchmark report saved: %s", saved)
        finally:
            if not args.keep_server:
                _stop_server(server_process)
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
