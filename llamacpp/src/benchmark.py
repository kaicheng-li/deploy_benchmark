"""llama.cpp Python 封装基准测试。

使用方式:
    python src/benchmark.py --config config.yaml
"""

import argparse
import csv
import json
import math
import statistics
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib import request as http_request

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from common.metrics import BenchmarkMetrics, TimingResult
from common.reporter import BenchmarkReporter
from common.config import load_config as load_yaml_config, resolve_model_path, resolve_path
from common.logger import setup_logger
from common.data_loader import DataLoader

from inference import image_to_data_url, resolve_mmproj_path

logger = setup_logger("llamacpp_benchmark")


def gpu_memory_used_mb() -> float:
    """当前 GPU 已用显存 (MB)，模型加载后读取即为 llama.cpp 占用量。"""
    out = subprocess.check_output(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"]
    ).decode()
    return sum(float(line.strip()) for line in out.splitlines() if line.strip())


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


def run_multimodal_benchmark(
    llm,
    prompts: list[str],
    image_paths: list[str],
    max_tokens: int = 128,
    warmup: int = 3,
) -> list[TimingResult]:
    """多模态基准：图片 + 文本，流式计时 TTFT / TPOT / E2E。"""
    image_parts = [
        {"type": "image_url", "image_url": {"url": image_to_data_url(path)}}
        for path in image_paths
    ]

    def complete(prompt: str, mt: int):
        content = [{"type": "text", "text": prompt}] + image_parts
        return llm.create_chat_completion(
            messages=[{"role": "user", "content": content}],
            max_tokens=mt,
            temperature=0.0,
            stream=True,
        )

    # 预热
    logger.info(f"预热 {warmup} 轮 (多模态)...")
    for prompt in prompts[:warmup]:
        for _ in complete(prompt, 32):
            pass

    logger.info(f"开始评测: {len(prompts)} 条多模态请求")
    timings: list[TimingResult] = []
    for i, prompt in enumerate(prompts):
        t0 = time.perf_counter()
        first_token_at: float | None = None
        input_tokens = 0
        output_tokens = 0
        generated = 0
        for chunk in complete(prompt, max_tokens):
            choices = chunk.get("choices") or [{}]
            delta = (choices[0].get("delta") or {}) if choices else {}
            if first_token_at is None and delta.get("content"):
                first_token_at = time.perf_counter()
            if delta.get("content"):
                generated += 1
            usage = chunk.get("usage") or {}
            if usage:
                input_tokens = usage.get("prompt_tokens", input_tokens)
                output_tokens = usage.get("completion_tokens", output_tokens)

        e2e_ms = (time.perf_counter() - t0) * 1000
        ttft_ms = (first_token_at - t0) * 1000 if first_token_at else e2e_ms
        output_tokens = output_tokens or generated
        timings.append(
            TimingResult(
                ttft=ttft_ms,
                tpot=(e2e_ms - ttft_ms) / max(output_tokens - 1, 1),
                e2e_latency=e2e_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )
        )
        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(prompts)}")

    logger.info(f"完成: {len(timings)} 条请求")
    return timings


API_PROMPT = "请用中文描述这张图片中的主要内容。"


def _api_percentile(values: list[float], percent: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = (len(ordered) - 1) * percent / 100.0
    low = int(rank)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (rank - low)


def _api_request(url: str, payload: bytes) -> dict:
    started = time.perf_counter()
    result = {"status_code": 0, "success": False, "input_tokens": -1,
              "output_tokens": 0, "ttft_ms": 0.0, "tpot_ms": 0.0,
              "e2e_ms": 0.0, "error": ""}
    first_token = None
    try:
        req = http_request.Request(url, data=payload,
                                   headers={"Content-Type": "application/json"})
        with http_request.urlopen(req, timeout=1800) as response:
            result["status_code"] = response.status
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    item = json.loads(data)
                except json.JSONDecodeError:
                    continue
                delta = ((item.get("choices") or [{}])[0].get("delta") or {})
                if delta.get("content"):
                    first_token = first_token or time.perf_counter()
                    result["output_tokens"] += 1
                usage = item.get("usage") or {}
                if usage:
                    result["input_tokens"] = usage.get("prompt_tokens", result["input_tokens"])
                    result["output_tokens"] = usage.get("completion_tokens", result["output_tokens"])
    except Exception as exc:  # noqa: BLE001
        result["error"] = str(exc)
    ended = time.perf_counter()
    result["e2e_ms"] = (ended - started) * 1000.0
    result["ttft_ms"] = ((first_token - started) * 1000.0
                          if first_token is not None else result["e2e_ms"])
    result["tpot_ms"] = (result["e2e_ms"] - result["ttft_ms"]) / max(result["output_tokens"] - 1, 1)
    result["success"] = result["status_code"] == 200 and not result["error"] and result["output_tokens"] > 0
    return result


class _ApiResourceMonitor:
    def __init__(self, pid: int):
        self.pid = pid
        self.stop_event = threading.Event()
        self.peak_rss = 0.0
        self.peak_gpu = 0.0
        self.thread = threading.Thread(target=self._sample, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self) -> tuple[float, float]:
        self.stop_event.set()
        self.thread.join()
        return self.peak_rss, self.peak_gpu

    def _sample(self):
        while not self.stop_event.is_set():
            try:
                status = Path(f"/proc/{self.pid}/status").read_text()
                self.peak_rss = max(self.peak_rss, float(status.split("VmRSS:")[1].split()[0]) / 1024.0)
            except (FileNotFoundError, IndexError, ValueError):
                pass
            try:
                output = subprocess.check_output(
                    ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL)
                for line in output.splitlines():
                    fields = [value.strip() for value in line.split(",")]
                    if len(fields) >= 2 and fields[0] == str(self.pid):
                        self.peak_gpu = max(self.peak_gpu, float(fields[1]))
            except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
                pass
            self.stop_event.wait(0.25)


def _api_run_level(url: str, payload: bytes, concurrency: int, warmup: int,
                   count: int, pid: int) -> dict:
    warm_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        list(pool.map(lambda _: _api_request(url, payload), range(warmup)))
    warmup_seconds = time.perf_counter() - warm_start
    monitor = _ApiResourceMonitor(pid)
    monitor.start()
    started = time.perf_counter()
    results = []
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(_api_request, url, payload) for _ in range(count)]
        for future in as_completed(futures):
            results.append(future.result())
    wall_seconds = time.perf_counter() - started
    peak_rss, peak_gpu = monitor.stop()
    return {"concurrency": concurrency, "requests": results,
            "total": len(results), "success": sum(item["success"] for item in results),
            "output_tokens": sum(item["output_tokens"] for item in results if item["success"]),
            "wall_seconds": wall_seconds, "warmup_seconds": warmup_seconds,
            "peak_cpu_rss_mb": peak_rss, "peak_gpu_memory_mb": peak_gpu}


def _api_write_reports(args, levels: list[dict]):
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    prefix = "llamacpp_python_cuda" if args.device == "cuda" else "llamacpp_python_cpu"
    for level in levels:
        with (output / f"{prefix}_c{level['concurrency']}_requests.csv").open("w", newline="", encoding="utf-8") as stream:
            fields = ["status_code", "success", "input_tokens", "output_tokens", "ttft_ms", "tpot_ms", "e2e_ms", "error"]
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(level["requests"])
    summary = {"model": args.model, "device": args.device, "prompt": API_PROMPT,
               "temperature": 0.0, "top_p": 1.0, "max_tokens": args.max_tokens,
               "max_model_len": args.max_model_len, "cold_start_ms": args.cold_start_ms,
               "model_load_gpu_mb": args.model_load_gpu_mb, "levels": []}
    lines = ["# llama.cpp Python API benchmark", "", f"- Model: `{args.model}`",
             f"- Device: `{args.device}`", f"- Prompt: {API_PROMPT}",
             f"- Parameters: stream=true, temperature=0, top_p=1, max_tokens={args.max_tokens}, max_model_len={args.max_model_len}",
             f"- Cold start: {args.cold_start_ms:.3f} ms", f"- Model load GPU memory: {args.model_load_gpu_mb:.3f} MiB", "",
             "| Concurrency | Success | TTFT avg/p50/p95/p99 (ms) | TPOT avg/p50/p95/p99 (ms) | E2E avg/p50/p95/p99 (ms) | req/s | tok/s | Warmup (s) | Peak GPU/RSS (MiB) |",
             "|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for level in levels:
        successful = [item for item in level["requests"] if item["success"]]
        def stats(key):
            values = [float(item[key]) for item in successful]
            return [statistics.fmean(values) if values else 0.0,
                    _api_percentile(values, 50), _api_percentile(values, 95), _api_percentile(values, 99)]
        ttft, tpot, e2e = stats("ttft_ms"), stats("tpot_ms"), stats("e2e_ms")
        req_s = level["success"] / level["wall_seconds"] if level["wall_seconds"] else 0.0
        tok_s = level["output_tokens"] / level["wall_seconds"] if level["wall_seconds"] else 0.0
        fmt = lambda values: "/".join(f"{value:.3f}" for value in values)
        lines.append(f"| {level['concurrency']} | {level['success']}/{level['total']} ({100 * level['success'] / max(level['total'], 1):.1f}%) | {fmt(ttft)} | {fmt(tpot)} | {fmt(e2e)} | {req_s:.3f} | {tok_s:.3f} | {level['warmup_seconds']:.3f} | {level['peak_gpu_memory_mb']:.3f}/{level['peak_cpu_rss_mb']:.3f} |")
        summary["levels"].append({"concurrency": level["concurrency"], "total": level["total"], "success": level["success"], "success_rate": level["success"] / max(level["total"], 1), "wall_seconds": level["wall_seconds"], "warmup_seconds": level["warmup_seconds"], "output_tokens": level["output_tokens"], "req_per_second": req_s, "tok_per_second": tok_s, "peak_gpu_memory_mb": level["peak_gpu_memory_mb"], "peak_cpu_rss_mb": level["peak_cpu_rss_mb"], "ttft_ms": dict(zip(("avg", "p50", "p95", "p99"), ttft)), "tpot_ms": dict(zip(("avg", "p50", "p95", "p99"), tpot)), "e2e_ms": dict(zip(("avg", "p50", "p95", "p99"), e2e))})
    (output / f"{prefix}_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (output / f"{prefix}_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _api_main():
    parser = argparse.ArgumentParser(description="llama.cpp OpenAI API benchmark")
    parser.add_argument("--server-pid", type=int, required=True)
    parser.add_argument("--image-base64", required=True)
    parser.add_argument("--model", default="Qwen3-VL-8B-Instruct")
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--concurrency", default="1,2,4")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--cold-start-ms", type=float, default=0.0)
    parser.add_argument("--model-load-gpu-mb", type=float, default=0.0)
    args = parser.parse_args()
    image = "".join(Path(args.image_base64).read_text(encoding="ascii").split())
    payload = json.dumps({"model": args.model, "messages": [{"role": "user", "content": [{"type": "text", "text": API_PROMPT}, {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image}"}}]}], "stream": True, "stream_options": {"include_usage": True}, "temperature": 0.0, "top_p": 1.0, "max_tokens": args.max_tokens}, ensure_ascii=False).encode()
    url = f"http://{args.host}:{args.port}/v1/chat/completions"
    levels = [_api_run_level(url, payload, int(value), args.warmup, args.requests, args.server_pid) for value in args.concurrency.split(",")]
    if not any(level["success"] for level in levels):
        raise SystemExit("all benchmark requests failed; refusing to write an empty report")
    _api_write_reports(args, levels)
    print(f"Reports: {args.output_dir}/llamacpp_python_{args.device}_summary.{{md,json}}")


def main():
    if "--server-pid" in sys.argv:
        _api_main()
        return

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
    parser.add_argument("--image", action="append", default=[], help="本地图片路径，可重复")
    parser.add_argument("--mmproj", help="视觉投影 GGUF 路径；默认取 config")
    parser.add_argument("--num-requests", type=int, help="Override benchmark.num_requests")
    parser.add_argument("--output", help="Override benchmark.output_dir")
    args = parser.parse_args()

    config, config_path = load_yaml_config(args.config)
    model_cfg = config["model"]
    runtime_cfg = config["runtime"]
    request_cfg = config["request"]
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
    max_samples = int(args.num_requests or benchmark_cfg.get("num_requests", 50))

    if not data_path.exists():
        logger.warning(f"数据文件不存在: {data_path}, 使用内置 prompts")
        prompts = [
            "Explain the attention mechanism in transformers.",
            "请用中文介绍深度学习的基本原理。",
            "Write a Python function to implement binary search.",
        ] * max(1, max_samples // 3)
    else:
        prompts = DataLoader(data_path).load_prompts(max_samples=max_samples)

    image_paths = args.image or request_cfg.get("images", [])
    llm = load_model(
        model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
    )

    logger.info(f"模型: {model_path}")
    gpu_mb = gpu_memory_used_mb() if n_gpu_layers != 0 else 0.0
    reporter = BenchmarkReporter(output_dir=output_dir)
    if image_paths:
        from llama_cpp.llama_chat_format import MTMDChatHandler

        llm.chat_handler = MTMDChatHandler(
            clip_model_path=resolve_mmproj_path(args.mmproj, model_path),
            verbose=False,
        )
        image_prompts = [
            "Describe this image in detail.",
            "What objects are visible in this image?",
            "请用中文描述这张图片的内容。",
        ] * max(1, math.ceil(max_samples / 3))
        timings = run_multimodal_benchmark(
            llm, image_prompts[:max_samples], image_paths,
            max_tokens=max_tokens, warmup=warmup,
        )
        prefix = "llamacpp_qwen3vl_benchmark"
    else:
        timings = run_benchmark(llm, prompts, max_tokens=max_tokens, warmup=warmup)
        prefix = "llamacpp_benchmark"

    if timings:
        metrics = BenchmarkMetrics.from_timings(
            timings,
            framework="llama.cpp",
            model_name=Path(model_path).stem,
            device="cuda" if n_gpu_layers != 0 else "cpu",
            gpu_memory_mb=gpu_mb,
        )
        reporter.add_result(metrics)
        reporter.print_comparison()

    saved = reporter.save_all(prefix=prefix)
    logger.info(f"报告已保存: {saved}")


if __name__ == "__main__":
    main()
