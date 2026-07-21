"""基准测试指标计算模块。

支持两类任务：
- text-generation: TTFT / TPOT / tok/s
- image-*: 单图推理延迟 / images/s
"""

import statistics
from dataclasses import dataclass, field
from typing import Optional, Literal

TaskType = Literal["text-generation", "image-classification", "object-detection"]


@dataclass
class TimingResult:
    """单次推理的计时数据。

    LLM 任务: 填充 ttft / tpot / e2e_latency / input_tokens / output_tokens
    CV 任务:   填充 e2e_latency / input_shape / output_shape
    """

    # ── 通用 ──
    e2e_latency: Optional[float] = None   # 端到端延迟 (ms)

    # ── LLM 专用 ──
    ttft: Optional[float] = None          # Time To First Token (ms)
    tpot: Optional[float] = None          # Time Per Output Token (ms)
    input_tokens: int = 0
    output_tokens: int = 0

    # ── CV 专用 ──
    preprocess_ms: float = 0.0            # 预处理耗时 (ms)
    inference_ms: float = 0.0             # 纯推理耗时 (ms)
    postprocess_ms: float = 0.0           # 后处理耗时 (ms)
    input_shape: Optional[tuple] = None   # e.g. (1, 3, 224, 224)
    output_shape: Optional[tuple] = None

    @property
    def tokens_per_second(self) -> float:
        if self.e2e_latency and self.output_tokens:
            return self.output_tokens / (self.e2e_latency / 1000)
        return 0.0


@dataclass
class BenchmarkMetrics:
    """聚合后的基准测试指标，按 task_type 切换关注字段。"""

    framework: str = ""
    model_name: str = ""
    device: str = "cuda"
    task_type: TaskType = "text-generation"

    # ── LLM 延迟 (ms) ──
    ttft_avg: float = 0.0
    ttft_p50: float = 0.0
    ttft_p95: float = 0.0
    ttft_p99: float = 0.0

    tpot_avg: float = 0.0
    tpot_p50: float = 0.0
    tpot_p95: float = 0.0
    tpot_p99: float = 0.0

    # ── 通用延迟 (ms) ──
    e2e_avg: float = 0.0
    e2e_p50: float = 0.0
    e2e_p95: float = 0.0
    e2e_p99: float = 0.0

    # ── CV 细分延迟 (ms) ──
    preprocess_avg: float = 0.0
    inference_avg: float = 0.0
    postprocess_avg: float = 0.0

    # ── 吞吐 ──
    tokens_per_second: float = 0.0
    requests_per_second: float = 0.0
    images_per_second: float = 0.0

    # ── 资源 ──
    gpu_memory_mb: float = 0.0
    cpu_memory_mb: float = 0.0

    # ── 元信息 ──
    total_requests: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    elapsed_seconds: float = 0.0

    raw_timings: list[TimingResult] = field(default_factory=list)

    # ───────────────────────────────────────────────────────────
    # Factory: 从一组 TimingResult 计算聚合指标
    # ───────────────────────────────────────────────────────────
    @classmethod
    def from_timings(
        cls,
        timings: list[TimingResult],
        framework: str = "",
        model_name: str = "",
        device: str = "cuda",
        task_type: TaskType = "text-generation",
        gpu_memory_mb: float = 0.0,
        cpu_memory_mb: float = 0.0,
    ) -> "BenchmarkMetrics":
        if not timings:
            return cls(framework=framework, model_name=model_name, device=device, task_type=task_type)

        e2e_values = sorted([t.e2e_latency for t in timings if t.e2e_latency is not None])

        def _pcts(vals: list[float]) -> tuple[float, float, float, float]:
            if not vals:
                return 0.0, 0.0, 0.0, 0.0
            n = len(vals)
            return (
                statistics.mean(vals),
                vals[int(n * 0.50)] if n > 1 else vals[0],
                vals[int(n * 0.95)] if n > 1 else vals[0],
                vals[int(n * 0.99)] if n > 1 else vals[0],
            )

        e2e_avg, e2e_p50, e2e_p95, e2e_p99 = _pcts(e2e_values)

        total_time = max(e2e_values) / 1000 if e2e_values else 1.0

        inst = cls(
            framework=framework,
            model_name=model_name,
            device=device,
            task_type=task_type,
            e2e_avg=e2e_avg, e2e_p50=e2e_p50, e2e_p95=e2e_p95, e2e_p99=e2e_p99,
            requests_per_second=len(timings) / total_time if total_time > 0 else 0.0,
            gpu_memory_mb=gpu_memory_mb,
            cpu_memory_mb=cpu_memory_mb,
            total_requests=len(timings),
            elapsed_seconds=total_time,
            raw_timings=timings,
        )

        if task_type == "text-generation":
            ttft_vals = sorted([t.ttft for t in timings if t.ttft is not None])
            tpot_vals = sorted([t.tpot for t in timings if t.tpot is not None])
            inst.ttft_avg, inst.ttft_p50, inst.ttft_p95, inst.ttft_p99 = _pcts(ttft_vals)
            inst.tpot_avg, inst.tpot_p50, inst.tpot_p95, inst.tpot_p99 = _pcts(tpot_vals)
            inst.total_input_tokens = sum(t.input_tokens for t in timings)
            inst.total_output_tokens = sum(t.output_tokens for t in timings)
            inst.tokens_per_second = inst.total_output_tokens / total_time if total_time > 0 else 0.0
        else:
            # CV 任务
            prep_vals  = sorted([t.preprocess_ms for t in timings if t.preprocess_ms > 0])
            infer_vals = sorted([t.inference_ms for t in timings if t.inference_ms > 0])
            post_vals  = sorted([t.postprocess_ms for t in timings if t.postprocess_ms > 0])
            if prep_vals:
                inst.preprocess_avg = statistics.mean(prep_vals)
            if infer_vals:
                inst.inference_avg = statistics.mean(infer_vals)
            if post_vals:
                inst.postprocess_avg = statistics.mean(post_vals)
            inst.images_per_second = len(timings) / total_time if total_time > 0 else 0.0

        return inst

    # ───────────────────────────────────────────────────────────
    # 输出
    # ───────────────────────────────────────────────────────────
    def summary(self) -> str:
        """按 task_type 生成单行摘要。"""
        if self.task_type == "text-generation":
            return (
                f"[{self.framework}] {self.model_name} | "
                f"TTFT avg={self.ttft_avg:.1f}ms p95={self.ttft_p95:.1f}ms | "
                f"TPOT avg={self.tpot_avg:.1f}ms p95={self.tpot_p95:.1f}ms | "
                f"tok/s={self.tokens_per_second:.1f} | "
                f"req/s={self.requests_per_second:.2f} | "
                f"GPU={self.gpu_memory_mb:.0f}MB"
            )
        else:
            return (
                f"[{self.framework}] {self.model_name} ({self.task_type}) | "
                f"e2e avg={self.e2e_avg:.1f}ms p95={self.e2e_p95:.1f}ms | "
                f"infer={self.inference_avg:.1f}ms | "
                f"img/s={self.images_per_second:.1f} | "
                f"GPU={self.gpu_memory_mb:.0f}MB"
            )

    def to_dict(self) -> dict:
        base = {
            "framework": self.framework,
            "model_name": self.model_name,
            "device": self.device,
            "task_type": self.task_type,
            "latency": {
                "e2e": {"avg": self.e2e_avg, "p50": self.e2e_p50, "p95": self.e2e_p95, "p99": self.e2e_p99},
            },
            "resource": {
                "gpu_memory_mb": self.gpu_memory_mb,
                "cpu_memory_mb": self.cpu_memory_mb,
            },
            "stats": {
                "total_requests": self.total_requests,
                "elapsed_seconds": self.elapsed_seconds,
            },
        }

        if self.task_type == "text-generation":
            base["latency"].update({
                "ttft": {"avg": self.ttft_avg, "p50": self.ttft_p50, "p95": self.ttft_p95, "p99": self.ttft_p99},
                "tpot": {"avg": self.tpot_avg, "p50": self.tpot_p50, "p95": self.tpot_p95, "p99": self.tpot_p99},
            })
            base["throughput"] = {
                "tokens_per_second": self.tokens_per_second,
                "requests_per_second": self.requests_per_second,
            }
            base["stats"].update({
                "total_input_tokens": self.total_input_tokens,
                "total_output_tokens": self.total_output_tokens,
            })
        else:
            base["latency"].update({
                "preprocess_avg": self.preprocess_avg,
                "inference_avg": self.inference_avg,
                "postprocess_avg": self.postprocess_avg,
            })
            base["throughput"] = {
                "images_per_second": self.images_per_second,
                "requests_per_second": self.requests_per_second,
            }

        return base
