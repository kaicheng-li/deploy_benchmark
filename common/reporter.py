"""基准测试报告生成器。

支持输出 Markdown 表格、JSON、CSV 格式的对比报告。
"""

import json
import csv
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Optional

from .metrics import BenchmarkMetrics


class BenchmarkReporter:
    """汇总多个框架的评测结果并生成对比报告。"""

    def __init__(self, output_dir: str | Path = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results: list[BenchmarkMetrics] = []

    def add_result(self, metrics: BenchmarkMetrics) -> None:
        """添加一个评测结果。"""
        self.results.append(metrics)

    def to_markdown(self) -> str:
        """生成 Markdown 对比表格。"""
        if not self.results:
            return "暂无评测数据。"

        buf = StringIO()
        buf.write("# 模型部署评测报告\n\n")
        buf.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        # 延迟对比
        buf.write("## 延迟对比 (ms)\n\n")
        buf.write("| 框架 | 模型 | TTFT avg | TTFT p95 | TPOT avg | TPOT p95 | E2E avg | E2E p95 |\n")
        buf.write("|------|------|----------|----------|----------|----------|---------|--------|\n")
        for r in self.results:
            buf.write(
                f"| {r.framework} | {r.model_name} | {r.ttft_avg:.1f} | {r.ttft_p95:.1f} | "
                f"{r.tpot_avg:.1f} | {r.tpot_p95:.1f} | {r.e2e_avg:.1f} | {r.e2e_p95:.1f} |\n"
            )

        # 吞吐对比
        buf.write("\n## 吞吐对比\n\n")
        buf.write("| 框架 | 模型 | tok/s | req/s | GPU 显存 (MB) | 总请求数 |\n")
        buf.write("|------|------|-------|-------|---------------|----------|\n")
        for r in self.results:
            buf.write(
                f"| {r.framework} | {r.model_name} | {r.tokens_per_second:.1f} | "
                f"{r.requests_per_second:.2f} | {r.gpu_memory_mb:.0f} | {r.total_requests} |\n"
            )

        return buf.getvalue()

    def to_json(self) -> str:
        """导出为 JSON。"""
        return json.dumps(
            [r.to_dict() for r in self.results], ensure_ascii=False, indent=2
        )

    def to_csv(self) -> str:
        """导出为 CSV。"""
        if not self.results:
            return ""

        buf = StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "framework", "model_name", "device",
            "ttft_avg", "ttft_p50", "ttft_p95", "ttft_p99",
            "tpot_avg", "tpot_p50", "tpot_p95", "tpot_p99",
            "e2e_avg", "e2e_p50", "e2e_p95", "e2e_p99",
            "tokens_per_second", "requests_per_second",
            "gpu_memory_mb", "cpu_memory_mb",
            "total_requests", "total_input_tokens", "total_output_tokens",
        ])
        for r in self.results:
            writer.writerow([
                r.framework, r.model_name, r.device,
                f"{r.ttft_avg:.2f}", f"{r.ttft_p50:.2f}", f"{r.ttft_p95:.2f}", f"{r.ttft_p99:.2f}",
                f"{r.tpot_avg:.2f}", f"{r.tpot_p50:.2f}", f"{r.tpot_p95:.2f}", f"{r.tpot_p99:.2f}",
                f"{r.e2e_avg:.2f}", f"{r.e2e_p50:.2f}", f"{r.e2e_p95:.2f}", f"{r.e2e_p99:.2f}",
                f"{r.tokens_per_second:.2f}", f"{r.requests_per_second:.2f}",
                f"{r.gpu_memory_mb:.2f}", f"{r.cpu_memory_mb:.2f}",
                r.total_requests, r.total_input_tokens, r.total_output_tokens,
            ])
        return buf.getvalue()

    def save_all(self, prefix: str = "benchmark") -> dict[str, str]:
        """同时保存 Markdown、JSON、CSV 报告。

        Returns:
            {格式: 文件路径} 的字典。
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        saved: dict[str, str] = {}

        md_path = self.output_dir / f"{prefix}_{timestamp}.md"
        md_path.write_text(self.to_markdown(), encoding="utf-8")
        saved["markdown"] = str(md_path)

        json_path = self.output_dir / f"{prefix}_{timestamp}.json"
        json_path.write_text(self.to_json(), encoding="utf-8")
        saved["json"] = str(json_path)

        csv_path = self.output_dir / f"{prefix}_{timestamp}.csv"
        csv_path.write_text(self.to_csv(), encoding="utf-8")
        saved["csv"] = str(csv_path)

        return saved

    def print_comparison(self) -> None:
        """在终端打印简洁对比。"""
        print("\n" + "=" * 80)
        print("  模型部署评测 — 框架对比摘要")
        print("=" * 80)
        for r in self.results:
            print(f"  {r.summary()}")
        print("=" * 80 + "\n")
