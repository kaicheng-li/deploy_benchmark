"""
deploy_benchmark 公共模块

提供跨框架共享的：
- 基准测试指标计算
- 数据加载器
- 统一日志
- 测试报告生成
"""

from .metrics import BenchmarkMetrics, TimingResult
from .data_loader import DataLoader
from .logger import setup_logger
from .reporter import BenchmarkReporter

__all__ = [
    "BenchmarkMetrics",
    "TimingResult",
    "DataLoader",
    "setup_logger",
    "BenchmarkReporter",
]
