"""统一日志配置。"""

import logging
import sys
from pathlib import Path


def setup_logger(
    name: str = "deploy_benchmark",
    level: int = logging.INFO,
    log_file: str | Path | None = None,
) -> logging.Logger:
    """创建带控制台和可选文件输出的 logger。

    Args:
        name: logger 名称。
        level: 日志级别。
        log_file: 可选日志文件路径。

    Returns:
        配置好的 Logger 实例。
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)

    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # 控制台输出
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(fmt)
    logger.addHandler(console_handler)

    # 文件输出
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    return logger
