"""HuggingFace/任意模型 → OpenVINO IR 格式转换。

支持 task: text-generation | image-classification | object-detection

使用方式:
    # LLM
    python convert_model.py --config config.yaml
    # CV 分类模型
    python convert_model.py --config config.yaml --task image-classification --model google/vit-base-patch16-224
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logger import setup_logger

logger = setup_logger("openvino_convert")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── LLM 转换 ──────────────────────────────────────────────────

def convert_llm_optimum(config: dict) -> None:
    """使用 Optimum Intel 转换 LLM 为 OpenVINO IR。"""
    convert_cfg = config["convert"]
    model_path = convert_cfg["model_path"]
    output_dir = Path(convert_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"转换 LLM: {model_path} -> {output_dir}")

    cmd = [
        sys.executable, "-m", "optimum_cli",
        "export", "openvino",
        "--model", model_path,
        "--output", str(output_dir),
        "--task", "text-generation-with-past",
        "--weight-format", convert_cfg.get("precision", "FP16").lower(),
    ]
    if convert_cfg.get("compress_to_fp16", True):
        cmd.extend(["--fp16"])

    logger.info(f"  执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"  ✓ OpenVINO LLM 模型已导出到 {output_dir}")


# ── CV 转换 ───────────────────────────────────────────────────

def convert_cv_optimum(config: dict) -> None:
    """使用 Optimum Intel 转换 CV 模型为 OpenVINO IR。"""
    convert_cfg = config["convert"]
    model_path = convert_cfg["model_path"]
    output_dir = Path(convert_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"转换 CV 模型: {model_path} -> {output_dir}")

    cmd = [
        sys.executable, "-m", "optimum_cli",
        "export", "openvino",
        "--model", model_path,
        "--output", str(output_dir),
        "--task", "image-classification",
        "--weight-format", convert_cfg.get("precision", "FP16").lower(),
    ]
    logger.info(f"  执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"  ✓ OpenVINO CV 模型已导出到 {output_dir}")


def convert_manual(config: dict) -> None:
    """手动转换（适用于非标准模型）。"""
    import torch
    import openvino as ov
    from transformers import AutoModel, AutoTokenizer

    convert_cfg = config["convert"]
    model_path = convert_cfg["model_path"]
    output_dir = Path(convert_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"手动转换: {model_path}")
    model = AutoModel.from_pretrained(model_path, torch_dtype=torch.float16, trust_remote_code=True)
    model.eval()

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        dummy_input = tokenizer("Hello world", return_tensors="pt")
    except Exception:
        # CV 模型回退
        input_shape = tuple(convert_cfg.get("input_shape", [1, 3, 224, 224]))
        dummy_input = {"pixel_values": torch.randn(*input_shape)}

    ov_model = ov.convert_model(
        model,
        example_input={k: v for k, v in dummy_input.items()},
    )
    ov_path = output_dir / "model.xml"
    ov.save_model(ov_model, str(ov_path))
    logger.info(f"  ✓ OpenVINO 模型已保存到 {ov_path}")


# ── 入口 ──────────────────────────────────────────────────────

TASK_CONVERTERS = {
    "text-generation":       convert_llm_optimum,
    "image-classification":  convert_cv_optimum,
    "object-detection":      convert_cv_optimum,
}


def main():
    parser = argparse.ArgumentParser(description="转换 OpenVINO 模型")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str, default=None,
                        choices=list(TASK_CONVERTERS.keys()),
                        help="任务类型 (覆盖配置文件)")
    parser.add_argument("--method", type=str, default="optimum",
                        choices=["optimum", "manual"],
                        help="转换方式: optimum (推荐) / manual")
    parser.add_argument("--model", type=str, default=None, help="模型路径 (覆盖 config)")
    parser.add_argument("--output", type=str, default=None, help="输出目录 (覆盖 config)")
    args = parser.parse_args()

    config = load_config(args.config)
    task = args.task or config.get("task", "text-generation")

    if args.model:
        config.setdefault("convert", {})["model_path"] = args.model
    if args.output:
        config.setdefault("convert", {})["output_dir"] = args.output

    if args.method == "manual":
        convert_manual(config)
    else:
        converter = TASK_CONVERTERS.get(task)
        if converter is None:
            logger.error(f"不支持的任务类型: {task}")
            sys.exit(1)
        converter(config)


if __name__ == "__main__":
    main()
