"""HuggingFace/任意模型 → ONNX 格式导出。

支持 task: text-generation | image-classification | object-detection

使用方式:
    # LLM
    python export_onnx.py --config config.yaml
    # CV 分类模型
    python export_onnx.py --config config.yaml --task image-classification --model resnet50
"""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logger import setup_logger

logger = setup_logger("onnx_export")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


# ── LLM 导出 ──────────────────────────────────────────────────

def export_llm_optimum(config: dict) -> None:
    """使用 Optimum 导出 LLM ONNX。"""
    export_cfg = config["export"]
    model_path = export_cfg["model_path"]
    onnx_dir = Path(export_cfg["onnx_dir"])
    onnx_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"导出 LLM: {model_path} -> {onnx_dir}")

    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_path,
        "--output", str(onnx_dir),
        "--opset", str(export_cfg.get("opset_version", 17)),
        "--task", "text-generation",
    ]
    if export_cfg.get("use_external_data_format", True):
        cmd.append("--for-ort")

    logger.info(f"  执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"  ✓ LLM ONNX 模型已导出到 {onnx_dir}")


# ── CV 模型导出 ───────────────────────────────────────────────

def export_cv_torch(config: dict) -> None:
    """导出 CV 分类/检测模型为 ONNX。"""
    import torch
    import torchvision.models as models

    export_cfg = config["export"]
    onnx_dir = Path(export_cfg["onnx_dir"])
    onnx_dir.mkdir(parents=True, exist_ok=True)
    onnx_path = onnx_dir / "model.onnx"

    model_name = export_cfg.get("model_path", "resnet50")
    logger.info(f"导出 CV 模型: {model_name} -> {onnx_path}")

    # 加载 torchvision 预训练模型
    if hasattr(models, model_name.lower()):
        model = getattr(models, model_name.lower())(pretrained=True)
    else:
        # 尝试从 torchvision 按名称加载
        model = getattr(models, model_name)(weights="IMAGENET1K_V1")
    model.eval()

    input_shape = export_cfg.get("input_shape", [1, 3, 224, 224])
    dummy_input = torch.randn(*input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        opset_version=export_cfg.get("opset_version", 17),
        do_constant_folding=True,
    )
    logger.info(f"  ✓ CV ONNX 模型已导出到 {onnx_path}")


def export_cv_optimum(config: dict) -> None:
    """使用 Optimum 导出 HuggingFace CV 模型 (ViT 等)。"""
    export_cfg = config["export"]
    model_path = export_cfg["model_path"]
    onnx_dir = Path(export_cfg["onnx_dir"])
    onnx_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"导出 HF CV 模型: {model_path} -> {onnx_dir}")

    cmd = [
        sys.executable, "-m", "optimum.exporters.onnx",
        "--model", model_path,
        "--output", str(onnx_dir),
        "--opset", str(export_cfg.get("opset_version", 17)),
        "--task", "image-classification",
    ]
    logger.info(f"  执行: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    logger.info(f"  ✓ CV ONNX 模型已导出到 {onnx_dir}")


# ── 入口 ──────────────────────────────────────────────────────

TASK_EXPORTERS = {
    "text-generation":       export_llm_optimum,
    "image-classification":  export_cv_optimum,
    "object-detection":      export_cv_optimum,
}

TORCH_CV_MODELS = {"resnet18", "resnet34", "resnet50", "resnet101",
                   "mobilenet_v2", "mobilenet_v3_small", "mobilenet_v3_large",
                   "efficientnet_b0", "efficientnet_b1",
                   "shufflenet_v2_x0_5", "shufflenet_v2_x1_0",
                   "squeezenet1_0", "squeezenet1_1",
                   "densenet121", "densenet169",
                   "vgg16", "vgg19",
                   "alexnet", "googlenet", "inception_v3",
                   "vit_b_16", "vit_b_32", "vit_l_16"}


def main():
    parser = argparse.ArgumentParser(description="导出 ONNX 模型")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件")
    parser.add_argument("--task", type=str, default=None,
                        choices=list(TASK_EXPORTERS.keys()),
                        help="任务类型 (覆盖配置文件)")
    parser.add_argument("--model", type=str, default=None, help="模型名称/路径 (覆盖配置文件)")
    parser.add_argument("--output", type=str, default=None, help="输出目录 (覆盖配置文件)")
    args = parser.parse_args()

    config = load_config(args.config)
    task = args.task or config.get("task", "text-generation")

    if args.model:
        config.setdefault("export", {})["model_path"] = args.model
    if args.output:
        config.setdefault("export", {})["onnx_dir"] = args.output

    model_name = config["export"].get("model_path", "")

    # 智能选择导出方式：torchvision 模型直接用 torch.onnx
    if model_name.lower() in TORCH_CV_MODELS:
        logger.info(f"检测到 torchvision 模型 '{model_name}'，使用 torch.onnx.export")
        export_cv_torch(config)
    else:
        exporter = TASK_EXPORTERS.get(task)
        if exporter is None:
            logger.error(f"不支持的任务类型: {task}")
            sys.exit(1)
        exporter(config)


if __name__ == "__main__":
    main()
