"""ONNX Runtime 基准测试 — 支持 LLM 和 CV。

使用方式:
    # LLM
    python benchmark.py --config config.yaml --task text-generation --data ../data/prompts.txt
    # CV
    python benchmark.py --config config.yaml --task image-classification --data ../data/images
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.metrics import BenchmarkMetrics, TimingResult, TaskType
from common.reporter import BenchmarkReporter
from common.logger import setup_logger
from common.data_loader import DataLoader

logger = setup_logger("onnx_benchmark")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def create_session(config: dict):
    import onnxruntime as ort

    runtime_cfg = config["runtime"]
    onnx_dir = Path(runtime_cfg["onnx_dir"])
    model_path = str(onnx_dir / "model.onnx")

    sess_opt = ort.SessionOptions()
    opt_cfg = runtime_cfg.get("session_options", {})
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess_opt.intra_op_num_threads = opt_cfg.get("intra_op_num_threads", 4)

    provider = runtime_cfg.get("provider", "CPUExecutionProvider")
    providers = [provider]
    if provider == "CUDAExecutionProvider":
        providers.append("CPUExecutionProvider")

    return ort.InferenceSession(model_path, sess_opt, providers=providers)


# ── LLM benchmark ──────────────────────────────────────────────

def run_text_benchmark(session, config: dict, prompts: list[str], warmup: int = 10) -> list[TimingResult]:
    from transformers import AutoTokenizer
    model_name = config["export"]["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    input_names = [inp.name for inp in session.get_inputs()]
    output_names = [out.name for out in session.get_outputs()]
    timings = []

    logger.info(f"预热 {warmup} 轮...")
    for i in range(min(warmup, len(prompts))):
        inputs = tokenizer(prompts[i][:1024], return_tensors="np")
        feed_dict = {n: inputs[n] for n in input_names if n in inputs}
        session.run(output_names, feed_dict)

    logger.info(f"开始评测: {len(prompts)} prompts")
    for i, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="np")
        feed_dict = {n: inputs[n] for n in input_names if n in inputs}
        input_len = inputs["input_ids"].shape[1] if "input_ids" in inputs else 0

        t0 = time.perf_counter()
        outputs = session.run(output_names, feed_dict)
        e2e_ms = (time.perf_counter() - t0) * 1000

        output_len = outputs[0].shape[1] if len(outputs[0].shape) > 1 else 0
        timings.append(TimingResult(
            ttft=e2e_ms, tpot=e2e_ms / max(output_len, 1),
            e2e_latency=e2e_ms, input_tokens=input_len, output_tokens=output_len,
        ))
        if (i + 1) % 10 == 0:
            logger.info(f"  进度: {i+1}/{len(prompts)}")
    return timings


# ── CV benchmark ───────────────────────────────────────────────

def preprocess_batch(image_paths: list[Path], input_shape: tuple = (3, 224, 224)) -> np.ndarray:
    """批量预处理图像。"""
    from PIL import Image

    batch = []
    for p in image_paths:
        img = Image.open(p).convert("RGB").resize(input_shape[1:], Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - np.array([0.485, 0.456, 0.406])) / np.array([0.229, 0.224, 0.225])
        arr = np.transpose(arr, (2, 0, 1))
        batch.append(arr)
    return np.stack(batch, axis=0)


def run_cv_benchmark(session, config: dict, images: list[Path], warmup: int = 10,
                     batch_size: int = 1) -> list[TimingResult]:
    export_cfg = config["export"]
    input_shape = tuple(export_cfg.get("input_shape", [1, 3, 224, 224]))
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    timings = []

    # 预热
    logger.info(f"预热 {warmup} 轮...")
    for i in range(min(warmup // max(batch_size, 1), len(images) // max(batch_size, 1))):
        batch_imgs = images[i * batch_size:(i + 1) * batch_size]
        tensor = preprocess_batch(batch_imgs, input_shape)
        session.run([output_name], {input_name: tensor.astype(np.float32)})

    logger.info(f"开始评测: {len(images)} images, batch={batch_size}")
    for i in range(0, len(images), batch_size):
        batch_imgs = images[i:i + batch_size]
        if len(batch_imgs) < batch_size and i + batch_size > len(images):
            break  # 跳过不完整的 batch

        t_pre0 = time.perf_counter()
        tensor = preprocess_batch(batch_imgs, input_shape)
        t_pre = (time.perf_counter() - t_pre0) * 1000

        t_inf0 = time.perf_counter()
        outputs = session.run([output_name], {input_name: tensor.astype(np.float32)})
        t_inf = (time.perf_counter() - t_inf0) * 1000

        e2e_ms = t_pre + t_inf
        per_image_e2e = e2e_ms / batch_size

        for j, img_path in enumerate(batch_imgs):
            timings.append(TimingResult(
                e2e_latency=per_image_e2e,
                preprocess_ms=t_pre / batch_size,
                inference_ms=t_inf / batch_size,
                postprocess_ms=0.0,
                input_shape=input_shape,
                output_shape=outputs[0].shape,
            ))

        if (i // batch_size + 1) % 10 == 0:
            logger.info(f"  进度: {min(i+batch_size, len(images))}/{len(images)}")
    return timings


# ── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ONNX Runtime 基准测试")
    parser.add_argument("--config", type=str, default="config.yaml")
    parser.add_argument("--task", type=str, default=None, choices=["text-generation", "image-classification", "object-detection"])
    parser.add_argument("--data", type=str, default=None, help="数据路径 (覆盖 config)")
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--output", type=str, default="../results")
    args = parser.parse_args()

    config = load_config(args.config)
    task: TaskType = args.task or config.get("task", "text-generation")  # type: ignore[assignment]
    bench_cfg = config.get("benchmark", {})
    warmup = bench_cfg.get("warmup_requests", 10)

    reporter = BenchmarkReporter(output_dir=args.output)

    session = create_session(config)
    model_name = Path(config["export"]["model_path"]).name

    if task == "text-generation":
        data_path = Path(args.data or bench_cfg.get("prompt_file", "../data/prompts.txt"))
        prompts = _load_or_fallback_text(data_path)
        timings = run_text_benchmark(session, config, prompts, warmup=warmup)
        device_label = "cuda" if "CUDA" in config["runtime"]["provider"] else "cpu"
    else:
        data_path = Path(args.data or bench_cfg.get("image_dir", "../data/images"))
        images = _load_or_fallback_images(data_path)
        batch_size = args.batch_size or bench_cfg.get("batch_size", 1)
        timings = run_cv_benchmark(session, config, images, warmup=warmup, batch_size=batch_size)
        device_label = "cuda" if "CUDA" in config["runtime"]["provider"] else "cpu"

    if timings:
        metrics = BenchmarkMetrics.from_timings(
            timings, framework="ONNX Runtime", model_name=model_name,
            device=device_label, task_type=task,
        )
        reporter.add_result(metrics)
        reporter.print_comparison()

    saved = reporter.save_all(prefix="onnx_benchmark")
    logger.info(f"报告已保存: {saved}")


def _load_or_fallback_text(data_path: Path) -> list[str]:
    if data_path.exists():
        return DataLoader(data_path).load_prompts()
    logger.warning(f"数据文件不存在: {data_path}, 使用内置 prompts")
    return ["Explain the attention mechanism."] * 20


def _load_or_fallback_images(data_path: Path) -> list[Path]:
    if data_path.exists():
        return DataLoader(data_path).load_images()
    logger.error(f"图像数据不存在: {data_path}")
    logger.error("请准备图像目录或通过 --data 指定")
    sys.exit(1)


if __name__ == "__main__":
    main()
