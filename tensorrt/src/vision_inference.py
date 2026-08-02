"""Run RF-DETR segmentation through a generic TensorRT engine."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import tensorrt as trt
import torch


LOGGER = trt.Logger(trt.Logger.WARNING)
TORCH_DTYPES = {
    trt.float32: torch.float32,
    trt.float16: torch.float16,
    trt.int32: torch.int32,
    trt.int64: torch.int64,
    trt.bool: torch.bool,
}


def load_labels(model_path: str | Path) -> dict[int, str]:
    with (Path(model_path) / "config.json").open(encoding="utf-8") as file:
        labels = json.load(file).get("id2label", {})
    return {int(key): str(value) for key, value in labels.items()}


def run_vision(cfg: dict, image_override: str | None = None) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Generic TensorRT inference requires a CUDA-enabled PyTorch installation")

    engine_path = Path(cfg["engine_file"])
    if not engine_path.is_file():
        raise FileNotFoundError(
            f"TensorRT engine does not exist: {engine_path}. Run engine_builder.py first."
        )
    image_path = Path(image_override).expanduser().resolve() if image_override else Path(cfg["image"])
    if not image_path.is_file():
        raise FileNotFoundError(f"Image does not exist: {image_path}")

    runtime = trt.Runtime(LOGGER)
    engine = runtime.deserialize_cuda_engine(engine_path.read_bytes())
    if engine is None:
        raise RuntimeError(f"Could not deserialize TensorRT engine: {engine_path}")
    context = engine.create_execution_context()
    if context is None:
        raise RuntimeError("Could not create TensorRT execution context")

    from transformers import AutoImageProcessor

    image = Image.open(image_path).convert("RGB")
    processor = AutoImageProcessor.from_pretrained(cfg["model_path"])
    encoded = processor(images=image, return_tensors="np")
    tensors: dict[str, torch.Tensor] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.INPUT:
            continue
        if name not in encoded:
            raise KeyError(f"Image processor did not produce TensorRT input: {name}")
        dtype = TORCH_DTYPES.get(engine.get_tensor_dtype(name))
        if dtype is None:
            raise TypeError(f"Unsupported TensorRT dtype for {name}: {engine.get_tensor_dtype(name)}")
        tensor = torch.from_numpy(np.ascontiguousarray(encoded[name])).to(device="cuda", dtype=dtype)
        context.set_input_shape(name, tuple(tensor.shape))
        context.set_tensor_address(name, tensor.data_ptr())
        tensors[name] = tensor

    outputs: dict[str, torch.Tensor] = {}
    for index in range(engine.num_io_tensors):
        name = engine.get_tensor_name(index)
        if engine.get_tensor_mode(name) != trt.TensorIOMode.OUTPUT:
            continue
        dtype = TORCH_DTYPES.get(engine.get_tensor_dtype(name))
        shape = tuple(context.get_tensor_shape(name))
        if dtype is None or any(size < 0 for size in shape):
            raise RuntimeError(f"Could not resolve output shape/dtype for {name}: {shape}")
        tensor = torch.empty(shape, device="cuda", dtype=dtype)
        context.set_tensor_address(name, tensor.data_ptr())
        outputs[name] = tensor

    if not context.execute_async_v3(torch.cuda.current_stream().cuda_stream):
        raise RuntimeError("TensorRT execution failed")
    torch.cuda.synchronize()

    result = processor.post_process_instance_segmentation(
        SimpleNamespace(
            logits=outputs["logits"].float().cpu(),
            pred_boxes=outputs["pred_boxes"].float().cpu(),
            pred_masks=outputs["pred_masks"].float().cpu(),
        ),
        target_sizes=[image.size[::-1]],
        threshold=float(cfg.get("threshold", 0.5)),
        mask_threshold=0.0,
    )[0]

    labels = load_labels(cfg["model_path"])
    print(f"Image: {image_path}")
    segments = result["segments_info"]
    print(f"Segments: {len(segments)}")
    for item in segments[:20]:
        label_id = int(item["label_id"])
        label = labels.get(label_id, f"class_{label_id}")
        print(f"  id={item['id']} label={label} score={float(item['score']):.4f}")


