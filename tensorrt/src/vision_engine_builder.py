"""Build the RF-DETR TensorRT engine from the exported ONNX artifact."""

from __future__ import annotations

from pathlib import Path

import tensorrt as trt


LOGGER = trt.Logger(trt.Logger.WARNING)


def build_engine(cfg: dict) -> Path:
    """Build the configured RF-DETR TensorRT engine."""

    onnx_path = Path(cfg["onnx_file"])
    engine_path = Path(cfg["engine_file"])
    if not onnx_path.is_file():
        raise FileNotFoundError(f"RF-DETR ONNX file does not exist: {onnx_path}")
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    builder = trt.Builder(LOGGER)
    flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    network = builder.create_network(flags)
    parser = trt.OnnxParser(network, LOGGER)
    if not parser.parse_from_file(str(onnx_path)):
        errors = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"TensorRT could not parse {onnx_path}:\n{errors}")

    build_cfg = cfg.get("build", {})
    builder_config = builder.create_builder_config()
    workspace_mb = int(build_cfg.get("workspace_size_mb", 2048))
    builder_config.set_memory_pool_limit(
        trt.MemoryPoolType.WORKSPACE, workspace_mb * 1024 * 1024
    )
    if build_cfg.get("precision", "fp16") == "fp16" and builder.platform_has_fast_fp16:
        builder_config.set_flag(trt.BuilderFlag.FP16)

    serialized = builder.build_serialized_network(network, builder_config)
    if serialized is None:
        raise RuntimeError("TensorRT failed to build the RF-DETR engine")
    engine_path.write_bytes(bytes(serialized))
    return engine_path
