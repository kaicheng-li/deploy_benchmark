"""Qwen3-VL 多模态 ONNX 链路的共享工具。

导出约束（与 Qwen3-VL 结构强相关，改动 transformers 版本时需要复核）：

1. Qwen3-VL 的 mRoPE position ids 计算依赖 token 数值（内部使用 ``tolist()``、
   ``.item()`` 和逐 token 循环），无法被 ONNX 图追踪（dynamo=False）或编译
   （dynamo=True 也会在数据相关分支上失败）。因此导出时把 ``position_ids``
   作为显式输入，运行时在 Python/numpy 里计算（本模块 ``compute_rope_index``）。

2. 模型内部的 ``masked_scatter``（图像 embedding 占位符合并）与
   ``torch.split``（按 ``image_grid_thw`` 切分视觉特征）都是数据相关算子。
   为让导出稳定，ONNX 图使用静态 shape：固定 ``seq_len`` 与固定图像
   （``pixel_values`` / ``image_grid_thw`` 尺寸固定）。推理时文本长度被
   padding 到 ``seq_len``，每次解码只前移一个 attention mask 位。

3. 导出时强制 eager attention（视觉塔和文本塔都关闭 SDPA/FlashAttention），
   避免 attention 算子在 ONNX 导出时无法分解。

推理阶段（``inference.py`` / ``benchmark.py``）没有 KV cache：每一步都把
完整序列重新过一遍图并重算视觉特征。这与本项目 Qwen3 文本链路（onnx
``run_qwen3`` / ``bench_qwen3``）的做法一致，功能正确但速度较慢。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class Qwen3VLConstants:
    """从 Qwen3-VL config 提取的运行期常量。"""

    spatial_merge_size: int
    image_token_id: int
    video_token_id: int
    vision_start_token_id: int

    @classmethod
    def from_config(cls, config: Any) -> "Qwen3VLConstants":
        return cls(
            spatial_merge_size=int(config.vision_config.spatial_merge_size),
            image_token_id=int(config.image_token_id),
            video_token_id=int(config.video_token_id),
            vision_start_token_id=int(config.vision_start_token_id),
        )


def compute_rope_index(
    input_ids: np.ndarray,
    attention_mask: np.ndarray,
    image_grid_thw: np.ndarray,
    consts: Qwen3VLConstants,
) -> tuple[np.ndarray, np.ndarray]:
    """numpy 移植的 ``Qwen3VLModel.get_rope_index``（仅图片，不支持视频）。

    Args:
        input_ids: (1, seq_len) int64，已 padding。
        attention_mask: (1, seq_len) int64，1 表示真实 token。
        image_grid_thw: (num_images, 3) int64，[t, h, w] 合并后的网格。
        consts: 模型常量。

    Returns:
        position_ids: (3, 1, seq_len) int64（mrope 的 t/h/w 三维位置）。
        rope_deltas: (1, 1) int64，与 HF 返回一致（本项目未使用）。
    """
    batch, seq = input_ids.shape
    position_ids = np.ones((3, batch, seq), dtype=input_ids.dtype)
    rope_deltas = []

    for i in range(batch):
        ids = input_ids[i][attention_mask[i] == 1]
        vision_start_indices = np.argwhere(ids == consts.vision_start_token_id).reshape(-1)
        vision_tokens = ids[vision_start_indices + 1]
        image_nums = int((vision_tokens == consts.image_token_id).sum())
        video_nums = int((vision_tokens == consts.video_token_id).sum())
        if video_nums:
            raise NotImplementedError("ONNX Qwen3-VL 链路目前只支持图片，不支持视频输入")

        input_tokens = ids.tolist()
        llm_pos_ids_list: list[np.ndarray] = []
        st = 0
        image_index = 0
        remain_images = image_nums
        for _ in range(image_nums):
            if consts.image_token_id in input_tokens and remain_images > 0:
                ed_image = input_tokens.index(consts.image_token_id, st)
            else:
                ed_image = len(input_tokens) + 1
            # 无视频 token 时 HF 等价分支不会命中，直接走图片分支
            t, h, w = image_grid_thw[image_index]
            image_index += 1
            remain_images -= 1
            ed = ed_image

            llm_grid_t = int(t)
            llm_grid_h = int(h) // consts.spatial_merge_size
            llm_grid_w = int(w) // consts.spatial_merge_size
            text_len = ed - st

            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len, dtype=input_ids.dtype), (3, text_len)) + st_idx
            )

            t_index = np.repeat(np.arange(llm_grid_t, dtype=input_ids.dtype), llm_grid_h * llm_grid_w)
            h_index = np.tile(
                np.repeat(np.arange(llm_grid_h, dtype=input_ids.dtype), llm_grid_w), llm_grid_t
            )
            w_index = np.tile(np.arange(llm_grid_w, dtype=input_ids.dtype), llm_grid_t * llm_grid_h)
            llm_pos_ids_list.append(np.stack([t_index, h_index, w_index]) + text_len + st_idx)

            st = ed + llm_grid_t * llm_grid_h * llm_grid_w

        if st < len(input_tokens):
            st_idx = int(llm_pos_ids_list[-1].max()) + 1 if llm_pos_ids_list else 0
            text_len = len(input_tokens) - st
            llm_pos_ids_list.append(
                np.broadcast_to(np.arange(text_len, dtype=input_ids.dtype), (3, text_len)) + st_idx
            )

        if llm_pos_ids_list:
            llm_positions = np.concatenate(llm_pos_ids_list, axis=1).reshape(3, -1)
            position_ids[..., i, attention_mask[i] == 1] = llm_positions
            # HF 用完整序列长度（含 padding）计算 rope delta
            rope_deltas.append(int(llm_positions.max()) + 1 - seq)
        else:
            rope_deltas.append(0)

    return position_ids, np.asarray(rope_deltas, dtype=input_ids.dtype)[:, None]


def prepare_inputs(
    processor: Any,
    tokenizer: Any,
    image_path: str | Path,
    prompt: str,
    seq_len: int,
) -> dict[str, np.ndarray]:
    """用 processor 编码图片+文本，并 padding 到导出图要求的固定 ``seq_len``。"""
    image = Image.open(image_path).convert("RGB")
    encoded = processor(images=image, text=prompt, return_tensors="np")

    input_ids = np.asarray(encoded["input_ids"], dtype=np.int64).reshape(1, -1)
    attention_mask = np.asarray(encoded["attention_mask"], dtype=np.int64).reshape(1, -1)
    pixel_values = np.asarray(encoded["pixel_values"], dtype=np.float32)
    image_grid_thw = np.asarray(encoded["image_grid_thw"], dtype=np.int64)

    length = input_ids.shape[1]
    if length > seq_len:
        input_ids = input_ids[:, :seq_len]
        attention_mask = attention_mask[:, :seq_len]
    elif length < seq_len:
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else (
            tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 0
        )
        input_ids = np.concatenate(
            [input_ids, np.full((1, seq_len - length), pad_id, dtype=np.int64)], axis=1
        )
        attention_mask = np.concatenate(
            [attention_mask, np.zeros((1, seq_len - length), dtype=np.int64)], axis=1
        )

    return {
        "input_ids": np.ascontiguousarray(input_ids),
        "attention_mask": np.ascontiguousarray(attention_mask),
        "pixel_values": np.ascontiguousarray(pixel_values),
        "image_grid_thw": np.ascontiguousarray(image_grid_thw),
    }


def generate(
    session: Any,
    feeds: dict[str, np.ndarray],
    tokenizer: Any,
    consts: Qwen3VLConstants,
    max_new_tokens: int = 64,
) -> tuple[str, int, int]:
    """基于固定 shape ONNX 图的贪心解码（无 KV cache，逐步前移 mask）。"""
    input_ids = feeds["input_ids"]
    attention_mask = feeds["attention_mask"]
    pixel_values = feeds["pixel_values"]
    image_grid_thw = feeds["image_grid_thw"]
    seq_len = input_ids.shape[1]
    start_len = int(attention_mask.sum())
    eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        if start_len + step >= seq_len:
            logger_warning("到达最大序列长度 %s，提前停止", seq_len)
            break
        position_ids, _ = compute_rope_index(input_ids, attention_mask, image_grid_thw, consts)
        logits = session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "image_grid_thw": image_grid_thw,
                "position_ids": position_ids,
            },
        )[0]
        next_id = int(np.argmax(logits[0, start_len + step]))
        if eos_token_id is not None and next_id == eos_token_id:
            break
        input_ids[0, start_len + step] = next_id
        attention_mask[0, start_len + step] = 1

    end_len = int(attention_mask.sum())
    generated = input_ids[0, start_len:end_len]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    return text, start_len, end_len - start_len


def logger_warning(message: str, *args: Any) -> None:
    import logging

    logging.getLogger("qwen3vl_utils").warning(message, *args)
