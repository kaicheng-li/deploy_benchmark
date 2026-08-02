"""Qwen3-VL 多模态 ONNX 链路的共享工具。

导出采用两段式设计（与 llm-export 等社区方案一致），原因：

1. torch 2.9 的 ``torch.onnx.export`` 已强制走 ``torch.export``（dynamo）导出器，
   它不允许图里出现依赖张量**数值**的 Python 调用。Qwen3-VL 视觉塔的
   ``torch.linspace(0, N-1, h)``（h 来自 ``image_grid_thw`` 的值）就属于这类，
   整模型导出会在 ``fast_pos_embed_interpolate`` 处报
   ``GuardOnDataDependentSymNode``。

2. 整模型 forward 里还有其它数据相关控制流：mRoPE position ids 的逐 token
   循环（``tolist()``/``.item()``）、图像占位符数量校验（``mask.sum()`` 参与
   Python ``if``）。这些都无法进图。

因此：

- **视觉塔**（vision.onnx）: ``pixel_values -> image_embeds + deepstack 特征``。
  导出时把 ``image_grid_thw`` 硬编码为 Python 常量（静态图：固定图片尺寸 ->
  固定 grid），绕开 ``linspace``/``torch.split`` 的数据相关问题。
- **文本解码器**（decoder.onnx）: ``input_ids + attention_mask + position_ids +
  image_embeds + deepstack_embeds -> logits``。position_ids 在 numpy 里算好作为
  显式输入；图像 embedding 合并（``masked_scatter``）和 deepstack 注入由
  ``Qwen3VLDecoderWrapper`` 用纯 tensor 算子完成，不触发数据相关 guard。

  ``attention_mask`` 也以 4D 浮点掩码（因果 + padding，0/-inf）作为显式输入：
  HF 的 ``create_causal_mask`` 遇到 4D 掩码会 early-exit 原样返回，从而绕开
  transformers 4.57 内部用 ``torch.vmap`` 构造掩码的路径（TorchScript 追踪器
  无法处理 vmap）。

推理阶段无 KV cache：vision 图每个请求跑一次，decoder 图每步跑一次
（padding 到固定 ``seq_len``，逐步前移 attention mask），与项目里 Qwen3
文本链路的做法一致。
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn


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
        image_grid_thw: (num_images, 3) int64，[t, h, w] 合并前的网格。
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
    """用 processor 编码图片+文本，并 padding 到导出图要求的固定 ``seq_len``。

    Qwen3-VL 的 fast image processor 只支持 ``return_tensors="pt"``
    （传 ``"np"`` 会抛 ValueError），因此先取 torch 张量再转 numpy。
    新版 transformers 的 processor 不再自动插入图像占位符，文本里必须自带
    ``<|vision_start|><|image_pad|><|vision_end|>``（chat template 就是这么做
    的），否则模型拿不到图像 token，position ids 也无法正确计算。
    """
    image = Image.open(image_path).convert("RGB")
    image_token = getattr(processor, "image_token", "<|image_pad|>")
    prompt_with_image = f"<|vision_start|>{image_token}<|vision_end|>{prompt}"
    encoded = processor(images=image, text=prompt_with_image, return_tensors="pt")

    input_ids = encoded["input_ids"].cpu().numpy().astype(np.int64).reshape(1, -1)
    attention_mask = encoded["attention_mask"].cpu().numpy().astype(np.int64).reshape(1, -1)
    pixel_values = encoded["pixel_values"]
    if isinstance(pixel_values, (tuple, list)):
        pixel_values = torch.cat(list(pixel_values), dim=0)
    pixel_values = pixel_values.cpu().numpy().astype(np.float32)
    image_grid_thw = encoded["image_grid_thw"]
    if isinstance(image_grid_thw, (tuple, list)):
        image_grid_thw = torch.cat(list(image_grid_thw), dim=0)
    image_grid_thw = image_grid_thw.cpu().numpy().astype(np.int64)

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


def _pos_embed_constants(
    num_grid_per_side: int, h: int, w: int
) -> tuple[np.ndarray, np.ndarray]:
    """numpy 预计算 ``fast_pos_embed_interpolate`` 的索引与权重（导出常量）。"""
    h_idxs = np.linspace(0, num_grid_per_side - 1, h)
    w_idxs = np.linspace(0, num_grid_per_side - 1, w)
    h_floor = np.floor(h_idxs).astype(np.int64)
    w_floor = np.floor(w_idxs).astype(np.int64)
    h_ceil = np.minimum(h_floor + 1, num_grid_per_side - 1)
    w_ceil = np.minimum(w_floor + 1, num_grid_per_side - 1)
    dh = h_idxs - h_floor
    dw = w_idxs - w_floor
    base_h = h_floor * num_grid_per_side
    base_h_ceil = h_ceil * num_grid_per_side
    indices = np.stack(
        [
            (base_h[:, None] + w_floor[None, :]).ravel(),
            (base_h[:, None] + w_ceil[None, :]).ravel(),
            (base_h_ceil[:, None] + w_floor[None, :]).ravel(),
            (base_h_ceil[:, None] + w_ceil[None, :]).ravel(),
        ]
    )
    weights = np.stack(
        [
            ((1 - dh)[:, None] * (1 - dw)[None, :]).ravel(),
            ((1 - dh)[:, None] * dw[None, :]).ravel(),
            (dh[:, None] * (1 - dw)[None, :]).ravel(),
            (dh[:, None] * dw[None, :]).ravel(),
        ]
    )
    return indices.astype(np.int64), weights.astype(np.float32)


def prepare_vision_for_export(visual: nn.Module, image_grid_thw: np.ndarray) -> None:
    """把视觉塔替换为 Python 常量驱动的前向，使 torch.export 可追踪。

    Qwen3-VL 视觉塔的 pos/rot embedding（``torch.linspace``、``.tolist()``）
    和 attention 分块（``torch.split(lengths.tolist())``）都依赖
    ``image_grid_thw`` 的数值，dynamo 下会触发 ``GuardOnDataDependentSymNode``。
    本项目导出为固定图像 -> 固定 grid，因此：
    - 用 numpy 预计算 position embedding 索引/权重，注册为 buffer（图中常量）；
    - 用等价常量版本替换 ``visual.forward`` 和各 block 的 ``attn.forward``。
    仅支持单帧图片（t=1）。
    """
    from types import MethodType

    from transformers.models.qwen3_vl.modeling_qwen3_vl import (
        apply_rotary_pos_emb_vision,
        eager_attention_forward,
    )

    grid = np.asarray(image_grid_thw, dtype=np.int64)
    t, h, w = int(grid[0, 0]), int(grid[0, 1]), int(grid[0, 2])
    if t != 1:
        raise NotImplementedError("ONNX Qwen3-VL 视觉塔目前只支持单帧图片 (t=1)")
    merge = int(visual.config.spatial_merge_size)
    num_grid = int(visual.num_grid_per_side)
    device = visual.pos_embed.weight.device
    dtype = visual.pos_embed.weight.dtype

    idx, weight = _pos_embed_constants(num_grid, h, w)
    visual.register_buffer("_export_pos_idx", torch.tensor(idx, dtype=torch.long, device=device))
    visual.register_buffer(
        "_export_pos_weight", torch.tensor(weight, dtype=dtype, device=device)
    )

    def vision_forward(self: nn.Module, pixel_values: torch.Tensor) -> tuple[torch.Tensor, list]:
        hidden_states = self.patch_embed(pixel_values)
        pos_embeds = self.pos_embed(self._export_pos_idx) * self._export_pos_weight[:, :, None]
        patch_pos_embeds = pos_embeds[0] + pos_embeds[1] + pos_embeds[2] + pos_embeds[3]
        patch_pos_embeds = (
            patch_pos_embeds.view(t, h // merge, merge, w // merge, merge, -1)
            .permute(0, 1, 3, 2, 4, 5)
            .flatten(0, 4)
        )
        hidden_states = hidden_states + patch_pos_embeds

        freq_table = self.rotary_pos_emb(max(h, w))
        block_rows = torch.arange(h // merge, device=device)
        block_cols = torch.arange(w // merge, device=device)
        intra_row = torch.arange(merge, device=device)
        intra_col = torch.arange(merge, device=device)
        row_idx = block_rows[:, None, None, None] * merge + intra_row[None, None, :, None]
        col_idx = block_cols[None, :, None, None] * merge + intra_col[None, None, None, :]
        row_idx = row_idx.expand(h // merge, w // merge, merge, merge).reshape(-1)
        col_idx = col_idx.expand(h // merge, w // merge, merge, merge).reshape(-1)
        coords = torch.stack((row_idx, col_idx), dim=-1)
        rotary_pos_emb = freq_table[coords].flatten(1)

        seq_len, _ = hidden_states.size()
        hidden_states = hidden_states.reshape(seq_len, -1)
        rotary_pos_emb = rotary_pos_emb.reshape(seq_len, -1)
        emb = torch.cat((rotary_pos_emb, rotary_pos_emb), dim=-1)
        position_embeddings = (emb.cos(), emb.sin())

        cu_seqlens = torch.tensor([0, h * w], dtype=torch.int32, device=hidden_states.device)
        deepstack_feature_lists = []
        for layer_num, blk in enumerate(self.blocks):
            hidden_states = blk(
                hidden_states,
                cu_seqlens=cu_seqlens,
                position_embeddings=position_embeddings,
            )
            if layer_num in self.deepstack_visual_indexes:
                deepstack_feature = self.deepstack_merger_list[
                    self.deepstack_visual_indexes.index(layer_num)
                ](hidden_states)
                deepstack_feature_lists.append(deepstack_feature)
        hidden_states = self.merger(hidden_states)
        return hidden_states, deepstack_feature_lists

    visual.forward = MethodType(vision_forward, visual)

    def vision_attn_forward(
        self: nn.Module,
        hidden_states: torch.Tensor,
        cu_seqlens: torch.Tensor,
        rotary_pos_emb: torch.Tensor | None = None,
        position_embeddings: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        # 单图场景等价于 HF 的 eager 分支，但跳过 torch.split(lengths.tolist())
        seq_length = hidden_states.shape[0]
        query_states, key_states, value_states = (
            self.qkv(hidden_states)
            .reshape(seq_length, 3, self.num_heads, -1)
            .permute(1, 0, 2, 3)
            .unbind(0)
        )
        cos, sin = position_embeddings
        query_states, key_states = apply_rotary_pos_emb_vision(query_states, key_states, cos, sin)
        query_states = query_states.transpose(0, 1).unsqueeze(0)
        key_states = key_states.transpose(0, 1).unsqueeze(0)
        value_states = value_states.transpose(0, 1).unsqueeze(0)
        attn_output, _ = eager_attention_forward(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask=None,
            scaling=self.scaling,
            dropout=0.0 if not self.training else self.attention_dropout,
            is_causal=False,
            **kwargs,
        )
        attn_output = attn_output.reshape(seq_length, -1).contiguous()
        return self.proj(attn_output)

    for block in visual.blocks:
        block.attn.forward = MethodType(vision_attn_forward, block.attn)


class Qwen3VLVisionWrapper(nn.Module):
    """视觉塔导出包装：pixel_values -> (image_embeds, deepstack_embeds)。

    构造时调用 ``prepare_vision_for_export`` 把视觉塔替换为常量驱动版本
    （grid 已固化），因此前向只接收 ``pixel_values``。
    """

    def __init__(self, visual: nn.Module, image_grid_thw: np.ndarray) -> None:
        super().__init__()
        self.visual = visual
        prepare_vision_for_export(visual, image_grid_thw)

    def forward(self, pixel_values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        image_embeds, deepstack_features = self.visual(pixel_values)
        return image_embeds, torch.stack(deepstack_features, dim=0)


class Qwen3VLDecoderWrapper(nn.Module):
    """文本解码器导出包装：input_ids + position_ids + 视觉特征 -> logits。

    图像 embedding 合并（``masked_scatter``）和 deepstack 注入都是纯 tensor
    算子，可由 torch.export 追踪；mRoPE position ids 由调用方算好传入。

    DeepStack 注入不使用 HF 原版的 ``hidden_states[mask]`` 动态 gather
    （ONNX shape inference 无法推导 gather 后的维度），而是把视觉特征展开成
    全序列形式（图像位置为特征值、其余为 0），用静态 shape 的 Add 完成。
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        language_model = model.model.language_model

        def _plain_deepstack(
            hidden_states: torch.Tensor,
            visual_pos_masks: torch.Tensor,
            visual_embeds: torch.Tensor,
        ) -> torch.Tensor:
            # visual_embeds 为全序列形式（非图像位置为 0），等价于 HF 的
            # hidden_states[mask] += features，但形状全程静态
            return hidden_states + visual_embeds

        language_model._deepstack_process = _plain_deepstack

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        image_embeds: torch.Tensor,
        deepstack_embeds: torch.Tensor,
    ) -> torch.Tensor:
        image_token_id = int(self.model.config.image_token_id)
        hidden_size = int(self.model.config.text_config.hidden_size)
        num_layers, _, _ = deepstack_embeds.shape
        image_mask2d = input_ids == image_token_id
        image_mask = image_mask2d.unsqueeze(-1).expand(-1, -1, hidden_size)

        inputs_embeds = self.model.get_input_embeddings()(input_ids)
        inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

        # 紧凑 (L, n_vis, hidden) -> 全序列 (L, seq_len, hidden)
        mask3d = (
            image_mask2d.unsqueeze(-1)
            .expand(-1, -1, hidden_size)
            .expand(num_layers, -1, -1)
        )
        deepstack_full = torch.zeros(
            num_layers,
            input_ids.shape[1],
            hidden_size,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )
        deepstack_full = deepstack_full.masked_scatter(mask3d, deepstack_embeds.reshape(-1))

        outputs = self.model.model.language_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=False,
            visual_pos_masks=image_mask[..., 0],
            deepstack_visual_embeds=deepstack_full,
        )
        return self.model.lm_head(outputs.last_hidden_state)


def build_causal_mask(attention_mask_2d: np.ndarray, dtype: np.dtype = np.float16) -> np.ndarray:
    """由 2D padding mask 构建 4D 因果掩码（0 参与注意力，-inf 屏蔽）。

    shape: (1, 1, seq_len, seq_len)。4D 掩码作为 decoder ONNX 图的显式输入，
    使 HF ``create_causal_mask`` early-exit，绕开 vmap 路径。
    """
    batch, seq_len = attention_mask_2d.shape
    q_idx = np.arange(seq_len, dtype=np.int64)[None, None, :, None]
    kv_idx = np.arange(seq_len, dtype=np.int64)[None, None, None, :]
    causal = kv_idx <= q_idx
    padding = attention_mask_2d[:, None, None, :].astype(bool)
    mask = causal & padding
    min_value = np.finfo(np.float16).min if np.dtype(dtype) == np.float16 else np.finfo(np.float32).min
    out = np.where(mask, np.zeros((), dtype=dtype), np.asarray(min_value, dtype=dtype))
    return np.ascontiguousarray(out.astype(dtype))


def run_vision(
    vision_session: Any,
    pixel_values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """跑视觉塔 ONNX 图，返回 (image_embeds, deepstack_embeds)。"""
    image_embeds, deepstack_embeds = vision_session.run(
        ["image_embeds", "deepstack_embeds"],
        {"pixel_values": np.ascontiguousarray(pixel_values.astype(np.float16))},
    )
    return np.ascontiguousarray(image_embeds), np.ascontiguousarray(deepstack_embeds)


def generate(
    vision_session: Any,
    decoder_session: Any,
    feeds: dict[str, np.ndarray],
    tokenizer: Any,
    consts: Qwen3VLConstants,
    max_new_tokens: int = 64,
) -> tuple[str, int, int]:
    """贪心解码：vision 图跑一次，decoder 图逐步跑（无 KV cache，前移 mask）。"""
    image_embeds, deepstack_embeds = run_vision(vision_session, feeds["pixel_values"])

    input_ids = feeds["input_ids"]
    attention_mask = feeds["attention_mask"]
    seq_len = input_ids.shape[1]
    start_len = int(attention_mask.sum())
    eos_token_id = tokenizer.eos_token_id

    for step in range(max_new_tokens):
        if start_len + step >= seq_len:
            logging.getLogger("qwen3vl_utils").warning(
                "到达最大序列长度 %s，提前停止", seq_len
            )
            break
        position_ids, _ = compute_rope_index(input_ids, attention_mask, feeds["image_grid_thw"], consts)
        mask_4d = build_causal_mask(attention_mask)
        logits = decoder_session.run(
            ["logits"],
            {
                "input_ids": input_ids,
                "attention_mask": mask_4d,
                "position_ids": position_ids,
                "image_embeds": image_embeds,
                "deepstack_embeds": deepstack_embeds,
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
