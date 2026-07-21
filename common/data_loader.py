"""统一数据加载器。

支持：
- 文本任务: .jsonl / .json / .txt (每行/每项一个 prompt)
- 图像任务: 目录下的图片文件, 或 .jsonl (每行含 "image" 路径)
"""

import json
from pathlib import Path
from typing import Iterator, Optional, Union


class DataLoader:
    """测试数据加载器。"""

    def __init__(self, data_path: str | Path):
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据文件不存在: {self.data_path}")

    # ── 文本 prompts ───────────────────────────────────────────

    def load_prompts(self, max_samples: Optional[int] = None) -> list[str]:
        """加载文本 prompt 列表。"""
        prompts: list[str] = []
        suffix = self.data_path.suffix.lower()

        if suffix == ".jsonl":
            prompts = self._load_jsonl()
        elif suffix == ".json":
            prompts = self._load_json()
        elif suffix == ".txt":
            prompts = self._load_txt()
        else:
            raise ValueError(f"不支持的文本格式: {suffix}")

        if max_samples and max_samples > 0:
            prompts = prompts[:max_samples]
        return prompts

    def iter_prompts(self, max_samples: Optional[int] = None) -> Iterator[str]:
        for i, prompt in enumerate(self.load_prompts()):
            if max_samples and i >= max_samples:
                break
            yield prompt

    # ── 图像 ───────────────────────────────────────────────────

    def load_images(self, max_samples: Optional[int] = None) -> list[Path]:
        """加载图像路径列表。

        支持两种输入:
        - 目录: 自动扫描 *.jpg, *.jpeg, *.png, *.bmp
        - .jsonl: 每行 {"image": "path/to/img.jpg", "label": 0}
        """
        if self.data_path.is_dir():
            images = self._scan_image_dir()
        elif self.data_path.suffix.lower() == ".jsonl":
            images = self._load_image_jsonl()
        else:
            raise ValueError(f"不支持的图像数据源: {self.data_path}")

        if max_samples and max_samples > 0:
            images = images[:max_samples]
        return images

    def iter_images(self, max_samples: Optional[int] = None) -> Iterator[Path]:
        for i, img in enumerate(self.load_images()):
            if max_samples and i >= max_samples:
                break
            yield img

    def _scan_image_dir(self) -> list[Path]:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        images = sorted(
            p for p in self.data_path.iterdir()
            if p.suffix.lower() in exts
        )
        return images

    def _load_image_jsonl(self) -> list[Path]:
        images = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                img_path = obj.get("image", obj.get("path", ""))
                if img_path:
                    p = Path(img_path)
                    if not p.is_absolute():
                        p = self.data_path.parent / p
                    images.append(p)
        return images

    # ── 内部 ───────────────────────────────────────────────────

    def _load_jsonl(self) -> list[str]:
        prompts = []
        with open(self.data_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                prompts.append(obj.get("prompt", obj.get("text", "")))
        return prompts

    def _load_json(self) -> list[str]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return [item if isinstance(item, str) else item.get("prompt", "") for item in data]
        if isinstance(data, dict) and "prompts" in data:
            return data["prompts"]
        raise ValueError("JSON 格式不匹配：期望列表或含 'prompts' 键的对象")

    def _load_txt(self) -> list[str]:
        with open(self.data_path, "r", encoding="utf-8") as f:
            return [line.strip() for line in f if line.strip()]
