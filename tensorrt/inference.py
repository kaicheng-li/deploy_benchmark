"""TensorRT-LLM 推理脚本。

使用方式:
    python inference.py --config config.yaml --prompt "Hello world"
"""

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from common.logger import setup_logger

logger = setup_logger("tensorrt_inference")


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def run_inference(config: dict, prompt: str, max_new_tokens: int = 512) -> str:
    """使用 TensorRT-LLM runtime 进行单次推理。"""
    runtime_cfg = config["runtime"]

    try:
        from tensorrt_llm.runtime import ModelRunner, ModelRunnerCpp
    except ImportError:
        logger.error("请安装 tensorrt_llm: pip install tensorrt_llm")
        raise

    runner_kwargs = dict(
        engine_dir=runtime_cfg["engine_dir"],
        rank=0,
    )

    runner = ModelRunnerCpp.from_dir(**runner_kwargs)
    logger.info("TensorRT-LLM runner 初始化完成")

    # 推理
    batch_input_ids = [runner.tokenizer.encode(prompt, add_special_tokens=True)]

    with runner.session as session:
        output_ids = session.generate(
            batch_input_ids,
            max_new_tokens=max_new_tokens,
            end_id=runner.tokenizer.eos_token_id,
            pad_id=runner.tokenizer.pad_token_id or runner.tokenizer.eos_token_id,
        )

    output_text = runner.tokenizer.decode(output_ids[0][0], skip_special_tokens=True)
    return output_text


def main():
    parser = argparse.ArgumentParser(description="TensorRT-LLM 推理")
    parser.add_argument("--config", type=str, default="config.yaml", help="配置文件")
    parser.add_argument("--prompt", type=str, required=True, help="输入提示")
    parser.add_argument("--max_tokens", type=int, default=512, help="最大生成长度")
    args = parser.parse_args()

    config = load_config(args.config)
    output = run_inference(config, args.prompt, args.max_tokens)
    print(f"\n[输入] {args.prompt}")
    print(f"[输出] {output}")


if __name__ == "__main__":
    main()
