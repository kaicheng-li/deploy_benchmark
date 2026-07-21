"""llama.cpp Python 封装推理脚本。

需要安装 llama-cpp-python:
    pip install llama-cpp-python --extra-index-url https://abetlen.github.io/llama-cpp-python/whl

使用方式:
    python inference.py --model ../../models/qwen2-7b-instruct-q4_k_m.gguf --prompt "Hello"
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from common.logger import setup_logger

logger = setup_logger("llamacpp_inference")


def load_model(
    model_path: str,
    n_ctx: int = 4096,
    n_threads: int = 8,
    n_gpu_layers: int = -1,
    verbose: bool = False,
):
    """加载 llama.cpp 模型。"""
    from llama_cpp import Llama

    llm = Llama(
        model_path=model_path,
        n_ctx=n_ctx,
        n_threads=n_threads,
        n_gpu_layers=n_gpu_layers,
        verbose=verbose,
    )
    logger.info(f"模型已加载: {model_path}")
    return llm


def generate(
    llm,
    prompt: str,
    max_tokens: int = 512,
    temperature: float = 0.0,
) -> dict:
    """单次生成。"""
    output = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        echo=False,
    )

    return {
        "text": output["choices"][0]["text"],
        "input_tokens": output["usage"]["prompt_tokens"],
        "output_tokens": output["usage"]["completion_tokens"],
    }


def main():
    parser = argparse.ArgumentParser(description="llama.cpp 推理")
    parser.add_argument("--model", type=str, required=True, help="GGUF 模型路径")
    parser.add_argument("--prompt", type=str, required=True, help="输入提示")
    parser.add_argument("--n_ctx", type=int, default=4096, help="上下文长度")
    parser.add_argument("--n_threads", type=int, default=8, help="CPU 线程数")
    parser.add_argument("--n_gpu_layers", type=int, default=-1, help="GPU 层数")
    parser.add_argument("--max_tokens", type=int, default=512, help="最大生成长度")
    parser.add_argument("--temperature", type=float, default=0.0)
    args = parser.parse_args()

    llm = load_model(
        args.model,
        n_ctx=args.n_ctx,
        n_threads=args.n_threads,
        n_gpu_layers=args.n_gpu_layers,
    )

    result = generate(llm, args.prompt, args.max_tokens, args.temperature)

    print(f"\n[输入] {args.prompt}")
    print(f"[输出] {result['text']}")
    print(f"[Tokens] input={result['input_tokens']}, output={result['output_tokens']}")


if __name__ == "__main__":
    main()
