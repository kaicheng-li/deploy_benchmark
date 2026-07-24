/**
 * ONNX Runtime C++ benchmark — vision (RF-DETR) or qwen.
 *
 * Usage:
 *   ./onnx_benchmark --model ./vision.onnx --mode vision --image test.jpg
 *   ./onnx_benchmark --model ./qwen.onnx --mode qwen --prompt "Hello"
 */

#include <iostream>
#include <string>

#include "benchmark_utils.h"

namespace onnx_bench {
deploy_bench::BenchResult run_vision(struct OrtSession& ctx,
    const std::string& image_path, int target_size, int num_iterations);
deploy_bench::BenchResult run_qwen(struct OrtSession& ctx,
    int seq_len, int num_iterations);
}

#include "inference.cpp"

struct Config {
    std::string model_path;
    std::string mode = "vision";
    std::string image_path = "test.jpg";
    std::string provider = "CPUExecutionProvider";
    int threads = 4;
    int seq_len = 128;
    int target_size = 640;
    int iterations = 100;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model <path>    ONNX model path (required)\n"
              << "  --mode <name>     vision | qwen (default: vision)\n"
              << "  --image <path>    Image for vision mode\n"
              << "  --provider <name> CPUExecutionProvider | CUDAExecutionProvider\n"
              << "  --threads <n>     Intra-op threads (default: 4)\n"
              << "  --seq-len <n>     Qwen input seq length (default: 128)\n"
              << "  --target-size <n> Vision input size (default: 640)\n"
              << "  --iterations <n>  Benchmark iterations (default: 100)\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)      cfg.model_path = argv[++i];
        else if (arg == "--mode" && i + 1 < argc)  cfg.mode = argv[++i];
        else if (arg == "--image" && i + 1 < argc) cfg.image_path = argv[++i];
        else if (arg == "--provider" && i + 1 < argc) cfg.provider = argv[++i];
        else if (arg == "--threads" && i + 1 < argc) cfg.threads = std::stoi(argv[++i]);
        else if (arg == "--seq-len" && i + 1 < argc) cfg.seq_len = std::stoi(argv[++i]);
        else if (arg == "--target-size" && i + 1 < argc) cfg.target_size = std::stoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) cfg.iterations = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        return 1;
    }

    std::cout << "=== ONNX Runtime C++ Benchmark ===\n";
    std::cout << "Model : " << cfg.model_path << "\n";
    std::cout << "Mode  : " << cfg.mode << "\n\n";

    onnx_bench::OrtSession ctx(cfg.model_path, cfg.threads, cfg.provider);

    deploy_bench::BenchResult result;

    if (cfg.mode == "vision") {
        result = onnx_bench::run_vision(ctx, cfg.image_path, cfg.target_size, cfg.iterations);
    } else if (cfg.mode == "qwen") {
        result = onnx_bench::run_qwen(ctx, cfg.seq_len, cfg.iterations);
    } else {
        std::cerr << "Error: unknown mode '" << cfg.mode << "'\n";
        return 1;
    }

    result.print();
    std::cout << "CPU Memory: "
              << deploy_bench::get_current_rss() / (1024.0 * 1024.0) << " MB\n";
    return 0;
}
