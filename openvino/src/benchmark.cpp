/**
 * OpenVINO C++ benchmark — vision (RF-DETR) or qwen3.
 *
 * Usage:
 *   ./ov_benchmark --model ./vision.xml --mode vision --image test.jpg
 *   ./ov_benchmark --model ./qwen3.xml --mode qwen3
 */

#include <iostream>
#include <string>

#include <openvino/openvino.hpp>

#include "benchmark_utils.h"

namespace ov_bench {
deploy_bench::BenchResult run_vision(const std::string& model_path,
    const std::string& device, const std::string& image_path,
    int target_size, int num_iterations);
deploy_bench::BenchResult run_qwen(const std::string& model_path,
    const std::string& device, int seq_len, int num_iterations);
}

#include "inference.cpp"

struct Config {
    std::string model_path;
    std::string mode = "vision";
    std::string device = "CPU";
    std::string image_path = "test.jpg";
    int seq_len = 128;
    int target_size = 640;
    int iterations = 100;
};

void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model <path>    OpenVINO .xml model (required)\n"
              << "  --mode <name>     vision | qwen3\n"
              << "  --device <name>   CPU | GPU | AUTO\n"
              << "  --image <path>    Image for vision mode\n"
              << "  --seq-len <n>     Qwen3 input seq length\n"
              << "  --target-size <n> Vision input size\n"
              << "  --iterations <n>  Benchmark iterations\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)      cfg.model_path = argv[++i];
        else if (arg == "--mode" && i + 1 < argc)  cfg.mode = argv[++i];
        else if (arg == "--device" && i + 1 < argc) cfg.device = argv[++i];
        else if (arg == "--image" && i + 1 < argc) cfg.image_path = argv[++i];
        else if (arg == "--seq-len" && i + 1 < argc) cfg.seq_len = std::stoi(argv[++i]);
        else if (arg == "--target-size" && i + 1 < argc) cfg.target_size = std::stoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc) cfg.iterations = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") { print_usage(argv[0]); return 0; }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        return 1;
    }

    std::cout << "=== OpenVINO C++ Benchmark ===\n";
    std::cout << "Model : " << cfg.model_path << "\n";
    std::cout << "Mode  : " << cfg.mode << "\n";
    std::cout << "Device: " << cfg.device << "\n\n";

    deploy_bench::BenchResult result;

    if (cfg.mode == "vision") {
        result = ov_bench::run_vision(cfg.model_path, cfg.device,
                                       cfg.image_path, cfg.target_size, cfg.iterations);
    } else if (cfg.mode == "qwen3" || cfg.mode == "qwen") {
        result = ov_bench::run_qwen(cfg.model_path, cfg.device,
                                     cfg.seq_len, cfg.iterations);
    } else {
        std::cerr << "Error: unknown mode '" << cfg.mode << "'\n";
        return 1;
    }

    result.print();
    std::cout << "CPU Memory: "
              << deploy_bench::get_current_rss() / (1024.0 * 1024.0) << " MB\n";
    return 0;
}
