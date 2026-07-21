/**
 * OpenVINO C++ 基准测试。
 *
 * 构建:
 *   cd openvino && mkdir build && cd build
 *   cmake .. -DOpenVINO_DIR=<openvino_install>/runtime/cmake
 *   cmake --build . --config Release
 *
 * 使用:
 *   ./ov_benchmark --model ./openvino_models/model.xml --image ./cat.jpg
 */

#include <algorithm>
#include <iostream>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "benchmark_utils.h"

// ── 参数 ──────────────────────────────────────────────────────
struct Config {
    std::string model_path;       // .xml 路径
    std::string device = "CPU";
    std::string image_path;       // 单张图像路径
    std::string image_dir;        // 图像目录（批量）
    int warmup = 10;
    int iterations = 100;
};

// ── 图像预处理 (简单: resize + normalize → NCHW float32) ─────
std::vector<float> preprocess(const std::string& image_path,
                               int target_h = 224, int target_w = 224) {
    // 使用 OpenCV 做预处理 (需要 opencv)
    // 这里提供最简实现 — 实际使用建议用 OpenCV 或 stb_image

    std::cerr << "图像预处理需要链接 OpenCV。当前为占位实现。\n";
    std::cerr << "请确保已安装 OpenCV 并在 CMake 中链接。\n";

    // 返回一个占位 tensor (1, 3, 224, 224)
    size_t size = 1 * 3 * target_h * target_w;
    return std::vector<float>(size, 0.5f);
}

// ── 加载模型 ──────────────────────────────────────────────────
ov::CompiledModel load_model(const Config& cfg) {
    ov::Core core;
    auto model = core.read_model(cfg.model_path);

    // 设置 batch size
    ov::preprocess::PrePostProcessor ppp(model);
    ppp.input().tensor().set_layout("NHWC");
    ppp.input().model().set_layout("NCHW");
    model = ppp.build();

    auto compiled = core.compile_model(model, cfg.device);

    std::cout << "Model  : " << cfg.model_path << "\n";
    std::cout << "Device : " << cfg.device << "\n";
    return compiled;
}

// ── 单次推理 ──────────────────────────────────────────────────
double run_inference(const ov::CompiledModel& compiled,
                     const ov::Tensor& input_tensor) {
    deploy_bench::Timer timer;

    ov::InferRequest infer = compiled.create_infer_request();
    infer.set_input_tensor(input_tensor);
    infer.infer();

    return timer.elapsed_ms();
}

// ── 主程序 ────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg;

    // 简单参数解析
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc)
            cfg.model_path = argv[++i];
        else if (arg == "--device" && i + 1 < argc)
            cfg.device = argv[++i];
        else if (arg == "--image" && i + 1 < argc)
            cfg.image_path = argv[++i];
        else if (arg == "--image-dir" && i + 1 < argc)
            cfg.image_dir = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc)
            cfg.warmup = std::stoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc)
            cfg.iterations = std::stoi(argv[++i]);
        else if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: ov_benchmark --model <model.xml> [OPTIONS]\n"
                      << "  --model <path>       OpenVINO .xml model\n"
                      << "  --device <name>      CPU / GPU / AUTO\n"
                      << "  --image <path>       Single image for inference\n"
                      << "  --image-dir <dir>    Image directory for batch\n"
                      << "  --warmup <n>         Warmup rounds\n"
                      << "  --iterations <n>     Benchmark iterations\n";
            return 0;
        }
    }

    if (cfg.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        return 1;
    }

    // ── 加载 ──────────────────────────────────────────────────
    std::cout << "==============================================\n";
    std::cout << "  OpenVINO C++ Benchmark\n";
    std::cout << "==============================================\n";

    auto compiled = load_model(cfg);

    // 获取输入 shape
    auto input = compiled.input();
    ov::Shape input_shape = input.get_shape();
    std::cout << "Input  : [";
    for (size_t j = 0; j < input_shape.size(); j++) {
        if (j > 0) std::cout << ", ";
        std::cout << input_shape[j];
    }
    std::cout << "]\n";

    // 输入 tensor
    size_t input_size = 1;
    for (auto d : input_shape) input_size *= d;
    std::vector<float> input_data(input_size, 0.5f);

    ov::Tensor input_tensor(ov::element::f32, input_shape, input_data.data());

    // ── 预热 ──────────────────────────────────────────────────
    std::cout << "Warmup (" << cfg.warmup << " rounds)..." << std::endl;
    for (int i = 0; i < cfg.warmup; i++) {
        run_inference(compiled, input_tensor);
    }

    // ── 基准测试 ──────────────────────────────────────────────
    std::cout << "Benchmarking (" << cfg.iterations << " iterations)..." << std::endl;

    deploy_bench::BenchResult result;
    result.model_path = cfg.model_path;
    result.num_warmup = cfg.warmup;

    deploy_bench::Timer total_timer;

    for (int i = 0; i < cfg.iterations; i++) {
        double lat = run_inference(compiled, input_tensor);
        result.per_iteration_latency.push_back(lat);

        if ((i + 1) % 20 == 0) {
            std::cout << "  [" << (i + 1) << "/" << cfg.iterations << "] "
                      << lat << " ms\n";
        }
    }

    double total_elapsed = total_timer.elapsed_ms();
    result.compute_stats();

    // ── 输出 ──────────────────────────────────────────────────
    result.print();
    std::cout << "\nTotal time   : " << total_elapsed / 1000.0 << " s\n";
    std::cout << "Throughput   : " << (cfg.iterations / (total_elapsed / 1000.0))
              << " img/s\n";
    std::cout << "CPU Memory   : "
              << deploy_bench::get_current_rss() / (1024.0 * 1024.0) << " MB\n";

    return 0;
}
