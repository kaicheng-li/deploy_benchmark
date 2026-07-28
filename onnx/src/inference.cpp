/**
 * ONNX Runtime C++ inference — vision (RF-DETR Seg) and qwen.
 *
 * Build with cmake and link onnxruntime.
 */

#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <onnxruntime_cxx_api.h>

#include "benchmark_utils.h"

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace onnx_bench {

// ══════════════════════════════════════════════════════════════════
//  OrtSession
// ══════════════════════════════════════════════════════════════════

struct OrtSession {
    Ort::Env env;
    Ort::SessionOptions opts;
    Ort::Session session;
    Ort::MemoryInfo memory_info;
    std::vector<std::string> input_names;
    std::vector<std::string> output_names;

    OrtSession(const std::string& model_path, int threads = 4,
               const std::string& provider = "CPUExecutionProvider")
        : env(ORT_LOGGING_LEVEL_WARNING, "onnx_bench")
        , memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
        , session(nullptr)
    {
        opts.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        opts.SetIntraOpNumThreads(threads);

        if (provider == "CUDAExecutionProvider") {
            OrtCUDAProviderOptions cuda_opts{};
            opts.AppendExecutionProvider_CUDA(cuda_opts);
        }

        session = Ort::Session(env, model_path.c_str(), opts);

        for (size_t i = 0; i < session.GetInputCount(); i++) {
            auto name = session.GetInputNameAllocated(i, Ort::Allocator::GetWithDefaultAllocator());
            input_names.push_back(name.get());
        }
        for (size_t i = 0; i < session.GetOutputCount(); i++) {
            auto name = session.GetOutputNameAllocated(i, Ort::Allocator::GetWithDefaultAllocator());
            output_names.push_back(name.get());
        }
    }

    std::vector<Ort::Value> run(const std::vector<Ort::Value>& inputs) {
        std::vector<const char*> in_names, out_names;
        for (auto& n : input_names)  in_names.push_back(n.c_str());
        for (auto& n : output_names) out_names.push_back(n.c_str());
        return session.Run(Ort::RunOptions{nullptr},
                           in_names.data(), inputs.data(), inputs.size(),
                           out_names.data(), out_names.size());
    }
};

// ══════════════════════════════════════════════════════════════════
//  vision — RF-DETR Seg
// ══════════════════════════════════════════════════════════════════

const float IMAGENET_MEAN[] = {0.485f, 0.456f, 0.406f};
const float IMAGENET_STD[]  = {0.229f, 0.224f, 0.225f};

std::vector<float> preprocess(const std::string& image_path, int h, int w) {
#ifdef HAS_OPENCV
    cv::Mat img = cv::imread(image_path, cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cerr << "Cannot read: " << image_path << std::endl;
        return std::vector<float>(3 * h * w, 0.0f);
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);
    cv::resize(img, img, cv::Size(w, h));
    img.convertTo(img, CV_32F, 1.0 / 255.0);

    std::vector<float> result(3 * h * w);
    for (int c = 0; c < 3; c++) {
        float* dst = result.data() + c * h * w;
        for (int y = 0; y < h; y++) {
            const float* src = img.ptr<float>(y);
            for (int x = 0; x < w; x++)
                dst[y * w + x] = (src[x * 3 + c] - IMAGENET_MEAN[c]) / IMAGENET_STD[c];
        }
    }
    return result;
#else
    std::cerr << "[WARN] OpenCV not available, using random input\n";
    size_t n = 3 * h * w;
    std::vector<float> result(n);
    for (size_t i = 0; i < n; i++) result[i] = static_cast<float>(rand()) / RAND_MAX;
    return result;
#endif
}

deploy_bench::BenchResult run_vision(OrtSession& ctx,
                                      const std::string& image_path,
                                      int target_size, int num_iterations) {
    deploy_bench::BenchResult result;
    result.model_path = "onnx_vision";

    auto pixel_values = preprocess(image_path, target_size, target_size);

    std::vector<int64_t> img_shape = {1, 3, target_size, target_size};
    std::vector<int64_t> mask_shape = {1, target_size, target_size};

    Ort::Value pixel_tensor = Ort::Value::CreateTensor<float>(
        ctx.memory_info, pixel_values.data(), pixel_values.size(),
        img_shape.data(), img_shape.size());

    std::vector<int64_t> mask_data(target_size * target_size, 1);
    Ort::Value mask_tensor = Ort::Value::CreateTensor<int64_t>(
        ctx.memory_info, mask_data.data(), mask_data.size(),
        mask_shape.data(), mask_shape.size());

    // warmup
    for (int i = 0; i < 10; i++) ctx.run({pixel_tensor, mask_tensor});

    // benchmark
    for (int i = 0; i < num_iterations; i++) {
        deploy_bench::Timer t;
        ctx.run({pixel_tensor, mask_tensor});
        result.per_iteration_latency.push_back(t.elapsed_ms());
    }

    result.num_iterations = num_iterations;
    result.num_warmup = 10;
    result.compute_stats();
    return result;
}

// ══════════════════════════════════════════════════════════════════
//  qwen
// ══════════════════════════════════════════════════════════════════

std::vector<int64_t> random_tokens(int seq_len, int vocab_size = 151936) {
    static std::mt19937 rng(42);
    std::uniform_int_distribution<int> dist(1, vocab_size - 1);
    std::vector<int64_t> tokens(seq_len);
    tokens[0] = 1;
    for (int i = 1; i < seq_len; i++) tokens[i] = dist(rng);
    return tokens;
}

deploy_bench::BenchResult run_qwen(OrtSession& ctx,
                                    int seq_len, int num_iterations) {
    deploy_bench::BenchResult result;
    result.model_path = "onnx_qwen";

    auto tokens = random_tokens(seq_len);
    std::vector<int64_t> shape = {1, static_cast<int64_t>(seq_len)};
    Ort::Value ids = Ort::Value::CreateTensor<int64_t>(
        ctx.memory_info, tokens.data(), tokens.size(), shape.data(), shape.size());

    std::vector<int64_t> attn(seq_len, 1);
    Ort::Value mask = Ort::Value::CreateTensor<int64_t>(
        ctx.memory_info, attn.data(), attn.size(), shape.data(), shape.size());

    for (int i = 0; i < 5; i++) ctx.run({ids, mask});

    for (int i = 0; i < num_iterations; i++) {
        deploy_bench::Timer t;
        ctx.run({ids, mask});
        result.per_iteration_latency.push_back(t.elapsed_ms());
    }

    result.num_iterations = num_iterations;
    result.num_warmup = 5;
    result.compute_stats();
    return result;
}

}  // namespace onnx_bench
