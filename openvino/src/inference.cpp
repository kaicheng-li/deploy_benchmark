/**
 * OpenVINO C++ inference — vision (RF-DETR Seg) and qwen.
 */

#include <algorithm>
#include <cmath>
#include <iostream>
#include <random>
#include <string>
#include <vector>

#include <openvino/openvino.hpp>

#include "benchmark_utils.h"

#ifdef HAS_OPENCV
#include <opencv2/opencv.hpp>
#endif

namespace ov_bench {

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
        return {};
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

deploy_bench::BenchResult run_vision(const std::string& model_path,
                                      const std::string& device,
                                      const std::string& image_path,
                                      int target_size, int num_iterations) {
    deploy_bench::BenchResult result;
    result.model_path = model_path;

    ov::Core core;
    auto compiled = core.compile_model(model_path, device);

    auto pixel_values = preprocess(image_path, target_size, target_size);
    if (pixel_values.empty()) return result;

    ov::Shape img_shape = {1, 3, static_cast<size_t>(target_size), static_cast<size_t>(target_size)};
    ov::Shape mask_shape = {1, static_cast<size_t>(target_size), static_cast<size_t>(target_size)};

    ov::Tensor pixel_tensor(ov::element::f32, img_shape, pixel_values.data());
    std::vector<int64_t> mask_data(target_size * target_size, 1);
    ov::Tensor mask_tensor(ov::element::i64, mask_shape, mask_data.data());

    ov::InferRequest infer = compiled.create_infer_request();
    infer.set_input_tensor(pixel_tensor);
    // set pixel_mask if model has it
    for (auto& input : compiled.inputs()) {
        std::string name = input.get_any_name();
        if (name.find("mask") != std::string::npos)
            infer.set_tensor(name, mask_tensor);
    }

    for (int i = 0; i < 10; i++) infer.infer();

    for (int i = 0; i < num_iterations; i++) {
        deploy_bench::Timer t;
        infer.infer();
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

deploy_bench::BenchResult run_qwen(const std::string& model_path,
                                    const std::string& device,
                                    int seq_len, int num_iterations) {
    deploy_bench::BenchResult result;
    result.model_path = model_path;

    ov::Core core;
    auto compiled = core.compile_model(model_path, device);

    auto tokens = random_tokens(seq_len);
    ov::Shape shape = {1, static_cast<size_t>(seq_len)};
    ov::Tensor ids(ov::element::i64, shape, tokens.data());

    std::vector<int64_t> attn(seq_len, 1);
    ov::Tensor mask(ov::element::i64, shape, attn.data());

    ov::InferRequest infer = compiled.create_infer_request();
    infer.set_input_tensor(ids);
    for (auto& input : compiled.inputs()) {
        std::string name = input.get_any_name();
        if (name.find("attention") != std::string::npos || name.find("mask") != std::string::npos)
            infer.set_tensor(name, mask);
    }

    for (int i = 0; i < 5; i++) infer.infer();

    for (int i = 0; i < num_iterations; i++) {
        deploy_bench::Timer t;
        infer.infer();
        result.per_iteration_latency.push_back(t.elapsed_ms());
    }

    result.num_iterations = num_iterations;
    result.num_warmup = 5;
    result.compute_stats();
    return result;
}

}  // namespace ov_bench
