/**
 * llama.cpp 基准测试主程序。
 */

#include <iostream>
#include <fstream>
#include <filesystem>
#include <string>
#include <vector>

#include "benchmark_utils.h"
#include "inference.cpp"  // 引用上面的推理实现

// M-RoPE 模型（如 Qwen3-VL）要求同一序列的位置只能向前推进；
// 每轮独立 prompt 前清空 KV cache 元数据，位置即可重新从 0 开始。
static void clear_kv_cache(llama_context* ctx) {
    llama_memory_clear(llama_get_memory(ctx), false);
}

// ── 命令行参数解析 ────────────────────────────────────────────
void print_usage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --model <path>      GGUF model path (required)\n"
              << "  --prompt <text>     Input prompt\n"
              << "  --prompt-file <f>   File with one prompt per line\n"
              << "  --n-ctx <n>         Context size (default: 4096)\n"
              << "  --n-threads <n>     CPU threads (default: 8)\n"
              << "  --n-gpu-layers <n>  GPU layers (default: -1 = all)\n"
              << "  --max-tokens <n>    Max tokens to generate (default: 512)\n"
              << "  --warmup <n>        Warmup rounds (default: 5)\n"
              << "  --iterations <n>    Benchmark iterations (default: 20)\n"
              << "  --output <dir>      Report output directory (default: ../results)\n"
              << "  --report-prefix <s> Report file prefix (default: llamacpp_text_benchmark)\n"
              << std::endl;
}

int main(int argc, char** argv) {
    LlamaParams params;
    std::string prompt = "Hello, how are you?";
    std::string prompt_file;
    int warmup = 5;
    int iterations = 20;
    std::string output_dir = "../results";
    std::string report_prefix = "llamacpp_text_benchmark";

    // 解析参数
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--model" && i + 1 < argc) {
            params.model_path = argv[++i];
        } else if (arg == "--prompt" && i + 1 < argc) {
            prompt = argv[++i];
        } else if (arg == "--prompt-file" && i + 1 < argc) {
            prompt_file = argv[++i];
        } else if (arg == "--n-ctx" && i + 1 < argc) {
            params.n_ctx = std::stoi(argv[++i]);
        } else if (arg == "--n-threads" && i + 1 < argc) {
            params.n_threads = std::stoi(argv[++i]);
        } else if (arg == "--n-gpu-layers" && i + 1 < argc) {
            params.n_gpu_layers = std::stoi(argv[++i]);
        } else if (arg == "--max-tokens" && i + 1 < argc) {
            params.max_tokens = std::stoi(argv[++i]);
        } else if (arg == "--warmup" && i + 1 < argc) {
            warmup = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            iterations = std::stoi(argv[++i]);
        } else if (arg == "--output" && i + 1 < argc) {
            output_dir = argv[++i];
        } else if (arg == "--report-prefix" && i + 1 < argc) {
            report_prefix = argv[++i];
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }

    if (params.model_path.empty()) {
        std::cerr << "Error: --model is required\n";
        print_usage(argv[0]);
        return 1;
    }

    // 加载 prompts
    std::vector<std::string> prompts;
    if (!prompt_file.empty()) {
        std::ifstream file(prompt_file);
        std::string line;
        while (std::getline(file, line)) {
            if (!line.empty()) prompts.push_back(line);
        }
    }
    if (prompts.empty()) {
        prompts.push_back(prompt);
    }

    std::cout << "==============================================\n";
    std::cout << "  llama.cpp Benchmark\n";
    std::cout << "  Model: " << params.model_path << "\n";
    std::cout << "  GPU layers: " << params.n_gpu_layers << "\n";
    std::cout << "  Threads: " << params.n_threads << "\n";
    std::cout << "==============================================\n";

    // 预热
    std::cout << "Warming up (" << warmup << " rounds)..." << std::endl;
    {
        LlamaContext warmup_ctx;
        if (!warmup_ctx.load(params)) return 1;
        for (int i = 0; i < warmup; i++) {
            clear_kv_cache(warmup_ctx.ctx);
            generate(warmup_ctx, prompts[i % prompts.size()], 32);
        }
    }

    // 基准测试
    deploy_bench::BenchResult total_result;
    total_result.model_path = params.model_path;
    total_result.num_warmup = warmup;

    LlamaContext bench_ctx;
    if (!bench_ctx.load(params)) return 1;

    std::cout << "\nBenchmarking (" << iterations << " iterations)..." << std::endl;
    deploy_bench::Timer total_timer;
    size_t total_output_tokens = 0;

    for (int i = 0; i < iterations; i++) {
        const std::string& p = prompts[i % prompts.size()];

        clear_kv_cache(bench_ctx.ctx);
        deploy_bench::Timer iter_timer;
        int gen_tokens = 0;
        std::string output = generate(bench_ctx, p, params.max_tokens, &gen_tokens);
        double lat = iter_timer.elapsed_ms();
        total_output_tokens += static_cast<size_t>(gen_tokens);

        total_result.per_iteration_latency.push_back(lat);
        std::cout << "  [" << (i + 1) << "/" << iterations << "] "
                  << lat << "ms, output=" << output.size() << " chars" << std::endl;
    }

    double total_elapsed = total_timer.elapsed_ms();
    total_result.compute_stats();

    // 打印结果
    total_result.print();

    std::cout << "\nTotal time: " << total_elapsed / 1000.0 << " s" << std::endl;
    std::cout << "Avg throughput: "
              << (iterations / (total_elapsed / 1000.0)) << " req/s" << std::endl;

    // ── 保存报告 (MD / JSON / CSV) ──
    total_result.throughput =
        total_output_tokens / (total_elapsed / 1000.0);
    deploy_bench::ReportMeta meta;
    meta.framework = "llama.cpp";
    meta.model_name = std::filesystem::path(params.model_path).stem().string();
    meta.device = params.n_gpu_layers != 0 ? "cuda" : "cpu";
    meta.task_type = "text-generation";
    meta.total_output_tokens = total_output_tokens;
    meta.elapsed_seconds = total_elapsed / 1000.0;

    auto saved = deploy_bench::save_benchmark_report(
        total_result, meta, output_dir, report_prefix);
    std::cout << "\n报告已保存:\n";
    for (const auto& path : saved) {
        std::cout << "  " << path << std::endl;
    }

    llama_backend_free();
    return 0;
}
