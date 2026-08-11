/**
 * llama.cpp C++ 推理实现。
 *
 * 构建:
 *   cd llamacpp
 *   mkdir build && cd build
 *   cmake .. -DGGML_CUDA=ON
 *   cmake --build . --config Release
 *
 * 使用:
 *   ./llama_benchmark --model /path/to/model.gguf --prompt "Hello"
 */

#include <iostream>
#include <string>
#include <vector>

#include "llama.h"
#include "benchmark_utils.h"

// ── 默认参数 ──────────────────────────────────────────────────
struct LlamaParams {
    std::string model_path;
    int n_ctx = 4096;
    int n_threads = 8;
    int n_gpu_layers = -1;    // -1 = 全部 GPU 层
    int batch_size = 512;
    int seed = 42;
    int max_tokens = 512;
    float temperature = 0.0f;
};

// ── 模型加载 ──────────────────────────────────────────────────
struct LlamaContext {
    llama_model* model = nullptr;
    llama_context* ctx = nullptr;
    const llama_vocab* vocab = nullptr;

    ~LlamaContext() {
        if (ctx) llama_free(ctx);
        if (model) llama_model_free(model);
    }

    bool load(const LlamaParams& params) {
        llama_backend_init();

        // 模型参数
        llama_model_params model_params = llama_model_default_params();
        model_params.n_gpu_layers = params.n_gpu_layers;

        model = llama_model_load_from_file(params.model_path.c_str(), model_params);
        if (!model) {
            std::cerr << "Failed to load model: " << params.model_path << std::endl;
            return false;
        }

        // 上下文参数
        llama_context_params ctx_params = llama_context_default_params();
        ctx_params.n_ctx = params.n_ctx;
        ctx_params.n_threads = params.n_threads;
        ctx_params.n_batch = params.batch_size;

        ctx = llama_init_from_model(model, ctx_params);
        if (!ctx) {
            std::cerr << "Failed to create context" << std::endl;
            return false;
        }

        vocab = llama_model_get_vocab(model);
        return true;
    }
};

// ── 分词 ──────────────────────────────────────────────────────
std::vector<llama_token> tokenize(const llama_vocab* vocab, const std::string& text, bool add_bos = true) {
    std::vector<llama_token> tokens(text.size() + 16);
    int n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), tokens.size(), add_bos, true);
    if (n_tokens < 0) {
        tokens.resize(-n_tokens);
        n_tokens = llama_tokenize(vocab, text.c_str(), text.size(),
                                   tokens.data(), tokens.size(), add_bos, true);
    }
    tokens.resize(n_tokens);
    return tokens;
}

// ── 生成 ──────────────────────────────────────────────────────
std::string generate(LlamaContext& lctx, const std::string& prompt, int max_tokens,
                     int* out_tokens = nullptr) {
    llama_context* ctx = lctx.ctx;
    const llama_vocab* vocab = lctx.vocab;

    // 分词 prompt
    auto tokens = tokenize(vocab, prompt, true);

    // 准备 batch
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (size_t i = 0; i < tokens.size(); i++) {
        batch.token[i] = tokens[i];
        batch.pos[i] = static_cast<llama_pos>(i);
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == tokens.size() - 1) ? 1 : 0;
    }

    // 编码 prompt
    if (llama_decode(ctx, batch) != 0) {
        std::cerr << "Decode failed" << std::endl;
        llama_batch_free(batch);
        return "";
    }
    llama_batch_free(batch);

    // 生成
    std::string result;
    int n_cur = tokens.size();
    int generated = 0;

    for (int i = 0; i < max_tokens; i++) {
        // 采样
        float* logits = llama_get_logits_ith(ctx, -1);

        // 贪婪采样
        int n_vocab = llama_vocab_n_tokens(vocab);
        llama_token next_id = 0;
        float max_logit = logits[0];
        for (int j = 1; j < n_vocab; j++) {
            if (logits[j] > max_logit) {
                max_logit = logits[j];
                next_id = j;
            }
        }

        if (llama_vocab_is_eog(vocab, next_id)) break;
        generated++;

        // 解码 token
        char buf[256];
        int n_chars = llama_token_to_piece(vocab, next_id, buf, sizeof(buf), 0, true);
        if (n_chars > 0) {
            result.append(buf, n_chars);
        }

        // 下一轮
        llama_batch next_batch = llama_batch_init(1, 0, 1);
        next_batch.n_tokens = 1;
        next_batch.token[0] = next_id;
        next_batch.pos[0] = static_cast<llama_pos>(n_cur);
        next_batch.n_seq_id[0] = 1;
        next_batch.seq_id[0][0] = 0;
        next_batch.logits[0] = 1;
        if (llama_decode(ctx, next_batch) != 0) {
            llama_batch_free(next_batch);
            break;
        }
        llama_batch_free(next_batch);
        n_cur++;
    }

    if (out_tokens) *out_tokens = generated;
    return result;
}

// ── 单次推理入口 ──────────────────────────────────────────────
deploy_bench::BenchResult run_single(const LlamaParams& params, const std::string& prompt) {
    deploy_bench::BenchResult result;
    result.model_path = params.model_path;

    LlamaContext lctx;
    if (!lctx.load(params)) {
        std::cerr << "Failed to load model" << std::endl;
        return result;
    }

    deploy_bench::Timer timer;
    std::string output = generate(lctx, prompt, params.max_tokens);
    double elapsed = timer.elapsed_ms();

    result.per_iteration_latency.push_back(elapsed);
    result.throughput = output.size() / (elapsed / 1000.0);  // chars/s ≈ tokens/s
    result.compute_stats();

    std::cout << "\n[Prompt] " << prompt.substr(0, 100) << "..." << std::endl;
    std::cout << "[Output] " << output.substr(0, 200) << "..." << std::endl;
    std::cout << "[Latency] " << elapsed << " ms" << std::endl;

    return result;
}
