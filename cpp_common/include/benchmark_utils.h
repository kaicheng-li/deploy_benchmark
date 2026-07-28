#pragma once

#include <chrono>
#include <string>
#include <vector>

namespace deploy_bench {

// ── 高精度计时器 ──────────────────────────────────────────────
class Timer {
public:
    Timer() { reset(); }

    void reset() { start_ = std::chrono::high_resolution_clock::now(); }

    // 返回已经过的时间 (毫秒)
    double elapsed_ms() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(now - start_).count();
    }

    // 返回已经过的时间 (微秒)
    double elapsed_us() const {
        auto now = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::micro>(now - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ── 基准测试结果 ──────────────────────────────────────────────
struct BenchResult {
    std::string model_path;
    int num_warmup = 0;
    int num_iterations = 0;

    // 延迟统计 (ms)
    double latency_avg = 0.0;
    double latency_min = 0.0;
    double latency_max = 0.0;
    double latency_p50 = 0.0;
    double latency_p95 = 0.0;
    double latency_p99 = 0.0;

    // 吞吐 (tokens/s)
    double throughput = 0.0;

    // 资源
    size_t memory_usage_bytes = 0;

    // 每轮延迟
    std::vector<double> per_iteration_latency;

    void compute_stats();
    void print() const;
};

// ── 工具函数 ──────────────────────────────────────────────────
// 从一组延迟值计算百分位
double percentile(const std::vector<double>& sorted_vals, double pct);

// 获取当前进程 RSS 内存 (bytes)
size_t get_current_rss();

// 获取 GPU 显存 (bytes) — 需要 CUDA
size_t get_gpu_memory_used(int device_id = 0);

// 预热运行
template <typename Func>
void warmup(Func&& func, int times = 5) {
    for (int i = 0; i < times; ++i) {
        func();
    }
}

}  // namespace deploy_bench
