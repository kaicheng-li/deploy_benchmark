#include "benchmark_utils.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>

#ifdef _WIN32
#include <windows.h>
#include <psapi.h>
#else
#include <sys/resource.h>
#include <unistd.h>
#endif

#ifdef HAS_CUDA
#include <cuda_runtime.h>
#endif

namespace deploy_bench {

// ── 百分位计算 ────────────────────────────────────────────────
double percentile(const std::vector<double>& sorted_vals, double pct) {
    if (sorted_vals.empty()) return 0.0;
    if (sorted_vals.size() == 1) return sorted_vals[0];

    double idx = pct / 100.0 * (sorted_vals.size() - 1);
    size_t lo = static_cast<size_t>(std::floor(idx));
    size_t hi = static_cast<size_t>(std::ceil(idx));
    if (lo == hi) return sorted_vals[lo];

    double frac = idx - lo;
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac;
}

// ── RSS 内存 ──────────────────────────────────────────────────
size_t get_current_rss() {
#ifdef _WIN32
    PROCESS_MEMORY_COUNTERS pmc;
    if (GetProcessMemoryInfo(GetCurrentProcess(), &pmc, sizeof(pmc))) {
        return pmc.WorkingSetSize;
    }
    return 0;
#else
    struct rusage usage;
    if (getrusage(RUSAGE_SELF, &usage) == 0) {
        return static_cast<size_t>(usage.ru_maxrss) * 1024;  // ru_maxrss is in KB on Linux
    }
    return 0;
#endif
}

// ── GPU 显存 ──────────────────────────────────────────────────
size_t get_gpu_memory_used(int device_id) {
#ifdef HAS_CUDA
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    cudaSetDevice(device_id);
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        return total_bytes - free_bytes;
    }
#endif
    (void)device_id;  // unused when no CUDA
    return 0;
}

// ── BenchResult 实现 ──────────────────────────────────────────
void BenchResult::compute_stats() {
    if (per_iteration_latency.empty()) return;

    std::vector<double> sorted = per_iteration_latency;
    std::sort(sorted.begin(), sorted.end());

    // 剔除预热轮次
    size_t count = sorted.size();
    size_t warmup_skip = static_cast<size_t>(num_warmup);
    if (warmup_skip >= count) warmup_skip = 0;

    std::vector<double> valid(sorted.begin() + warmup_skip, sorted.end());
    if (valid.empty()) valid = sorted;  // fallback

    latency_min = valid.front();
    latency_max = valid.back();
    latency_avg = std::accumulate(valid.begin(), valid.end(), 0.0) / valid.size();
    latency_p50 = percentile(valid, 50.0);
    latency_p95 = percentile(valid, 95.0);
    latency_p99 = percentile(valid, 99.0);

    num_iterations = static_cast<int>(valid.size());
    memory_usage_bytes = get_current_rss();
}

void BenchResult::print() const {
    std::cout << "========================================\n";
    std::cout << "  Benchmark: " << model_path << "\n";
    std::cout << "========================================\n";
    std::cout << "  Iterations : " << num_iterations << " (+" << num_warmup << " warmup)\n";
    std::cout << "  Latency    : avg=" << latency_avg << "ms, min=" << latency_min
              << "ms, max=" << latency_max << "ms\n";
    std::cout << "  Percentiles: p50=" << latency_p50 << "ms, p95=" << latency_p95
              << "ms, p99=" << latency_p99 << "ms\n";
    std::cout << "  Throughput : " << throughput << " tok/s\n";
    std::cout << "  Memory     : " << memory_usage_bytes / (1024.0 * 1024.0) << " MB\n";
    std::cout << "========================================\n";
}

}  // namespace deploy_bench
