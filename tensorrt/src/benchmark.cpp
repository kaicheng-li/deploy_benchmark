/**
 * TensorRT C++ 基准测试。
 *
 * 流程:
 *   1. 从 .engine 文件反序列化 TensorRT engine
 *   2. 分配 GPU 输入/输出 buffer
 *   3. 预热 + 多轮推理计时
 *   4. 输出延迟 / 吞吐 / 显存统计
 *
 * 构建:
 *   cd tensorrt && mkdir build && cd build
 *   cmake .. -DTensorRT_ROOT=/usr/src/tensorrt
 *   cmake --build . --config Release
 *
 * 使用:
 *   ./trt_benchmark --engine ./trt_engines/model.engine --iterations 100
 */

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <NvInfer.h>

#include "benchmark_utils.h"

// ── RAII 封装 ─────────────────────────────────────────────────
struct CudaBuffer {
    void* data = nullptr;
    size_t size = 0;

    bool alloc(size_t bytes) {
        size = bytes;
        cudaError_t err = cudaMalloc(&data, bytes);
        if (err != cudaSuccess) {
            std::cerr << "cudaMalloc failed: " << cudaGetErrorString(err) << "\n";
            return false;
        }
        return true;
    }

    bool copyFromHost(const void* src, size_t bytes) {
        cudaError_t err = cudaMemcpy(data, src, bytes, cudaMemcpyHostToDevice);
        return err == cudaSuccess;
    }

    bool copyToHost(void* dst, size_t bytes) {
        cudaError_t err = cudaMemcpy(dst, data, bytes, cudaMemcpyDeviceToHost);
        return err == cudaSuccess;
    }

    ~CudaBuffer() {
        if (data) cudaFree(data);
    }
};

struct TrtLogger : public nvinfer1::ILogger {
    void log(Severity severity, const char* msg) noexcept override {
        if (severity <= Severity::kWARNING) {
            std::cout << "[TensorRT] " << msg << std::endl;
        }
    }
};

// ── Engine 反序列化 ──────────────────────────────────────────
nvinfer1::ICudaEngine* loadEngine(const std::string& engine_path,
                                   nvinfer1::IRuntime* runtime) {
    std::ifstream file(engine_path, std::ios::binary | std::ios::ate);
    if (!file.is_open()) {
        std::cerr << "无法打开 engine 文件: " << engine_path << "\n";
        return nullptr;
    }

    std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);

    std::vector<char> buffer(size);
    if (!file.read(buffer.data(), size)) {
        std::cerr << "读取 engine 文件失败\n";
        return nullptr;
    }

    return runtime->deserializeCudaEngine(buffer.data(), size);
}

// ── 打印 tensor 信息 ─────────────────────────────────────────
void printTensorInfo(const nvinfer1::ICudaEngine* engine) {
    int nb = engine->getNbIOTensors();
    std::cout << "Engine tensors (" << nb << "):\n";
    for (int i = 0; i < nb; i++) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        auto dtype = engine->getTensorDataType(name);
        auto shape = engine->getTensorShape(name);
        std::cout << "  " << (mode == nvinfer1::TensorIOMode::kINPUT ? "[IN] " : "[OUT]")
                  << name << " | dtype=";
        switch (dtype) {
            case nvinfer1::DataType::kFLOAT:  std::cout << "float32"; break;
            case nvinfer1::DataType::kHALF:   std::cout << "float16"; break;
            case nvinfer1::DataType::kINT32:  std::cout << "int32";   break;
            case nvinfer1::DataType::kINT8:   std::cout << "int8";    break;
            default: std::cout << "unknown";
        }
        std::cout << " | shape=[";
        for (int j = 0; j < shape.nbDims; j++) {
            if (j > 0) std::cout << ",";
            std::cout << shape.d[j];
        }
        std::cout << "]\n";
    }
}

// ── 计算 tensor 字节数 ───────────────────────────────────────
size_t tensorBytes(const nvinfer1::ICudaEngine* engine, const char* name) {
    auto shape = engine->getTensorShape(name);
    auto dtype = engine->getTensorDataType(name);

    size_t elem_size = 4;  // default float32
    switch (dtype) {
        case nvinfer1::DataType::kFLOAT:  elem_size = 4; break;
        case nvinfer1::DataType::kHALF:   elem_size = 2; break;
        case nvinfer1::DataType::kINT32:  elem_size = 4; break;
        case nvinfer1::DataType::kINT8:   elem_size = 1; break;
        case nvinfer1::DataType::kBOOL:   elem_size = 1; break;
    }

    size_t total = elem_size;
    for (int i = 0; i < shape.nbDims; i++) {
        total *= shape.d[i];
    }
    return total;
}

// ── 准备输入数据 (随机填充) ──────────────────────────────────
std::vector<float> generateRandomInput(const nvinfer1::ICudaEngine* engine) {
    int nb = engine->getNbIOTensors();
    size_t max_size = 0;

    for (int i = 0; i < nb; i++) {
        const char* name = engine->getIOTensorName(i);
        if (engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT) {
            max_size = std::max(max_size, tensorBytes(engine, name));
        }
    }

    // float32 占 4 字节
    size_t num_floats = max_size / 4;
    std::vector<float> data(num_floats);

    // 用正态分布随机数填充
    for (size_t i = 0; i < num_floats; i++) {
        data[i] = static_cast<float>(rand()) / RAND_MAX * 2.0f - 1.0f;
    }
    return data;
}

// ── 命令行参数 ────────────────────────────────────────────────
struct Config {
    std::string engine_path;
    int warmup = 10;
    int iterations = 100;
    bool use_cuda_graph = false;
    bool verbose = false;
};

void printUsage(const char* prog) {
    std::cout << "Usage: " << prog << " [OPTIONS]\n"
              << "  --engine <path>       TensorRT .engine file (required)\n"
              << "  --warmup <n>          Warmup iterations (default: 10)\n"
              << "  --iterations <n>      Benchmark iterations (default: 100)\n"
              << "  --cuda-graph          Enable CUDA graph capture\n"
              << "  --verbose             Print detailed tensor info\n"
              << "  --help, -h            Show this message\n";
}

// ──────────────────────────────────────────────────────────────
//  主程序
// ──────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    Config cfg;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc)
            cfg.engine_path = argv[++i];
        else if (arg == "--warmup" && i + 1 < argc)
            cfg.warmup = std::stoi(argv[++i]);
        else if (arg == "--iterations" && i + 1 < argc)
            cfg.iterations = std::stoi(argv[++i]);
        else if (arg == "--cuda-graph")
            cfg.use_cuda_graph = true;
        else if (arg == "--verbose")
            cfg.verbose = true;
        else if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            return 0;
        }
    }

    if (cfg.engine_path.empty()) {
        std::cerr << "Error: --engine is required\n\n";
        printUsage(argv[0]);
        return 1;
    }

    // ── 1. 初始化 ────────────────────────────────────────────
    std::cout << "==============================================\n";
    std::cout << "  TensorRT C++ Benchmark\n";
    std::cout << "==============================================\n";
    std::cout << "Engine : " << cfg.engine_path << "\n\n";

    TrtLogger logger;
    auto* runtime = nvinfer1::createInferRuntime(logger);
    if (!runtime) {
        std::cerr << "创建 TensorRT runtime 失败\n";
        return 1;
    }

    auto* engine = loadEngine(cfg.engine_path, runtime);
    if (!engine) {
        std::cerr << "加载 engine 失败\n";
        runtime->destroy();
        return 1;
    }

    if (cfg.verbose) {
        printTensorInfo(engine);
    }

    auto* context = engine->createExecutionContext();
    if (!context) {
        std::cerr << "创建 execution context 失败\n";
        engine->destroy();
        runtime->destroy();
        return 1;
    }

    // ── 2. 分配 buffer ───────────────────────────────────────
    // 收集所有 IO tensors
    int nb_tensors = engine->getNbIOTensors();
    std::vector<std::string> input_names, output_names;
    std::vector<CudaBuffer> input_bufs, output_bufs;
    std::vector<std::vector<float>> host_inputs, host_outputs;

    for (int i = 0; i < nb_tensors; i++) {
        const char* name = engine->getIOTensorName(i);
        auto mode = engine->getTensorIOMode(name);
        size_t bytes = tensorBytes(engine, name);

        if (mode == nvinfer1::TensorIOMode::kINPUT) {
            input_names.push_back(name);
            CudaBuffer buf;
            buf.alloc(bytes);
            input_bufs.push_back(std::move(buf));

            // 生成随机输入
            auto input_data = generateRandomInput(engine);
            input_bufs.back().copyFromHost(input_data.data(), bytes);
            host_inputs.push_back(std::move(input_data));

            // 设置 tensor address
            context->setTensorAddress(name, input_bufs.back().data);

            std::cout << "[IN]  " << name << " | " << bytes / 1024.0f / 1024.0f << " MB\n";
        } else {
            output_names.push_back(name);
            CudaBuffer buf;
            buf.alloc(bytes);
            output_bufs.push_back(std::move(buf));

            std::vector<float> out(bytes / 4, 0.0f);
            host_outputs.push_back(out);

            context->setTensorAddress(name, output_bufs.back().data);

            std::cout << "[OUT] " << name << " | " << bytes / 1024.0f / 1024.0f << " MB\n";
        }
    }

    std::cout << "\n";

    // ── CUDA Graph (可选) ────────────────────────────────────
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t graph_exec = nullptr;

    if (cfg.use_cuda_graph) {
        std::cout << "Capturing CUDA graph..." << std::endl;
        cudaStream_t capture_stream;
        cudaStreamCreate(&capture_stream);

        cudaStreamBeginCapture(capture_stream, cudaStreamCaptureModeGlobal);
        context->enqueueV3(capture_stream);
        cudaStreamEndCapture(capture_stream, &graph);
        cudaGraphInstantiate(&graph_exec, graph, nullptr, nullptr, 0);

        cudaStreamDestroy(capture_stream);
        std::cout << "CUDA graph captured.\n\n";
    }

    // ── 3. 预热 ──────────────────────────────────────────────
    std::cout << "Warmup (" << cfg.warmup << " rounds)..." << std::endl;
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    for (int i = 0; i < cfg.warmup; i++) {
        if (cfg.use_cuda_graph && graph_exec) {
            cudaGraphLaunch(graph_exec, stream);
        } else {
            context->enqueueV3(stream);
        }
        cudaStreamSynchronize(stream);
    }

    // ── 4. Benchmark ─────────────────────────────────────────
    std::cout << "Benchmarking (" << cfg.iterations << " iterations)..." << std::endl;

    deploy_bench::BenchResult result;
    result.model_path = cfg.engine_path;
    result.num_warmup = cfg.warmup;

    deploy_bench::Timer total_timer;

    for (int i = 0; i < cfg.iterations; i++) {
        deploy_bench::Timer iter_timer;

        if (cfg.use_cuda_graph && graph_exec) {
            cudaGraphLaunch(graph_exec, stream);
        } else {
            context->enqueueV3(stream);
        }
        cudaStreamSynchronize(stream);

        double lat = iter_timer.elapsed_ms();
        result.per_iteration_latency.push_back(lat);

        if ((i + 1) % 50 == 0) {
            std::cout << "  [" << (i + 1) << "/" << cfg.iterations << "] "
                      << lat << " ms\n";
        }
    }

    double total_elapsed = total_timer.elapsed_ms();
    result.compute_stats();

    // ── 5. 输出 ──────────────────────────────────────────────
    result.print();

    // 额外统计
    double total_flops = cfg.iterations * 1e9;  // placeholder
    std::cout << "\n--- Additional Stats ---\n";
    std::cout << "Total time       : " << total_elapsed / 1000.0 << " s\n";
    std::cout << "Throughput       : " << (cfg.iterations / (total_elapsed / 1000.0))
              << " inf/s\n";
    std::cout << "GPU memory       : "
              << deploy_bench::get_gpu_memory_used(0) / (1024.0 * 1024.0) << " MB\n";
    std::cout << "CUDA Graph       : " << (cfg.use_cuda_graph ? "ON" : "OFF") << "\n";

    // ── 6. 清理 ──────────────────────────────────────────────
    if (graph_exec) cudaGraphExecDestroy(graph_exec);
    if (graph) cudaGraphDestroy(graph);
    cudaStreamDestroy(stream);

    context->destroy();
    engine->destroy();
    runtime->destroy();

    return 0;
}
