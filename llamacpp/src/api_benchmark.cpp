// OpenAI-compatible SSE benchmark client for llama.cpp's native C++ server.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <functional>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <netdb.h>
#include <numeric>
#include <optional>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include <arpa/inet.h>
#include <sys/socket.h>
#include <unistd.h>

namespace {

using Clock = std::chrono::steady_clock;

struct Config {
    std::string host = "127.0.0.1";
    int port = 8080;
    int server_pid = 0;
    std::string image_base64_path;
    std::string output_dir = "results";
    std::string device;
    std::string model = "Qwen3-VL-8B-Instruct";
    std::string prompt = "请用中文描述这张图片中的主要内容。";
    std::vector<int> concurrencies = {1, 2, 4};
    int warmup = 10;
    int requests = 100;
    int max_tokens = 32;
    int max_model_len = 4096;
    double cold_start_ms = 0.0;
    double model_load_gpu_mb = 0.0;
};

struct RequestResult {
    int status_code = 0;
    bool success = false;
    int input_tokens = -1;
    int output_tokens = -1;
    double ttft_ms = 0.0;
    double tpot_ms = 0.0;
    double e2e_ms = 0.0;
    std::string error;
};

struct ResourceSample {
    double cpu_rss_mb = 0.0;
    double gpu_memory_mb = 0.0;
};

struct Summary {
    int concurrency = 1;
    int total = 0;
    int success = 0;
    int output_tokens = 0;
    double wall_seconds = 0.0;
    double warmup_seconds = 0.0;
    ResourceSample peak;
    std::vector<RequestResult> requests;
};

void print_usage(const char * program) {
    std::cout
        << "Usage: " << program << " --server-pid PID --image-base64 FILE --device cpu|cuda [options]\n"
        << "  --host HOST                 API host (default: 127.0.0.1)\n"
        << "  --port PORT                 API port (default: 8080)\n"
        << "  --output-dir DIR            Result directory (default: results)\n"
        << "  --concurrency 1,2,4         Comma-separated concurrency levels\n"
        << "  --warmup N                  Warmup requests per level (default: 10)\n"
        << "  --requests N                Measured requests per level (default: 100)\n"
        << "  --cold-start-ms N           Server process start to /health ready\n"
        << "  --model-load-gpu-mb N       GPU memory immediately after model load\n";
}

bool read_option(int argc, char ** argv, int & index, std::string & value) {
    if (index + 1 >= argc) return false;
    value = argv[++index];
    return true;
}

std::vector<int> parse_concurrencies(const std::string & input) {
    std::vector<int> values;
    std::stringstream stream(input);
    std::string item;
    while (std::getline(stream, item, ',')) {
        const int value = std::atoi(item.c_str());
        if (value > 0) values.push_back(value);
    }
    return values;
}

bool parse_args(int argc, char ** argv, Config & cfg) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        std::string value;
        if (arg == "--host" && read_option(argc, argv, i, cfg.host)) continue;
        if (arg == "--port" && read_option(argc, argv, i, value)) { cfg.port = std::atoi(value.c_str()); continue; }
        if (arg == "--server-pid" && read_option(argc, argv, i, value)) { cfg.server_pid = std::atoi(value.c_str()); continue; }
        if (arg == "--image-base64" && read_option(argc, argv, i, cfg.image_base64_path)) continue;
        if (arg == "--output-dir" && read_option(argc, argv, i, cfg.output_dir)) continue;
        if (arg == "--device" && read_option(argc, argv, i, cfg.device)) continue;
        if (arg == "--model" && read_option(argc, argv, i, cfg.model)) continue;
        if (arg == "--prompt" && read_option(argc, argv, i, cfg.prompt)) continue;
        if (arg == "--concurrency" && read_option(argc, argv, i, value)) { cfg.concurrencies = parse_concurrencies(value); continue; }
        if (arg == "--warmup" && read_option(argc, argv, i, value)) { cfg.warmup = std::atoi(value.c_str()); continue; }
        if (arg == "--requests" && read_option(argc, argv, i, value)) { cfg.requests = std::atoi(value.c_str()); continue; }
        if (arg == "--max-tokens" && read_option(argc, argv, i, value)) { cfg.max_tokens = std::atoi(value.c_str()); continue; }
        if (arg == "--max-model-len" && read_option(argc, argv, i, value)) { cfg.max_model_len = std::atoi(value.c_str()); continue; }
        if (arg == "--cold-start-ms" && read_option(argc, argv, i, value)) { cfg.cold_start_ms = std::atof(value.c_str()); continue; }
        if (arg == "--model-load-gpu-mb" && read_option(argc, argv, i, value)) { cfg.model_load_gpu_mb = std::atof(value.c_str()); continue; }
        if (arg == "--help" || arg == "-h") { print_usage(argv[0]); std::exit(0); }
        return false;
    }
    return cfg.server_pid > 0 && !cfg.image_base64_path.empty() &&
        (cfg.device == "cpu" || cfg.device == "cuda") && !cfg.concurrencies.empty() &&
        cfg.warmup >= 0 && cfg.requests > 0;
}

std::string read_file_without_newlines(const std::string & path) {
    std::ifstream file(path, std::ios::binary);
    std::string output((std::istreambuf_iterator<char>(file)), std::istreambuf_iterator<char>());
    output.erase(std::remove(output.begin(), output.end(), '\n'), output.end());
    output.erase(std::remove(output.begin(), output.end(), '\r'), output.end());
    return output;
}

std::string json_escape(const std::string & input) {
    std::string output;
    for (const char c : input) {
        if (c == '"') output += "\\\"";
        else if (c == '\\') output += "\\\\";
        else if (c == '\n') output += "\\n";
        else output += c;
    }
    return output;
}

std::string make_payload(const Config & cfg, const std::string & image_base64) {
    std::ostringstream output;
    output << "{\"model\":\"" << json_escape(cfg.model) << "\","
           << "\"stream\":true,\"stream_options\":{\"include_usage\":true},"
           << "\"temperature\":0,\"top_p\":1,\"max_tokens\":" << cfg.max_tokens << ','
           << "\"messages\":[{\"role\":\"user\",\"content\":["
           << "{\"type\":\"text\",\"text\":\"" << json_escape(cfg.prompt) << "\"},"
           << "{\"type\":\"image_url\",\"image_url\":{\"url\":\"data:image/png;base64,"
           << image_base64 << "\"}}]}]}";
    return output.str();
}

int connect_socket(const std::string & host, int port, std::string & error) {
    addrinfo hints{};
    hints.ai_family = AF_UNSPEC;
    hints.ai_socktype = SOCK_STREAM;
    addrinfo * result = nullptr;
    const std::string port_text = std::to_string(port);
    if (getaddrinfo(host.c_str(), port_text.c_str(), &hints, &result) != 0) {
        error = "getaddrinfo failed";
        return -1;
    }
    int fd = -1;
    for (addrinfo * address = result; address; address = address->ai_next) {
        fd = socket(address->ai_family, address->ai_socktype, address->ai_protocol);
        if (fd >= 0 && connect(fd, address->ai_addr, address->ai_addrlen) == 0) break;
        if (fd >= 0) close(fd);
        fd = -1;
    }
    freeaddrinfo(result);
    if (fd < 0) error = "connect failed";
    return fd;
}

bool send_all(int fd, const std::string & data) {
    size_t sent = 0;
    while (sent < data.size()) {
        const ssize_t count = send(fd, data.data() + sent, data.size() - sent, 0);
        if (count <= 0) return false;
        sent += static_cast<size_t>(count);
    }
    return true;
}

std::optional<int> json_int(const std::string & data, const std::string & key) {
    const size_t found = data.find("\"" + key + "\":");
    if (found == std::string::npos) return std::nullopt;
    char * end = nullptr;
    const long value = std::strtol(data.c_str() + found + key.size() + 3, &end, 10);
    return end == data.c_str() + found + key.size() + 3 ? std::nullopt : std::optional<int>(static_cast<int>(value));
}

bool contains_content(const std::string & data) {
    const std::string marker = "\"content\":\"";
    const size_t found = data.find(marker);
    return found != std::string::npos && found + marker.size() < data.size() && data[found + marker.size()] != '"';
}

class ChunkedSseReader {
public:
    explicit ChunkedSseReader(std::function<void(const std::string &)> on_event)
        : on_event_(std::move(on_event)) {}

    void append(const char * data, size_t size) {
        raw_.append(data, size);
        while (true) {
            const size_t line_end = raw_.find("\r\n");
            if (line_end == std::string::npos) return;
            const std::string length_text = raw_.substr(0, line_end);
            char * end = nullptr;
            const unsigned long length = std::strtoul(length_text.c_str(), &end, 16);
            if (end == length_text.c_str()) return;
            const size_t content_start = line_end + 2;
            if (raw_.size() < content_start + length + 2) return;
            if (length == 0) { raw_.clear(); return; }
            sse_.append(raw_, content_start, length);
            raw_.erase(0, content_start + length + 2);
            consume_sse();
        }
    }

private:
    void consume_sse() {
        while (true) {
            const size_t line_end = sse_.find('\n');
            if (line_end == std::string::npos) return;
            std::string line = sse_.substr(0, line_end);
            sse_.erase(0, line_end + 1);
            if (!line.empty() && line.back() == '\r') line.pop_back();
            if (line.rfind("data: ", 0) == 0) on_event_(line.substr(6));
        }
    }

    std::function<void(const std::string &)> on_event_;
    std::string raw_;
    std::string sse_;
};

RequestResult perform_request(const Config & cfg, const std::string & payload) {
    RequestResult result;
    std::string error;
    const int fd = connect_socket(cfg.host, cfg.port, error);
    if (fd < 0) { result.error = error; return result; }

    const std::string request = "POST /v1/chat/completions HTTP/1.1\r\nHost: " + cfg.host +
        "\r\nContent-Type: application/json\r\nAccept: text/event-stream\r\nConnection: close\r\nContent-Length: " +
        std::to_string(payload.size()) + "\r\n\r\n" + payload;
    const auto started = Clock::now();
    if (!send_all(fd, request)) {
        close(fd);
        result.error = "send failed";
        return result;
    }

    bool headers_ready = false;
    bool done = false;
    std::string headers;
    std::optional<Clock::time_point> first_token;
    std::optional<Clock::time_point> last_token;
    ChunkedSseReader reader([&](const std::string & event) {
        if (event == "[DONE]") { done = true; return; }
        const auto now = Clock::now();
        if (contains_content(event)) {
            if (!first_token) first_token = now;
            last_token = now;
        }
        if (const auto value = json_int(event, "prompt_tokens")) result.input_tokens = *value;
        if (const auto value = json_int(event, "completion_tokens")) result.output_tokens = *value;
    });

    char buffer[16384];
    while (!done) {
        const ssize_t count = recv(fd, buffer, sizeof(buffer), 0);
        if (count <= 0) break;
        if (!headers_ready) {
            headers.append(buffer, static_cast<size_t>(count));
            const size_t header_end = headers.find("\r\n\r\n");
            if (header_end == std::string::npos) continue;
            const size_t first_space = headers.find(' ');
            if (first_space != std::string::npos) result.status_code = std::atoi(headers.c_str() + first_space + 1);
            const size_t body_start = header_end + 4;
            if (headers.size() > body_start) reader.append(headers.data() + body_start, headers.size() - body_start);
            headers.clear();
            headers_ready = true;
        } else {
            reader.append(buffer, static_cast<size_t>(count));
        }
    }
    close(fd);
    const auto finished = Clock::now();
    result.e2e_ms = std::chrono::duration<double, std::milli>(finished - started).count();
    result.ttft_ms = first_token
        ? std::chrono::duration<double, std::milli>(*first_token - started).count()
        : result.e2e_ms;
    const int generated = std::max(result.output_tokens, 0);
    result.tpot_ms = last_token && first_token && generated > 1
        ? std::chrono::duration<double, std::milli>(*last_token - *first_token).count() / (generated - 1)
        : 0.0;
    result.success = done && result.status_code == 200 && result.input_tokens >= 0 && result.output_tokens >= 0;
    if (!result.success && result.error.empty()) result.error = done ? "missing usage or non-200 response" : "stream ended before [DONE]";
    return result;
}

double read_rss_mb(int pid) {
    std::ifstream file("/proc/" + std::to_string(pid) + "/status");
    std::string key;
    while (file >> key) {
        if (key == "VmRSS:") {
            double kb = 0.0;
            file >> kb;
            return kb / 1024.0;
        }
        std::string rest;
        std::getline(file, rest);
    }
    return 0.0;
}

double read_gpu_memory_mb(int pid) {
    std::string command = "nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader,nounits 2>/dev/null";
    FILE * pipe = popen(command.c_str(), "r");
    if (!pipe) return 0.0;
    char line[256];
    double result = 0.0;
    while (fgets(line, sizeof(line), pipe)) {
        int listed_pid = 0;
        double memory = 0.0;
        if (std::sscanf(line, "%d, %lf", &listed_pid, &memory) == 2 && listed_pid == pid) result = memory;
    }
    pclose(pipe);
    return result;
}

class ResourceMonitor {
public:
    ResourceMonitor(int pid, bool read_gpu) : pid_(pid), read_gpu_(read_gpu) {}

    void start() {
        running_ = true;
        thread_ = std::thread([this] {
            while (running_) {
                peak_.cpu_rss_mb = std::max(peak_.cpu_rss_mb, read_rss_mb(pid_));
                if (read_gpu_) peak_.gpu_memory_mb = std::max(peak_.gpu_memory_mb, read_gpu_memory_mb(pid_));
                std::this_thread::sleep_for(std::chrono::milliseconds(250));
            }
        });
    }

    ResourceSample stop() {
        running_ = false;
        if (thread_.joinable()) thread_.join();
        return peak_;
    }

private:
    int pid_;
    bool read_gpu_;
    std::atomic<bool> running_{false};
    std::thread thread_;
    ResourceSample peak_;
};

std::vector<RequestResult> run_requests(const Config & cfg, const std::string & payload, int count, int concurrency, double & wall_seconds) {
    std::vector<RequestResult> results(static_cast<size_t>(count));
    std::atomic<int> next{0};
    const auto started = Clock::now();
    std::vector<std::thread> workers;
    for (int worker = 0; worker < concurrency; ++worker) {
        workers.emplace_back([&] {
            while (true) {
                const int index = next.fetch_add(1);
                if (index >= count) return;
                results[static_cast<size_t>(index)] = perform_request(cfg, payload);
            }
        });
    }
    for (auto & worker : workers) worker.join();
    wall_seconds = std::chrono::duration<double>(Clock::now() - started).count();
    return results;
}

double percentile(std::vector<double> values, double ratio) {
    if (values.empty()) return 0.0;
    std::sort(values.begin(), values.end());
    const double index = ratio * (values.size() - 1);
    const size_t low = static_cast<size_t>(std::floor(index));
    const size_t high = static_cast<size_t>(std::ceil(index));
    return values[low] + (values[high] - values[low]) * (index - low);
}

struct LatencyStats { double average = 0.0; double p50 = 0.0; double p95 = 0.0; double p99 = 0.0; };

LatencyStats latency_stats(const std::vector<RequestResult> & requests, double RequestResult::* member) {
    std::vector<double> values;
    for (const auto & request : requests) if (request.success) values.push_back(request.*member);
    LatencyStats result;
    if (values.empty()) return result;
    result.average = std::accumulate(values.begin(), values.end(), 0.0) / values.size();
    result.p50 = percentile(values, 0.50);
    result.p95 = percentile(values, 0.95);
    result.p99 = percentile(values, 0.99);
    return result;
}

std::string escape_csv(const std::string & value) {
    std::string result = "\"";
    for (const char c : value) result += c == '"' ? "\"\"" : std::string(1, c);
    return result + "\"";
}

void write_raw_csv(const Config & cfg, const Summary & summary) {
    const std::string path = cfg.output_dir + "/llamacpp_api_" + cfg.device + "_c" + std::to_string(summary.concurrency) + "_requests.csv";
    std::ofstream file(path);
    file << "request_index,status_code,success,input_tokens,output_tokens,ttft_ms,tpot_ms,e2e_ms,error\n";
    for (size_t i = 0; i < summary.requests.size(); ++i) {
        const auto & request = summary.requests[i];
        file << i << ',' << request.status_code << ',' << (request.success ? 1 : 0) << ','
             << request.input_tokens << ',' << request.output_tokens << ',' << std::fixed << std::setprecision(3)
             << request.ttft_ms << ',' << request.tpot_ms << ',' << request.e2e_ms << ',' << escape_csv(request.error) << '\n';
    }
}

void write_report(const Config & cfg, const std::vector<Summary> & summaries) {
    const std::string base = cfg.output_dir + "/llamacpp_api_" + cfg.device + "_summary";
    std::ofstream markdown(base + ".md");
    std::ofstream json(base + ".json");
    markdown << "# llama.cpp C++ OpenAI API Benchmark\n\n"
             << "- Model: " << cfg.model << "\n- Device: " << cfg.device
             << "\n- Endpoint: /v1/chat/completions\n- stream: true\n- temperature: 0\n- top_p: 1\n"
             << "- max_tokens: " << cfg.max_tokens << "\n- max_model_len: " << cfg.max_model_len
             << "\n- cold start: " << std::fixed << std::setprecision(3) << cfg.cold_start_ms << " ms\n"
             << "- model-load GPU memory: " << cfg.model_load_gpu_mb << " MiB\n\n"
             << "| Concurrency | Success | TTFT avg/p50/p95/p99 (ms) | TPOT avg/p50/p95/p99 (ms) | E2E avg/p50/p95/p99 (ms) | req/s | tok/s | Warmup (s) | Steady peak GPU/RSS (MiB) |\n"
             << "|---:|---:|---|---|---|---:|---:|---:|---:|\n";
    json << "{\n  \"model\": \"" << cfg.model << "\",\n  \"device\": \"" << cfg.device
         << "\",\n  \"cold_start_ms\": " << cfg.cold_start_ms
         << ",\n  \"model_load_gpu_mb\": " << cfg.model_load_gpu_mb << ",\n  \"levels\": [\n";
    for (size_t index = 0; index < summaries.size(); ++index) {
        const auto & summary = summaries[index];
        const auto ttft = latency_stats(summary.requests, &RequestResult::ttft_ms);
        const auto tpot = latency_stats(summary.requests, &RequestResult::tpot_ms);
        const auto e2e = latency_stats(summary.requests, &RequestResult::e2e_ms);
        const double req_per_second = summary.wall_seconds > 0.0 ? summary.success / summary.wall_seconds : 0.0;
        const double tok_per_second = summary.wall_seconds > 0.0 ? summary.output_tokens / summary.wall_seconds : 0.0;
        markdown << '|' << summary.concurrency << " | " << summary.success << '/' << summary.total << " ("
                 << std::setprecision(1) << (100.0 * summary.success / summary.total) << "%) | "
                 << std::setprecision(3) << ttft.average << '/' << ttft.p50 << '/' << ttft.p95 << '/' << ttft.p99 << " | "
                 << tpot.average << '/' << tpot.p50 << '/' << tpot.p95 << '/' << tpot.p99 << " | "
                 << e2e.average << '/' << e2e.p50 << '/' << e2e.p95 << '/' << e2e.p99 << " | "
                 << req_per_second << " | " << tok_per_second << " | " << summary.warmup_seconds << " | "
                 << summary.peak.gpu_memory_mb << '/' << summary.peak.cpu_rss_mb << " |\n";
        json << "    {\"concurrency\": " << summary.concurrency << ", \"total\": " << summary.total
             << ", \"success\": " << summary.success << ", \"success_rate\": " << (100.0 * summary.success / summary.total)
             << ", \"wall_seconds\": " << summary.wall_seconds << ", \"warmup_seconds\": " << summary.warmup_seconds
             << ", \"output_tokens\": " << summary.output_tokens << ", \"req_per_second\": " << req_per_second
             << ", \"tok_per_second\": " << tok_per_second
             << ", \"peak_gpu_memory_mb\": " << summary.peak.gpu_memory_mb
             << ", \"peak_cpu_rss_mb\": " << summary.peak.cpu_rss_mb
             << ", \"ttft_ms\": {\"avg\": " << ttft.average << ", \"p50\": " << ttft.p50 << ", \"p95\": " << ttft.p95 << ", \"p99\": " << ttft.p99 << "}"
             << ", \"tpot_ms\": {\"avg\": " << tpot.average << ", \"p50\": " << tpot.p50 << ", \"p95\": " << tpot.p95 << ", \"p99\": " << tpot.p99 << "}"
             << ", \"e2e_ms\": {\"avg\": " << e2e.average << ", \"p50\": " << e2e.p50 << ", \"p95\": " << e2e.p95 << ", \"p99\": " << e2e.p99 << "}}"
             << (index + 1 == summaries.size() ? "\n" : ",\n");
    }
    json << "  ]\n}\n";
}

}  // namespace

int main(int argc, char ** argv) {
    Config cfg;
    if (!parse_args(argc, argv, cfg)) { print_usage(argv[0]); return 1; }
    const std::string image_base64 = read_file_without_newlines(cfg.image_base64_path);
    if (image_base64.empty()) { std::cerr << "Unable to read image Base64: " << cfg.image_base64_path << '\n'; return 1; }
    const std::string payload = make_payload(cfg, image_base64);
    std::filesystem::create_directories(cfg.output_dir);

    std::vector<Summary> summaries;
    for (const int concurrency : cfg.concurrencies) {
        std::cout << "Running concurrency " << concurrency << ": warmup=" << cfg.warmup << ", requests=" << cfg.requests << std::endl;
        Summary summary;
        summary.concurrency = concurrency;
        double ignored = 0.0;
        run_requests(cfg, payload, cfg.warmup, concurrency, summary.warmup_seconds);
        ResourceMonitor monitor(cfg.server_pid, true);
        monitor.start();
        summary.requests = run_requests(cfg, payload, cfg.requests, concurrency, summary.wall_seconds);
        summary.peak = monitor.stop();
        summary.total = static_cast<int>(summary.requests.size());
        for (const auto & request : summary.requests) {
            if (request.success) {
                ++summary.success;
                summary.output_tokens += request.output_tokens;
            }
        }
        write_raw_csv(cfg, summary);
        std::cout << "  success=" << summary.success << '/' << summary.total
                  << ", wall=" << std::fixed << std::setprecision(3) << summary.wall_seconds << " s" << std::endl;
        summaries.push_back(std::move(summary));
    }
    write_report(cfg, summaries);
    std::cout << "Reports: " << cfg.output_dir << "/llamacpp_api_" << cfg.device << "_summary.{md,json}" << std::endl;
    return 0;
}
