#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <thread>
#include <future>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <memory>
#include <iomanip>
#include <limits>
#include <cstdlib>

// C++20 single-file CIFAR-VGG8 inference implementation
// No external dependencies, FP32, multithreaded

namespace cifar_vgg8 {

// Basic tensor class for 4D data (batch, height, width, channels)
class Tensor {
public:
    std::vector<float> data;
    std::array<int, 4> shape; // [batch, height, width, channels]
    
    Tensor(int b, int h, int w, int c) : shape{b, h, w, c} {
        data.resize(b * h * w * c);
    }
    
    float& at(int b, int h, int w, int c) {
        return data[b * shape[1] * shape[2] * shape[3] + 
                   h * shape[2] * shape[3] + 
                   w * shape[3] + c];
    }
    
    const float& at(int b, int h, int w, int c) const {
        return data[b * shape[1] * shape[2] * shape[3] + 
                   h * shape[2] * shape[3] + 
                   w * shape[3] + c];
    }
    
    int size() const {
        return shape[0] * shape[1] * shape[2] * shape[3];
    }
};

// Convolution layer
class Conv2D {
public:
    Tensor weights;
    std::vector<float> bias;
    int stride;
    int padding;
    
    Conv2D(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0)
        : weights(out_channels, kernel_size, kernel_size, in_channels)
        , bias(out_channels)
        , stride(stride)
        , padding(padding) {
        
        // Xavier initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = std::sqrt(2.0f / (kernel_size * kernel_size * in_channels));
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (auto& w : weights.data) {
            w = dist(gen);
        }
        
        std::fill(bias.begin(), bias.end(), 0.0f);
    }
    
    Tensor forward(const Tensor& input) {
        int batch = input.shape[0];
        int in_h = input.shape[1];
        int in_w = input.shape[2];
        int in_c = input.shape[3];
        
        int out_h = (in_h + 2 * padding - weights.shape[1]) / stride + 1;
        int out_w = (in_w + 2 * padding - weights.shape[2]) / stride + 1;
        int out_c = weights.shape[0];
        
        Tensor output(batch, out_h, out_w, out_c);
        
        // Parallel convolution across batch and output channels
        const int num_threads = std::thread::hardware_concurrency();
        std::vector<std::future<void>> futures;
        
        auto conv_worker = [&](int start_idx, int end_idx) {
            for (int idx = start_idx; idx < end_idx; ++idx) {
                int b = idx / out_c;
                int oc = idx % out_c;
                
                for (int oh = 0; oh < out_h; ++oh) {
                    for (int ow = 0; ow < out_w; ++ow) {
                        float sum = bias[oc];
                        
                        for (int kh = 0; kh < weights.shape[1]; ++kh) {
                            for (int kw = 0; kw < weights.shape[2]; ++kw) {
                                for (int ic = 0; ic < in_c; ++ic) {
                                    int ih = oh * stride - padding + kh;
                                    int iw = ow * stride - padding + kw;
                                    
                                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                        sum += input.at(b, ih, iw, ic) * weights.at(oc, kh, kw, ic);
                                    }
                                }
                            }
                        }
                        
                        output.at(b, oh, ow, oc) = sum;
                    }
                }
            }
        };
        
        int total_work = batch * out_c;
        int work_per_thread = std::max(1, total_work / num_threads);
        
        for (int i = 0; i < num_threads; ++i) {
            int start = i * work_per_thread;
            int end = (i == num_threads - 1) ? total_work : (i + 1) * work_per_thread;
            if (start < total_work) {
                futures.push_back(std::async(std::launch::async, conv_worker, start, end));
            }
        }
        
        for (auto& future : futures) {
            future.wait();
        }
        
        return output;
    }
};

// ReLU activation
Tensor relu(const Tensor& input) {
    Tensor output = input;
    for (auto& val : output.data) {
        val = std::max(0.0f, val);
    }
    return output;
}

// 2x2 Max Pooling
Tensor max_pool2d(const Tensor& input, int pool_size = 2, int stride = 2) {
    int batch = input.shape[0];
    int in_h = input.shape[1];
    int in_w = input.shape[2];
    int channels = input.shape[3];
    
    int out_h = in_h / stride;
    int out_w = in_w / stride;
    
    Tensor output(batch, out_h, out_w, channels);
    
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            for (int oh = 0; oh < out_h; ++oh) {
                for (int ow = 0; ow < out_w; ++ow) {
                    float max_val = -std::numeric_limits<float>::infinity();
                    
                    for (int kh = 0; kh < pool_size; ++kh) {
                        for (int kw = 0; kw < pool_size; ++kw) {
                            int ih = oh * stride + kh;
                            int iw = ow * stride + kw;
                            if (ih < in_h && iw < in_w) {
                                max_val = std::max(max_val, input.at(b, ih, iw, c));
                            }
                        }
                    }
                    
                    output.at(b, oh, ow, c) = max_val;
                }
            }
        }
    }
    
    return output;
}

// Global Average Pooling
Tensor global_avg_pool(const Tensor& input) {
    int batch = input.shape[0];
    int height = input.shape[1];
    int width = input.shape[2];
    int channels = input.shape[3];
    
    Tensor output(batch, 1, 1, channels);
    
    for (int b = 0; b < batch; ++b) {
        for (int c = 0; c < channels; ++c) {
            float sum = 0.0f;
            for (int h = 0; h < height; ++h) {
                for (int w = 0; w < width; ++w) {
                    sum += input.at(b, h, w, c);
                }
            }
            output.at(b, 0, 0, c) = sum / (height * width);
        }
    }
    
    return output;
}

// Fully Connected layer
class Dense {
public:
    std::vector<std::vector<float>> weights;
    std::vector<float> bias;
    
    Dense(int input_size, int output_size) 
        : weights(output_size, std::vector<float>(input_size))
        , bias(output_size) {
        
        // Xavier initialization
        std::random_device rd;
        std::mt19937 gen(rd());
        float scale = std::sqrt(2.0f / input_size);
        std::normal_distribution<float> dist(0.0f, scale);
        
        for (auto& row : weights) {
            for (auto& w : row) {
                w = dist(gen);
            }
        }
        
        std::fill(bias.begin(), bias.end(), 0.0f);
    }
    
    std::vector<float> forward(const std::vector<float>& input) {
        std::vector<float> output(weights.size());
        
        for (size_t i = 0; i < weights.size(); ++i) {
            output[i] = bias[i];
            for (size_t j = 0; j < input.size(); ++j) {
                output[i] += weights[i][j] * input[j];
            }
        }
        
        return output;
    }
};

// VGG8 Model for CIFAR-10
class VGG8 {
private:
    // VGG8 architecture: 
    // Conv(64,3x3) -> ReLU -> Conv(64,3x3) -> ReLU -> MaxPool(2x2) ->
    // Conv(128,3x3) -> ReLU -> Conv(128,3x3) -> ReLU -> MaxPool(2x2) ->
    // Conv(256,3x3) -> ReLU -> Conv(256,3x3) -> ReLU -> GAP -> FC(256->10)
    
    Conv2D conv1{3, 64, 3, 1, 1};    // 32x32x3 -> 32x32x64
    Conv2D conv2{64, 64, 3, 1, 1};   // 32x32x64 -> 32x32x64
    // MaxPool 32x32x64 -> 16x16x64
    Conv2D conv3{64, 128, 3, 1, 1};  // 16x16x64 -> 16x16x128
    Conv2D conv4{128, 128, 3, 1, 1}; // 16x16x128 -> 16x16x128
    // MaxPool 16x16x128 -> 8x8x128
    Conv2D conv5{128, 256, 3, 1, 1}; // 8x8x128 -> 8x8x256
    Conv2D conv6{256, 256, 3, 1, 1}; // 8x8x256 -> 8x8x256
    // GAP 8x8x256 -> 1x1x256
    Dense fc{256, 10};               // 256 -> 10
    
public:
    std::vector<float> forward(const Tensor& input) {
        auto x = relu(conv1.forward(input));
        x = relu(conv2.forward(x));
        x = max_pool2d(x);
        
        x = relu(conv3.forward(x));
        x = relu(conv4.forward(x));
        x = max_pool2d(x);
        
        x = relu(conv5.forward(x));
        x = relu(conv6.forward(x));
        x = global_avg_pool(x);
        
        // Flatten for FC layer
        std::vector<float> flattened(x.data.begin(), x.data.end());
        
        return fc.forward(flattened);
    }
};

// Performance benchmarking
class Benchmark {
public:
    static void run_inference_benchmark(int num_iterations = 100, int batch_size = 1) {
        std::cout << "CIFAR-VGG8 Inference Benchmark\n";
        std::cout << "==============================\n";
        std::cout << "Batch size: " << batch_size << "\n";
        std::cout << "Iterations: " << num_iterations << "\n";
        std::cout << "Hardware threads: " << std::thread::hardware_concurrency() << "\n\n";
        
        VGG8 model;
        
        // Create random input data (CIFAR-10 format: 32x32x3)
        Tensor input(batch_size, 32, 32, 3);
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        for (auto& val : input.data) {
            val = dist(gen);
        }
        
        // Warmup runs
        std::cout << "Warming up...\n";
        for (int i = 0; i < 5; ++i) {
            auto output = model.forward(input);
        }
        
        // Benchmark runs
        std::cout << "Running benchmark...\n";
        auto start_time = std::chrono::high_resolution_clock::now();
        
        for (int i = 0; i < num_iterations; ++i) {
            auto output = model.forward(input);
            
            // Print progress every 10 iterations
            if ((i + 1) % 10 == 0) {
                std::cout << "Completed " << (i + 1) << "/" << num_iterations << " iterations\n";
            }
        }
        
        auto end_time = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
        
        // Calculate metrics
        double total_time_ms = duration.count() / 1000.0;
        double avg_latency_ms = total_time_ms / num_iterations;
        double total_images = num_iterations * batch_size;
        double images_per_second = total_images / (total_time_ms / 1000.0);
        
        // Print results
        std::cout << "\nBenchmark Results:\n";
        std::cout << "==================\n";
        std::cout << "Total time: " << total_time_ms << " ms\n";
        std::cout << "Average latency: " << avg_latency_ms << " ms/batch\n";
        std::cout << "Throughput: " << images_per_second << " images/s\n";
        std::cout << "Per-image latency: " << avg_latency_ms / batch_size << " ms/image\n";
        
        // Run a single inference to show output format
        std::cout << "\nSample inference output (logits):\n";
        auto sample_output = model.forward(input);
        for (size_t i = 0; i < sample_output.size(); ++i) {
            std::cout << "Class " << i << ": " << sample_output[i] << "\n";
        }
    }
};

} // namespace cifar_vgg8

void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [OPTIONS]\n\n";
    std::cout << "CIFAR-VGG8 C++20 Implementation\n";
    std::cout << "Single-file, no external dependencies\n";
    std::cout << "FP32 precision, multithreaded\n\n";
    std::cout << "Options:\n";
    std::cout << "  --batch-size <size>    Set batch size for benchmarking (default: 1)\n";
    std::cout << "  --iterations <num>     Set number of iterations for benchmarking (default: 100)\n";
    std::cout << "  --help, -h             Show this help message\n\n";
    std::cout << "Examples:\n";
    std::cout << "  " << program_name << "                           # Run with default settings\n";
    std::cout << "  " << program_name << " --batch-size 4 --iterations 50\n";
    std::cout << "  " << program_name << " --batch-size 8 --iterations 25\n";
}

int main(int argc, char* argv[]) {
    // Default values
    int batch_size = 1;
    int num_iterations = 100;
    
    // Parse command line arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        
        if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else if (arg == "--batch-size") {
            if (i + 1 < argc) {
                batch_size = std::atoi(argv[++i]);
                if (batch_size <= 0) {
                    std::cerr << "Error: batch size must be positive\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --batch-size requires a value\n";
                return 1;
            }
        }
        else if (arg == "--iterations") {
            if (i + 1 < argc) {
                num_iterations = std::atoi(argv[++i]);
                if (num_iterations <= 0) {
                    std::cerr << "Error: iterations must be positive\n";
                    return 1;
                }
            } else {
                std::cerr << "Error: --iterations requires a value\n";
                return 1;
            }
        }
        else {
            std::cerr << "Error: unknown option '" << arg << "'\n";
            std::cerr << "Use --help for usage information\n";
            return 1;
        }
    }
    
    std::cout << "CIFAR-VGG8 C++20 Implementation\n";
    std::cout << "Single-file, no external dependencies\n";
    std::cout << "FP32 precision, multithreaded\n\n";
    
    // Run benchmark with user-specified or default configurations
    cifar_vgg8::Benchmark::run_inference_benchmark(num_iterations, batch_size);
    
    return 0;
}