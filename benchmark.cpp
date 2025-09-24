#include "cifar_vgg8_model.h"
#include "cifar_resnet8_model.h"
#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <string>
#include <cstring>
#include <iomanip>
#ifdef _OPENMP
#include <omp.h>
#endif

// Command line argument parsing and benchmarking logic
class BenchmarkRunner {
private:
    int batch_size = 1;
    int num_iterations = 100;
    bool help = false;
    std::string model_type = "vgg8";
    
public:
    void parse_arguments(int argc, char* argv[]) {
        for (int i = 1; i < argc; ++i) {
            if (std::strcmp(argv[i], "--batch-size") == 0 || std::strcmp(argv[i], "-b") == 0) {
                if (i + 1 < argc) {
                    batch_size = std::stoi(argv[++i]);
                } else {
                    std::cerr << "Error: --batch-size requires a value\n";
                    print_usage();
                    exit(1);
                }
            } else if (std::strcmp(argv[i], "--iterations") == 0 || std::strcmp(argv[i], "-i") == 0) {
                if (i + 1 < argc) {
                    num_iterations = std::stoi(argv[++i]);
                } else {
                    std::cerr << "Error: --iterations requires a value\n";
                    print_usage();
                    exit(1);
                }
            } else if (std::strcmp(argv[i], "--model") == 0 || std::strcmp(argv[i], "-m") == 0) {
                if (i + 1 < argc) {
                    model_type = argv[++i];
                    if (model_type != "vgg8" && model_type != "resnet8") {
                        std::cerr << "Error: model must be 'vgg8' or 'resnet8'\n";
                        print_usage();
                        exit(1);
                    }
                } else {
                    std::cerr << "Error: --model requires a value\n";
                    print_usage();
                    exit(1);
                }
            } else if (std::strcmp(argv[i], "--help") == 0 || std::strcmp(argv[i], "-h") == 0) {
                help = true;
            } else {
                std::cerr << "Error: Unknown argument " << argv[i] << "\n";
                print_usage();
                exit(1);
            }
        }
        
        // Validate arguments
        if (batch_size <= 0) {
            std::cerr << "Error: batch size must be positive\n";
            exit(1);
        }
        if (num_iterations <= 0) {
            std::cerr << "Error: iterations must be positive\n";
            exit(1);
        }
    }
    
    void print_usage() {
        std::cout << "CIFAR CNN Benchmark\n";
        std::cout << "Usage: ./benchmark [OPTIONS]\n\n";
        std::cout << "OPTIONS:\n";
        std::cout << "  -b, --batch-size NUM    Set batch size (default: 1)\n";
        std::cout << "  -i, --iterations NUM    Set number of iterations (default: 100)\n";
        std::cout << "  -m, --model MODEL       Set model type: vgg8, resnet8 (default: vgg8)\n";
        std::cout << "  -h, --help              Show this help message\n\n";
        std::cout << "Examples:\n";
        std::cout << "  ./benchmark --model resnet8 --batch-size 4 --iterations 50\n";
        std::cout << "  ./benchmark -m vgg8 -b 8 -i 25\n";
    }
    
    void run_benchmark() {
        if (help) {
            print_usage();
            return;
        }
        
        std::string model_name = (model_type == "vgg8") ? "CIFAR-VGG8" : "CIFAR-ResNet8";
        std::cout << model_name << " Inference Benchmark\n";
        std::cout << "==============================\n";
        std::cout << "Model: " << model_type << "\n";
        std::cout << "Batch size: " << batch_size << "\n";
        std::cout << "Iterations: " << num_iterations << "\n";
        #ifdef _OPENMP
        std::cout << "OpenMP threads: " << omp_get_max_threads() << "\n\n";
        #else
        std::cout << "OpenMP not available\n\n";
        #endif
        
        // Create random input data (CIFAR-10 format: 32x32x3)
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        
        if (model_type == "vgg8") {
            cifar_vgg8::VGG8 model;
            cifar_vgg8::Tensor input(batch_size, 32, 32, 3);
            
            for (auto& val : input.data) {
                val = dist(gen);
            }
            
            run_model_benchmark(model, input);
        } else { // resnet8
            cifar_resnet8::ResNet8 model;
            cifar_resnet8::Tensor input(batch_size, 32, 32, 3);
            
            for (auto& val : input.data) {
                val = dist(gen);
            }
            
            run_model_benchmark(model, input);
        }
    }

private:
    template<typename ModelType, typename TensorType>
    void run_model_benchmark(ModelType& model, TensorType& input) {
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
        std::cout << "Total time: " << std::fixed << std::setprecision(1) << total_time_ms << " ms\n";
        std::cout << "Average latency: " << std::fixed << std::setprecision(3) << avg_latency_ms << " ms/batch\n";
        std::cout << "Throughput: " << std::fixed << std::setprecision(2) << images_per_second << " images/s\n";
        std::cout << "Per-image latency: " << std::fixed << std::setprecision(3) << avg_latency_ms / batch_size << " ms/image\n";
        
        // Run a single inference to show output format
        std::cout << "\nSample inference output (logits):\n";
        auto sample_output = model.forward(input);
        for (size_t i = 0; i < sample_output.size(); ++i) {
            std::cout << "Class " << i << ": " << std::fixed << std::setprecision(5) << sample_output[i] << "\n";
        }
    }
};

int main(int argc, char* argv[]) {
    std::cout << "CIFAR CNN C++20 Implementation\n";
    std::cout << "Single-file, no external dependencies\n";
    std::cout << "FP32 precision, multithreaded\n\n";
    
    BenchmarkRunner benchmark;
    
    try {
        benchmark.parse_arguments(argc, argv);
        benchmark.run_benchmark();
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << "\n";
        return 1;
    }
    
    return 0;
}