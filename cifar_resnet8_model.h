#pragma once

#include <iostream>
#include <vector>
#include <array>
#include <random>
#include <chrono>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <memory>
#include <iomanip>
#include <limits>
#ifdef _OPENMP
#include <omp.h>
#endif

// C++20 CIFAR-ResNet8 model implementation
// No external dependencies, FP32, multithreaded with OpenMP

namespace cifar_resnet8 {

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
        
        // Parallel convolution using OpenMP
        int total_work = batch * out_c;
        
        #pragma omp parallel for schedule(dynamic)
        for (int idx = 0; idx < total_work; ++idx) {
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
        
        return output;
    }
};

// ReLU activation
Tensor relu(const Tensor& input) {
    Tensor output = input;
    #pragma omp parallel for
    for (size_t i = 0; i < output.data.size(); ++i) {
        output.data[i] = std::max(0.0f, output.data[i]);
    }
    return output;
}

// Element-wise addition for residual connections
Tensor add(const Tensor& a, const Tensor& b) {
    if (a.shape[0] != b.shape[0] || a.shape[1] != b.shape[1] || 
        a.shape[2] != b.shape[2] || a.shape[3] != b.shape[3]) {
        throw std::runtime_error("Tensor shapes must match for addition");
    }
    
    Tensor output = a;
    #pragma omp parallel for
    for (size_t i = 0; i < output.data.size(); ++i) {
        output.data[i] += b.data[i];
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

// Residual Block
class ResidualBlock {
private:
    Conv2D conv1;
    Conv2D conv2;
    std::unique_ptr<Conv2D> shortcut_conv; // For dimension matching
    
public:
    ResidualBlock(int in_channels, int out_channels, int stride = 1)
        : conv1(in_channels, out_channels, 3, stride, 1)
        , conv2(out_channels, out_channels, 3, 1, 1) {
        
        // Add shortcut connection if dimensions don't match
        if (stride != 1 || in_channels != out_channels) {
            shortcut_conv = std::make_unique<Conv2D>(in_channels, out_channels, 1, stride, 0);
        }
    }
    
    Tensor forward(const Tensor& input) {
        auto x = relu(conv1.forward(input));
        x = conv2.forward(x);
        
        // Shortcut connection
        Tensor residual = input;
        if (shortcut_conv) {
            residual = shortcut_conv->forward(input);
        }
        
        // Add residual connection and apply ReLU
        return relu(add(x, residual));
    }
};

// ResNet8 Model for CIFAR-10
class ResNet8 {
private:
    // ResNet8 architecture:
    // Conv(16,3x3) -> ReLU ->
    // ResBlock(16->16, stride=1) ->
    // ResBlock(16->32, stride=2) ->
    // ResBlock(32->64, stride=2) ->
    // GAP -> FC(64->10)
    
    Conv2D initial_conv{3, 16, 3, 1, 1};     // 32x32x3 -> 32x32x16
    ResidualBlock res_block1{16, 16, 1};     // 32x32x16 -> 32x32x16
    ResidualBlock res_block2{16, 32, 2};     // 32x32x16 -> 16x16x32
    ResidualBlock res_block3{32, 64, 2};     // 16x16x32 -> 8x8x64
    Dense fc{64, 10};                        // 64 -> 10
    
public:
    std::vector<float> forward(const Tensor& input) {
        auto x = relu(initial_conv.forward(input));
        x = res_block1.forward(x);
        x = res_block2.forward(x);
        x = res_block3.forward(x);
        x = global_avg_pool(x);
        
        // Flatten for FC layer
        std::vector<float> flattened(x.data.begin(), x.data.end());
        
        return fc.forward(flattened);
    }
};

} // namespace cifar_resnet8