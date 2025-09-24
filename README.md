# tiny-cnns

A C++20 implementation of CIFAR CNN inference with no external dependencies and configurable benchmarking. Supports both VGG8 and ResNet8 architectures.

## Features

- **Modular implementation**: Network model in header file, benchmarking in separate executable
- **Command-line interface**: Configurable batch size and iterations via flags
- **C++20 standard**: Modern C++ features for clean, efficient code
- **No external dependencies**: Only standard library
- **FP32 precision**: 32-bit floating point operations
- **Multithreaded**: Parallel convolution operations using **OpenMP**
- **CIFAR-10 compatible**: 32×32×3 input images
- **Multiple architectures**: VGG8 and ResNet8 implementations

## Architecture

### VGG8 Model
The VGG8 model implements the following architecture:
```
Input: 32×32×3
├─ Conv(64, 3×3) → ReLU
├─ Conv(64, 3×3) → ReLU
├─ MaxPool(2×2) → 16×16×64
├─ Conv(128, 3×3) → ReLU
├─ Conv(128, 3×3) → ReLU
├─ MaxPool(2×2) → 8×8×128
├─ Conv(256, 3×3) → ReLU
├─ Conv(256, 3×3) → ReLU
├─ GlobalAvgPool → 1×1×256
└─ FC(256→10) → Output logits
```

### ResNet8 Model
The ResNet8 model implements the following architecture:
```
Input: 32×32×3
├─ Conv(16, 3×3) → ReLU → 32×32×16
├─ ResBlock(16→16, stride=1) → 32×32×16
├─ ResBlock(16→32, stride=2) → 16×16×32
├─ ResBlock(32→64, stride=2) → 8×8×64
├─ GlobalAvgPool → 1×1×64
└─ FC(64→10) → Output logits
```

## Compilation

```bash
make                 # Build benchmark with command-line args
make all            # Same as above
```

### Manual compilation:
```bash
g++ -std=c++20 -O3 -fopenmp -march=native benchmark.cpp -o benchmark
```

## Usage

```bash
./benchmark [OPTIONS]
```

**Options:**
- `-b, --batch-size NUM`    Set batch size (default: 1)
- `-i, --iterations NUM`    Set number of iterations (default: 100)
- `-m, --model MODEL`       Set model type: vgg8, resnet8 (default: vgg8)
- `-h, --help`              Show help message

**Examples:**
```bash
./benchmark                              # Use defaults (vgg8, batch=1, iterations=100)
./benchmark --model resnet8 --batch-size 4 --iterations 50
./benchmark -m vgg8 -b 8 -i 25          # Short flags
./benchmark --help                       # Show usage information
```

## Performance

The implementation automatically detects available hardware threads and parallelizes convolution operations. Performance will vary based on hardware.

### VGG8 Performance
Three experiments were conducted on an Intel Xeon Gold 6248 (2.5 GHz) processor:
- 1 CPU, 1 Thread
    - Throughput: 1.56 images/s
    - Per-image latency: 641.801 ms/image
- 8 CPUs, 8 Threads
    - Throughput: 11.06 images/s
    - Per-image latency: 90.388 ms/image
- 8 CPUs, 16 Threads
    - Throughput: 20.84 images/s
    - Per-image latency: 47.986 ms/image

### ResNet8 Performance
ResNet8 shows significantly better performance due to its more efficient architecture:
- Typical performance: ~3x faster than VGG8
- Example: ~57 images/s vs ~6 images/s for VGG8 (single thread)


## Implementation Details

- **cifar_vgg8_model.h**: Contains VGG8 model classes (Tensor, Conv2D, VGG8, etc.)
- **cifar_resnet8_model.h**: Contains ResNet8 model classes (ResidualBlock, ResNet8, etc.)
- **benchmark.cpp**: Command-line interface and benchmarking logic
- **Tensor class**: 4D tensor operations (batch, height, width, channels)
- **Conv2D**: Parallel convolution with Xavier weight initialization
- **ResidualBlock**: Residual connections with shortcut matching
- **Pooling**: 2×2 max pooling and global average pooling
- **Threading**: Work distribution across available CPU cores
- **Memory layout**: NHWC format for cache-friendly access patterns