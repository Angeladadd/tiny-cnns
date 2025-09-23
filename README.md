# tiny-cnns

A C++20 implementation of CIFAR-VGG8 inference with no external dependencies and configurable benchmarking.

## Features

- **Modular implementation**: Network model in header file, benchmarking in separate executable
- **Command-line interface**: Configurable batch size and iterations via flags
- **C++20 standard**: Modern C++ features for clean, efficient code
- **No external dependencies**: Only standard library
- **FP32 precision**: 32-bit floating point operations
- **Multithreaded**: Parallel convolution operations using `std::async`
- **CIFAR-10 compatible**: 32×32×3 input images
- **VGG8 architecture**: 6 conv layers + 2 pooling + GAP + FC head

## Architecture

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

## Compilation

### New modular version (recommended):
```bash
make                 # Build benchmark with command-line args
make all            # Same as above
```

### Legacy single-file version:
```bash
make legacy         # Build original cifar_vgg8 executable
```

### Manual compilation:
```bash
# New version
g++ -std=c++20 -O3 -pthread -march=native benchmark.cpp -o benchmark

# Legacy version  
g++ -std=c++20 -O3 -pthread -march=native cifar_vgg8.cpp -o cifar_vgg8
```

## Usage

### New benchmark executable with command-line args:

```bash
./benchmark [OPTIONS]
```

**Options:**
- `-b, --batch-size NUM`    Set batch size (default: 1)
- `-i, --iterations NUM`    Set number of iterations (default: 100)  
- `-h, --help`              Show help message

**Examples:**
```bash
./benchmark                              # Use defaults (batch=1, iterations=100)
./benchmark --batch-size 4 --iterations 50
./benchmark -b 8 -i 25                  # Short flags
./benchmark --help                       # Show usage information
```

### Legacy version:
```bash
./cifar_vgg8
```
Runs with fixed parameters: 50 iterations (batch=1) + 20 iterations (batch=4).

## Performance

The implementation automatically detects available hardware threads and parallelizes convolution operations. Performance will vary based on hardware, but typical results include:

- **Latency**: ~300-400ms per image on modern CPUs
- **Throughput**: ~3-4 images/second
- **Memory**: Minimal memory footprint with efficient tensor operations

## Implementation Details

- **cifar_vgg8_model.h**: Contains all model classes (Tensor, Conv2D, VGG8, etc.)
- **benchmark.cpp**: Command-line interface and benchmarking logic
- **cifar_vgg8.cpp**: Legacy single-file implementation (maintained for compatibility)
- **Tensor class**: 4D tensor operations (batch, height, width, channels)
- **Conv2D**: Parallel convolution with Xavier weight initialization
- **Pooling**: 2×2 max pooling and global average pooling
- **Threading**: Work distribution across available CPU cores
- **Memory layout**: NHWC format for cache-friendly access patterns