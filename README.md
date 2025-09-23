# tiny-cnns

A single-file C++20 implementation of CIFAR-VGG8 inference with no external dependencies.

## Features

- **Single-file implementation**: Everything contained in `cifar_vgg8.cpp`
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

```bash
g++ -std=c++20 -O3 -pthread -march=native cifar_vgg8.cpp -o cifar_vgg8
```

## Usage

```bash
./cifar_vgg8 [OPTIONS]
```

### Command Line Options

- `--batch-size <size>`: Set batch size for benchmarking (default: 1)
- `--iterations <num>`: Set number of iterations for benchmarking (default: 100)
- `--help, -h`: Show help message

### Examples

```bash
# Run with default settings (100 iterations, batch size 1)
./cifar_vgg8

# Custom batch size and iterations
./cifar_vgg8 --batch-size 4 --iterations 50

# Large batch size with fewer iterations
./cifar_vgg8 --batch-size 8 --iterations 25

# Show help
./cifar_vgg8 --help
```

The program will run a benchmark with the specified configuration and display:
- Performance metrics (latency, throughput)
- Sample inference output

## Performance

The implementation automatically detects available hardware threads and parallelizes convolution operations. Performance will vary based on hardware, but typical results include:

- **Latency**: ~300-400ms per image on modern CPUs
- **Throughput**: ~3-4 images/second
- **Memory**: Minimal memory footprint with efficient tensor operations

## Implementation Details

- **Tensor class**: 4D tensor operations (batch, height, width, channels)
- **Conv2D**: Parallel convolution with Xavier weight initialization
- **Pooling**: 2×2 max pooling and global average pooling
- **Threading**: Work distribution across available CPU cores
- **Memory layout**: NHWC format for cache-friendly access patterns