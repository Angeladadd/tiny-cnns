# Makefile for CIFAR-VGG8 implementation

CXX = g++
CXXFLAGS = -std=c++20 -O3 -fopenmp -march=native -Wall -Wextra
TARGET = benchmark
LEGACY_TARGET = cifar_vgg8
SOURCE = benchmark.cpp
LEGACY_SOURCE = cifar_vgg8.cpp
HEADER = cifar_vgg8_model.h

.PHONY: all clean run debug legacy

all: $(TARGET)

$(TARGET): $(SOURCE) $(HEADER)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

# Build the legacy single-file version for compatibility
legacy: $(LEGACY_TARGET)

$(LEGACY_TARGET): $(LEGACY_SOURCE)
	$(CXX) $(CXXFLAGS) $(LEGACY_SOURCE) -o $(LEGACY_TARGET)

debug: CXXFLAGS = -std=c++20 -g -fopenmp -Wall -Wextra -DDEBUG
debug: $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET) $(LEGACY_TARGET)

# Additional targets for different optimization levels
fast: CXXFLAGS = -std=c++20 -O3 -fopenmp -march=native -DNDEBUG -flto
fast: $(TARGET)

profile: CXXFLAGS = -std=c++20 -O2 -fopenmp -march=native -pg
profile: $(TARGET)

help:
	@echo "Available targets:"
	@echo "  all      - Build new benchmark binary with command-line args (default)"
	@echo "  legacy   - Build legacy single-file cifar_vgg8 binary"
	@echo "  debug    - Build debug version"
	@echo "  fast     - Build with maximum optimizations"
	@echo "  profile  - Build with profiling support"
	@echo "  run      - Build and run the program"
	@echo "  clean    - Remove built files"
	@echo "  help     - Show this help message"