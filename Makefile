# Makefile for CIFAR-VGG8 implementation

CXX ?= g++
CXXFLAGS = -std=c++20 -O3 -fopenmp -march=native -Wall -Wextra
TARGET = benchmark
SOURCE = benchmark.cpp
HEADER = cifar_vgg8_model.h cifar_resnet8_model.h

.PHONY: all clean run debug legacy

all: $(TARGET)

$(TARGET): $(SOURCE) $(HEADER)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

debug: CXXFLAGS = -std=c++20 -g -fopenmp -Wall -Wextra -DDEBUG
debug: $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

# Additional targets for different optimization levels
fast: CXXFLAGS = -std=c++20 -O3 -fopenmp -march=native -DNDEBUG -flto
fast: $(TARGET)

profile: CXXFLAGS = -std=c++20 -O2 -fopenmp -march=native -pg
profile: $(TARGET)

help:
	@echo "Available targets:"
	@echo "  all      - Build new benchmark binary with command-line args (default)"
	@echo "  debug    - Build debug version"
	@echo "  fast     - Build with maximum optimizations"
	@echo "  profile  - Build with profiling support"
	@echo "  run      - Build and run the program"
	@echo "  clean    - Remove built files"
	@echo "  help     - Show this help message"
