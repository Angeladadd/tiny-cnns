# Makefile for CIFAR-VGG8 implementation

CXX = g++
CXXFLAGS = -std=c++20 -O3 -pthread -march=native -Wall -Wextra
TARGET = cifar_vgg8
SOURCE = cifar_vgg8.cpp

.PHONY: all clean run debug

all: $(TARGET)

$(TARGET): $(SOURCE)
	$(CXX) $(CXXFLAGS) $(SOURCE) -o $(TARGET)

debug: CXXFLAGS = -std=c++20 -g -pthread -Wall -Wextra -DDEBUG
debug: $(TARGET)

run: $(TARGET)
	./$(TARGET)

clean:
	rm -f $(TARGET)

# Additional targets for different optimization levels
fast: CXXFLAGS = -std=c++20 -O3 -pthread -march=native -DNDEBUG -flto
fast: $(TARGET)

profile: CXXFLAGS = -std=c++20 -O2 -pthread -march=native -pg
profile: $(TARGET)

help:
	@echo "Available targets:"
	@echo "  all      - Build optimized binary (default)"
	@echo "  debug    - Build debug version"
	@echo "  fast     - Build with maximum optimizations"
	@echo "  profile  - Build with profiling support"
	@echo "  run      - Build and run the program"
	@echo "  clean    - Remove built files"
	@echo "  help     - Show this help message"