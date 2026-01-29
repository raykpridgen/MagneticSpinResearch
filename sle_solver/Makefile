# SLE Solver Makefile
# Builds CPU and/or GPU versions of the solver

# Compilers
CXX = g++
NVCC = nvcc

# Directories
COMP_DIR = computation
SRC_DIR = $(COMP_DIR)/src
INC_DIR = $(COMP_DIR)/include
BUILD_DIR = $(COMP_DIR)/build
OUTPUT_DIR = output

# Compiler flags
CXXFLAGS = -std=c++17 -O2 -Wall -I$(INC_DIR) -I/usr/include/eigen3
NVCCFLAGS = -std=c++17 -O2 -I$(INC_DIR) -I/usr/include/eigen3

# CUDA libraries
CUDA_LIBS = -L/usr/local/cuda/lib64 -lcusolver -lcudart -lcublas

# Source files
SOURCES = $(SRC_DIR)/operators.cpp \
          $(SRC_DIR)/hamiltonian.cpp \
          $(SRC_DIR)/sle_solver.cpp \
          $(SRC_DIR)/main.cpp

CUDA_SRC = $(SRC_DIR)/cuda_solver.cu

# Output binaries
CPU_BIN = $(BUILD_DIR)/sle_solver_cpu
GPU_BIN = $(BUILD_DIR)/sle_solver_cuda

# Default target
.PHONY: all
all: cpu

# CPU version
.PHONY: cpu
cpu: $(CPU_BIN)

$(CPU_BIN): $(SOURCES) | $(BUILD_DIR) $(OUTPUT_DIR)
	@echo "Building CPU version..."
	$(CXX) $(CXXFLAGS) $(SOURCES) -o $@
	@echo "CPU binary created: $@"

# GPU version
.PHONY: gpu
gpu: $(GPU_BIN)

$(GPU_BIN): $(SOURCES) $(CUDA_SRC) | $(BUILD_DIR) $(OUTPUT_DIR)
	@echo "Building GPU version..."
	# Compile CUDA code to object file
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -c $(CUDA_SRC) -o $(BUILD_DIR)/cuda_solver.o
	# Compile and link everything
	$(CXX) $(CXXFLAGS) -DUSE_CUDA $(SOURCES) $(BUILD_DIR)/cuda_solver.o -o $@ $(CUDA_LIBS)
	@echo "GPU binary created: $@"

# Build both versions
.PHONY: both
both: cpu gpu

# Create directories
$(BUILD_DIR):
	@mkdir -p $(BUILD_DIR)

$(OUTPUT_DIR):
	@mkdir -p $(OUTPUT_DIR)

# Clean
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -rf $(BUILD_DIR)

# Install Python dependencies
.PHONY: python-deps
python-deps:
	pip install -r requirements.txt

# Test CPU version
.PHONY: test-cpu
test-cpu: cpu
	@echo "\nTesting CPU solver (2 electrons)..."
	./$(CPU_BIN) --cpu -n 2 -v

# Test GPU version
.PHONY: test-gpu
test-gpu: gpu
	@echo "\nTesting GPU solver (2 electrons)..."
	./$(GPU_BIN) --gpu -n 2 -v

# Benchmark comparison
.PHONY: benchmark
benchmark: both
	@echo "\n=== CPU vs GPU Benchmark ===\n"
	@echo "2-electron system (64x64 linear system):"
	@echo "CPU:"
	@time -p ./$(CPU_BIN) --cpu -n 2 -o $(OUTPUT_DIR)/bench_cpu_2.csv 2>&1 | grep real || true
	@echo "GPU:"
	@time -p ./$(GPU_BIN) --gpu -n 2 -o $(OUTPUT_DIR)/bench_gpu_2.csv 2>&1 | grep real || true
	@echo "\n3-electron system (256x256 linear system):"
	@echo "CPU:"
	@time -p ./$(CPU_BIN) --cpu -n 3 -o $(OUTPUT_DIR)/bench_cpu_3.csv 2>&1 | grep real || true
	@echo "GPU:"
	@time -p ./$(GPU_BIN) --gpu -n 3 -o $(OUTPUT_DIR)/bench_gpu_3.csv 2>&1 | grep real || true

# Help
.PHONY: help
help:
	@echo "SLE Solver Build System"
	@echo ""
	@echo "Build targets:"
	@echo "  make cpu         Build CPU-only version (default)"
	@echo "  make gpu         Build GPU (CUDA) version"
	@echo "  make both        Build both versions"
	@echo "  make clean       Remove build artifacts"
	@echo ""
	@echo "Test targets:"
	@echo "  make test-cpu    Build and test CPU version"
	@echo "  make test-gpu    Build and test GPU version"
	@echo "  make benchmark   Compare CPU vs GPU performance"
	@echo ""
	@echo "Python:"
	@echo "  make python-deps Install Python visualization dependencies"
	@echo ""
	@echo "CLI usage:"
	@echo "  ./$(CPU_BIN) --cpu [options]"
	@echo "  ./$(GPU_BIN) --gpu [options]"
	@echo ""
	@echo "Run './$(CPU_BIN) --help' for full option list"
