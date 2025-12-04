# Makefile for Stochastic Liouville Equation Solver
# Supports both CPU-only and CUDA-accelerated builds, plus Qt GUI

# Compiler settings
CXX = g++
NVCC = nvcc
MOC = $(shell which moc-qt6 2>/dev/null || which moc-qt5 2>/dev/null || find /usr -name moc 2>/dev/null | head -1)

# Common flags
CXXFLAGS = -std=c++17 -O2 -I/usr/include/eigen3 -Isrc
NVCCFLAGS = -std=c++17 -O2 -I/usr/include/eigen3 -Isrc

# Qt settings
QT_CXXFLAGS = $(shell pkg-config --cflags Qt6Widgets Qt6PrintSupport 2>/dev/null || pkg-config --cflags Qt5Widgets Qt5PrintSupport)
QT_LDFLAGS = $(shell pkg-config --libs Qt6Widgets Qt6PrintSupport 2>/dev/null || pkg-config --libs Qt5Widgets Qt5PrintSupport)
QT_AVAILABLE = $(shell pkg-config --exists Qt6Widgets 2>/dev/null && echo "Qt6" || pkg-config --exists Qt5Widgets 2>/dev/null && echo "Qt5" || echo "")

# Directories
SRC_DIR = src
GUI_DIR = $(SRC_DIR)/gui
BUILD_DIR = build
DATA_DIR = data

# Source files
CPP_SRC = $(SRC_DIR)/operators.cpp
CUDA_SRC = $(SRC_DIR)/cuda_solver.cu

# GUI source files
GUI_MAIN = $(GUI_DIR)/main.cpp
GUI_MAINWINDOW = $(GUI_DIR)/main_window.cpp
GUI_WORKER = $(GUI_DIR)/simulation_worker.cpp
GUI_QCUSTOMPLOT = $(GUI_DIR)/qcustomplot.cpp

# Output binaries
CPU_BIN = $(BUILD_DIR)/operators_cpu
CUDA_BIN = $(BUILD_DIR)/operators_cuda
GUI_CPU_BIN = $(BUILD_DIR)/magspin_gui
GUI_CUDA_BIN = $(BUILD_DIR)/magspin_gui_cuda
MATRIX_SOLVER_CPU = $(BUILD_DIR)/matrix_solver
MATRIX_SOLVER_CUDA = $(BUILD_DIR)/matrix_solver_cuda

# Default target (CPU version)
.PHONY: all
all: cpu

# CPU-only version
.PHONY: cpu
cpu: $(CPU_BIN)

$(CPU_BIN): $(CPP_SRC)
	@echo "Building CPU-only version..."
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@
	@echo "CPU binary created: $@"

# CUDA-accelerated version
.PHONY: cuda
cuda: $(CUDA_BIN)

$(CUDA_BIN): $(CPP_SRC) $(CUDA_SRC)
	@echo "Building CUDA-accelerated version..."
	@mkdir -p $(BUILD_DIR)
	# First compile CUDA code to object file
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -c $(CUDA_SRC) -o $(BUILD_DIR)/cuda_solver.o
	# Then compile C++ and link with CUDA object
	$(CXX) $(CXXFLAGS) -DUSE_CUDA $(CPP_SRC) $(BUILD_DIR)/cuda_solver.o \
		-o $@ -L/usr/local/cuda/lib64 -lcusolver -lcudart -lcublas
	@echo "CUDA binary created: $@"

# Build both versions
.PHONY: both
both: cpu cuda

# ============================================
# Qt GUI Build Targets
# ============================================

# Check if Qt is available
check-qt:
	@if [ -z "$(QT_AVAILABLE)" ]; then \
		echo "Error: Qt not found. Please install Qt6 or Qt5 development packages:"; \
		echo "  Ubuntu/Debian: sudo apt install qt6-base-dev libqt6charts6-dev"; \
		echo "  or:            sudo apt install qtbase5-dev libqt5charts5-dev"; \
		exit 1; \
	else \
		echo "Found $(QT_AVAILABLE)"; \
	fi

# MOC (Meta Object Compiler) targets
$(BUILD_DIR)/moc_main_window.cpp: $(GUI_DIR)/main_window.h | check-qt
	@mkdir -p $(BUILD_DIR)
	$(MOC) $(QT_CXXFLAGS) $< -o $@

$(BUILD_DIR)/moc_simulation_worker.cpp: $(GUI_DIR)/simulation_worker.h | check-qt
	@mkdir -p $(BUILD_DIR)
	$(MOC) $(QT_CXXFLAGS) $< -o $@

# MOC for QCustomPlot
$(BUILD_DIR)/moc_qcustomplot.cpp: $(GUI_DIR)/qcustomplot.h | check-qt
	@mkdir -p $(BUILD_DIR)
	$(MOC) $(QT_CXXFLAGS) $< -o $@

# Compile simulation library object (shared between CLI and GUI)
$(BUILD_DIR)/operators_lib.o: $(CPP_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DNO_MAIN -c $< -o $@

$(BUILD_DIR)/operators_lib_cuda.o: $(CPP_SRC)
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) -DUSE_CUDA -DNO_MAIN -c $< -o $@

# GUI CPU version
.PHONY: gui
gui: $(GUI_CPU_BIN)

$(GUI_CPU_BIN): $(BUILD_DIR)/moc_main_window.cpp $(BUILD_DIR)/moc_simulation_worker.cpp $(BUILD_DIR)/moc_qcustomplot.cpp $(BUILD_DIR)/operators_lib.o
	@echo "Building Qt GUI (CPU version)..."
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $(QT_CXXFLAGS) -fPIC \
		$(GUI_MAIN) $(GUI_MAINWINDOW) $(GUI_WORKER) $(GUI_QCUSTOMPLOT) \
		$(BUILD_DIR)/moc_main_window.cpp $(BUILD_DIR)/moc_simulation_worker.cpp $(BUILD_DIR)/moc_qcustomplot.cpp \
		$(BUILD_DIR)/operators_lib.o \
		$(QT_LDFLAGS) -o $@
	@echo "GUI binary created: $@"
	@echo "Run with: ./$(GUI_CPU_BIN)"

# GUI CUDA version
.PHONY: gui-cuda
gui-cuda: $(GUI_CUDA_BIN)

$(GUI_CUDA_BIN): $(BUILD_DIR)/moc_main_window.cpp $(BUILD_DIR)/moc_simulation_worker.cpp $(BUILD_DIR)/moc_qcustomplot.cpp $(BUILD_DIR)/operators_lib_cuda.o $(CUDA_SRC)
	@echo "Building Qt GUI (CUDA version)..."
	@mkdir -p $(BUILD_DIR)
	# Compile CUDA code
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -c $(CUDA_SRC) -o $(BUILD_DIR)/cuda_solver_gui.o
	# Link everything together
	$(CXX) $(CXXFLAGS) $(QT_CXXFLAGS) -DUSE_CUDA -fPIC \
		$(GUI_MAIN) $(GUI_MAINWINDOW) $(GUI_WORKER) $(GUI_QCUSTOMPLOT) \
		$(BUILD_DIR)/moc_main_window.cpp $(BUILD_DIR)/moc_simulation_worker.cpp $(BUILD_DIR)/moc_qcustomplot.cpp \
		$(BUILD_DIR)/operators_lib_cuda.o $(BUILD_DIR)/cuda_solver_gui.o \
		$(QT_LDFLAGS) -L/usr/local/cuda/lib64 -lcusolver -lcudart -lcublas -o $@
	@echo "GUI binary (CUDA) created: $@"
	@echo "Run with: ./$(GUI_CUDA_BIN)"

# Build all GUI versions
.PHONY: gui-all
gui-all: gui gui-cuda

# ============================================
# Matrix Solver for Mathematica Integration
# ============================================

# Matrix solver CPU version
.PHONY: matrix-solver
matrix-solver: $(MATRIX_SOLVER_CPU)

$(MATRIX_SOLVER_CPU): $(SRC_DIR)/matrix_solver.cpp
	@echo "Building matrix solver (CPU version)..."
	@mkdir -p $(BUILD_DIR)
	$(CXX) $(CXXFLAGS) $< -o $@
	@echo "Matrix solver binary created: $@"
	@echo "Usage: $@ <matrix_file.txt> [--cuda]"

# Matrix solver CUDA version
.PHONY: matrix-solver-cuda
matrix-solver-cuda: $(MATRIX_SOLVER_CUDA)

$(MATRIX_SOLVER_CUDA): $(SRC_DIR)/matrix_solver.cpp $(CUDA_SRC)
	@echo "Building matrix solver (CUDA version)..."
	@mkdir -p $(BUILD_DIR)
	# Compile CUDA code
	$(NVCC) $(NVCCFLAGS) -DUSE_CUDA -c $(CUDA_SRC) -o $(BUILD_DIR)/cuda_solver_matrix.o
	# Compile and link matrix solver with CUDA
	$(CXX) $(CXXFLAGS) -DUSE_CUDA $< $(BUILD_DIR)/cuda_solver_matrix.o \
		-o $@ -L/usr/local/cuda/lib64 -lcusolver -lcudart -lcublas
	@echo "Matrix solver (CUDA) binary created: $@"
	@echo "Usage: $@ <matrix_file.txt> --cuda"

# Build both matrix solver versions
.PHONY: matrix-solver-all
matrix-solver-all: matrix-solver matrix-solver-cuda

# ============================================
# Clean build artifacts
.PHONY: clean
clean:
	@echo "Cleaning build artifacts..."
	rm -f $(BUILD_DIR)/operators_cpu $(BUILD_DIR)/operators_cuda $(BUILD_DIR)/*.o
	rm -f $(BUILD_DIR)/magspin_gui $(BUILD_DIR)/magspin_gui_cuda
	rm -f $(BUILD_DIR)/matrix_solver $(BUILD_DIR)/matrix_solver_cuda
	rm -f $(BUILD_DIR)/moc_*.cpp

# Test CPU version
.PHONY: test-cpu
test-cpu: cpu
	@echo "Testing CPU version with 2-electron system..."
	./$(CPU_BIN) 2 2.003 5.788e-8 1.0 1.0 4e6 1e6 6.582e-16

# Test CUDA version (if CUDA is available)
.PHONY: test-cuda
test-cuda: cuda
	@echo "Testing CUDA version with 3-electron system..."
	./$(CUDA_BIN) 3 2.003 5.788e-8 1.0 1.0 4e6 1e6 6.582e-16 --cuda

# Performance comparison
.PHONY: benchmark
benchmark: both
	@echo "Running performance comparison..."
	@echo "\n=== 2-electron system (64x64) ==="
	@echo "CPU:"
	@./$(CPU_BIN) 2 2.003 5.788e-8 1.0 1.0 4e6 1e6 6.582e-16 | grep "Solve time"
	@echo "CUDA:"
	@./$(CUDA_BIN) 2 2.003 5.788e-8 1.0 1.0 4e6 1e6 6.582e-16 --cuda | grep "Solve time"
	@echo "\n=== 3-electron system (256x256) ==="
	@echo "CPU:"
	@./$(CPU_BIN) 3 2.003 5.788e-8 1.0 1.0 4e6 1e6 6.582e-16 | grep "Solve time"
	@echo "CUDA:"
	@./$(CUDA_BIN) 3 2.003 5.788e-8 1.0 1.0 4e6 1e6 6.582e-16 --cuda | grep "Solve time"
	@echo "\n=== 4-electron system (1024x1024) ==="
	@echo "CPU:"
	@./$(CPU_BIN) 4 2.003 5.788e-8 1.0 1.0 4e6 1e6 6.582e-16 | grep "Solve time"
	@echo "CUDA:"
	@./$(CUDA_BIN) 4 2.003 5.788e-8 1.0 1.0 4e6 1e6 6.582e-16 --cuda | grep "Solve time"

# Help
.PHONY: help
help:
	@echo "Stochastic Liouville Equation Solver - Build System"
	@echo ""
	@echo "Command-line Interface Targets:"
	@echo "  make cpu         - Build CPU-only version (default)"
	@echo "  make cuda        - Build CUDA-accelerated version"
	@echo "  make both        - Build both CLI versions"
	@echo "  make test-cpu    - Build and test CPU version"
	@echo "  make test-cuda   - Build and test CUDA version"
	@echo "  make benchmark   - Compare CPU vs CUDA performance"
	@echo ""
	@echo "Qt GUI Targets:"
	@echo "  make gui         - Build Qt GUI (CPU version)"
	@echo "  make gui-cuda    - Build Qt GUI (CUDA version)"
	@echo "  make gui-all     - Build both GUI versions"
	@echo ""
	@echo "Mathematica Integration Targets:"
	@echo "  make matrix-solver      - Build matrix solver (CPU version)"
	@echo "  make matrix-solver-cuda - Build matrix solver (CUDA version)"
	@echo "  make matrix-solver-all  - Build both matrix solver versions"
	@echo ""
	@echo "Utility Targets:"
	@echo "  make clean       - Remove build artifacts"
	@echo "  make help        - Show this help message"
	@echo ""
	@echo "CLI Usage examples:"
	@echo "  ./build/operators_cpu <n_electrons> <g> <mu> <Bz> <a> <Ks> <Kd> <hbar>"
	@echo "  ./build/operators_cuda <n_electrons> <g> <mu> <Bz> <a> <Ks> <Kd> <hbar> --cuda"
	@echo ""
	@echo "GUI Usage examples:"
	@echo "  ./build/magspin_gui              # CPU version"
	@echo "  ./build/magspin_gui_cuda         # CUDA version"
	@echo ""
	@echo "Matrix Solver Usage examples:"
	@echo "  ./build/matrix_solver data/mma_out/matrix_data_1.txt"
	@echo "  ./build/matrix_solver_cuda data/mma_out/matrix_data_1.txt --cuda"
	@echo ""
	@echo "Requirements:"
	@echo "  - Eigen3 library (required for all builds)"
	@echo "  - CUDA Toolkit (required for CUDA builds)"
	@echo "  - Qt6 or Qt5 (required for GUI builds)"
