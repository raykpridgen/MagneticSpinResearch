#!/bin/bash
# MagSpin GUI Dependencies Installation Script
# This script installs all required dependencies for building the Qt GUI

set -e  # Exit on error

echo "=========================================="
echo "MagSpin GUI Dependencies Installer"
echo "=========================================="
echo ""

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_info() {
    echo -e "${YELLOW}[INFO]${NC} $1"
}

# Detect OS
if [ -f /etc/os-release ]; then
    . /etc/os-release
    OS=$ID
    VERSION=$VERSION_ID
else
    print_error "Cannot detect OS. This script supports Ubuntu/Debian-based systems."
    exit 1
fi

print_info "Detected OS: $OS $VERSION"
echo ""

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Please do not run this script as root. It will ask for sudo when needed."
    exit 1
fi

# Update package lists
print_info "Updating package lists..."
sudo apt update

echo ""
print_info "Installing essential build tools..."
sudo apt install -y build-essential g++ make pkg-config

# Install Eigen3
echo ""
print_info "Installing Eigen3 library (required for linear algebra)..."
sudo apt install -y libeigen3-dev

# Install Qt
echo ""
print_info "Installing Qt development packages..."
echo "Attempting to install Qt6 first (recommended)..."

if sudo apt install -y qt6-base-dev qt6-tools-dev qt6-tools-dev-tools libqt6printsupport6 2>/dev/null; then
    print_success "Qt6 installed successfully!"
    QT_VERSION="Qt6"
else
    print_info "Qt6 not available, trying Qt5..."
    if sudo apt install -y qtbase5-dev qt5-qmake qttools5-dev-tools libqt5printsupport5 2>/dev/null; then
        print_success "Qt5 installed successfully!"
        QT_VERSION="Qt5"
    else
        print_error "Failed to install Qt. Please install Qt manually."
        echo "Try one of these commands:"
        echo "  sudo apt install qt6-base-dev qt6-tools-dev"
        echo "  sudo apt install qtbase5-dev qt5-qmake"
        exit 1
    fi
fi

# Ask about CUDA installation
echo ""
echo "=========================================="
echo "CUDA Installation (Optional)"
echo "=========================================="
echo "CUDA is needed for GPU-accelerated computations."
echo "This is optional - the GUI works fine with CPU-only mode."
echo ""
read -p "Do you want to install CUDA support? (y/N): " -n 1 -r
echo ""

if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_info "Checking for NVIDIA GPU..."

    if lspci | grep -i nvidia > /dev/null; then
        print_success "NVIDIA GPU detected!"

        print_info "Installing CUDA Toolkit..."
        print_info "This may take a while..."

        # Check if CUDA is already installed
        if command -v nvcc &> /dev/null; then
            print_success "CUDA is already installed!"
            nvcc --version
        else
            # Install CUDA
            sudo apt install -y nvidia-cuda-toolkit

            # Add CUDA to PATH if not already there
            if ! grep -q "/usr/local/cuda/bin" ~/.bashrc; then
                echo "" >> ~/.bashrc
                echo "# CUDA paths" >> ~/.bashrc
                echo "export PATH=/usr/local/cuda/bin:\$PATH" >> ~/.bashrc
                echo "export LD_LIBRARY_PATH=/usr/local/cuda/lib64:\$LD_LIBRARY_PATH" >> ~/.bashrc
                print_info "Added CUDA to PATH in ~/.bashrc"
            fi

            print_success "CUDA Toolkit installed!"
            print_info "You may need to run 'source ~/.bashrc' or restart your terminal."
        fi
    else
        print_error "No NVIDIA GPU detected. Skipping CUDA installation."
        print_info "The GUI will work in CPU-only mode."
    fi
else
    print_info "Skipping CUDA installation. You can build the CPU-only GUI version."
fi

# Verify installations
echo ""
echo "=========================================="
echo "Verifying Installation"
echo "=========================================="

# Check g++
if command -v g++ &> /dev/null; then
    print_success "g++ compiler: $(g++ --version | head -n1)"
else
    print_error "g++ not found!"
fi

# Check make
if command -v make &> /dev/null; then
    print_success "make: $(make --version | head -n1)"
else
    print_error "make not found!"
fi

# Check Eigen3
if [ -d "/usr/include/eigen3" ]; then
    print_success "Eigen3 library found"
else
    print_error "Eigen3 not found at /usr/include/eigen3"
fi

# Check Qt
if pkg-config --exists Qt6Widgets 2>/dev/null; then
    print_success "Qt6 found: $(pkg-config --modversion Qt6Widgets)"
elif pkg-config --exists Qt5Widgets 2>/dev/null; then
    print_success "Qt5 found: $(pkg-config --modversion Qt5Widgets)"
else
    print_error "Qt not found!"
fi

# Check moc (Qt Meta-Object Compiler)
if command -v moc-qt6 &> /dev/null || command -v moc-qt5 &> /dev/null || command -v moc &> /dev/null; then
    print_success "Qt MOC (Meta-Object Compiler) found"
else
    print_error "Qt MOC not found!"
fi

# Check CUDA (if requested)
if command -v nvcc &> /dev/null; then
    print_success "CUDA Toolkit: $(nvcc --version | grep release | awk '{print $5}' | cut -d',' -f1)"
fi

# Summary
echo ""
echo "=========================================="
echo "Installation Complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Build the GUI (CPU version):"
echo "   make gui"
echo ""
echo "2. Run the GUI:"
echo "   ./build/magspin_gui"
echo ""
if command -v nvcc &> /dev/null; then
    echo "3. Or build the CUDA-accelerated GUI version:"
    echo "   make gui-cuda"
    echo ""
    echo "4. Run the CUDA version:"
    echo "   ./build/magspin_gui_cuda"
    echo ""
fi
echo "For more build options, run:"
echo "   make help"
echo ""
print_success "All dependencies have been installed successfully!"
