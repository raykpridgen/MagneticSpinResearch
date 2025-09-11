# Setup for CUDA to begin solving systems of equations

## Check for NVIDIA GPU
lspci | grep -i nvidia

nvidia-smi

## If 2 doesn't work, install drivers:
sudo apt update

apt search nvidia-driver
- search for latest driver

sudo apt install nvidia-driver-535
- (Or latest found in search)

## Install CUDA toolkit
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring.deb

sudo dpkg -i cuda-keyring.deb

sudo apt update

sudo apt install cuda

## Update environment Variables
export PATH=/usr/local/cuda/bin:$PATH

export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

## Reload
source ~/.bashrc

## Verify
nvcc --version

