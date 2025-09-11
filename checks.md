# Checks to run when interfacing with the GPU initially

## GPU visibility 
nvidia-smi
- shows version, etc

## Query Device being used
- run check.cpp with: 

nvcc device_query.cu -o device_query

./device_query

- Tells max block size, warp size, grid size

## Toolkit Info

nvcc --version
- Compiler version

cat /usr/local/cuda/version.txt

ls /usr/local/ | grep cuda
- Runtime library version

## Helpful commands

deviceQuery | grep "Capability"
-  GPU architecture 

lspci | grep -i nvidia
- GPU hardware visibility

# Capture Specs
- Get outputs of nvidia-smi, device_query to get limits


