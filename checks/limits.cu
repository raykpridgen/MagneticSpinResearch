#include <iostream>
#include <cuda_runtime.h>

int main()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);  // 0 = first GPU
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << "\n";
    std::cout << "Max threads per multiprocessor: " << prop.maxThreadsPerMultiProcessor << "\n";
    std::cout << "Max block dimensions: " 
              << prop.maxThreadsDim[0] << " x "
              << prop.maxThreadsDim[1] << " x "
              << prop.maxThreadsDim[2] << "\n";
    std::cout << "Max grid dimensions: " 
              << prop.maxGridSize[0] << " x "
              << prop.maxGridSize[1] << " x "
              << prop.maxGridSize[2] << "\n";
    std::cout << "Shared memory per block: " << prop.sharedMemPerBlock / 1024.0 << " KB\n";
    std::cout << "Registers per block: " << prop.regsPerBlock << "\n";
    return 0;
}
