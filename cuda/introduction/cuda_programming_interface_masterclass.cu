/**
 * =================================================================================================
 * FILE: cuda_programming_interface_masterclass.cu
 * AUTHOR: World's Best Coder & Tutor (IQ 300+)
 * TOPIC: 6. Programming Interface (CUDA C++ Extensions & Runtime API)
 * =================================================================================================
 *
 * COMPILATION INSTRUCTIONS:
 * -------------------------
 * This source file contains CUDA C++ extensions and must be compiled with the NVCC compiler.
 * NVCC acts as a compiler driver, separating device code (GPU) from host code (CPU).
 * 
 * Command: nvcc -o cuda_interface cuda_programming_interface_masterclass.cu
 *
 * =================================================================================================
 * DETAILED EXPLANATION OF THE PROGRAMMING INTERFACE:
 * =================================================================================================
 * 
 * 1. THE CUDA C++ PARADIGM:
 *    CUDA provides a seamless path for C++ developers to utilize GPU acceleration. It is not
 *    a new language, but a minimal set of EXTENSIONS to C++ combined with a Runtime Library.
 *
 * 2. LANGUAGE EXTENSIONS (The Frontend):
 *    These allow the definition of kernels and execution configuration.
 *    - Function Execution Space Specifiers: __global__, __device__, __host__.
 *    - Built-in Variables: gridDim, blockDim, blockIdx, threadIdx (for coordinate mapping).
 *    - Kernel Launch Syntax: kernelName<<<grid, block>>>(args...).
 *
 * 3. THE RUNTIME API (The Backend):
 *    Provided via <cuda_runtime.h>. It handles:
 *    - Device Memory Management (cudaMalloc, cudaFree).
 *    - Data Transfer (cudaMemcpy).
 *    - Device Management (querying device properties).
 *    - Synchronization.
 *
 * 4. RUNTIME VS. DRIVER API:
 *    - The Runtime API is built ON TOP of the low-level Driver API (cuda.h).
 *    - Driver API: Exposes low-level concepts like CUDA Contexts (analogue to CPU processes)
 *      and Modules (analogue to dynamic libraries/DLLs).
 *    - Runtime API: Implicitly manages contexts and modules. This leads to significantly
 *      more concise code.
 *    - Interoperability: Applications can mix both, but the Runtime is preferred for 99% of 
 *      use cases due to ease of use and type safety.
 *
 * =================================================================================================
 */

#include <iostream>
#include <vector>
#include <cuda_runtime.h> // The header for the CUDA Runtime API

/*
 * =================================================================================================
 * ERROR HANDLING UTILITY
 * =================================================================================================
 * As a best practice, every CUDA Runtime API call returns a generic error code (cudaError_t).
 * A robust system MUST check these codes. We define a macro to wrap API calls.
 */
#define GPU_ERR_CHK(ans) { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

/*
 * =================================================================================================
 * KERNEL DEFINITION (C++ Language Extension)
 * =================================================================================================
 * This function is the "Kernel". It is defined using the C++ extension keyword `__global__`.
 * 
 * ATTRIBUTES:
 * - __global__: Indicates the function runs on the Device (GPU) and is callable from the Host (CPU).
 * - void return: Kernels cannot return values directly; they write to global memory.
 * 
 * LOGIC:
 * Performs SAXPY (Single-Precision A*X + Y) operation.
 */
__global__ void saxpyKernel(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
    /*
     * BUILT-IN VARIABLES (Language Extension):
     * These variables are implicitly defined by the CUDA hardware/compiler for every thread.
     * 
     * threadIdx: The index of the thread within a block.
     * blockIdx:  The index of the block within the grid.
     * blockDim:  The size (dimensions) of the block.
     * 
     * We calculate a global index `i` to map this specific thread to a specific data element.
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * BOUNDARY CHECK:
     * Essential for robust code. The total number of threads launched (GridSize * BlockSize)
     * might exceed the array size `n`. We must ensure we don't access invalid memory.
     */
    if (i < n) {
        y[i] = a * x[i] + y[i];
    }
}

/*
 * =================================================================================================
 * HOST CODE (Main Execution Flow)
 * =================================================================================================
 */
int main() {
    // 1. PROBLEM SIZE CONFIGURATION
    // We will process 1 million elements.
    const int N = 1 << 20; 
    const size_t bytes = N * sizeof(float);
    const float scalar_a = 2.0f;

    std::cout << "[BestCoder] Initializing data for " << N << " elements..." << std::endl;

    // 2. HOST MEMORY ALLOCATION (Standard C++)
    // We use std::vector for automatic memory management on the Host (CPU side).
    // Alternatively, standard malloc() or new[] could be used.
    std::vector<float> h_x(N, 1.0f); // Fill x with 1.0
    std::vector<float> h_y(N, 2.0f); // Fill y with 2.0

    // 3. DEVICE MEMORY ALLOCATION (Runtime API)
    // We must explicitly allocate memory on the GPU global memory.
    // The Runtime API function `cudaMalloc` is the standard interface for this.
    float *d_x = nullptr;
    float *d_y = nullptr;

    // Notice the pointer-to-pointer cast. cudaMalloc allocates a linear address space on the device.
    GPU_ERR_CHK(cudaMalloc((void**)&d_x, bytes));
    GPU_ERR_CHK(cudaMalloc((void**)&d_y, bytes));

    // 4. DATA TRANSFER: HOST -> DEVICE (Runtime API)
    // We move input data from system memory to GPU memory.
    // `cudaMemcpy` is a synchronous call (blocks CPU until copy is complete).
    std::cout << "[BestCoder] Transferring data Host -> Device..." << std::endl;
    GPU_ERR_CHK(cudaMemcpy(d_x, h_x.data(), bytes, cudaMemcpyHostToDevice));
    GPU_ERR_CHK(cudaMemcpy(d_y, h_y.data(), bytes, cudaMemcpyHostToDevice));

    // 5. EXECUTION CONFIGURATION (Language Extension)
    // We must define the dimensions of the "Grid" and "Blocks".
    // Threads are grouped into blocks, blocks are grouped into a grid.
    
    int threadsPerBlock = 256; // Standard multiple of 32 (warp size) for occupancy.
    
    // Calculate grid size required to cover N elements.
    // Formula: (N + threadsPerBlock - 1) / threadsPerBlock performs a ceiling division.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "[BestCoder] Launching Kernel with Grid: " << blocksPerGrid 
              << ", Block: " << threadsPerBlock << std::endl;

    /*
     * 6. KERNEL LAUNCH (Language Extension)
     * The <<< ... >>> syntax is the core C++ extension for execution.
     * 
     * Architecture Note: 
     * This launch is ASYNCHRONOUS with respect to the host. The CPU continues execution
     * immediately after issuing the command to the GPU command queue.
     * 
     * Driver API Note:
     * If using the Driver API, this would require manually loading the module, 
     * getting the function handle, setting arguments, and calling cuLaunchKernel.
     * The Runtime API simplifies this dramatically.
     */
    saxpyKernel<<<blocksPerGrid, threadsPerBlock>>>(N, scalar_a, d_x, d_y);

    // 7. ERROR CHECKING FOR KERNEL LAUNCH
    // PeekAtLastError checks if the launch parameters were invalid.
    GPU_ERR_CHK(cudaPeekAtLastError());
    
    // DeviceSynchronize waits for the GPU to finish. This is necessary here to catch
    // asynchronous execution errors and to ensure data is ready before copying back.
    GPU_ERR_CHK(cudaDeviceSynchronize());

    // 8. DATA TRANSFER: DEVICE -> HOST (Runtime API)
    // Retrieve the results (y = a*x + y).
    std::cout << "[BestCoder] Transferring results Device -> Host..." << std::endl;
    GPU_ERR_CHK(cudaMemcpy(h_y.data(), d_y, bytes, cudaMemcpyDeviceToHost));

    // 9. VERIFICATION
    // Check if the calculation 2.0 * 1.0 + 2.0 = 4.0 is correct.
    float maxError = 0.0f;
    for (int i = 0; i < N; i++) {
        maxError = std::max(maxError, std::abs(h_y[i] - 4.0f));
    }

    std::cout << "[BestCoder] Max Error: " << maxError << std::endl;
    if(maxError == 0.0f) 
        std::cout << "[BestCoder] SUCCESS: Logic verified." << std::endl;
    else 
        std::cout << "[BestCoder] FAILURE: Calculation deviation detected." << std::endl;

    // 10. CLEANUP (Runtime API)
    // Just like malloc/free, we must free device resources to prevent memory leaks.
    // Context Management Note: When the process exits, the CUDA Context is destroyed 
    // and resources are freed, but explicit freeing is the standard of a great coder.
    GPU_ERR_CHK(cudaFree(d_x));
    GPU_ERR_CHK(cudaFree(d_y));

    return 0;
}