/**
 * @file 01_cuda_architecture_overview.cu
 * @brief Comprehensive Overview of CUDA (Compute Unified Device Architecture).
 * @author Master Architect - AI Generated
 *
 * =================================================================================================
 *                                      SECTION 1: CUDA OVERVIEW
 * =================================================================================================
 *
 * 1. WHAT IS CUDA?
 *    CUDA is a parallel computing platform and programming model developed by NVIDIA. It exposes
 *    the massive computational power of the Graphics Processing Unit (GPU) for general-purpose
 *    computing (GPGPU).
 *
 * 2. ARCHITECTURAL PHILOSOPHY (SIMT):
 *    CUDA employs the Single Instruction, Multiple Threads (SIMT) architecture.
 *    - The CPU (Host) is a Latency-Oriented design (large caches, complex flow control, few cores).
 *    - The GPU (Device) is a Throughput-Oriented design (small caches, massive parallelism, thousands of cores).
 *    
 *    The GPU handles parallel tasks by launching a grid of thread blocks. Threads within a block
 *    can cooperate via Shared Memory. Threads are physically executed in groups of 32 called "Warps".
 *
 * 3. THE CUDA PROGRAMMING MODEL:
 *    - KERNELS: Functions defined with `__global__` executed N times in parallel by N different CUDA threads.
 *    - THREAD HIERARCHY:
 *      - Thread: The fundamental unit of execution.
 *      - Block: A collection of threads that can synchronize and share memory.
 *      - Grid: A collection of blocks that execute a kernel.
 *    - MEMORY HIERARCHY:
 *      - Global Memory: High latency, accessible by all threads (DRAM).
 *      - Shared Memory: Low latency, on-chip, accessible by threads in a block (User-managed cache).
 *      - Registers: Zero latency, private to a thread.
 *
 * =================================================================================================
 *                                      COMPILATION INSTRUCTIONS
 * =================================================================================================
 * Compiler: nvcc (NVIDIA CUDA Compiler)
 * Command:  nvcc -O3 -arch=sm_80 01_cuda_architecture_overview.cu -o cuda_overview
 * Note:     Adjust -arch=sm_XX based on your specific GPU capability (e.g., sm_75 for Turing, sm_80 for Ampere).
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <assert.h>

/*
 * =================================================================================================
 *                                ERROR HANDLING MACROS
 * =================================================================================================
 * Professional coding requires rigorous exception handling. CUDA API calls return error codes
 * that must be checked. Kernel launches are asynchronous, requiring specific checks.
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
 *                                      KERNEL DEFINITION
 * =================================================================================================
 * 
 * @brief: A simple SAXPY kernel (Single Precision A*X + Y) to demonstrate parallelism.
 * 
 * KEY CONCEPTS:
 * 1. __global__: Indicates this function runs on the Device (GPU) and is called from Host (CPU).
 * 2. __restrict__: Optimization hint to the compiler that pointers do not alias, allowing 
 *                  better read-only caching.
 * 3. Index Calculation: Mapping the 1D hardware thread index to the data array.
 */
__global__ void saxpy_kernel(int n, float a, const float* __restrict__ x, float* __restrict__ y) {
    
    /*
     * THREAD ID CALCULATION:
     * blockIdx.x : The ID of the block within the grid.
     * blockDim.x : The number of threads per block.
     * threadIdx.x: The ID of the thread within the block.
     *
     * Global ID determines which element of the array this specific thread will process.
     * This eliminates the need for a loop (loop unrolling via hardware parallelism).
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /*
     * BOUNDARY CHECK:
     * The total number of threads launched usually exceeds the array size 'n' because
     * we launch blocks of fixed sizes (e.g., 256). We must ensure we don't access memory out of bounds.
     */
    if (i < n) {
        // The core computation
        y[i] = a * x[i] + y[i];
    }
}

/*
 * =================================================================================================
 *                                      HOST CODE (MAIN)
 * =================================================================================================
 * The Host orchestrates the data movement and computation instructions sent to the Device.
 */
int main(void) {
    printf("[CUDA OVERVIEW] Starting Execution...\n");

    // 1. DEFINE PROBLEM SIZE
    // We will process 1 Million elements. On a CPU, this is sequential. On GPU, massive parallel.
    int N = 1 << 20; // 1,048,576 elements
    size_t bytes = N * sizeof(float);

    // 2. QUERY DEVICE PROPERTIES
    // A master coder always understands the hardware topology before executing.
    int devId = 0;
    cudaDeviceProp deviceProp;
    GPU_ERR_CHK(cudaGetDeviceProperties(&deviceProp, devId));
    printf("Device: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Multiprocessors (SMs): %d\n", deviceProp.multiProcessorCount);
    printf("Warp Size: %d\n", deviceProp.warpSize);

    // 3. ALLOCATE HOST MEMORY (Pinned Memory vs Pageable)
    // Standard malloc allocates pageable memory. cudaHostAlloc allows for faster DMA transfers via PCIe.
    float *h_x, *h_y, *h_ref;
    h_x = (float*)malloc(bytes);
    h_y = (float*)malloc(bytes);
    h_ref = (float*)malloc(bytes); // For CPU verification

    // Initialize data
    for (int i = 0; i < N; i++) {
        h_x[i] = 1.0f;
        h_y[i] = 2.0f;
        h_ref[i] = 2.0f; // Reference copy
    }

    // 4. ALLOCATE DEVICE MEMORY
    // Allocation on the GPU Global Memory (DRAM).
    float *d_x, *d_y;
    GPU_ERR_CHK(cudaMalloc((void**)&d_x, bytes));
    GPU_ERR_CHK(cudaMalloc((void**)&d_y, bytes));

    // 5. DATA TRANSFER (HOST -> DEVICE)
    // This is often the bottleneck (PCIe bus latency).
    GPU_ERR_CHK(cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice));
    GPU_ERR_CHK(cudaMemcpy(d_y, h_y, bytes, cudaMemcpyHostToDevice));

    // 6. EXECUTION CONFIGURATION (Grid and Block Dimensions)
    /*
     * OCCUPANCY:
     * To hide memory latency, we need enough active warps per SM.
     * Standard block size is often 128, 256, or 512.
     * 
     * threadsPerBlock: 256 is a safe heuristic for most modern architectures to maximize occupancy.
     * blocksPerGrid: Total elements divided by threads per block, rounded up.
     */
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching Kernel with Grid Size: %d, Block Size: %d\n", blocksPerGrid, threadsPerBlock);

    // 7. LAUNCH KERNEL
    // Syntax: kernel_name<<<gridDim, blockDim, sharedMemSize, stream>>>(args);
    float scalar_a = 2.0f;
    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(N, scalar_a, d_x, d_y);

    // 8. ERROR CHECKING FOR KERNEL
    // Kernel launches are asynchronous. errors may not appear immediately.
    GPU_ERR_CHK(cudaPeekAtLastError()); // Check for invalid launch parameters
    GPU_ERR_CHK(cudaDeviceSynchronize()); // Wait for GPU to finish (check for execution errors)

    // 9. DATA TRANSFER (DEVICE -> HOST)
    // Retrieve results
    GPU_ERR_CHK(cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost));

    // 10. VERIFICATION (CPU vs GPU)
    // Never trust hardware blindly. Verify correctness.
    // CPU Calculation for comparison
    int error_count = 0;
    for (int i = 0; i < N; i++) {
        float expected = scalar_a * h_x[i] + h_ref[i]; // 2.0 * 1.0 + 2.0 = 4.0
        if (fabs(h_y[i] - expected) > 1e-5) {
            error_count++;
            if(error_count < 5) printf("Error at index %d: GPU %f != CPU %f\n", i, h_y[i], expected);
        }
    }

    if (error_count == 0) {
        printf("PASSED: Result Verified. Calculation Successful.\n");
    } else {
        printf("FAILED: %d errors found.\n", error_count);
    }

    // 11. RESOURCE CLEANUP
    // Memory leaks in VRAM are fatal for long-running HPC applications.
    cudaFree(d_x);
    cudaFree(d_y);
    free(h_x);
    free(h_y);
    free(h_ref);

    // Reset device to flush profiling data
    cudaDeviceReset();

    return 0;
}