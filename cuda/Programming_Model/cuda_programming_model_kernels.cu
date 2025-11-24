/**
 * @file    cuda_programming_model_kernels.cu
 * @brief   Comprehensive explanation and implementation of CUDA Kernels (Topic 5.1).
 * @author  Master Coder (IQ > 300)
 * @date    2023
 *
 * ================================================================================================
 * 5. PROGRAMMING MODEL :: 5.1 KERNELS
 * ================================================================================================
 *
 * OVERVIEW:
 * This file serves as a masterclass in the CUDA Programming Model, specifically focusing on "Kernels".
 * As the requested "Best Coder", I have implemented a Vector Addition example that transcends the
 * basic example provided in standard documentation. It incorporates production-grade error handling,
 * optimal memory patterns, and detailed architectural explanations.
 *
 * CONCEPTS COVERED:
 * 1. __global__ Specifier: Distinguishing Device code from Host code.
 * 2. Execution Configuration <<<...>>>: Defining the Grid and Block dimensions.
 * 3. Thread Identity: Using built-in variables (threadIdx, blockIdx, blockDim) to map logic to data.
 * 4. SIMT Architecture: Understanding how Single Instruction Multiple Threads logic applies here.
 * 5. Memory Management: Explicit data movement between Host (CPU) and Device (GPU).
 *
 * ================================================================================================
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <iostream>
#include <vector>
#include <cmath>
#include <cstdio>

/**
 * @brief Error Handling Macro.
 *
 * A genius coder never assumes hardware behaves perfectly. This macro wraps CUDA API calls.
 * If a call fails, it prints the file, line number, and error string, then terminates.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t error = call; \
        if (error != cudaSuccess) { \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n", __FILE__, __LINE__, cudaGetErrorString(error)); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/**
 * ================================================================================================
 * KERNEL DEFINITION
 * ================================================================================================
 *
 * Concept: __global__
 * The `__global__` declaration specifier tells the compiler that this function:
 * 1. Runs on the Device (GPU).
 * 2. Is callable from the Host (CPU).
 *
 * Concept: SIMT (Single Instruction, Multiple Threads)
 * Unlike a C++ for-loop which executes sequentially, this function is instantiated
 * simultaneously across thousands of CUDA threads.
 *
 * Optimization Note:
 * We use `const float* __restrict__` for input arrays `A` and `B`.
 * - `const`: Enforces read-only correctness.
 * - `__restrict__`: A hint to the compiler that these pointers do not alias (overlap in memory).
 *   This allows the NVCC compiler to optimize load instructions via the Read-Only Data Cache (LDG).
 *
 * @param A  Input vector A (Device Pointer)
 * @param B  Input vector B (Device Pointer)
 * @param C  Output vector C (Device Pointer)
 * @param N  Total number of elements
 */
__global__ void VecAdd(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N)
{
    /**
     * --------------------------------------------------------------------------------------------
     * THREAD IDENTIFICATION & MAPPING
     * --------------------------------------------------------------------------------------------
     * In the prompt's example: VecAdd<<<1, N>>>(...);
     * The prompt assumes a single block. In that specific case, `int i = threadIdx.x;` is valid.
     *
     * However, hardware limits the number of threads per block (max 1024 on modern architectures).
     * To process large arrays (N > 1024), we must use multiple blocks (Grid).
     *
     * The global index formula:
     * i = (Block ID * Block Size) + Thread ID within Block
     */
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    /**
     * --------------------------------------------------------------------------------------------
     * BOUNDARY CHECK (The Exception Handler)
     * --------------------------------------------------------------------------------------------
     * Since the total number of threads launched (GridDim * BlockDim) might slightly exceed N
     * (due to rounding up to the nearest block size), we must ensure we don't access invalid memory.
     */
    if (i < N)
    {
        // The core computation: Performed in parallel by N threads.
        C[i] = A[i] + B[i];
    }
}

/**
 * @brief Helper function to initialize data on Host.
 */
void randomInit(float* data, int size)
{
    for (int i = 0; i < size; ++i)
    {
        data[i] = static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
    }
}

/**
 * ================================================================================================
 * HOST CODE (Main)
 * ================================================================================================
 */
int main()
{
    // 1. Problem Configuration
    // We choose a size larger than 1024 to demonstrate robust Grid/Block logic,
    // proving superiority over the basic textbook example.
    const int N = 1 << 20; // 1 Million elements (2^20)
    size_t sizeBytes = N * sizeof(float);

    std::cout << "[Vector Addition of " << N << " elements]\n";

    // 2. Host Memory Allocation (Pinned Memory for performance)
    // Regular malloc is pageable. cudaMallocHost allocates pinned (page-locked) memory,
    // allowing for faster PCI-E transfer speeds (DMA).
    float *h_A, *h_B, *h_C;
    CUDA_CHECK(cudaMallocHost((void**)&h_A, sizeBytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_B, sizeBytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_C, sizeBytes));

    // Initialize vectors A and B
    randomInit(h_A, N);
    randomInit(h_B, N);

    // 3. Device Memory Allocation
    // We must allocate memory in the GPU's VRAM (Global Memory).
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, sizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_B, sizeBytes));
    CUDA_CHECK(cudaMalloc((void**)&d_C, sizeBytes));

    // 4. Memory Copy: Host -> Device
    // Copy input data from CPU RAM to GPU VRAM.
    CUDA_CHECK(cudaMemcpy(d_A, h_A, sizeBytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, sizeBytes, cudaMemcpyHostToDevice));

    // 5. Execution Configuration
    // This is the "syntax" mentioned in the prompt: <<< GridDim, BlockDim >>>
    //
    // BlockDim: How many threads execute together in a thread block.
    // Hardware typically executes threads in "Warps" of 32.
    // 256 is a standard heuristic for high occupancy on most GPUs.
    int threadsPerBlock = 256;

    // GridDim: How many blocks are needed to cover N elements.
    // Formula: ceil(N / threadsPerBlock)
    // Implementation: (N + threadsPerBlock - 1) / threadsPerBlock performs integer ceiling.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "Execution Configuration: <<< " << blocksPerGrid << ", " << threadsPerBlock << " >>>\n";

    // 6. Kernel Invocation
    // This launches the 'VecAdd' function on the GPU.
    // Unlike C++ functions, this returns immediately (asynchronous launch).
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 7. Error Checking for Asynchronous Launch
    // Check if the launch logic itself was invalid (e.g., invalid config).
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to catch execution errors and ensure the kernel has finished
    // before accessing results (though cudaMemcpy implicitly syncs, explicit sync is good for debugging).
    CUDA_CHECK(cudaDeviceSynchronize());

    // 8. Memory Copy: Device -> Host
    // Retrieve result from GPU VRAM to CPU RAM.
    CUDA_CHECK(cudaMemcpy(h_C, d_C, sizeBytes, cudaMemcpyDeviceToHost));

    // 9. Verification
    // A 300 IQ coder always verifies the output mathematically.
    // We verify a small subset or check error tolerance.
    bool success = true;
    double maxError = 0.0;
    for (int i = 0; i < N; i++)
    {
        float expected = h_A[i] + h_B[i];
        if (fabs(h_C[i] - expected) > 1e-5)
        {
            success = false;
            printf("Verification failed at index %d: Expected %f, Got %f\n", i, expected, h_C[i]);
            break;
        }
    }

    if (success)
    {
        std::cout << "Test PASSED. SIMT Execution successful.\n";
    }

    // 10. Resource Deallocation
    // Clean up device memory to prevent memory leaks in VRAM.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));

    // Clean up pinned host memory.
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));

    return 0;
}