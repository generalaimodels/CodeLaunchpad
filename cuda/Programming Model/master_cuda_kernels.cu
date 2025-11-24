/**
 * ====================================================================================================
 * FILE: master_cuda_kernels.cu
 * AUTHOR: The World's Best Coder (AI Entity)
 * TOPIC: CUDA Kernels, Execution Model, and Memory Hierarchy (End-to-End Mastery)
 * ARCHITECTURE: CUDA C++ (Compute Unified Device Architecture)
 *
 * ----------------------------------------------------------------------------------------------------
 * PREFACE & PHILOSOPHY:
 * To master CUDA, one must stop thinking sequentially. The CPU is a latency-optimized scalar processor.
 * The GPU is a throughput-optimized vector processor.
 *
 * When we write a "Kernel", we are writing code for a single thread. However, we must inherently
 * visualize that code running on thousands of threads simultaneously.
 *
 * This file covers:
 * 1. The Anatomy of a Kernel (__global__, __device__, __host__).
 * 2. The Execution Configuration (<<<Grid, Block>>>).
 * 3. Thread Hierarchy & Global Indexing (Mapping logical threads to physical data).
 * 4. Memory spaces (Host vs. Device).
 * 5. Exception Handling & Boundary Checks (The Grid-Stride Loop pattern).
 * 6. Asynchronous Error Handling.
 * ====================================================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

// ====================================================================================================
// PART 1: UTILITY & ERROR HANDLING
// ====================================================================================================
/**
 * BEST PRACTICE: CUDA API calls return error codes. Ignoring them is fatal.
 * This macro wraps CUDA calls to catch runtime errors (e.g., Out of Memory, Invalid Configuration).
 */
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess)
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// ====================================================================================================
// PART 2: KERNEL DEFINITIONS
// ====================================================================================================

/**
 * CONCEPT: Function Execution Space Qualifiers
 *
 * __global__ : Defines a "Kernel".
 *              - Callable from: Host (CPU) (and Device in Compute Capability 5.0+ "Dynamic Parallelism").
 *              - Executes on: Device (GPU).
 *              - Must return void.
 *
 * __device__ : Defines a helper function.
 *              - Callable from: Device only.
 *              - Executes on: Device.
 *
 * __host__   : Standard C++ function (default).
 *              - Callable from: Host.
 *              - Executes on: Host.
 */

/**
 * ----------------------------------------------------------------------------------------------------
 * KERNEL 1: The "Textbook" Implementation (Naive)
 * ----------------------------------------------------------------------------------------------------
 * LIMITATIONS:
 * 1. Assumes 1D Grid and 1D Blocks.
 * 2. Fails if N > (GridDim * BlockDim).
 * 3. Fails to access memory efficiently if not coalesced (though linear access here is fine).
 *
 * INPUTS:
 * - A, B: Input arrays (Must reside in Device Memory).
 * - C: Output array (Must reside in Device Memory).
 * - N: Total number of elements.
 */
__global__ void vecAddNaive(const float* A, const float* B, float* C, int N)
{
    // Built-in Variables:
    // threadIdx.x : The ID of the thread within the current block.
    // blockIdx.x  : The ID of the block within the grid.
    // blockDim.x  : The number of threads in a block.

    // GLOBAL INDEX CALCULATION:
    // This maps the hierarchical hardware (Blocks/Threads) to a linear memory address space.
    int i = blockIdx.x * blockDim.x + threadIdx.x;

    // BOUNDARY CHECK (CRITICAL):
    // A GPU launches threads in multiples of the "warp" size (32) and block size.
    // If N = 100 and we launch 128 threads, threads 100-127 must NOT access memory.
    // Accessing C[101] results in undefined behavior or segfaults.
    if (i < N)
    {
        C[i] = A[i] + B[i];
    }
}

/**
 * ----------------------------------------------------------------------------------------------------
 * KERNEL 2: The "World-Class" Implementation (Grid-Stride Loop)
 * ----------------------------------------------------------------------------------------------------
 * WHY THIS IS BETTER (IQ 300 APPROACH):
 * 1. Decoupling: The number of threads launched is decoupled from the data size N.
 * 2. Scalability: Handles arrays larger than the max grid size.
 * 3. Reusability: One thread can compute multiple elements, increasing instruction intensity
 *    and hiding memory latency.
 * 4. Debugging: You can launch 1 block with 1 thread to debug serially without changing code.
 * 5. Keyword '__restrict__': Promises compiler that pointers A, B, C do not alias (overlap),
 *    enabling aggressive load/store optimizations.
 */
__global__ void vecAddRobust(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N)
{
    // Calculate the unique index of this thread in the global grid
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    // Calculate the total stride (total threads in the grid)
    // gridDim.x : Total number of blocks in the grid
    int stride = blockDim.x * gridDim.x;

    // EXCEPTION HANDLING & LOOPING:
    // Instead of 'if (index < N)', we loop.
    // If the array is larger than the grid, the thread wraps around.
    // Example: 1000 elements, 256 threads.
    // Thread 0 processes indices: 0, 256, 512, 768.
    for (int i = index; i < N; i += stride)
    {
        C[i] = A[i] + B[i];
    }
}

// ====================================================================================================
// PART 3: HOST CODE (DRIVER)
// ====================================================================================================

int main()
{
    printf("[MASTER CODER TUTORIAL] CUDA Kernel Execution Concepts.\n");

    // 1. DEFINING PROBLEM SIZE
    // Let's use a large number to justify GPU usage.
    // 1 Million elements.
    int N = 1 << 20;
    size_t size = N * sizeof(float);

    // 2. HOST MEMORY ALLOCATION
    // This memory lives in the system RAM (accessible by CPU).
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size); // Result container

    // Initialize data on Host (CPU side)
    // We use random deterministic data for verification.
    for (int i = 0; i < N; ++i) {
        h_A[i] = static_cast<float>(rand()) / RAND_MAX;
        h_B[i] = static_cast<float>(rand()) / RAND_MAX;
    }

    // 3. DEVICE MEMORY ALLOCATION
    // Pointers d_A, d_B, d_C act as handles to memory on the GPU VRAM.
    // Attempting to dereference these on the CPU (*d_A) will cause a segfault.
    float *d_A, *d_B, *d_C;

    // cudaMalloc allocates linear memory on the device.
    gpuErrchk(cudaMalloc((void**)&d_A, size));
    gpuErrchk(cudaMalloc((void**)&d_B, size));
    gpuErrchk(cudaMalloc((void**)&d_C, size));

    // 4. DATA TRANSFER (HOST -> DEVICE)
    // The PCIe bus is the bottleneck. We must copy data to the GPU to process it.
    // cudaMemcpy is a blocking call (CPU waits until copy finishes).
    printf("Copying data to device...\n");
    gpuErrchk(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 5. EXECUTION CONFIGURATION (The <<<...>>> Syntax)
    // We must decide how to organize our threads.
    // CUDA organizes threads into Blocks, and Blocks into a Grid.

    // Block Size: Usually a multiple of 32 (Warp size).
    // 256 or 512 are common heuristics for occupancy.
    int threadsPerBlock = 256;

    // Grid Size: How many blocks do we need to cover N elements?
    // Formula: (N + threadsPerBlock - 1) / threadsPerBlock
    // This creates a "Ceiling" division ensuring we have enough threads for N.
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching Kernel with Grid: %d, Block: %d\n", blocksPerGrid, threadsPerBlock);

    // 6. KERNEL LAUNCH
    // Syntax: KernelName<<<Blocks, Threads, SharedMemBytes, StreamID>>>(Args...);
    // Note: SharedMemBytes and StreamID are optional (default to 0).
    // This call is ASYNCHRONOUS. Control returns to CPU immediately after launch.
    vecAddRobust<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 7. ERROR CHECKING (KERNEL SPECIFIC)
    // Because launches are async, cudaPeekAtLastError checks if the *launch* was valid.
    gpuErrchk(cudaPeekAtLastError());

    // cudaDeviceSynchronize waits for the GPU to finish.
    // Only needed here because we want to check execution errors or measure time immediately.
    // In production, cudaMemcpy acts as an implicit synchronization point.
    gpuErrchk(cudaDeviceSynchronize());

    // 8. DATA TRANSFER (DEVICE -> HOST)
    // Retrieve results.
    printf("Copying results to host...\n");
    gpuErrchk(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // 9. VERIFICATION
    // Always verify GPU results against a CPU reference implementation (or math logic)
    // to ensure the kernel logic is sound.
    bool success = true;
    double epsilon = 1e-5; // FP32 precision tolerance
    for (int i = 0; i < N; i++) {
        float expected = h_A[i] + h_B[i];
        if (abs(h_C[i] - expected) > epsilon) {
            printf("ERROR at index %d: GPU %f != CPU %f\n", i, h_C[i], expected);
            success = false;
            break;
        }
    }

    if (success) {
        printf("SUCCESS: Kernel execution and validation passed.\n");
    }

    // 10. CLEANUP
    // Freeing device memory is crucial to prevent memory leaks in VRAM.
    gpuErrchk(cudaFree(d_A));
    gpuErrchk(cudaFree(d_B));
    gpuErrchk(cudaFree(d_C));

    // Free host memory
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}