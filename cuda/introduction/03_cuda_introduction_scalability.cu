/**
 * @file 03_cuda_introduction_scalability.cu
 * @brief Introduction to CUDA: GPU Architecture, Benefits, and the Scalable Programming Model.
 * @author Master Architect (IQ > 300)
 * @version 1.0
 * @date 2023-10-27
 *
 * =================================================================================================
 *                                         THEORY: CPU VS GPU
 * =================================================================================================
 * 
 * 3.1. THE BENEFITS OF USING GPUS
 * -------------------------------
 * The CPU is a "Latency-Oriented" processor. It excels at executing a single thread of instructions
 * very quickly. It dedicates massive transistor budget to:
 *   1. Large Caches (L1/L2/L3) -> To reduce effective memory latency for sequential access.
 *   2. Complex Control Logic (Branch Prediction, Out-of-Order Execution) -> To handle conditional logic.
 * 
 * The GPU is a "Throughput-Oriented" processor. It assumes that for every thread waiting on memory,
 * there are thousands of others ready to execute. It dedicates transistors to:
 *   1. Massive numbers of Compute Units (ALUs/FPUs).
 *   2. Large Register Files (to support thousands of active threads).
 * 
 * @note FIGURE 1 LOGIC: The GPU uses transistors for Data Processing, not caching/flow control.
 * This allows the GPU to hide memory latency by swapping active warps (groups of 32 threads) 
 * rather than relying on caches.
 *
 * 3.3. A SCALABLE PROGRAMMING MODEL
 * ---------------------------------
 * CUDA solves the challenge of diverse hardware (different core counts) via "Transparent Scalability".
 * 
 * KEY ABSTRACTIONS:
 * 1. Hierarchy of Thread Groups: Grid -> Block -> Warp -> Thread.
 * 2. Shared Memories: Fast, user-managed cache for threads within a block to cooperate.
 * 3. Barrier Synchronization: `__syncthreads()` ensures order within a block.
 *
 * SCALABILITY MECHANISM (FIGURE 3 LOGIC):
 * A compiled CUDA program defines a Grid of Blocks. The GPU hardware (Runtime + Scheduler) 
 * assigns these Blocks to available Streaming Multiprocessors (SMs).
 * - High-end GPU (100 SMs): Executes 100 blocks in parallel.
 * - Low-end GPU (10 SMs): Executes 10 blocks in parallel, serializing the rest.
 * *The code remains unchanged.*
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// =================================================================================================
//                                  UTILITY: BEST PRACTICES
// =================================================================================================

/**
 * @brief Macro for robust error checking.
 * A Master Coder never assumes success. Every API call is verified.
 */
#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) exit(code);
    }
}

// Constants for the problem size
#define ARRAY_SIZE (1 << 24) // ~16 Million elements
#define THREADS_PER_BLOCK 256

// =================================================================================================
//                          KERNEL: DEMONSTRATING SCALABILITY & COOPERATION
// =================================================================================================

/**
 * @brief A Kernel performing a Dot Product (Partial).
 * 
 * This kernel demonstrates the three key abstractions of Section 3.3:
 * 1. **Thread Hierarchy**: Used to calculate global index `idx`.
 * 2. **Shared Memory**: Used (`cache`) to store intermediate results for the block.
 * 3. **Barrier Synchronization**: Used (`__syncthreads`) to ensure data validity during reduction.
 *
 * SCALABILITY NOTE:
 * This kernel runs as independent Blocks. Whether the GPU has 1 SM or 80 SMs, 
 * the hardware scheduler dispatches blocks to SMs. The logic inside the kernel 
 * is agnostic to the total hardware count.
 *
 * @param A Input vector A
 * @param B Input vector B
 * @param partial_C Output vector containing partial sums (one per block)
 * @param N Total number of elements
 */
__global__ void scalable_dot_product_kernel(const float* __restrict__ A, 
                                            const float* __restrict__ B, 
                                            float* __restrict__ partial_C, 
                                            int N) 
{
    // 1. SHARED MEMORY: The "Cooperation" Abstraction.
    // This memory is on-chip, extremely fast, and private to the Block.
    __shared__ float cache[THREADS_PER_BLOCK];

    // 2. THREAD HIERARCHY: Global Index Calculation
    int tid = threadIdx.x;                       // Local Thread ID
    int grid_idx = blockIdx.x * blockDim.x + tid; // Global Array Index

    // Grid-Stride Loop: Ensures scalability even if N > Total Threads
    // Although usually configured such that grid covers N, this loop handles arbitrary sizes.
    float temp_sum = 0.0f;
    while (grid_idx < N) {
        // The core compute: Heavy floating point operation
        // This is where GPU throughput shines over CPU latency.
        temp_sum += A[grid_idx] * B[grid_idx];
        grid_idx += gridDim.x * blockDim.x;
    }

    // Store local result into shared memory
    cache[tid] = temp_sum;

    // 3. BARRIER SYNCHRONIZATION
    // Wait for all threads in this block to populate 'cache'.
    __syncthreads();

    // PARALLEL REDUCTION (Tree-based)
    // Threads cooperate to sum up the values in 'cache' to cache[0].
    // This is "Fine-grained data parallelism" nested within the coarse block parallelism.
    
    int i = blockDim.x / 2;
    while (i != 0) {
        if (tid < i) {
            cache[tid] += cache[tid + i];
        }
        // We must synchronize at every step of reduction to prevent race conditions
        __syncthreads(); 
        i /= 2;
    }

    // The first thread of the block writes the block's partial result to global memory.
    // This creates "Coarse-grained" sub-problems (one result per block).
    if (tid == 0) {
        partial_C[blockIdx.x] = cache[0];
    }
}

// =================================================================================================
//                                          HOST CODE
// =================================================================================================

int main(void) {
    printf("[CUDA 3.0 INTRODUCTION] Scalability & Architecture Demo\n");
    printf("=======================================================\n");

    // --- 1. QUERY DEVICE ARCHITECTURE ---
    // Understanding the hardware is crucial for understanding scalability (Figure 3 logic).
    int dev = 0;
    cudaDeviceProp deviceProp;
    GPU_CHECK(cudaGetDeviceProperties(&deviceProp, dev));

    printf("Device Name: %s\n", deviceProp.name);
    printf("Compute Capability: %d.%d\n", deviceProp.major, deviceProp.minor);
    printf("Multiprocessors (SMs): %d\n", deviceProp.multiProcessorCount);
    printf("  -> This GPU can execute %d Blocks concurrently (approx).\n", deviceProp.multiProcessorCount * (2048/THREADS_PER_BLOCK));
    printf("  -> Scalability: If we upgrade to a GPU with %d SMs, performance increases automatically.\n\n", deviceProp.multiProcessorCount * 2);

    // --- 2. MEMORY ALLOCATION ---
    size_t bytes = ARRAY_SIZE * sizeof(float);
    
    // Host memory
    float *h_A = (float*)malloc(bytes);
    float *h_B = (float*)malloc(bytes);
    float *h_ref = (float*)malloc(bytes); // Not used for full verification to save time, but standard practice.

    // Initialize vectors
    for(int i=0; i<ARRAY_SIZE; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // Device memory
    float *d_A, *d_B, *d_partial_C;
    GPU_CHECK(cudaMalloc(&d_A, bytes));
    GPU_CHECK(cudaMalloc(&d_B, bytes));

    // Calculate Grid Dimensions
    // To saturate the GPU, we need enough blocks.
    int blocksPerGrid = (ARRAY_SIZE + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    
    // Allocate memory for partial sums (one float per block)
    float *h_partial_C = (float*)malloc(blocksPerGrid * sizeof(float));
    GPU_CHECK(cudaMalloc(&d_partial_C, blocksPerGrid * sizeof(float)));

    printf("Problem Size: %d Elements\n", ARRAY_SIZE);
    printf("Execution Configuration: Grid Size = %d Blocks, Block Size = %d Threads\n", blocksPerGrid, THREADS_PER_BLOCK);
    printf("Total Threads: %d\n\n", blocksPerGrid * THREADS_PER_BLOCK);

    // --- 3. DATA TRANSFER (Host -> Device) ---
    GPU_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    GPU_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // --- 4. EXECUTION & TIMING ---
    // Using CUDA Events to measure pure kernel execution time
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    // LAUNCH THE SCALABLE KERNEL
    // The runtime system decomposes this Grid into blocks and schedules them
    // onto the available SMs (Multiprocessors).
    scalable_dot_product_kernel<<<blocksPerGrid, THREADS_PER_BLOCK>>>(d_A, d_B, d_partial_C, ARRAY_SIZE);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);

    // --- 5. FINALIZE COMPUTATION ON CPU ---
    // The GPU did the heavy lifting (Parallel Reduction).
    // We finish the coarse-grained sum on the CPU (since the result array is small now).
    GPU_CHECK(cudaMemcpy(h_partial_C, d_partial_C, blocksPerGrid * sizeof(float), cudaMemcpyDeviceToHost));

    double total_sum = 0.0;
    for(int i=0; i<blocksPerGrid; i++) {
        total_sum += h_partial_C[i];
    }

    // --- 6. RESULTS & ANALYSIS ---
    // Theoretical throughput calculation
    // 2 FLOPs per element (Multiply + Add) roughly, though reduction adds more.
    double gigaFlops = (2.0 * ARRAY_SIZE) / (milliseconds * 1e-3) / 1e9;

    printf("Execution Time: %.4f ms\n", milliseconds);
    printf("Estimated Throughput: %.2f GFLOPs\n", gigaFlops);
    printf("Total Dot Product: %.2f (Expected: %.2f)\n", total_sum, (double)ARRAY_SIZE * 2.0);

    // --- 7. CLEANUP ---
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_partial_C);
    free(h_A);
    free(h_B);
    free(h_partial_C);
    free(h_ref);
    
    printf("\n[CONCLUSION]\n");
    printf("The code demonstrated how fine-grained threads cooperate within a block\n");
    printf("using Shared Memory, and how the Grid of blocks enables transparent scalability\n");
    printf("across any number of Multiprocessors.\n");

    return 0;
}