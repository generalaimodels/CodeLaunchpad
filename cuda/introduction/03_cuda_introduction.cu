/**
 * @file 03_cuda_introduction.cu
 * @brief Deep Dive into CUDA Introduction, GPU Benefits, and Scalable Programming Models.
 * @author Master Architect (IQ > 300)
 * @date 2023-10-27
 *
 * =================================================================================================
 *                                     TOPIC: CUDA INTRODUCTION
 * =================================================================================================
 *
 * This file serves as a definitive guide and implementation of the concepts outlined in:
 * 3.1. The Benefits of Using GPUs
 * 3.2. CUDA: A General-Purpose Parallel Computing Platform
 * 3.3. A Scalable Programming Model
 *
 * THEORETICAL FOUNDATION (The Master's Perspective):
 * --------------------------------------------------
 *
 * 1. THE TRANSISTOR ECONOMY (Section 3.1):
 *    - CPU (Host): Designed for LATENCY. Large caches (L1/L2/L3) and complex control logic 
 *      (Branch Prediction) consume most transistors. Great for sequential tasks.
 *    - GPU (Device): Designed for THROUGHPUT. Most transistors are dedicated to ALUs (Arithmetic 
 *      Logic Units). It hides latency not with cache, but with massive parallelism (switching 
 *      between active warps).
 *
 * 2. THE PLATFORM (Section 3.2):
 *    - CUDA allows C++ developers to treat the GPU as a coprocessor. It is not just for graphics;
 *      it is for General-Purpose GPU (GPGPU) computing.
 *
 * 3. SCALABILITY & ABSTRACTION (Section 3.3):
 *    - The "Secret Sauce" of CUDA is its transparent scalability.
 *    - CORE ABSTRACTIONS:
 *      A. Hierarchy of Thread Groups (Grid -> Block -> Thread).
 *      B. Shared Memories (Fast, user-managed on-chip cache).
 *      C. Barrier Synchronization (__syncthreads()).
 *    - Why Scalable? A compiled CUDA kernel can run on a GPU with 2 SMs or 100 SMs without
 *      changing a single line of code. The hardware scheduler assigns Blocks to SMs.
 *
 * =================================================================================================
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

/*
 * =================================================================================================
 *                                  ERROR HANDLING (Best Practices)
 * =================================================================================================
 * A superior coder never ignores return codes. CUDA APIs are asynchronous and error-prone
 * if configurations are mismatched.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (Code: %d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/*
 * =================================================================================================
 *                               KERNEL 1: DEMONSTRATING THROUGHPUT
 *                               (Topic 3.1 & 3.2)
 * =================================================================================================
 * 
 * CONCEPT: High Arithmetic Intensity.
 * GPUs excel when the ratio of Math Operations to Memory Accesses is high.
 * This kernel performs heavy floating-point math to utilize the massive number of ALUs
 * mentioned in Section 3.1 (Figure 1 logic).
 *
 * @param data   Pointer to the array in Global Memory.
 * @param N      Total number of elements.
 */
__global__ void heavy_compute_kernel(float* __restrict__ data, int N) {
    // 1. Thread Identification
    // Unique global index for the thread.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // 2. Boundary Check
    if (idx < N) {
        float val = data[idx];
        
        // Heavy Computation simulation (Trigonometry + Roots)
        // On a CPU, this sequential loop would be slow.
        // On a GPU, thousands of threads execute this in parallel.
        for (int k = 0; k < 50; k++) {
            val = sinf(val) * cosf(val) + sqrtf(fabsf(val));
        }
        
        data[idx] = val;
    }
}

/*
 * =================================================================================================
 *                        KERNEL 2: SCALABLE PROGRAMMING MODEL
 *                                   (Topic 3.3)
 * =================================================================================================
 * 
 * CONCEPT: The Three Abstractions.
 * 1. Thread Hierarchy: We calculate `tid` (local) and `idx` (global).
 * 2. Shared Memory: We use `__shared__ temp[]` to allow threads in a block to cooperate.
 * 3. Synchronization: We use `__syncthreads()` to prevent race conditions.
 *
 * SCALABILITY EXPLANATION:
 * This kernel reduces an array (sums it up).
 * The logic operates on a "Block" level. The GPU hardware scheduler can dispatch these blocks
 * to any available Multiprocessor (SM). If a GPU has more SMs, more blocks run concurrently.
 * The code does not change.
 */
__global__ void scalable_reduction_kernel(const float* __restrict__ input, 
                                          float* __restrict__ output, 
                                          int N) {
    // Dynamic Shared Memory: Allocated at kernel launch.
    // This is the "User-Managed Cache" allowing fine-grained cooperation.
    extern __shared__ float s_data[];

    // Global Index (Coarse-grained)
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;

    // Load input into shared memory (Cooperative Load)
    // If within bounds, load data; otherwise pad with 0 (identity for sum).
    s_data[tid] = (i < N) ? input[i] : 0.0f;

    // ABSTRACTION 3: BARRIER SYNCHRONIZATION
    // Ensure all threads in this block have loaded their data before computing.
    __syncthreads();

    // Reduction in Shared Memory
    // This is "Fine-grained thread parallelism" solving a sub-problem.
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        // Barrier needed at every step to ensure data consistency
        __syncthreads();
    }

    // Write result for this block to global memory
    // Only one thread per block (Thread 0) writes the result.
    if (tid == 0) {
        output[blockIdx.x] = s_data[0];
    }
}

/*
 * =================================================================================================
 *                                          HOST CODE
 * =================================================================================================
 */
int main() {
    printf("=== CUDA ARCHITECTURE & SCALABILITY DEMO ===\n");
    printf("Demonstrating concepts from Section 3.1, 3.2, and 3.3\n\n");

    // --- PART 1: DEVICE QUERY (Understanding the Scalable Hardware) ---
    int devId = 0;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, devId));

    printf("[Hardware Info]\n");
    printf("  GPU: %s\n", prop.name);
    printf("  Multiprocessors (SMs): %d\n", prop.multiProcessorCount);
    printf("  Explanation: The CUDA Runtime will distribute Blocks across these %d SMs.\n\n", 
           prop.multiProcessorCount);


    // --- PART 2: DATA PREPARATION ---
    int N = 1 << 20; // 1 Million elements
    size_t bytes = N * sizeof(float);

    // Allocate Host Memory
    float *h_in = (float*)malloc(bytes);
    float *h_out_ref = (float*)malloc(bytes);

    // Initialize Data
    for(int i = 0; i < N; i++) {
        h_in[i] = 1.0f; // Simple value for verification
    }

    // Allocate Device Memory
    float *d_in, *d_out, *d_partial_sums;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    // Copy Data Host -> Device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));


    // --- PART 3: EXECUTION - HEAVY COMPUTE (Section 3.1 Demo) ---
    printf("[Section 3.1 Demo] Launching Heavy Compute Kernel...\n");
    printf("  Purpose: Demonstrate GPU throughput capabilities via massive parallelism.\n");

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // Launch Kernel
    heavy_compute_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_in, N);
    CUDA_CHECK(cudaDeviceSynchronize());
    printf("  Status: Compute Complete.\n\n");


    // --- PART 4: EXECUTION - SCALABLE REDUCTION (Section 3.3 Demo) ---
    printf("[Section 3.3 Demo] Launching Scalable Reduction Kernel...\n");
    printf("  Purpose: Demonstrate Thread Groups, Shared Memory, and Synchronization.\n");

    // Calculate Grid size for Reduction
    // We launch enough blocks to cover the array.
    // Each block solves a "Sub-problem" (summing 256 elements).
    int numBlocks = (N + threadsPerBlock - 1) / threadsPerBlock;
    
    // Allocate memory for partial sums (one result per block)
    CUDA_CHECK(cudaMalloc((void**)&d_partial_sums, numBlocks * sizeof(float)));

    // Dynamic Shared Memory size: floats * threads per block
    size_t sharedMemSize = threadsPerBlock * sizeof(float);

    // LAUNCH CONFIGURATION:
    // This logic enables the scalability. We define the problem into 'numBlocks' chunks.
    // The GPU executes these chunks based on available hardware resources.
    scalable_reduction_kernel<<<numBlocks, threadsPerBlock, sharedMemSize>>>(d_in, d_partial_sums, N);
    
    CUDA_CHECK(cudaPeekAtLastError());
    CUDA_CHECK(cudaDeviceSynchronize());


    // --- PART 5: VERIFICATION & CLEANUP ---
    
    // Retrieve partial sums to Host
    float* h_partial_sums = (float*)malloc(numBlocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, numBlocks * sizeof(float), cudaMemcpyDeviceToHost));

    // Finish reduction on CPU (since the data is now small enough)
    float total_sum = 0.0f;
    for(int i = 0; i < numBlocks; i++) {
        total_sum += h_partial_sums[i];
    }

    // Note: We ran heavy_compute on d_in first, which modified values.
    // So the sum won't be exactly N * 1.0, but the reduction logic holds.
    printf("  Total Blocks executed: %d\n", numBlocks);
    printf("  Final Reduced Sum: %.2f\n", total_sum);
    printf("  Logic: The %d blocks ran cooperatively internally, but independently externally.\n", numBlocks);

    // Free Memory
    cudaFree(d_in);
    cudaFree(d_out);
    cudaFree(d_partial_sums);
    free(h_in);
    free(h_out_ref);
    free(h_partial_sums);

    printf("=== EXECUTION SUCCESSFUL ===\n");

    return 0;
}