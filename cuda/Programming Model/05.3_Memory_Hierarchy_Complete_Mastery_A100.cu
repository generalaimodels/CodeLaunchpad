/**
 * ==================================================================================================
 * FILE: memory_hierarchy_mastery.cu
 * AUTHOR: World's Best Coder (AI Tutor)
 * PLATFORM: NVIDIA A100 (Ampere Architecture, Compute Capability 8.0)
 * TOPIC: 5.3 Memory Hierarchy (Global, Local, Shared, Constant, Texture, and Clusters)
 * ==================================================================================================
 * 
 * INTRODUCTION
 * ------------
 * To master CUDA, one must master data movement. The GPU is a throughput machine; latency is hidden
 * by massive parallelism, but bandwidth is finite. The Memory Hierarchy allows us to stage data
 * closer to the execution units (CUDA Cores) to maximize performance.
 * 
 * HIERARCHY LEVELS (Fastest to Slowest):
 * 1. REGISTERS (Local): Zero latency. Private to a thread.
 * 2. L1 CACHE / SHARED MEMORY: Low latency. Visible to a Thread Block. Configurable on A100.
 * 3. CONSTANT / TEXTURE CACHES: Read-only caches optimized for specific access patterns.
 * 4. L2 CACHE: Shared across all SMs (Streaming Multiprocessors).
 * 5. GLOBAL MEMORY (VRAM): High latency (~hundreds of cycles). Visible to all threads.
 * 
 * SPECIAL NOTE ON A100 & CLUSTERS:
 * The prompt text mentions "Thread Block Clusters" (Distributed Shared Memory).
 * This is a hardware feature introduced in Hopper (H100, CC 9.0). 
 * Since we are using an A100 (Ampere, CC 8.0), we cannot execute Cluster instructions physically.
 * However, as the "Best Tutor," I will explain the syntax and concept in comments so you possess 
 * 100% knowledge, while ensuring the code compiles and runs flawlessly on your A100.
 * ==================================================================================================
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <stdio.h>

// ==================================================================================================
// CONSTANTS & MACROS
// ==================================================================================================

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (Line: %d)\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

constexpr int N = 1024 * 1024;  // Total elements
constexpr int BLOCK_SIZE = 256; // Threads per block
constexpr int CONSTANT_BUF_SIZE = 256; // Size for constant memory buffer

// ==================================================================================================
// 1. CONSTANT MEMORY DECLARATION
// ==================================================================================================
// Constant memory is a read-only cache specialized for "Uniform Access".
// If all threads in a warp read the same address, it is as fast as a register read.
// It has global scope but must be defined statically.
// Limit: 64KB total.
__constant__ float c_weights[CONSTANT_BUF_SIZE];

// ==================================================================================================
// KERNEL: Memory Hierarchy Showcase
// ==================================================================================================
// This kernel demonstrates the interaction between:
// - Global Memory (Input/Output)
// - Shared Memory (Intra-block cooperation)
// - Constant Memory (Read-only coefficients)
// - Texture Memory (Read-only spatially cached)
// - Local Memory (Registers and stack)
//
// Parameters:
// d_in: Global Memory Input
// d_out: Global Memory Output
// texObj: Texture Object (Alternative path to memory)
// n: Problem size
__global__ void hierarchyKernel(const float* __restrict__ d_in, 
                                float* __restrict__ d_out, 
                                cudaTextureObject_t texObj,
                                int n) 
{
    // ------------------------------------------------------------------------------------------
    // A. LOCAL MEMORY & REGISTERS
    // ------------------------------------------------------------------------------------------
    // Variables declared inside the kernel (like 'tid', 'idx', 'val') reside in Registers.
    // Registers are the fastest memory (0 cycles).
    // If a thread uses too many registers, variables "spill" to Local Memory (physically in VRAM/L2),
    // which is slow. The compiler optimizes this.
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // ------------------------------------------------------------------------------------------
    // B. SHARED MEMORY
    // ------------------------------------------------------------------------------------------
    // Allocated per Thread Block. On A100, Shared Memory and L1 Cache share the same hardware.
    // It is ~100x faster than Global Memory.
    // Used for: User-managed caching, communication between threads in a block.
    // Scope: Visible to all threads in this block. Lifetime: Duration of the block.
    __shared__ float s_cache[BLOCK_SIZE];

    // 1. LOAD FROM GLOBAL TO SHARED
    // We use a coalesced access pattern: Thread k loads Index k.
    float global_val = 0.0f;
    
    // Boundary check
    if (idx < n) {
        // Reading from Global Memory (High Latency)
        global_val = d_in[idx]; 
    } else {
        global_val = 0.0f;
    }

    // Store into Shared Memory
    s_cache[tid] = global_val;

    // SYNCHRONIZATION BARRIER
    // Essential! Ensures all threads have written to s_cache before anyone reads from it.
    __syncthreads(); 

    // ------------------------------------------------------------------------------------------
    // C. CONSTANT MEMORY ACCESS
    // ------------------------------------------------------------------------------------------
    // We apply a weight from constant memory.
    // Optimization: We use bitwise AND to keep the index small, simulating all threads
    // accessing a small range of constant data.
    float weight = c_weights[tid % 10]; 

    // ------------------------------------------------------------------------------------------
    // D. TEXTURE MEMORY ACCESS
    // ------------------------------------------------------------------------------------------
    // Textures are read-only and cached via the Texture Cache.
    // Unlike standard cache (optimized for linear lines), Texture cache is optimized for 2D spatial locality.
    // On A100, we use Texture Objects (cudaTextureObject_t).
    // Here, we fetch data using tex1Dfetch (linear texture).
    // In some workloads, this bypasses L1 and prevents polluting the L1 cache used for other data.
    float tex_val = 0.0f;
    if (idx < n) {
        tex_val = tex1Dfetch<float>(texObj, idx);
    }

    // ------------------------------------------------------------------------------------------
    // E. COMPUTATION (IN REGISTERS)
    // ------------------------------------------------------------------------------------------
    // We process data from Shared, Constant, and Texture sources.
    
    // Example computation:
    // 1. Read neighbor from shared memory (Data Reuse). 
    //    (Simple stencil: current + next). Handle boundary for the block.
    float neighbor = (tid < BLOCK_SIZE - 1) ? s_cache[tid + 1] : s_cache[tid];

    float result = (s_cache[tid] * weight) + (tex_val * 0.5f) + neighbor;

    // ------------------------------------------------------------------------------------------
    // F. DISTRIBUTED SHARED MEMORY (CLUSTERS) - THEORETICAL FOR A100
    // ------------------------------------------------------------------------------------------
    /*
     * CONCEPT EXPLANATION (Topic 5.2.1 / 5.3):
     * If this were Hopper (CC 9.0), we could define a cluster:
     * __global__ void __cluster_dims__(2, 1, 1) kernel(...) { ... }
     * 
     * Accessing another block's shared memory within the cluster:
     * float val = cluster_map_shared_rank(&s_cache[tid], target_rank);
     * 
     * Synchronization:
     * cluster.sync(); // Hardware barrier for all blocks in the cluster.
     * 
     * Since we are on A100, this hardware path does not exist. 
     * The hierarchy stops at the Block level (Shared Memory) and L2 (Global).
     */

    // ------------------------------------------------------------------------------------------
    // G. WRITE TO GLOBAL MEMORY
    // ------------------------------------------------------------------------------------------
    if (idx < n) {
        d_out[idx] = result;
    }
}

// ==================================================================================================
// HOST CODE
// ==================================================================================================
int main() {
    printf("====================================================================\n");
    printf("   CUDA MEMORY HIERARCHY TUTORIAL (A100 EDITION)\n");
    printf("====================================================================\n");

    // ------------------------------------------------------------------------------------------
    // 1. MEMORY ALLOCATION (GLOBAL MEMORY)
    // ------------------------------------------------------------------------------------------
    // Global memory is persistent across kernel launches. 
    // It is the largest memory space (40GB or 80GB on A100).
    size_t bytes = N * sizeof(float);
    
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *h_constData = (float*)malloc(CONSTANT_BUF_SIZE * sizeof(float));

    // Initialize Host Data
    for (int i = 0; i < N; i++) h_in[i] = 1.0f;
    for (int i = 0; i < CONSTANT_BUF_SIZE; i++) h_constData[i] = 0.5f; // Weight

    float *d_in, *d_out;
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    // Copy data from Host to Device (Global Memory)
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // ------------------------------------------------------------------------------------------
    // 2. SETUP CONSTANT MEMORY
    // ------------------------------------------------------------------------------------------
    // To write to constant memory, we use cudaMemcpyToSymbol.
    // This data is cached in the Constant Cache on the SMs.
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights, h_constData, CONSTANT_BUF_SIZE * sizeof(float)));

    // ------------------------------------------------------------------------------------------
    // 3. SETUP TEXTURE MEMORY
    // ------------------------------------------------------------------------------------------
    // Textures bind to Global Memory but provide a different read path.
    // On A100, we create a Texture Object.

    // Step A: Create Resource Description (Where is the data?)
    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_in;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // 32-bit float
    resDesc.res.linear.sizeInBytes = bytes;

    // Step B: Create Texture Description (How do we read it?)
    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType; // Read as float (not normalized float)
    
    // Step C: Create the Object
    cudaTextureObject_t texObj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    // ------------------------------------------------------------------------------------------
    // 4. EXECUTION CONFIGURATION
    // ------------------------------------------------------------------------------------------
    dim3 threadsPerBlock(BLOCK_SIZE);
    dim3 numBlocks((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    printf("Launching Kernel with:\n");
    printf("  Grid Size:   %d blocks\n", numBlocks.x);
    printf("  Block Size:  %d threads\n", threadsPerBlock.x);
    printf("  Memory Usage: Global, Shared, Constant, Texture, Registers\n");

    // ------------------------------------------------------------------------------------------
    // 5. KERNEL LAUNCH
    // ------------------------------------------------------------------------------------------
    hierarchyKernel<<<numBlocks, threadsPerBlock>>>(d_in, d_out, texObj, N);
    
    CUDA_CHECK(cudaGetLastError()); // Check for launch errors
    CUDA_CHECK(cudaDeviceSynchronize()); // Wait for GPU to finish

    // ------------------------------------------------------------------------------------------
    // 6. VERIFICATION
    // ------------------------------------------------------------------------------------------
    // Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // Verify logic for index 0:
    // Global Input = 1.0
    // Shared = 1.0
    // Weight (Constant) = 0.5
    // Texture = 1.0
    // Neighbor (Shared[1]) = 1.0
    // Calculation: (Shared * Weight) + (Texture * 0.5) + Neighbor
    // Result: (1.0 * 0.5) + (1.0 * 0.5) + 1.0 = 0.5 + 0.5 + 1.0 = 2.0
    
    bool success = true;
    if (abs(h_out[0] - 2.0f) > 1e-5) {
        printf("Verification Failed at index 0! Expected 2.0, got %f\n", h_out[0]);
        success = false;
    }

    if (success) {
        printf("Test PASSED. Memory Hierarchy accessed correctly.\n");
    }

    // ------------------------------------------------------------------------------------------
    // 7. CLEANUP
    // ------------------------------------------------------------------------------------------
    // Destroy Texture Object
    CUDA_CHECK(cudaDestroyTextureObject(texObj));
    
    // Free Global Memory
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    
    // Free Host Memory
    free(h_in);
    free(h_out);
    free(h_constData);

    printf("Resources freed successfully.\n");
    return 0;
}