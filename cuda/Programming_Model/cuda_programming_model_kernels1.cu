/**
 * @file thread_hierarchy_clusters.cu
 * @brief Advanced CUDA Programming: Thread Hierarchy, Clusters, and Blocks-as-Clusters.
 *
 * @details
 * --------------------------------------------------------------------------------------------
 * AUTHOR: 
 * DATE: 2023
 * TARGET ARCHITECTURE: NVIDIA Hopper (Compute Capability 9.0+) for Cluster features.
 * STANDARD: CUDA C++ (ISO C++17 compatible)
 * --------------------------------------------------------------------------------------------
 *
 * PURPOSE:
 * This file serves as a comprehensive masterclass on the CUDA Thread Hierarchy.
 * It covers the foundational concepts of Grids and Blocks, and extends into the 
 * cutting-edge "Thread Block Clusters" introduced in Compute Capability 9.0.
 *
 * TOPICS COVERED:
 * 1. Thread Hierarchy (Grids, Blocks, Threads).
 * 2. Index Calculation (Mapping 2D/3D indices to linear memory).
 * 3. Matrix Addition (Classic Example).
 * 4. Thread Block Clusters (Static Compile-time definition).
 * 5. Thread Block Clusters (Dynamic Runtime definition via cudaLaunchKernelEx).
 * 6. Blocks as Clusters (__block_size__ attribute).
 *
 * --------------------------------------------------------------------------------------------
 * THEORETICAL FOUNDATION:
 *
 * 1. THE GRID:
 *    The highest level of the hierarchy. A kernel launch creates a Grid. 
 *    The Grid is composed of Thread Blocks.
 *    
 * 2. THE THREAD BLOCK:
 *    A group of threads that execute on the same Streaming Multiprocessor (SM).
 *    - Shared Resources: They share L1 Cache/Shared Memory.
 *    - Synchronization: They can sync using __syncthreads().
 *    - Limit: Max 1024 threads per block (hardware limit).
 *
 * 3. THE CLUSTER (CC 9.0+):
 *    A group of Thread Blocks guaranteed to be scheduled on the same 
 *    GPU Processing Cluster (GPC).
 *    - Hierarchy: Grid -> Cluster -> Block -> Thread.
 *    - Benefit: Faster synchronization and Distributed Shared Memory (DSMEM) access 
 *      between blocks in the same cluster.
 *
 * 4. EXECUTION MODEL:
 *    CUDA is SIMT (Single Instruction, Multiple Threads). Threads execute the same code
 *    independently. The hardware scheduler maps these to physical cores.
 * --------------------------------------------------------------------------------------------
 */

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>

// CUDA 9.0+ includes for Cluster Groups (if available in your SDK)
// <cooperative_groups.h> would be used for cluster.sync(), 
// strictly following the prompt's focus on hierarchy definition over complex synchronization logic.

// ============================================================================================
// UTILITY: ERROR CHECKING
// ============================================================================================
// We use a macro to wrap CUDA API calls. This is industry standard for robust code.
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// Dimensions for our Matrix Addition Example
#define N 2048 

// ============================================================================================
// KERNEL 1: STANDARD THREAD HIERARCHY (Matrix Addition)
// ============================================================================================
/**
 * @brief Performs Matrix Addition C = A + B using standard Grid/Block hierarchy.
 * 
 * @param A Pointer to Matrix A (Linearized 1D array representing 2D N x N)
 * @param B Pointer to Matrix B
 * @param C Pointer to Output Matrix C
 * @param n Dimension of the matrix (n x n)
 * 
 * @note
 * LOGIC EXPLANATION:
 * 1. We launch a 2D grid of 2D blocks.
 * 2. blockIdx: Identifies the block within the grid (x, y).
 * 3. threadIdx: Identifies the thread within the block (x, y).
 * 4. blockDim: The size of the block (set by host).
 * 
 * GLOBAL INDEX CALCULATION:
 * To map a 2D position (row, col) to a 1D memory address:
 * int col = blockIdx.x * blockDim.x + threadIdx.x;
 * int row = blockIdx.y * blockDim.y + threadIdx.y;
 * int index = row * width + col;
 */
__global__ void MatAdd_Standard(const float* __restrict__ A, 
                                const float* __restrict__ B, 
                                float* __restrict__ C, 
                                int n)
{
    // Calculate global column index (x-dimension)
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    // Calculate global row index (y-dimension)
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    // Boundary Check: Ideally grid covers exactly N, but always safe to check
    if (col < n && row < n)
    {
        // Linearize 2D coordinate to 1D index
        int idx = row * n + col;
        
        // Perform addition
        // Memory Coalescing Note: Threads with adjacent threadIdx.x access adjacent memory.
        // This is optimal for global memory bandwidth.
        C[idx] = A[idx] + B[idx];
    }
}

// ============================================================================================
// KERNEL 2: THREAD BLOCK CLUSTERS (Compile-Time Definition)
// ============================================================================================
/**
 * @brief Matrix Addition using Static Cluster Dimensioning.
 * 
 * @note 
 * COMPUTE CAPABILITY 9.0+ FEATURE.
 * The attribute __cluster_dims__(X, Y, Z) tells the compiler/scheduler to 
 * group blocks into clusters.
 * 
 * Here, __cluster_dims__(2, 1, 1) means:
 * Every 2 blocks in the X dimension form a single Cluster.
 * 
 * Why? 
 * On Hopper architecture, these 2 blocks are scheduled on the same GPC.
 * They can share data via Distributed Shared Memory (DSMEM) and sync faster.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__global__ void __cluster_dims__(2, 1, 1) MatAdd_Cluster_Static(const float* __restrict__ A, 
                                                                const float* __restrict__ B, 
                                                                float* __restrict__ C, 
                                                                int n)
{
    // The logic for calculating global index REMAINS THE SAME as standard kernels.
    // The 'Cluster' concept is a scheduling hint and capability enabler, 
    // it does not change the calculation of global ID relative to the Grid.
    
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < n)
    {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}
#endif

// ============================================================================================
// KERNEL 3: DYNAMIC CLUSTERS (Runtime Definition)
// ============================================================================================
/**
 * @brief Standard Kernel to be launched with dynamic cluster attributes.
 * 
 * @note
 * This kernel has NO __cluster_dims__ attribute.
 * We will define the cluster size in the HOST code using cudaLaunchKernelEx.
 */
__global__ void MatAdd_Cluster_Dynamic(const float* __restrict__ A, 
                                       const float* __restrict__ B, 
                                       float* __restrict__ C, 
                                       int n)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (col < n && row < n)
    {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}

// ============================================================================================
// KERNEL 4: BLOCKS AS CLUSTERS (Advanced Syntax)
// ============================================================================================
/**
 * @brief Implicit Cluster Launch using __block_size__.
 * 
 * @details
 * SYNTAX: __block_size__( (Bx, By, Bz), (Cx, Cy, Cz) )
 * 
 * Tuple 1 (Bx, By, Bz): Standard Block Dimensions (threadIdx range).
 * Tuple 2 (Cx, Cy, Cz): Cluster Dimensions (how many blocks per cluster).
 * 
 * BEHAVIOR CHANGE:
 * When this is used, the <<<GridDim, ...>>> launch configuration interprets 
 * the first argument (GridDim) as the number of CLUSTERS, not Blocks.
 * This is a significant semantic shift in the Programming Model.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
// Example: 16x16 threads per block, grouped into 2x1x1 clusters
__block_size__((16, 16, 1), (2, 1, 1))
__global__ void MatAdd_BlocksAsClusters(const float* __restrict__ A, 
                                        const float* __restrict__ B, 
                                        float* __restrict__ C, 
                                        int n)
{
    // CAUTION: Index calculation strategies might differ depending on how the user
    // interprets the grid. Standard blockIdx/threadIdx built-ins still function.
    
    // blockDim is fixed at compile time (16, 16, 1) by the attribute.
    int col = blockIdx.x * 16 + threadIdx.x;
    int row = blockIdx.y * 16 + threadIdx.y;

    if (col < n && row < n)
    {
        int idx = row * n + col;
        C[idx] = A[idx] + B[idx];
    }
}
#endif

// ============================================================================================
// MAIN HOST CODE
// ============================================================================================
int main()
{
    // ------------------------------------------------------------------------
    // 1. DATA PREPARATION
    // ------------------------------------------------------------------------
    size_t bytes = N * N * sizeof(float);
    
    // Use Unified Memory for tutorial cleanliness (accessible by CPU and GPU)
    // In high-performance scenarios, use explicit Device malloc and pinned Host memory.
    float *h_A, *h_B, *h_C;
    gpuErrchk(cudaMallocManaged(&h_A, bytes));
    gpuErrchk(cudaMallocManaged(&h_B, bytes));
    gpuErrchk(cudaMallocManaged(&h_C, bytes));

    // Initialize Matrices
    for(int i = 0; i < N * N; i++) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // ------------------------------------------------------------------------
    // 2. CONFIGURATION: STANDARD THREAD HIERARCHY
    // ------------------------------------------------------------------------
    // We choose a block size of 16x16 = 256 threads.
    // This is a common occupancy optimization choice (multiple of 32/warp size).
    dim3 threadsPerBlock(16, 16);
    
    // Calculate Grid Size.
    // We need enough blocks to cover N elements in X and Y.
    // Formula: (N + blockDim - 1) / blockDim ensures ceiling division.
    dim3 numBlocks((N + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (N + threadsPerBlock.y - 1) / threadsPerBlock.y);

    printf("Matrix Size: %dx%d\n", N, N);
    printf("Grid Config: (%d, %d)\n", numBlocks.x, numBlocks.y);
    printf("Block Config: (%d, %d)\n", threadsPerBlock.x, threadsPerBlock.y);

    // ------------------------------------------------------------------------
    // 3. EXECUTION: STANDARD LAUNCH
    // ------------------------------------------------------------------------
    printf("Launching Standard Kernel...\n");
    MatAdd_Standard<<<numBlocks, threadsPerBlock>>>(h_A, h_B, h_C, N);
    gpuErrchk(cudaDeviceSynchronize()); // Wait for GPU to finish

    // ------------------------------------------------------------------------
    // 4. EXECUTION: STATIC CLUSTER LAUNCH (Requires CC 9.0+)
    // ------------------------------------------------------------------------
    // For the static kernel, we specified __cluster_dims__(2, 1, 1).
    // The Grid dimensions must be divisible by the cluster size in the respective dimension.
    // Our numBlocks.x is (2048/16) = 128. 128 is divisible by 2. This is valid.
    
    int deviceId;
    cudaGetDevice(&deviceId);
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, deviceId);

    if (props.major >= 9) 
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        printf("Launching Static Cluster Kernel (CC 9.0+)...\n");
        MatAdd_Cluster_Static<<<numBlocks, threadsPerBlock>>>(h_A, h_B, h_C, N);
        gpuErrchk(cudaDeviceSynchronize());
#endif
    }
    else
    {
        printf("Skipping Cluster kernels (Requires Compute Capability 9.0+)\n");
    }

    // ------------------------------------------------------------------------
    // 5. EXECUTION: DYNAMIC CLUSTER LAUNCH (cudaLaunchKernelEx)
    // ------------------------------------------------------------------------
    // This method allows setting cluster size at runtime without recompiling.
    // It uses the Extensible Launch API.
    
    if (props.major >= 9)
    {
        printf("Launching Dynamic Cluster Kernel via cudaLaunchKernelEx...\n");

        cudaLaunchConfig_t config = {0};
        
        // Set Grid and Block dimensions in the config struct
        config.gridDim = numBlocks;
        config.blockDim = threadsPerBlock;
        config.stream = 0; // Default stream
        config.dynamicSmemBytes = 0;

        // Define Attributes
        cudaLaunchAttribute attribute[1];
        attribute[0].id = cudaLaunchAttributeClusterDimension;
        attribute[0].val.clusterDim.x = 2; // Cluster size X = 2 blocks
        attribute[0].val.clusterDim.y = 1;
        attribute[0].val.clusterDim.z = 1;

        config.attrs = attribute;
        config.numAttrs = 1;

        // Launch
        gpuErrchk(cudaLaunchKernelEx(&config, MatAdd_Cluster_Dynamic, h_A, h_B, h_C, N));
        gpuErrchk(cudaDeviceSynchronize());
    }

    // ------------------------------------------------------------------------
    // 6. EXECUTION: BLOCKS AS CLUSTERS
    // ------------------------------------------------------------------------
    // When using __block_size__, the launch config changes.
    // The kernel attribute __block_size__((16,16,1), (2,1,1)) handles block dim and cluster dim.
    // The launch configuration <<< >>> now expects Number of Clusters, not Blocks.
    
    if (props.major >= 9)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        printf("Launching Blocks-as-Clusters Kernel...\n");
        
        // Grid Calculation for CLUSTERS:
        // Original Blocks X: 128. Cluster X size: 2. -> 64 Clusters.
        // Original Blocks Y: 128. Cluster Y size: 1. -> 128 Clusters.
        dim3 numClusters(numBlocks.x / 2, numBlocks.y / 1);
        
        // NOTE: Second argument in dim3() for block dimension should be (1,1,1) 
        // or derived strictly as per documentation when using __block_size__.
        // However, the documentation states the compiler recognizes the first arg as clusters.
        // We pass a placeholder 1 for threads, as it is overridden by the attribute.
        
        MatAdd_BlocksAsClusters<<<numClusters, dim3(1,1,1)>>>(h_A, h_B, h_C, N);
        gpuErrchk(cudaDeviceSynchronize());
#endif
    }

    // ------------------------------------------------------------------------
    // 7. VERIFICATION & CLEANUP
    // ------------------------------------------------------------------------
    // Verify one element
    // Result should be 3.0 (1.0 + 2.0) * Number of launches (since we overwrote C each time, actually just 3.0)
    // Actually, in this code we didn't clear C between runs, but we assigned C = A + B, 
    // not C += A + B, so the result is always 3.0f.
    
    if (h_C[0] != 3.0f) {
        printf("Verification FAILED. Result: %f\n", h_C[0]);
    } else {
        printf("Verification SUCCESS. Result: %f\n", h_C[0]);
    }

    // Free Memory
    gpuErrchk(cudaFree(h_A));
    gpuErrchk(cudaFree(h_B));
    gpuErrchk(cudaFree(h_C));

    printf("Done.\n");
    return 0;
}

/*
 * COMPILATION INSTRUCTIONS:
 * To compile this for Hopper Architecture (supporting Clusters):
 * 
 * nvcc -arch=sm_90 -o thread_hierarchy thread_hierarchy_clusters.cu
 * 
 * For older architectures (will skip cluster logic based on macros):
 * nvcc -o thread_hierarchy thread_hierarchy_clusters.cu
 */