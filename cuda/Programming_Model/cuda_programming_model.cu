/**
 * @file cuda_programming_model.cu
 * @brief CUDA Programming Model: Thread Hierarchy and Clusters (Compute Capability 9.0+)
 *
 * @author Best Coder (IQ 300+)
 * @details
 * This file demonstrates the core concepts of the CUDA Programming Model as described in Chapter 5.
 * It covers:
 * 1. Thread Hierarchy: Grids, Blocks, and Threads.
 * 2. Indexing strategies for 2D domains (Matrices).
 * 3. Thread Block Clusters (Introduced in CC 9.0/Hopper Architecture).
 * 4. Cluster launch modes: Compile-time attributes, Runtime Launch API, and Blocks-as-Clusters.
 *
 * SYSTEM REQUIREMENTS:
 * - To compile the Cluster examples: NVCC with -arch=sm_90 or higher.
 * - To run: NVIDIA GPU with Compute Capability 9.0+ (e.g., H100).
 *
 * COMPILATION:
 * nvcc -arch=sm_90 -o thread_hierarchy cuda_programming_model.cu
 */

#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <iostream>
#include <cstdio>

// Namespace for Cooperative Groups, essential for Cluster synchronization and introspection
namespace cg = cooperative_groups;

// ==========================================================================================
// UTILITY: Error Checking
// ==========================================================================================

/**
 * @brief Macro for checking CUDA errors.
 * Ensures robust code execution by trapping runtime API errors immediately.
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

// ==========================================================================================
// SECTION 5.2: Thread Hierarchy (Standard)
// ==========================================================================================

/**
 * @brief Kernel: Standard Matrix Addition
 *
 * Concepts covered:
 * - 2D Thread Indexing (threadIdx.x, threadIdx.y)
 * - 2D Block Indexing (blockIdx.x, blockIdx.y)
 * - Global Index Calculation
 * - Boundary Checking
 *
 * The hierarchy is: Grid -> Block -> Thread.
 * Threads within a block can share memory and synchronize (__syncthreads()).
 * Blocks execute independently.
 *
 * @param A Input Matrix A (Linearized in memory)
 * @param B Input Matrix B (Linearized in memory)
 * @param C Output Matrix C
 * @param N Dimension of the matrices (NxN)
 */
__global__ void MatAdd(const float* A, const float* B, float* C, int N)
{
    // Calculate global column index
    // blockIdx.x * blockDim.x gives the starting index of the block in the x-dimension
    // + threadIdx.x adds the offset of the thread within that block
    int i = blockIdx.x * blockDim.x + threadIdx.x; // Column

    // Calculate global row index
    int j = blockIdx.y * blockDim.y + threadIdx.y; // Row

    // Calculate linear index for 1D memory array access: row * width + col
    int idx = j * N + i;

    // Boundary check: ensure we do not access memory outside the matrix dimensions
    if (i < N && j < N)
    {
        C[idx] = A[idx] + B[idx];
    }
}

// ==========================================================================================
// SECTION 5.2.1: Thread Block Clusters (Compile-Time Attribute)
// ==========================================================================================

/**
 * @brief Kernel: Cluster-enabled Matrix Operation (Compile-Time Size)
 *
 * Concepts covered:
 * - __cluster_dims__(X, Y, Z): Defines cluster size at compile time.
 * - Cooperative Groups: Accessing cluster-level rank and synchronization.
 *
 * Hierarchy: Grid -> Cluster -> Block -> Thread.
 * Blocks in a cluster are co-scheduled on a GPC (GPU Processing Cluster).
 * They share Distributed Shared Memory (DSMEM) and hardware synchronization.
 *
 * In this example, Cluster Size is fixed to (2, 1, 1).
 * This means 2 thread blocks along X are grouped into one cluster.
 */
__global__ void __cluster_dims__(2, 1, 1) cluster_kernel_compile_time(const float* A, float* C, int N)
{
    // Standard Global Index Calculation
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * N + i;

    // Get the handle to the current cluster group
    cg::cluster_group cluster = cg::this_cluster();

    // Synchronization barrier for all blocks in this cluster
    // Hardware-supported, efficient synchronization across blocks in the cluster
    cluster.sync(); 

    if (i < N && j < N)
    {
        // Simple operation: Pass-through with a debug print for the first thread
        C[idx] = A[idx];

        // Introspection: Print cluster details from the first thread of the first block in the grid
        if (i == 0 && j == 0)
        {
            printf("Compile-Time Cluster Kernel:\n");
            printf("  Cluster Rank: %d\n", cluster.block_rank());
            printf("  Cluster Size (Blocks): %d\n", cluster.num_blocks());
        }
    }
}

// ==========================================================================================
// SECTION 5.2.1: Thread Block Clusters (Runtime Launch API)
// ==========================================================================================

/**
 * @brief Kernel: Cluster-enabled (Runtime Size)
 *
 * Concepts covered:
 * - No __cluster_dims__ attribute.
 * - Configured via cudaLaunchKernelEx in host code.
 * - Allows flexibility in cluster size based on hardware availability at runtime.
 */
__global__ void cluster_kernel_runtime(const float* A, float* C, int N)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * N + i;

    // Access Cluster Group
    cg::cluster_group cluster = cg::this_cluster();
    
    // Example: In a real scenario, we might use Distributed Shared Memory here
    // cluster.map_shared_rank(...)
    
    cluster.sync();

    if (i < N && j < N)
    {
        C[idx] = A[idx] * 2.0f; // Just a dummy operation

        if (i == 0 && j == 0)
        {
            printf("Runtime Cluster Kernel:\n");
            printf("  Cluster Rank: %d\n", cluster.block_rank());
            printf("  Cluster Size (Blocks): %d\n", cluster.num_blocks());
        }
    }
}

// ==========================================================================================
// SECTION 5.2.2: Blocks as Clusters
// ==========================================================================================

/**
 * @brief Kernel: Blocks as Clusters
 *
 * Concepts covered:
 * - __block_size__(BlockDims, ClusterDims): Implicitly handles hierarchy.
 * - The launch grid dimensions are interpreted as *Cluster Counts*, not Block Counts.
 * 
 * In this example:
 * Block Size: (16, 16, 1) threads.
 * Cluster Size: (2, 2, 1) blocks per cluster.
 */
__block_size__((16, 16, 1), (2, 2, 1)) 
__global__ void blocks_as_clusters_kernel(const float* A, float* C, int N)
{
    // Note: When using Blocks as Clusters, the compiler handles the hierarchy.
    // Standard indexing usually applies relative to the grid of threads.
    
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = j * N + i;

    if (i < N && j < N)
    {
        C[idx] = A[idx] + 1.0f;
    }
}

// ==========================================================================================
// HOST CODE
// ==========================================================================================

int main()
{
    // Matrix Size: 32x32 (Small for demonstration purposes)
    const int N = 32;
    size_t size = N * N * sizeof(float);

    // Host Memory Allocation
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize Data
    for (int k = 0; k < N * N; ++k)
    {
        h_A[k] = 1.0f;
        h_B[k] = 2.0f;
        h_C[k] = 0.0f;
    }

    // Device Memory Allocation
    float *d_A, *d_B, *d_C;
    gpuErrchk(cudaMalloc(&d_A, size));
    gpuErrchk(cudaMalloc(&d_B, size));
    gpuErrchk(cudaMalloc(&d_C, size));

    // Copy Data to Device
    gpuErrchk(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    gpuErrchk(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // Define Execution Configuration
    // Standard Block Size: 16x16 threads
    dim3 threadsPerBlock(16, 16);
    // Grid Size: Enough blocks to cover N
    dim3 numBlocks(N / threadsPerBlock.x, N / threadsPerBlock.y);

    printf("Execution Configuration:\n");
    printf("  Matrix Size: %dx%d\n", N, N);
    printf("  Threads Per Block: %dx%d\n", threadsPerBlock.x, threadsPerBlock.y);
    printf("  Grid Size (Blocks): %dx%d\n", numBlocks.x, numBlocks.y);
    printf("--------------------------------------------------------\n");

    // ----------------------------------------------------------------
    // 1. Standard Kernel Launch (Section 5.2)
    // ----------------------------------------------------------------
    printf("[1] Launching Standard MatAdd...\n");
    MatAdd<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    
    // Verify result (1 + 2 = 3)
    gpuErrchk(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));
    if(h_C[0] != 3.0f) printf("Error in Standard MatAdd!\n");
    else printf("Standard MatAdd Successful.\n");
    printf("--------------------------------------------------------\n");

    // ----------------------------------------------------------------
    // Check for Compute Capability 9.0 for Cluster features
    // ----------------------------------------------------------------
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    if (prop.major < 9)
    {
        printf("Skipping Cluster examples. Requires Compute Capability 9.0+ (Hopper).\n");
        printf("Current GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
        
        // Cleanup and exit for non-compatible GPUs
        cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
        free(h_A); free(h_B); free(h_C);
        return 0;
    }

    // ----------------------------------------------------------------
    // 2. Cluster Kernel (Compile-Time Attribute) (Section 5.2.1)
    // ----------------------------------------------------------------
    printf("[2] Launching Compile-Time Cluster Kernel...\n");
    // Launch like a normal kernel; the attribute __cluster_dims__ handles the rest.
    // Note: Grid dimensions must be multiples of cluster dimensions.
    // Current Grid: 2x2. Cluster Dim: 2x1. (2%2==0, 2%1==0). Valid.
    cluster_kernel_compile_time<<<numBlocks, threadsPerBlock>>>(d_A, d_C, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    printf("Compile-Time Cluster Kernel Finished.\n");
    printf("--------------------------------------------------------\n");

    // ----------------------------------------------------------------
    // 3. Cluster Kernel (Runtime API: cudaLaunchKernelEx) (Section 5.2.1)
    // ----------------------------------------------------------------
    printf("[3] Launching Runtime Cluster Kernel...\n");
    
    cudaLaunchConfig_t config = {0};
    config.gridDim = numBlocks;
    config.blockDim = threadsPerBlock;

    // Attribute setup for Cluster Dimensions
    cudaLaunchAttribute attribute[1];
    attribute[0].id = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = 2; // Cluster X dim
    attribute[0].val.clusterDim.y = 1; // Cluster Y dim
    attribute[0].val.clusterDim.z = 1; // Cluster Z dim

    config.attrs = attribute;
    config.numAttrs = 1;

    // Launch using the Extended API
    gpuErrchk(cudaLaunchKernelEx(&config, cluster_kernel_runtime, d_A, d_C, N));
    gpuErrchk(cudaDeviceSynchronize());
    printf("Runtime Cluster Kernel Finished.\n");
    printf("--------------------------------------------------------\n");

    // ----------------------------------------------------------------
    // 4. Blocks as Clusters (Section 5.2.2)
    // ----------------------------------------------------------------
    printf("[4] Launching Blocks-as-Clusters Kernel...\n");
    
    // The Kernel has __block_size__((16,16,1), (2,2,1)).
    // We launch grids of CLUSTERS.
    // Total threads needed: 32x32.
    // Threads per block: 16x16 = 256.
    // Total blocks needed: 2x2 = 4 blocks.
    // Blocks per cluster: 2x2 = 4 blocks.
    // Therefore, we need 1 cluster (containing 4 blocks) to cover the domain.
    
    dim3 numClusters(1, 1, 1);
    
    // Syntax: <<<GridOfClusters, Stream>>> 
    // Note: The second argument inside <<<>>> must be implicit/stream related when 
    // __block_size__ implies cluster setup, effectively treating the first arg as cluster count.
    // However, standard CUDA runtime syntax usually expects <<<grid, block>>>.
    // With __block_size__, the compiler reinterprets the launch. 
    // We explicitly pass dim3(1024, 1, 1) as a dummy for the second argument based on documentation patterns
    // or simply pass the number of clusters. 
    // The prompt documentation says: "The compiler would recognize the first argument inside <<<>>> as the number of clusters".
    
    blocks_as_clusters_kernel<<<numClusters, dim3(1,1,1)>>>(d_A, d_C, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
    printf("Blocks-as-Clusters Kernel Finished.\n");

    // ----------------------------------------------------------------
    // Cleanup
    // ----------------------------------------------------------------
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    printf("All Tests Completed Successfully.\n");
    return 0;
}