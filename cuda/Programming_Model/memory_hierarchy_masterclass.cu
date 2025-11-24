/**
 * @file memory_hierarchy_masterclass.cu
 * @brief CUDA Memory Hierarchy Masterclass: From Registers to Global Memory & Clusters.
 *
 * @details
 * --------------------------------------------------------------------------------------------
 * AUTHOR: [World's Best Coder / AI Tutor]
 * IQ LEVEL: 300+
 * DATE: 2023
 * ARCHITECTURE: Generic (Includes Hopper SM90+ features for Distributed Shared Memory)
 * STANDARD: CUDA C++ (ISO C++17 compatible)
 * --------------------------------------------------------------------------------------------
 *
 * CONCEPTUAL OVERVIEW:
 * The CUDA Memory Hierarchy is a pyramid of trade-offs between speed, size, and scope.
 * To write world-class HPC code, one must minimize latency by moving data as close 
 * to the compute units (ALUs) as possible.
 *
 * HIERARCHY LAYERS (Fastest/Smallest -> Slowest/Largest):
 * 1. REGISTERS: 
 *    - Scope: Thread-private. 
 *    - Speed: Zero latency (part of the instruction pipeline).
 *    - Lifetime: Thread lifetime.
 *
 * 2. LOCAL MEMORY:
 *    - Scope: Thread-private.
 *    - Physical Location: Global Memory (DRAM/L2 Cache), but scoped per thread. 
 *    - Usage: Used for register spilling or addressing arrays with non-constant indices.
 *
 * 3. SHARED MEMORY (SMem):
 *    - Scope: Block-private (Visible to all threads in a block).
 *    - Speed: Near-register speeds (L1 Cache equivalent).
 *    - Usage: User-managed cache for data reuse and inter-thread communication.
 *
 * 4. DISTRIBUTED SHARED MEMORY (DSMem) [Compute Capability 9.0+]:
 *    - Scope: Cluster-private (Visible to all blocks in a Thread Block Cluster).
 *    - Speed: High bandwidth peer-to-peer via SM-to-SM interconnect in a GPC.
 *
 * 5. CONSTANT MEMORY:
 *    - Scope: Global (Read-only).
 *    - Speed: Cached. Fast if all threads read the same address (broadcast).
 *    - Usage: Coefficients, configuration constants.
 *
 * 6. TEXTURE MEMORY:
 *    - Scope: Global (Read-only).
 *    - Speed: Cached via dedicated Texture Cache.
 *    - Usage: Spatial locality (2D/3D), boundary handling, interpolation.
 *
 * 7. GLOBAL MEMORY:
 *    - Scope: Global (Visible to all grids).
 *    - Speed: High latency (hundreds of cycles), High bandwidth.
 *    - Persistence: Across kernel launches.
 *
 * --------------------------------------------------------------------------------------------
 */

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>
#include <cooperative_groups.h>

// Namespace for cooperative groups to handle cluster synchronization
namespace cg = cooperative_groups;

// ============================================================================================
// ERROR HANDLING MACROS (Industry Standard)
// ============================================================================================
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// ============================================================================================
// 1. CONSTANT MEMORY DECLARATION
// ============================================================================================
/**
 * @brief Constant Memory resides in device DRAM but is cached in the Constant Cache.
 * 
 * LIMITATION: 64 KB total size.
 * OPTIMIZATION: Best used when all threads in a warp read the SAME address simultaneously.
 * This triggers a broadcast mechanism, reducing bandwidth pressure.
 */
__constant__ float c_filterKernel[9]; // Example: A 3x3 convolution filter

// ============================================================================================
// KERNEL 1: THE HIERARCHY DEMONSTRATION
// ============================================================================================
/**
 * @brief Demonstrates Registers, Local, Shared, Constant, and Global Memory.
 * 
 * @param global_in   Pointer to Global Memory (Read)
 * @param global_out  Pointer to Global Memory (Write)
 * @param n           Size of the array
 */
__global__ void MemoryHierarchyKernel(const float* __restrict__ global_in, 
                                      float* __restrict__ global_out, 
                                      int n)
{
    // --------------------------------------------------------------------
    // A. GLOBAL MEMORY & REGISTERS
    // --------------------------------------------------------------------
    // 'tid' is stored in a register.
    // Accessing 'global_in[tid]' pulls data from Global Memory (DRAM/L2) into a Register.
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (tid >= n) return;

    // 'val' is a Register variable. Fastest access.
    float val = global_in[tid];

    // --------------------------------------------------------------------
    // B. SHARED MEMORY
    // --------------------------------------------------------------------
    // __shared__ allocates memory in the On-Chip L1/Shared RAM.
    // This is manually managed by the programmer.
    // Critical for reducing Global Memory bandwidth when data is shared among threads.
    extern __shared__ float s_data[]; 

    // Load global data into shared memory
    s_data[threadIdx.x] = val;

    // BARRIER: Mandatory synchronization to ensure all threads have loaded data
    // before any thread reads neighbors' data.
    __syncthreads(); 

    // Example: Simple smoothing using neighbors in Shared Memory.
    // Because s_data is fast, repeated access here is cheap compared to global memory.
    float smoothed = 0.0f;
    if (threadIdx.x > 0 && threadIdx.x < blockDim.x - 1)
    {
        smoothed = s_data[threadIdx.x - 1] + s_data[threadIdx.x] + s_data[threadIdx.x + 1];
    }
    else 
    {
        smoothed = s_data[threadIdx.x];
    }

    // --------------------------------------------------------------------
    // C. CONSTANT MEMORY
    // --------------------------------------------------------------------
    // Accessing the global __constant__ array.
    // If all threads read c_filterKernel[0] at the same time, it's 1 memory transaction.
    float factor = c_filterKernel[0]; 
    smoothed *= factor;

    // --------------------------------------------------------------------
    // D. LOCAL MEMORY
    // --------------------------------------------------------------------
    // Local memory is a misleading name. It implies "fast/local", but it actually
    // lives in Global Memory (DRAM). It is used when registers spill or for 
    // arrays indexed dynamically where the compiler can't place them in registers.
    
    // To force local memory usage, we define an array and index it somewhat unpredictably
    // (though modern compilers are smart, 'volatile' helps simulate the spill).
    float local_array[5]; 
    for(int i=0; i<5; i++) local_array[i] = (float)i * 0.1f;
    
    // Accessing local_array involves DRAM latency (hopefully cached in L1/L2).
    smoothed += local_array[tid % 5];

    // --------------------------------------------------------------------
    // E. WRITE BACK TO GLOBAL
    // --------------------------------------------------------------------
    global_out[tid] = smoothed;
}

// ============================================================================================
// KERNEL 2: DISTRIBUTED SHARED MEMORY (CLUSTERS)
// ============================================================================================
/**
 * @brief Demonstrates Thread Block Clusters (Compute Capability 9.0+).
 * 
 * CONCEPT:
 * A Cluster is a group of Blocks. Blocks in a cluster can access each other's 
 * Shared Memory directly via the Distributed Shared Memory (DSMEM) network.
 * 
 * Note: This code requires -arch=sm_90 or higher.
 */
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
__global__ void __cluster_dims__(2, 1, 1) ClusterMemoryKernel(float* data)
{
    // Get handle to the cluster group
    cg::cluster_group cluster = cg::this_cluster();
    
    // Rank of the current block within the cluster
    unsigned int blockRank = cluster.block_rank();

    // Shared memory for THIS block
    extern __shared__ int smem_local[];
    
    // Initialize local shared memory
    if (threadIdx.x == 0)
    {
        smem_local[0] = blockRank + 100; // distinct value based on rank
    }
    cluster.sync(); // Ensure all blocks in cluster initialized their smem

    // ----------------------------------------------------------------
    // ACCESSING NEIGHBOR BLOCK'S SHARED MEMORY
    // ----------------------------------------------------------------
    // Let's read from the "other" block in this cluster of size 2.
    // If I am rank 0, I read rank 1. If I am rank 1, I read rank 0.
    unsigned int neighborRank = 1 - blockRank; 

    // Map the pointer to the neighbor's shared memory space
    int* smem_remote = cluster.map_shared_rank(smem_local, neighborRank);

    // Read from remote shared memory (DSMEM Access)
    int remote_val = 0;
    if (threadIdx.x == 0)
    {
        remote_val = *smem_remote;
        // Just for demonstration, write to global memory to verify
        data[blockIdx.x] = (float)remote_val;
    }
}
#endif

// ============================================================================================
// KERNEL 3: TEXTURE MEMORY
// ============================================================================================
/**
 * @brief Demonstrates Texture Memory Objects.
 * 
 * TEXTURE MEMORY:
 * Optimized for 2D spatial locality.
 * Features:
 * 1. Caching optimized for 2D access patterns.
 * 2. Handling out-of-bounds coordinates (Clamp, Wrap, Mirror).
 * 3. Linear interpolation (filtering) for float coords.
 * 
 * @param texObj The texture object handle
 * @param output Output buffer
 */
__global__ void TextureKernel(cudaTextureObject_t texObj, float* output, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;

    if (x < width && y < height)
    {
        // tex2D<T>(textureObject, x, y)
        // Fetching data through the texture unit.
        // Even if we access slightly scattered indices, texture cache handles 
        // spatial locality better than L1/L2 line buffers in some cases.
        float val = tex2D<float>(texObj, (float)x, (float)y);
        
        output[y * width + x] = val;
    }
}

// ============================================================================================
// HOST MAIN FUNCTION
// ============================================================================================
int main()
{
    // Setup Dimensions
    const int N = 1024;
    const size_t bytes = N * sizeof(float);

    // 1. ALLOCATE HOST & DEVICE MEMORY (GLOBAL)
    float *h_in = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *d_in, *d_out;

    gpuErrchk(cudaMalloc(&d_in, bytes));
    gpuErrchk(cudaMalloc(&d_out, bytes));

    // Initialize Data
    for(int i=0; i<N; i++) h_in[i] = 1.0f;
    gpuErrchk(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // 2. SETUP CONSTANT MEMORY
    // We copy data from host to the global __constant__ symbol
    float h_constData[9] = {0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f, 0.5f};
    gpuErrchk(cudaMemcpyToSymbol(c_filterKernel, h_constData, 9 * sizeof(float)));

    // 3. LAUNCH HIERARCHY KERNEL
    // Configuration: 128 threads per block.
    // Shared Memory Size: 128 * sizeof(float).
    int threadsPerBlock = 128;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    printf("Launching MemoryHierarchyKernel...\n");
    MemoryHierarchyKernel<<<blocksPerGrid, threadsPerBlock, threadsPerBlock * sizeof(float)>>>(d_in, d_out, N);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

    // 4. TEXTURE MEMORY SETUP & LAUNCH
    // Create a CUDA Array (optimized layout for textures) or use Linear Memory.
    // Here we use Linear Memory for simplicity.
    cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeLinear;
    resDesc.res.linear.devPtr = d_in;
    resDesc.res.linear.desc.f = cudaChannelFormatKindFloat;
    resDesc.res.linear.desc.x = 32; // 32 bits
    resDesc.res.linear.sizeInBytes = bytes;

    cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.readMode = cudaReadModeElementType; // Read as float
    texDesc.addressMode[0] = cudaAddressModeClamp; // Clamp out of bounds
    texDesc.filterMode = cudaFilterModePoint; // Point sampling (no interpolation)

    cudaTextureObject_t texObj = 0;
    gpuErrchk(cudaCreateTextureObject(&texObj, &resDesc, &texDesc, NULL));

    printf("Launching TextureKernel...\n");
    dim3 texBlock(16, 16);
    dim3 texGrid(2, 2); // Small grid for demo
    TextureKernel<<<texGrid, texBlock>>>(texObj, d_out, 32, 32); // Treat N as 32x32
    gpuErrchk(cudaDeviceSynchronize());

    // Clean up Texture
    gpuErrchk(cudaDestroyTextureObject(texObj));

    // 5. CLUSTER LAUNCH (Conditional on Architecture)
    cudaDeviceProp props;
    cudaGetDeviceProperties(&props, 0);
    if (props.major >= 9)
    {
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
        printf("Launching ClusterMemoryKernel (Hopper+)...\n");
        // Launching 2 blocks which will form 1 cluster (defined by __cluster_dims__(2,1,1))
        ClusterMemoryKernel<<<2, 32, 1024>>>(d_out);
        gpuErrchk(cudaDeviceSynchronize());
#else
        printf("Compiled without sm_90 support. Skipping Cluster execution.\n");
#endif
    }

    // Cleanup Global Memory
    gpuErrchk(cudaFree(d_in));
    gpuErrchk(cudaFree(d_out));
    free(h_in);
    free(h_out);

    printf("Memory Hierarchy Demonstration Complete.\n");
    return 0;
}