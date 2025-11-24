// File: thread_hierarchy_and_clusters.cu
// -----------------------------------------------------------------------------
// This CUDA source file explains, via heavily-commented reference code, the CUDA
// programming model hierarchy (grids, blocks, threads), and modern extensions
// for thread block clusters and "blocks as clusters".
//
// The focus is on section 5.2 of the CUDA Programming Guide and subsections
// 5.2.1 (Thread Block Clusters) and 5.2.2 (Blocks as Clusters).
//
// All explanatory text is embedded as comments inside this file, as requested.
// No attempt is made to hide low-level details; the comments aim at the level
// of an advanced CUDA programmer or a performance engineer.
// -----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// -----------------------------------------------------------------------------
// Utility: macro for CUDA error checking
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call)                                                         \
    do {                                                                         \
        cudaError_t cuda_status__ = (call);                                      \
        if (cuda_status__ != cudaSuccess) {                                      \
            std::fprintf(stderr,                                                \
                          "CUDA error at %s:%d: %s\n",                           \
                          __FILE__, __LINE__, cudaGetErrorString(cuda_status__));\
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)


// -----------------------------------------------------------------------------
// Section 5.2: Thread Hierarchy
// -----------------------------------------------------------------------------
// CUDA exposes a strictly hierarchical execution model:
//
//   - A kernel launch creates a "grid" of thread blocks.
//   - Each "block" is a 1D/2D/3D array of threads.
//   - Each "thread" has a unique index within its block: threadIdx.{x,y,z}.
//   - Each "block" has a unique index within the grid: blockIdx.{x,y,z}.
//   - The size of blocks and grids is described by blockDim and gridDim.
//
// Dimensionality:
//   - threadIdx and blockIdx are 3-component vectors (x, y, z).
//   - gridDim and blockDim are also 3-component vectors describing the number
//     of blocks and threads in each dimension.
//
// Threads per block limit:
//   - There is a hardware limit on the total number of threads in a single
//     block: on current GPUs this is 1024 threads.
//   - This is a *total* limit across x*y*z, i.e. blockDim.x * blockDim.y *
//     blockDim.z <= 1024 typically (exact value is device-dependent).
//
// Blocks per grid:
//   - The number of blocks in a grid is much larger (and device-dependent).
//   - The product gridDim.x * gridDim.y * gridDim.z is the number of blocks.
//
// Block independence:
//   - Thread blocks are *required* to be independent in general CUDA kernels.
//   - Any block can run in parallel with any other block or in any order.
//   - There is no built-in cross-block synchronization primitive in the basic
//     programming model; synchronization is only:
//       * within a block, via __syncthreads() or cooperative groups,
//       * or at the kernel launch boundary (i.e., between kernels).
//   - Thread block clusters (see section 5.2.1) and cooperative launches add
//     optional capabilities for limited cross-block cooperation.
//
// The following device helpers demonstrate how to compute linear indices out
// of multi-dimensional thread/block indices, as described in the Programming
// Guide.
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// Device helpers for computing linear thread IDs within a block
// -----------------------------------------------------------------------------
__device__ __forceinline__ unsigned int linear_thread_id_1d()
{
    // For a 1D block, the linear thread ID is simply threadIdx.x.
    return threadIdx.x;
}

__device__ __forceinline__ unsigned int linear_thread_id_2d()
{
    // For a 2D block of size (Dx, Dy), the linear thread ID of a thread at
    // (x, y) is: x + y * Dx.
    //
    // Here blockDim.x plays the role of Dx.
    return threadIdx.x + threadIdx.y * blockDim.x;
}

__device__ __forceinline__ unsigned int linear_thread_id_3d()
{
    // For a 3D block of size (Dx, Dy, Dz), the Programming Guide states that
    // the thread ID of a thread of index (x, y, z) is:
    //     x + y * Dx + z * Dx * Dy
    //
    // Here we use blockDim.{x,y,z} for (Dx,Dy,Dz).
    return threadIdx.x +
           threadIdx.y * blockDim.x +
           threadIdx.z * blockDim.x * blockDim.y;
}


// -----------------------------------------------------------------------------
// Device helpers for computing linear global thread IDs across the grid
// -----------------------------------------------------------------------------
// Note: these helpers assume the grid is logically 1D, 2D, or 3D. There is
// no inherent requirement that your data layout exactly matches grid layout,
// but such mappings are convenient.

/**
 * @brief Linear global thread ID for a 1D grid of 1D blocks.
 *
 * Mapping:
 *   - blockIdx.x ranges from 0 to gridDim.x - 1.
 *   - threadIdx.x ranges from 0 to blockDim.x - 1.
 */
__device__ __forceinline__ unsigned int global_thread_id_1d()
{
    return blockIdx.x * blockDim.x + threadIdx.x;
}

/**
 * @brief Linear global thread ID for a 2D grid of 2D blocks.
 *
 * We interpret:
 *   - global "row" index:    gy = blockIdx.y * blockDim.y + threadIdx.y
 *   - global "column" index: gx = blockIdx.x * blockDim.x + threadIdx.x
 *
 * And then we flatten (gx, gy) into a single index using a provided extent
 * (width) at call site; see comments in kernels below.
 *
 * Here we just demonstrate 2D block + 2D grid decomposition.
 */
__device__ __forceinline__ unsigned int global_linear_tid_in_2d_grid(int extent_x,
                                                                     int extent_y)
{
    // extent_x and extent_y are logical problem dimensions in x/y.
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;

    if (gx >= static_cast<unsigned int>(extent_x) ||
        gy >= static_cast<unsigned int>(extent_y)) {
        // Out-of-bounds threads may exist when the problem size is not an
        // exact multiple of the block size. The caller must typically guard
        // against OOB accesses; here we simply return 0 for such threads.
        return 0;
    }

    // The typical row-major flattening: index = gy * extent_x + gx.
    return gy * extent_x + gx;
}

/**
 * @brief Linear global thread ID for a 3D grid of 3D blocks.
 *
 * This demonstrates conceptually how to flatten a 3D grid into a 1D index,
 * where the extent parameters describe overall problem dimensions.
 */
__device__ __forceinline__ unsigned int global_linear_tid_in_3d_grid(int extent_x,
                                                                     int extent_y,
                                                                     int extent_z)
{
    unsigned int gx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int gy = blockIdx.y * blockDim.y + threadIdx.y;
    unsigned int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if (gx >= static_cast<unsigned int>(extent_x) ||
        gy >= static_cast<unsigned int>(extent_y) ||
        gz >= static_cast<unsigned int>(extent_z)) {
        return 0;
    }

    // For a 3D volume in row-major layout:
    // index = gz * (extent_x * extent_y) + gy * extent_x + gx.
    return gz * (extent_x * extent_y) + gy * extent_x + gx;
}


// -----------------------------------------------------------------------------
// Example: Matrix Addition Using a Single Block (Conceptual Intro)
// -----------------------------------------------------------------------------
// This example mirrors the basic MatAdd kernel from the Programming Guide,
// using a single block of N*N*1 threads. Note:
//
//   - The example is only valid when N*N <= maxThreadsPerBlock, which is
//     typically 1024. Thus N <= 32 is required on typical hardware.
//
//   - For realistic N, we instead use multiple blocks; see the next kernel.
// -----------------------------------------------------------------------------

/**
 * @brief Matrix addition using a *single* 2D block (threadIdx.{x,y}) for an N×N matrix.
 *
 * Constraints:
 *   - N * N <= device maxThreadsPerBlock (usually 1024).
 *   - Matrices are stored in row-major 1D layout of length N*N.
 */
__global__ void MatAddSingleBlock(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int N)
{
    // threadIdx.x in [0, blockDim.x), threadIdx.y in [0, blockDim.y).
    // We assume blockDim.x == N and blockDim.y == N here.
    int i = threadIdx.x;  // Column index
    int j = threadIdx.y;  // Row index

    if (i < N && j < N) {
        int idx = j * N + i;
        C[idx] = A[idx] + B[idx];
    }
}


// -----------------------------------------------------------------------------
// Example: Matrix Addition Using a Grid of Blocks
// -----------------------------------------------------------------------------
// This is the practical version using potentially many blocks, each with a
// fixed thread block size (e.g., 16×16). It matches the Programming Guide's
// multi-block example, generalized to arbitrary N via ceiling division.
//
// Mapping:
//   - Each thread is responsible for computing exactly one matrix element
//     C[row, col] = A[row, col] + B[row, col].
//   - We treat x as column dimension, y as row dimension.
// -----------------------------------------------------------------------------

__global__ void MatAddGrid2D(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* __restrict__ C,
                             int N)
{
    // Compute the global row/column indices of this thread.
    //   - blockDim.{x,y} is the extent of the block in each dimension.
    //   - blockIdx.{x,y} is the index of the block in the grid.
    //   - threadIdx.{x,y} is the local thread index in the block.
    int col = blockIdx.x * blockDim.x + threadIdx.x;  // column index (i)
    int row = blockIdx.y * blockDim.y + threadIdx.y;  // row index    (j)

    // Boundary check is essential when N is not an integer multiple of
    // blockDim.x or blockDim.y.
    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }
}


// -----------------------------------------------------------------------------
// Example: Demonstrating __syncthreads() and Shared Memory Within a Block
// -----------------------------------------------------------------------------
// __syncthreads():
//   - Acts as a barrier within a block; all threads in the block must reach
//     the barrier before any of them can proceed.
//   - Used to coordinate accesses to shared memory or to ensure ordering.
//   - It is *undefined behavior* if only a subset of threads in a block
//     execute __syncthreads(), e.g., due to divergent control flow.
//
// Shared memory:
//   - Low-latency, explicitly managed memory shared among threads in the same
//     block.
//   - Does not span across different blocks (for that, see distributed shared
//     memory in clusters).
//
// Kernel below is a small demonstration: it computes a per-block sum of input
// elements and writes that partial sum out. This is not a fully optimized
// reduction, but it clearly demonstrates use of shared memory and barriers.
// -----------------------------------------------------------------------------

__global__ void BlockSumExample(const float* __restrict__ input,
                                float* __restrict__ blockSums,
                                int numElements)
{
    // 1D block and 1D grid for simplicity:
    //   - global thread index = blockIdx.x * blockDim.x + threadIdx.x.
    int global_tid = blockIdx.x * blockDim.x + threadIdx.x;

    // Shared memory allocation per block; the size is known at compile time.
    extern __shared__ float sdata[];

    // Initialize shared memory with the per-thread contribution.
    float val = 0.0f;
    if (global_tid < numElements) {
        val = input[global_tid];
    }
    sdata[threadIdx.x] = val;

    // Make sure every thread has written its value to shared memory before
    // proceeding to the reduction stage.
    __syncthreads();

    // Parallel reduction within the block (naive, not unrolled):
    // This loop iteratively halves the active range.
    for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }

        // Another barrier is required: all threads must complete the addition
        // before we move to the next stride.
        __syncthreads();
    }

    // After the loop, sdata[0] holds the sum of the block's assigned elements.
    if (threadIdx.x == 0) {
        blockSums[blockIdx.x] = sdata[0];
    }
}


// -----------------------------------------------------------------------------
// Section 5.2.1: Thread Block Clusters (Compute Capability 9.0+)
// -----------------------------------------------------------------------------
// Thread Block Clusters introduce an *optional* level of hierarchy:
//
//   - A "thread block cluster" is a collection of thread blocks.
//   - All blocks in a cluster are guaranteed to be co-scheduled on a single
//     GPU Processing Cluster (GPC).
//   - This enables:
//
//        * Hardware-supported synchronization across blocks in the cluster.
//        * Access to Distributed Shared Memory (DSM) across blocks.
//
//   - Clusters are organized into a 1D/2D/3D *grid of clusters* conceptually.
//   - However, for backward compatibility, in cluster-enabled kernels:
//
//        * gridDim still denotes the total number of thread blocks (not
//          clusters).
//        * The rank (index) of a block within its cluster is exposed via the
//          Cluster Group API.
//
// Cluster size:
//   - The maximum portable cluster size is 8 blocks per cluster. Some GPUs
//     support more, some less; the precise limit can be queried via
//     cudaOccupancyMaxPotentialClusterSize.
//
//   - The grid dimension (in blocks) *must* be a multiple of the cluster size
//     in each dimension for a cluster-enabled launch.
//
// Programming interfaces for clusters:
//   - Compile-time cluster size: kernel attribute __cluster_dims__(X, Y, Z).
//   - Runtime cluster size: via cudaLaunchKernelEx and launch attributes.
//
//   - Within the kernel, the Cooperative Groups API exposes cluster-group
//     collectives, including:
//         cg::this_cluster()
//         cluster_group::sync()
//         cluster_group::num_threads(), num_blocks()
//         cluster_group::thread_rank(), block_rank().
//
// Distributed Shared Memory (DSM):
//   - Blocks in a cluster can read/write and perform atomics on a region of
//     memory that is shared across the cluster, beyond the per-block shared
//     memory. See the CUDA Programming Guide DSM section for advanced usage.
//   - DSM is not demonstrated here with real code; this file focuses on
//     structure and launch mechanics.
//
// IMPORTANT:
//   - Thread block clusters require GPUs with compute capability 9.0+ and
//     appropriate CUDA toolchain support (e.g., CUDA 12.x, subject to change).
//   - Attempting to compile or launch cluster-enabled kernels on older GPUs
//     or toolchains will fail.
// -----------------------------------------------------------------------------


// -----------------------------------------------------------------------------
// Example: Compile-Time Cluster Size via __cluster_dims__
// -----------------------------------------------------------------------------
// We demonstrate a cluster-aware kernel that performs matrix addition exactly
// like MatAddGrid2D, but with a compile-time cluster size of 2×1×1. In
// practice, the use of clusters primarily matters when we exploit cross-block
// features like cluster-wide barriers or DSM.
//
// The __cluster_dims__(X,Y,Z) attribute:
//
//   - Specifies a *fixed* cluster size at compile time; it cannot be changed
//     at launch time.
//   - The grid dimension (in blocks) must be a multiple of (X,Y,Z).
//   - The launch syntax MatAddClusterCompileTime<<<grid, block>>>() remains
//     unchanged; the runtime enumerates blocks, not clusters.
//
// In this example, we only call cluster_group::sync() to illustrate hardware
// synchronization across blocks in the same cluster, conditioned on
// __CUDA_ARCH__ >= 900 to avoid compilation issues on older devices.
// -----------------------------------------------------------------------------

__global__ void __cluster_dims__(2, 1, 1)
MatAddClusterCompileTime(const float* __restrict__ A,
                         const float* __restrict__ B,
                         float* __restrict__ C,
                         int N)
{
    // Standard 2D grid / 2D block indexing, identical to MatAddGrid2D.
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }

    // Optional cluster-wide synchronization demo.
    // This only compiles for architectures that support cluster_group;
    // we guard it with __CUDA_ARCH__ >= 900.
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    cg::cluster_group cluster = cg::this_cluster();

    // cluster.sync() is a hardware-supported barrier across *all* threads
    // in *all* blocks of this cluster.
    //
    // A typical use case would be:
    //   - Step 1: all blocks write to distributed shared memory or global
    //             memory.
    //   - Step 2: cluster-wide barrier.
    //   - Step 3: designated block (e.g., rank 0) performs a reduction or
    //             some other aggregation across the cluster's data.
    cluster.sync();

    // Example of querying cluster properties. These calls are shown for
    // completeness; their return values are unused in this demo.
    unsigned int num_blocks_in_cluster = cluster.num_blocks();
    unsigned int num_threads_in_cluster = cluster.num_threads();
    unsigned int block_rank             = cluster.block_rank();
    unsigned int thread_rank            = cluster.thread_rank();

    (void)num_blocks_in_cluster;
    (void)num_threads_in_cluster;
    (void)block_rank;
    (void)thread_rank;
#endif
}


// -----------------------------------------------------------------------------
// Example: Runtime Cluster Size via cudaLaunchKernelEx
// -----------------------------------------------------------------------------
// This example demonstrates setting the cluster size at *runtime* using the
// extensible kernel launch API cudaLaunchKernelEx.
//
// Key points:
//
//   - The kernel itself does *not* have a __cluster_dims__ attribute.
//   - At launch, we specify a cluster dimension using cudaLaunchAttribute
//     with id = cudaLaunchAttributeClusterDimension.
//   - gridDim remains a count of blocks, not clusters.
//   - The grid dimension in each axis must be a multiple of the cluster
//     dimension in that axis.
//
// The kernel body is the same as MatAddGrid2D; the difference is how we
// launch it.
//
// Note: cudaLaunchKernelEx is a relatively recent API and may require a
// modern CUDA toolkit. Always consult the relevant CUDA documentation for
// version-appropriate usage.
// -----------------------------------------------------------------------------

__global__ void MatAddClusterRuntime(const float* __restrict__ A,
                                     const float* __restrict__ B,
                                     float* __restrict__ C,
                                     int N)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    if (row < N && col < N) {
        int idx = row * N + col;
        C[idx] = A[idx] + B[idx];
    }

    // Additional cluster-specific cooperative behavior could be added here
    // via cg::this_cluster(), similar to MatAddClusterCompileTime.
}


// -----------------------------------------------------------------------------
// Section 5.2.2: Blocks as Clusters
// -----------------------------------------------------------------------------
// "Blocks as Clusters" is a further abstraction that allows the programmer
// to specify both:
//
//   - The number of threads per block, and
//   - The number of blocks per cluster
//
// as *kernel attributes* via __block_size__((Bx,By,Bz), (Cx,Cy,Cz)).
//
// With this feature enabled:
//
//   - The kernel launch uses <<<numClusters>>>, where the first triple in
//     <<<>>> is interpreted as the number of clusters in each dimension,
//     not the number of blocks.
//   - The compiler uses the kernel attributes to derive:
//        * Threads per block  = (Bx, By, Bz)
//        * Blocks per cluster = (Cx, Cy, Cz)
//        * Total blocks       = numClusters * blocksPerCluster
//
// Important rules:
//
//   - The second tuple (cluster size) defaults to (1,1,1) if omitted.
//   - It is illegal to specify the second tuple of __block_size__ together
//     with a __cluster_dims__ attribute on the same kernel:
//        * When second tuple of __block_size__ is specified, it implies that
//          "Blocks as Clusters" is enabled.
//   - To specify a CUDA stream when using "Blocks as Clusters":
//        * The second and third arguments in <<<>>> must be 1 and 0,
//          respectively; the fourth argument is then the stream.
//          Example: kernel<<<numClusters, 1, 0, stream>>>(...);
//        * Passing other values leads to undefined behavior.
// -----------------------------------------------------------------------------

// -----------------------------------------------------------------------------
// Example: __cluster_dims__ for implicit clusters
// -----------------------------------------------------------------------------
// This is a conceptual declaration from the Programming Guide:
//
//   __cluster_dims__((2, 2, 2)) __global__ void foo();
//
//   // 8x8x8 clusters each with 2x2x2 thread blocks.
//   foo<<<dim3(16, 16, 16), dim3(1024, 1, 1)>>>();
//
// Explanation:
//   - The grid is declared with 16×16×16 blocks.
//   - Each cluster has 2×2×2 blocks (i.e., 8 blocks per cluster).
//   - Therefore, there are (16/2) × (16/2) × (16/2) = 8×8×8 clusters in total.
//   - Each block has 1024×1×1 threads.
// -----------------------------------------------------------------------------
//
// The following dummy declaration demonstrates syntactic structure; we do not
// define a body to keep it conceptual.
//
// Note: This is *not* used directly in this file; it is shown purely to
// reflect the Programming Guide's example.
// -----------------------------------------------------------------------------

__global__ void __cluster_dims__(2, 2, 2) FooImplicitClusters();


// -----------------------------------------------------------------------------
// Example: __block_size__ for "Blocks as Clusters"
// -----------------------------------------------------------------------------
// Now we provide a concrete example that uses __block_size__ to encode both
// the threads-per-block and blocks-per-cluster as attributes of the kernel.
//
//   __block_size__((1024, 1, 1), (2, 2, 2)) __global__ void FooBlocksAsClusters();
//
//   // 8x8x8 clusters.
//   FooBlocksAsClusters<<<dim3(8, 8, 8)>>>();
//
// Interpretation:
//   - Each block contains 1024 threads (1024×1×1).
//   - Each cluster contains 2×2×2 = 8 blocks.
//   - The grid is launched with 8×8×8 clusters.
//   - Built-in variables:
//
//        * blockDim.{x,y,z}  -> (1024,1,1)  (threads per block)
//        * gridDim.{x,y,z}   -> (#blocksX, #blocksY, #blocksZ),
//                               where #blocks = #clusters * blocksPerCluster.
//
//   - The number of clusters in each dimension is given by the first triple in
//     <<<>>>.
// -----------------------------------------------------------------------------

__block_size__((1024, 1, 1), (2, 2, 2))
__global__ void FooBlocksAsClusters(float* __restrict__ data,
                                    int numElementsPerCluster)
{
    // Although we are launching in terms of clusters, within the kernel,
    // blockIdx and threadIdx behave as usual, indexing blocks and threads,
    // not clusters directly. The Cluster Group API can be used to derive the
    // cluster semantics if needed.

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Obtain the cluster group to query cluster-wide properties.
    cg::cluster_group cluster = cg::this_cluster();

    unsigned int cluster_block_rank  = cluster.block_rank();
    unsigned int cluster_thread_rank = cluster.thread_rank();
    unsigned int cluster_num_blocks  = cluster.num_blocks();
    unsigned int cluster_num_threads = cluster.num_threads();

    (void)cluster_block_rank;
    (void)cluster_thread_rank;
    (void)cluster_num_blocks;
    (void)cluster_num_threads;

    // A typical usage might be to have one block (e.g., rank 0) in each
    // cluster perform an aggregation over the data owned by that cluster,
    // after all blocks have produced partial results in shared or distributed
    // shared memory. This is merely conceptual here.
    cluster.sync();
#else
    (void)data;
    (void)numElementsPerCluster;
#endif
}


// -----------------------------------------------------------------------------
// Host-side helpers for demonstrations
// -----------------------------------------------------------------------------
// We now provide simple host-side helpers to:
//
//   - Initialize matrices and verify results.
//   - Launch the kernels demonstrating thread hierarchy and clusters.
// -----------------------------------------------------------------------------

/**
 * @brief Initialize matrix with deterministic data for testing.
 *
 * We fill A[i,j] = i + j for convenience.
 */
void init_matrix(float* mat, int N)
{
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            mat[row * N + col] = static_cast<float>(row + col);
        }
    }
}

/**
 * @brief Verify that C == A + B for N×N matrices, within a small tolerance.
 */
bool verify_matrix_add(const float* A, const float* B, const float* C, int N)
{
    const float eps = 1e-5f;
    for (int row = 0; row < N; ++row) {
        for (int col = 0; col < N; ++col) {
            int idx = row * N + col;
            float expected = A[idx] + B[idx];
            float diff = std::abs(C[idx] - expected);
            if (diff > eps) {
                std::fprintf(stderr,
                             "Mismatch at (%d,%d): C=%f, expected=%f, diff=%f\n",
                             row, col, C[idx], expected, diff);
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Demonstrate matrix addition with a single-block launch.
 *
 * This is only valid for N such that N*N <= maxThreadsPerBlock.
 */
void demo_single_block_matadd(int N)
{
    std::printf("=== demo_single_block_matadd, N=%d ===\n", N);

    // Query device limit for threads per block.
    int maxThreadsPerBlock = 0;
    CUDA_CHECK(cudaDeviceGetAttribute(&maxThreadsPerBlock,
                                      cudaDevAttrMaxThreadsPerBlock, 0));
    if (N * N > maxThreadsPerBlock) {
        std::printf("Skipping single-block demo: N*N=%d exceeds maxThreadsPerBlock=%d\n",
                    N * N, maxThreadsPerBlock);
        return;
    }

    const size_t numElements = static_cast<size_t>(N) * N;
    const size_t bytes = numElements * sizeof(float);

    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));
    if (!h_A || !h_B || !h_C) {
        std::fprintf(stderr, "Host memory allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Launch with a single block of size (N, N, 1).
    dim3 threadsPerBlock(N, N, 1);
    dim3 numBlocks(1, 1, 1);
    MatAddSingleBlock<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify_matrix_add(h_A, h_B, h_C, N);
    std::printf("Single-block MatAdd verification: %s\n", ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
}

/**
 * @brief Demonstrate matrix addition using a grid of 2D blocks.
 *
 * This is the scalable, general technique where we use a fixed block size
 * (e.g., 16×16) and compute a grid dimension large enough to cover the N×N
 * matrix, using ceiling division.
 */
void demo_grid2d_matadd(int N)
{
    std::printf("=== demo_grid2d_matadd, N=%d ===\n", N);

    const size_t numElements = static_cast<size_t>(N) * N;
    const size_t bytes = numElements * sizeof(float);

    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));
    if (!h_A || !h_B || !h_C) {
        std::fprintf(stderr, "Host memory allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Use a common 16×16 = 256 threads per block.
    dim3 threadsPerBlock(16, 16, 1);

    // Ceiling division to compute the number of blocks in each dimension:
    auto ceil_div = [](int a, int b) {
        return (a + b - 1) / b;
    };
    dim3 numBlocks(ceil_div(N, threadsPerBlock.x),
                   ceil_div(N, threadsPerBlock.y),
                   1);

    MatAddGrid2D<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify_matrix_add(h_A, h_B, h_C, N);
    std::printf("Grid-2D MatAdd verification: %s\n", ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
}

/**
 * @brief Demonstrate the BlockSumExample kernel with shared memory and barriers.
 */
void demo_block_sum(int numElements)
{
    std::printf("=== demo_block_sum, numElements=%d ===\n", numElements);

    const size_t bytes = static_cast<size_t>(numElements) * sizeof(float);
    float* h_input = static_cast<float*>(std::malloc(bytes));
    if (!h_input) {
        std::fprintf(stderr, "Host memory allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < numElements; ++i) {
        h_input[i] = 1.0f;  // simplify: all ones, so sums are easy to reason about
    }

    float* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Choose a block size; for demo, 256 threads is common.
    int blockSize = 256;
    int gridSize = (numElements + blockSize - 1) / blockSize;

    float* d_blockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums, gridSize * sizeof(float)));

    // Dynamic shared memory size in bytes: one float per thread.
    size_t sharedMemSize = blockSize * sizeof(float);

    BlockSumExample<<<gridSize, blockSize, sharedMemSize>>>(d_input,
                                                            d_blockSums,
                                                            numElements);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_blockSums = static_cast<float*>(std::malloc(gridSize * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(h_blockSums, d_blockSums,
                          gridSize * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Verify: since every input element is 1.0f, each block's sum is
    // the number of elements that actually belonged to that block.
    // For simplicity, we just print a few sums rather than fully verifying.
    std::printf("First few per-block sums (up to 10 blocks):\n");
    for (int i = 0; i < gridSize && i < 10; ++i) {
        std::printf("  Block %d sum = %f\n", i, h_blockSums[i]);
    }

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_blockSums));
    std::free(h_input);
    std::free(h_blockSums);
}

/**
 * @brief Demonstrate a compile-time cluster launch (__cluster_dims__).
 *
 * Note:
 *   - This requires hardware and a CUDA toolkit that support thread block
 *     clusters (compute capability 9.0+).
 *   - On unsupported hardware, this code will fail at compile or runtime.
 *   - This function is provided as a structural reference.
 */
void demo_compile_time_cluster_matadd(int N)
{
    std::printf("=== demo_compile_time_cluster_matadd, N=%d ===\n", N);

    const size_t numElements = static_cast<size_t>(N) * N;
    const size_t bytes = numElements * sizeof(float);

    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));
    if (!h_A || !h_B || !h_C) {
        std::fprintf(stderr, "Host memory allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // Use 16×16 blocks as before.
    dim3 threadsPerBlock(16, 16, 1);
    auto ceil_div = [](int a, int b) {
        return (a + b - 1) / b;
    };
    dim3 numBlocks(ceil_div(N, threadsPerBlock.x),
                   ceil_div(N, threadsPerBlock.y),
                   1);

    // __cluster_dims__(2,1,1) in the kernel declaration requires that the
    // grid dimension be a multiple of (2,1,1) in each axis:
    if ((numBlocks.x % 2) != 0 || (numBlocks.y % 1) != 0 || (numBlocks.z % 1) != 0) {
        std::printf("Grid dimension not multiple of cluster size (2,1,1). "
                    "Adjust N or threadsPerBlock for a valid cluster launch.\n");
    }

    MatAddClusterCompileTime<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify_matrix_add(h_A, h_B, h_C, N);
    std::printf("Compile-time cluster MatAdd verification: %s\n",
                ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
}

/**
 * @brief Demonstrate a runtime cluster launch via cudaLaunchKernelEx.
 *
 * Note:
 *   - This demo assumes availability of cudaLaunchKernelEx and
 *     cudaLaunchAttributeClusterDimension. Check your CUDA version.
 *   - Hardware must support thread block clusters.
 */
void demo_runtime_cluster_matadd(int N)
{
    std::printf("=== demo_runtime_cluster_matadd, N=%d ===\n", N);

    const size_t numElements = static_cast<size_t>(N) * N;
    const size_t bytes = numElements * sizeof(float);

    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));
    if (!h_A || !h_B || !h_C) {
        std::fprintf(stderr, "Host memory allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    init_matrix(h_A, N);
    init_matrix(h_B, N);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(16, 16, 1);
    auto ceil_div = [](int a, int b) {
        return (a + b - 1) / b;
    };
    dim3 numBlocks(ceil_div(N, threadsPerBlock.x),
                   ceil_div(N, threadsPerBlock.y),
                   1);

    // Cluster dimensions selected at runtime: for example, 2×1×1 blocks
    // per cluster, exactly like the compile-time example.
    cudaLaunchConfig_t config = {};
    config.gridDim  = numBlocks;
    config.blockDim = threadsPerBlock;

    cudaLaunchAttribute attribute[1];
    attribute[0].id                 = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x   = 2;  // cluster size in X
    attribute[0].val.clusterDim.y   = 1;
    attribute[0].val.clusterDim.z   = 1;
    config.attrs    = attribute;
    config.numAttrs = 1;

    // As per Programming Guide, the grid dimension must be a multiple of
    // the cluster dimension in each axis.
    if ((numBlocks.x % attribute[0].val.clusterDim.x) != 0 ||
        (numBlocks.y % attribute[0].val.clusterDim.y) != 0 ||
        (numBlocks.z % attribute[0].val.clusterDim.z) != 0) {
        std::printf("Grid dimension not multiple of runtime clusterDim; "
                    "adjust N or threadsPerBlock.\n");
    }

    // Launch the kernel via the extensible API. The preferred usage is the
    // templated convenience overload, which accepts the kernel symbol and
    // its arguments directly.
    CUDA_CHECK(cudaLaunchKernelEx(&config,
                                  MatAddClusterRuntime,
                                  d_A, d_B, d_C, N));

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify_matrix_add(h_A, h_B, h_C, N);
    std::printf("Runtime cluster MatAdd verification: %s\n",
                ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
}

/**
 * @brief Demonstrate "Blocks as Clusters" usage via FooBlocksAsClusters.
 *
 * This is a structural demonstration; because FooBlocksAsClusters is a dummy
 * kernel that does not perform real work, this function only illustrates
 * launch syntax. In a real application, you would allocate and pass
 * cluster-partitioned data.
 */
void demo_blocks_as_clusters(int numClustersX,
                             int numClustersY,
                             int numClustersZ)
{
    std::printf("=== demo_blocks_as_clusters, clusters=(%d,%d,%d) ===\n",
                numClustersX, numClustersY, numClustersZ);

    // For demonstration, we only allocate a trivial buffer. In a real use
    // case, you would likely allocate data sized according to:
    //
    //   totalBlocks = numClustersX * numClustersY * numClustersZ
    //                 * clusterBlocksX * clusterBlocksY * clusterBlocksZ
    //
    // and partition it accordingly.
    float* d_buffer = nullptr;
    CUDA_CHECK(cudaMalloc(&d_buffer, sizeof(float) * 1));

    // Launch syntax for "Blocks as Clusters" when using __block_size__ with
    // a non-trivial second tuple:
    //
    //   - First triple in <<<>>>: number of clusters (not blocks).
    //   - Second and third arguments must be 1 and 0 respectively when you
    //     want to specify a stream as the fourth argument; otherwise undefined.
    //
    // We use the simple version without custom stream here:
    dim3 numClusters(numClustersX, numClustersY, numClustersZ);

    // Note: For more control (e.g., specifying a stream), one would call:
    //   FooBlocksAsClusters<<<numClusters, 1, 0, stream>>>(...);
    //
    FooBlocksAsClusters<<<numClusters>>>(d_buffer, /*numElementsPerCluster=*/0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaFree(d_buffer));
}


// -----------------------------------------------------------------------------
// main(): orchestrates all demonstrations
// -----------------------------------------------------------------------------
// The main function runs several small demos illustrating the thread hierarchy
// and thread block cluster concepts. It is not intended as a performance
// benchmark, only as a correctness and structure reference.
// -----------------------------------------------------------------------------

int main()
{
    // Ensure we are on device 0; in a multi-GPU system, the user may select
    // another device as needed.
    CUDA_CHECK(cudaSetDevice(0));

    // Demonstration sizes; chosen to be moderately small so that the program
    // runs quickly while fully exercising the thread hierarchy.
    const int N_small_for_single_block = 16;   // 16×16=256 threads, fits easily.
    const int N_general                = 256;  // For grid-based MatAdd examples.
    const int numElementsForBlockSum   = 1 << 16; // 65536 elements.

    demo_single_block_matadd(N_small_for_single_block);
    demo_grid2d_matadd(N_general);
    demo_block_sum(numElementsForBlockSum);

    // Cluster-related demonstrations are provided for completeness. They
    // require appropriate hardware and toolchain support.
    //
    // These calls may be commented out if you are compiling on GPUs or CUDA
    // versions that do not support clusters.
    //
    // Uncomment them when targeting compute capability 9.0+ with a recent
    // toolkit.
    //
    // demo_compile_time_cluster_matadd(N_general);
    // demo_runtime_cluster_matadd(N_general);
    // demo_blocks_as_clusters(8, 8, 8);  // Example: 8×8×8 clusters.

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}