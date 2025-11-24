// File: memory_hierarchy.cu
// -----------------------------------------------------------------------------
// This CUDA source file explains, via commented reference code, the CUDA memory
// hierarchy as described in section 5.3 of the CUDA Programming Guide.
//
// Memory spaces covered:
// 
//   * Per-thread:
//       - Registers (implicit, not directly addressable in CUDA C++)
//       - Local memory (per-thread, in device DRAM, used for large arrays,
//         register spills, and variables with certain access patterns)
//
//   * Per-thread-block:
//       - Shared memory (__shared__), low-latency, explicitly managed,
//         lifetime = block, scope = threads of the block only
//
//   * Per-thread-block-cluster (Compute Capability 9.0+):
//       - Distributed shared memory (DSM): shared memory of all blocks in a
//         cluster forms a single addressable region; blocks can read/write/atomics
//         each other's shared memory via cooperative groups API
//
//   * Per-device (all threads):
//       - Global memory: main device DRAM, read/write, persistent across kernels
//       - Constant memory: read-only from device, cached, persistent across kernels
//       - Texture memory: read-only from device, cached with dedicated texture
//         cache and sampling hardware, persistent across kernels
//
// Design goals of this file:
//   * Show how each memory space is declared and accessed in CUDA C++.
//   * Highlight lifetime, visibility, and performance characteristics.
//   * Provide small correctness checks on the host side.
//   * Keep explanations as comments inside this file only.
// -----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

// -----------------------------------------------------------------------------
// Error-checking helper
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
// Global memory example
// -----------------------------------------------------------------------------
// Global memory:
//
//   - Backed by device DRAM; largest memory space, high latency.
//   - Read/write by all threads, across all grids and kernels.
//   - Lifetime: application-level (until freed or context destroyed).
//   - Visible to host via cudaMemcpy, cudaMemset, etc.
//   - All data allocated via cudaMalloc resides in global memory.
//
// Vector addition demonstration:
//   - Each thread reads one element from A and B (global memory) and writes
//     the result to C (global memory).
//   - Straightforward mapping: one thread per element.
//   - Proper thread indexing leads to coalesced accesses when arrays are
//     contiguous and aligned.
// -----------------------------------------------------------------------------

__global__ void vector_add_global(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int n)
{
    // 1D grid of 1D blocks for simplicity.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // a, b, c are all in global memory. The __restrict__ qualifier
        // communicates to the compiler that these pointers do not alias,
        // enabling better optimization.
        c[idx] = a[idx] + b[idx];
    }
}


// -----------------------------------------------------------------------------
// Shared memory example
// -----------------------------------------------------------------------------
// Shared memory:
//
//   - On-chip, low-latency, explicitly managed by the programmer.
//   - Allocated per thread block, lifetime equals block lifetime.
//   - Accessible by all threads in the same block.
//   - Not visible outside the block (except as distributed shared memory in
//     clusters; see cluster example below).
//
// Usage pattern:
//   - Load a subset of global memory into shared memory cooperatively.
//   - Perform multiple operations on shared data.
//   - Optional: write reduced or transformed results back to global memory.
//
// Here we demonstrate a simple per-block sum reduction:
//
//   - Each block gets a subset of the input array.
//   - Threads in the block cooperatively reduce their elements into a single
//     value in shared memory.
//   - The result is written to a per-block result array in global memory.
// -----------------------------------------------------------------------------

__global__ void block_sum_shared(const float* __restrict__ input,
                                 float* __restrict__ block_sums,
                                 int n)
{
    // Dynamic shared memory allocation:
    //   extern __shared__ float sdata[];
    // The actual size in bytes is specified at kernel launch.
    extern __shared__ float sdata[];

    // 1D global thread index.
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Step 1: Each thread loads one element from global to shared memory.
    float value = 0.0f;
    if (global_idx < n) {
        value = input[global_idx];
    }

    sdata[threadIdx.x] = value;

    // Step 2: Synchronize threads to ensure shared memory is fully populated.
    __syncthreads();

    // Step 3: Reduce in shared memory. This is a simple binary tree reduction.
    //         For performance-critical production code, one would apply
    //         additional optimizations (loop unrolling, warp shuffles, etc.).
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata[threadIdx.x] += sdata[threadIdx.x + stride];
        }
        // All threads in the block must reach this barrier; divergent execution
        // where some threads skip __syncthreads() leads to undefined behavior.
        __syncthreads();
    }

    // Step 4: Thread 0 writes the block's sum to global memory.
    if (threadIdx.x == 0) {
        block_sums[blockIdx.x] = sdata[0];
    }
}


// -----------------------------------------------------------------------------
// Local memory example
// -----------------------------------------------------------------------------
// Local memory:
//
//   - Per-thread address space, logically "private" to each thread.
//   - Resides in device DRAM (same physical memory as global), but accessed
//     via a distinct local memory path.
//   - Used when:
//       * The compiler spills registers, e.g., due to high register pressure.
//       * A per-thread array or variable cannot be placed in registers.
//   - Lifetime: thread lifetime.
//   - Not directly addressable by other threads.
//
// In CUDA C++ syntax, there is no explicit "local" memory qualifier for user-
// declared variables; instead, local memory is automatically used when needed.
// A typical trigger is a large per-thread array.
//
// The kernel below allocates a per-thread array "local_buffer". On most
// architectures, such an array of moderate size will reside in local memory.
// -----------------------------------------------------------------------------

__global__ void local_memory_example(float* __restrict__ output,
                                     int num_elements_per_thread)
{
    // THREAD-PRIVATE array: logically local to each thread. The compiler
    // may allocate this in registers or in local memory depending on size,
    // access pattern, and available registers.
    const int LOCAL_CAPACITY = 128;
    float local_buffer[LOCAL_CAPACITY];

    // We limit the number of elements to LOCAL_CAPACITY to avoid OOB access.
    int count = num_elements_per_thread;
    if (count > LOCAL_CAPACITY) {
        count = LOCAL_CAPACITY;
    }

    // Initialize the per-thread local array.
    for (int i = 0; i < count; ++i) {
        // Arbitrary per-thread initialization; here we simply use a function
        // of the thread's index to emphasize independence between threads.
        local_buffer[i] = static_cast<float>(threadIdx.x + i);
    }

    // Perform a simple reduction over the per-thread local array.
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) {
        sum += local_buffer[i];
    }

    // Write result to global memory for visibility.
    int global_idx = blockIdx.x * blockDim.x + threadIdx.x;
    output[global_idx] = sum;
}


// -----------------------------------------------------------------------------
// Constant memory example
// -----------------------------------------------------------------------------
// Constant memory:
//
//   - Declared with the __constant__ qualifier at file scope.
//   - Read-only from device code; writable from host using cudaMemcpyToSymbol.
//   - Cached in a dedicated constant cache; designed for broadcast patterns
//     where many threads read the same location.
//   - Lifetime: application-level; persistent across kernel launches until
//     overwritten by the host.
//
// Example:
//   - A single scaling factor stored in constant memory.
//   - All threads multiply their input by this constant.
//   - When all threads read the same address, constant cache broadcast is very
//     efficient.
// -----------------------------------------------------------------------------

// Device-side constant memory symbol.
__constant__ float g_scale_constant;

// Kernel that uses constant memory to scale a global array.
__global__ void scale_with_constant(const float* __restrict__ input,
                                    float* __restrict__ output,
                                    int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // g_scale_constant is read-only and cached in the constant cache.
        output[idx] = g_scale_constant * input[idx];
    }
}


// -----------------------------------------------------------------------------
// Texture memory example (using texture objects)
// -----------------------------------------------------------------------------
// Texture memory:
//
//   - Read-only memory space, cached by a specialized texture cache.
//   - Best suited for access patterns with 2D / 3D locality, or when using
//     hardware-accelerated addressing modes and filtering.
//   - In modern CUDA, texture objects (cudaTextureObject_t) are the preferred
//     interface:
//
//       * Resource description (cudaResourceDesc) describes underlying memory.
//       * Texture description (cudaTextureDesc) describes sampling behavior.
//
//   - Texture memory is persistent across kernel launches as long as the
//     underlying resource (e.g., a cudaMalloc allocation) remains valid and
//     the texture object is not destroyed.
//
// Here we demonstrate:
//
//   - Binding a 1D float array in global memory to a texture object.
//   - Fetching elements via tex1Dfetch inside the kernel.
// -----------------------------------------------------------------------------

__global__ void texture_read_example(cudaTextureObject_t tex_obj,
                                     float* __restrict__ output,
                                     int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n) {
        // tex1Dfetch performs a point sample (no filtering) from a 1D
        // texture object bound to linear memory. The address is an integer
        // element index.
        float val = tex1Dfetch<float>(tex_obj, idx);

        // Write to global memory so we can verify results on host.
        output[idx] = val;
    }
}


// -----------------------------------------------------------------------------
// Distributed shared memory (thread block clusters, DSM) example (conceptual)
// -----------------------------------------------------------------------------
// Thread block clusters and distributed shared memory (DSM) are available on
// GPUs with Compute Capability 9.0+.
//
//   - Blocks in a cluster are co-scheduled on a GPU Processing Cluster (GPC).
//   - Each block has its own shared memory, but in a cluster these memories
//     are mapped into a unified distributed shared memory address space.
//   - Blocks can read/write/perform atomics on each other's shared memory by
//     using the Cooperative Groups Cluster API, e.g.:
//
//       cg::cluster_group cluster = cg::this_cluster();
//       T* remote_ptr = cluster.map_shared_rank(local_ptr, remote_block_rank);
//
//   - This enables new programming patterns, such as:
//       * Multi-block reductions without global memory traffic.
//       * Cooperative producer-consumer schemes across blocks.
//
// The kernel below is a *conceptual* demonstration:
//
//   - Each block in a cluster initializes a shared integer to its block rank
//     within the cluster.
//   - After a cluster-wide barrier, block 0 in the cluster sums these shared
//     integers from all blocks in the cluster by mapping their shared memory
//     into its own address space.
//   - The result is written to global memory.
//
// Note:
//   - This kernel is guarded by __CUDA_ARCH__ >= 900; on lower architectures,
//     the cluster-specific code is compiled out.
//   - This kernel is not invoked from main() by default, to keep the example
//     compatible with a wide range of devices and toolchains.
// -----------------------------------------------------------------------------

__global__ void __cluster_dims__(2, 1, 1)
cluster_dsm_example(int* __restrict__ cluster_sums)
{
    // Per-block shared variable that will participate in DSM.
    __shared__ int s_block_value;

#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ >= 900)
    // Obtain the cluster group abstraction.
    cg::cluster_group cluster = cg::this_cluster();

    // The rank of this block within its cluster (0 .. num_blocks-1).
    unsigned int block_rank = cluster.block_rank();
    unsigned int num_blocks = cluster.num_blocks();

    // Step 1: each block writes its rank into its shared memory.
    if (threadIdx.x == 0) {
        s_block_value = static_cast<int>(block_rank);
    }
    __syncthreads();

    // Step 2: synchronize all blocks in the cluster.
    cluster.sync();

    // Step 3: block 0 in the cluster reads all blocks' shared values via DSM.
    if (block_rank == 0 && threadIdx.x == 0) {
        int sum = 0;
        for (unsigned int remote_rank = 0; remote_rank < num_blocks; ++remote_rank) {
            // Map our local shared variable "s_block_value" into the address
            // space of the block with rank = remote_rank. The returned pointer
            // refers to that block's shared memory location corresponding to
            // s_block_value.
            int* remote_ptr = cluster.map_shared_rank(&s_block_value, remote_rank);
            sum += *remote_ptr;
        }

        // We now have the sum of block ranks in this cluster.
        //
        // For demonstration, we store the result indexed by the blockIdx.x
        // of the first block in the cluster. The mapping from blockIdx to
        // cluster index depends on launch configuration; this is just a
        // conceptual write-out.
        int base_block_index = blockIdx.x - static_cast<int>(block_rank);
        cluster_sums[base_block_index] = sum;
    }

    cluster.sync();
#else
    // On architectures that do not support clusters, avoid unused-variable
    // warnings by referencing variables in a no-op way.
    (void)cluster_sums;
    (void)s_block_value;
#endif
}


// -----------------------------------------------------------------------------
// Host-side helper functions and demonstrations
// -----------------------------------------------------------------------------
// These functions demonstrate how to:
//
//   - Allocate / initialize data.
//   - Launch kernels accessing different memory spaces.
//   - Verify correctness where applicable.
// -----------------------------------------------------------------------------


// Initialize a 1D float array with a simple pattern for testing.
void init_sequence(float* data, int n, float scale = 1.0f)
{
    for (int i = 0; i < n; ++i) {
        data[i] = scale * static_cast<float>(i);
    }
}


// Verify that c[i] == a[i] + b[i] for all i in [0, n).
bool verify_vector_add(const float* a, const float* b, const float* c, int n)
{
    const float eps = 1e-5f;
    for (int i = 0; i < n; ++i) {
        float expected = a[i] + b[i];
        float diff = std::fabs(c[i] - expected);
        if (diff > eps) {
            std::fprintf(stderr,
                         "VectorAdd mismatch at %d: c=%f, expected=%f, diff=%f\n",
                         i, c[i], expected, diff);
            return false;
        }
    }
    return true;
}


// Verify that out[i] == scale * in[i] for all i in [0, n).
bool verify_scaling(const float* in, const float* out, float scale, int n)
{
    const float eps = 1e-5f;
    for (int i = 0; i < n; ++i) {
        float expected = scale * in[i];
        float diff = std::fabs(out[i] - expected);
        if (diff > eps) {
            std::fprintf(stderr,
                         "Scale mismatch at %d: out=%f, expected=%f, diff=%f\n",
                         i, out[i], expected, diff);
            return false;
        }
    }
    return true;
}


// Verify that out[i] == in[i] for all i in [0, n).
bool verify_copy(const float* in, const float* out, int n)
{
    const float eps = 1e-5f;
    for (int i = 0; i < n; ++i) {
        float diff = std::fabs(out[i] - in[i]);
        if (diff > eps) {
            std::fprintf(stderr,
                         "Copy mismatch at %d: out=%f, in=%f, diff=%f\n",
                         i, out[i], in[i], diff);
            return false;
        }
    }
    return true;
}


// Demonstrate global memory via vector_add_global.
void demo_global_memory(int n)
{
    std::printf("=== demo_global_memory (vector_add_global), n=%d ===\n", n);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* h_a = static_cast<float*>(std::malloc(bytes));
    float* h_b = static_cast<float*>(std::malloc(bytes));
    float* h_c = static_cast<float*>(std::malloc(bytes));
    if (!h_a || !h_b || !h_c) {
        std::fprintf(stderr, "Host malloc failed in demo_global_memory\n");
        std::exit(EXIT_FAILURE);
    }

    init_sequence(h_a, n, 1.0f);
    init_sequence(h_b, n, 2.0f);

    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    vector_add_global<<<grid_size, block_size>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify_vector_add(h_a, h_b, h_c, n);
    std::printf("Global memory vector_add_global verification: %s\n",
                ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c);
}


// Demonstrate shared memory via block_sum_shared.
void demo_shared_memory(int n)
{
    std::printf("=== demo_shared_memory (block_sum_shared), n=%d ===\n", n);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* h_input = static_cast<float*>(std::malloc(bytes));
    if (!h_input) {
        std::fprintf(stderr, "Host malloc failed in demo_shared_memory\n");
        std::exit(EXIT_FAILURE);
    }

    // Use 1.0f so that sums equal the number of contributing elements.
    for (int i = 0; i < n; ++i) {
        h_input[i] = 1.0f;
    }

    float* d_input = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    float* d_block_sums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_block_sums, grid_size * sizeof(float)));

    // Dynamic shared memory size: one float per thread.
    size_t shared_bytes = static_cast<size_t>(block_size) * sizeof(float);

    block_sum_shared<<<grid_size, block_size, shared_bytes>>>(d_input,
                                                              d_block_sums,
                                                              n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_block_sums = static_cast<float*>(std::malloc(grid_size * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(h_block_sums, d_block_sums,
                          grid_size * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // Basic sanity check and partial print.
    float total_sum = 0.0f;
    for (int b = 0; b < grid_size; ++b) {
        total_sum += h_block_sums[b];
    }

    std::printf("Shared memory block sums (first up to 5 blocks):\n");
    for (int b = 0; b < grid_size && b < 5; ++b) {
        std::printf("  block %d: sum = %f\n", b, h_block_sums[b]);
    }
    std::printf("Total sum over all blocks: %f (expected approx. %d)\n",
                total_sum, n);

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_block_sums));
    std::free(h_input);
    std::free(h_block_sums);
}


// Demonstrate local memory via local_memory_example.
void demo_local_memory(int n_threads, int num_elements_per_thread)
{
    std::printf("=== demo_local_memory (local_memory_example), "
                "threads=%d, elems/thread=%d ===\n",
                n_threads, num_elements_per_thread);

    const size_t bytes = static_cast<size_t>(n_threads) * sizeof(float);

    float* d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    const int block_size = 256;
    const int grid_size = (n_threads + block_size - 1) / block_size;

    local_memory_example<<<grid_size, block_size>>>(d_output,
                                                    num_elements_per_thread);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    float* h_output = static_cast<float*>(std::malloc(bytes));
    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // Print a few sample results to illustrate per-thread independence.
    std::printf("Local memory per-thread reductions (first up to 8 threads):\n");
    for (int i = 0; i < n_threads && i < 8; ++i) {
        std::printf("  thread %d: sum = %f\n", i, h_output[i]);
    }

    CUDA_CHECK(cudaFree(d_output));
    std::free(h_output);
}


// Demonstrate constant memory via g_scale_constant and scale_with_constant.
void demo_constant_memory(int n, float scale)
{
    std::printf("=== demo_constant_memory (scale_with_constant), n=%d, scale=%f ===\n",
                n, scale);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* h_input  = static_cast<float*>(std::malloc(bytes));
    float* h_output = static_cast<float*>(std::malloc(bytes));
    if (!h_input || !h_output) {
        std::fprintf(stderr, "Host malloc failed in demo_constant_memory\n");
        std::exit(EXIT_FAILURE);
    }

    init_sequence(h_input, n, 1.0f);

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Copy scaling factor into constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(g_scale_constant, &scale, sizeof(scale)));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    scale_with_constant<<<grid_size, block_size>>>(d_input, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify_scaling(h_input, h_output, scale, n);
    std::printf("Constant memory scaling verification: %s\n",
                ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    std::free(h_input);
    std::free(h_output);
}


// Demonstrate texture memory via texture_read_example and a 1D texture object.
void demo_texture_memory(int n)
{
    std::printf("=== demo_texture_memory (texture_read_example), n=%d ===\n", n);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    float* h_input  = static_cast<float*>(std::malloc(bytes));
    float* h_output = static_cast<float*>(std::malloc(bytes));
    if (!h_input || !h_output) {
        std::fprintf(stderr, "Host malloc failed in demo_texture_memory\n");
        std::exit(EXIT_FAILURE);
    }

    init_sequence(h_input, n, 1.0f);

    float *d_input = nullptr, *d_output = nullptr;
    CUDA_CHECK(cudaMalloc(&d_input, bytes));
    CUDA_CHECK(cudaMalloc(&d_output, bytes));

    CUDA_CHECK(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // Create a channel descriptor for a 32-bit float.
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc<float>();

    // Describe the resource: 1D linear memory.
    cudaResourceDesc res_desc;
    std::memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType = cudaResourceTypeLinear;
    res_desc.res.linear.devPtr = d_input;
    res_desc.res.linear.desc = channel_desc;
    res_desc.res.linear.sizeInBytes = bytes;

    // Describe the texture behavior.
    cudaTextureDesc tex_desc;
    std::memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.readMode = cudaReadModeElementType;     // no normalization
    tex_desc.addressMode[0] = cudaAddressModeClamp;  // clamp out-of-range
    tex_desc.filterMode = cudaFilterModePoint;       // point sampling
    tex_desc.normalizedCoords = 0;                   // integer addressing

    // Create texture object.
    cudaTextureObject_t tex_obj = 0;
    CUDA_CHECK(cudaCreateTextureObject(&tex_obj, &res_desc, &tex_desc, nullptr));

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    texture_read_example<<<grid_size, block_size>>>(tex_obj, d_output, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    bool ok = verify_copy(h_input, h_output, n);
    std::printf("Texture memory read verification: %s\n",
                ok ? "PASSED" : "FAILED");

    // Destroy texture object and free resources.
    CUDA_CHECK(cudaDestroyTextureObject(tex_obj));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_output));
    std::free(h_input);
    std::free(h_output);
}


// Demonstration stub for cluster_dsm_example (not launched by default).
void demo_cluster_dsm_concept()
{
    std::printf("=== demo_cluster_dsm_concept (cluster_dsm_example) ===\n");
    std::printf("This is a conceptual example and is not launched by default.\n");
    std::printf("To use it, compile for Compute Capability 9.0+ and add a\n");
    std::printf("launch configuration consistent with the __cluster_dims__\n");
    std::printf("attribute, then allocate an int* for cluster_sums and pass\n");
    std::printf("it to cluster_dsm_example<<<...>>>().\n");
}


// -----------------------------------------------------------------------------
// main(): orchestrate demonstrations of CUDA memory hierarchy
// -----------------------------------------------------------------------------
// This main function:
//
//   - Selects device 0 (for multi-GPU systems, adjust as needed).
//   - Runs a series of small demos, each focusing on different memory spaces.
//   - Leaves cluster DSM demo as a conceptual placeholder.
//
// Note on persistence:
//
//   - Global memory (d_* allocations), constant memory symbols, and texture
//     objects remain valid across multiple kernel launches until explicitly
//     freed or destroyed.
//   - Shared memory, local memory, and registers are not persistent across
//     kernels; they are re-initialized for each kernel launch.
// -----------------------------------------------------------------------------

int main()
{
    CUDA_CHECK(cudaSetDevice(0));

    // Global memory demo: simple vector addition.
    demo_global_memory(1 << 16);  // 65,536 elements

    // Shared memory demo: per-block reduction.
    demo_shared_memory(1 << 16);

    // Local memory demo: per-thread local array and reduction.
    demo_local_memory(1 << 10, 64);  // 1024 threads, 64 elements/thread

    // Constant memory demo: scaling with a device-side constant.
    demo_constant_memory(1 << 16, 3.0f);

    // Texture memory demo: cached, read-only access via texture object.
    demo_texture_memory(1 << 16);

    // Conceptual DSM / cluster demo (not launched by default).
    demo_cluster_dsm_concept();

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}