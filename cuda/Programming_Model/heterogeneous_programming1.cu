// File: heterogeneous_programming.cu
// -----------------------------------------------------------------------------
// This CUDA source file illustrates heterogeneous programming as described in
// section 5.4 of the CUDA Programming Guide.
//
// Goals:
//
//   * Demonstrate the separation of HOST (CPU) and DEVICE (GPU) computation.
//   * Show that host and device have distinct DRAM spaces:
//         - host memory (CPU DRAM)
//         - device memory (GPU DRAM)
//   * Show how the host (C/C++ code) manages device-visible memory via the
//     CUDA Runtime API:
//         - cudaMalloc / cudaFree
//         - cudaMemcpy / cudaMemcpyAsync
//         - cudaMallocHost (pinned memory)
//   * Introduce Unified Memory (managed memory) via cudaMallocManaged as a
//     single coherent memory image shared between CPU(s) and GPU(s).
//
// Organization:
//
//   1. Utility macros and helpers.
//   2. Basic global-memory kernel (explicit host/device memory separation).
//   3. Host demo: explicit separate memories with synchronous copies.
//   4. Host demo: pinned host memory + asynchronous copies + streams.
//   5. Unified Memory kernels and demos.
//   6. main(): orchestrate the demonstrations.
//
// Throughout the file, all explanations are provided as comments.
// -----------------------------------------------------------------------------

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// -----------------------------------------------------------------------------
// Error-checking macro to enforce robust CUDA API usage
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
// Section 1: Helper functions for host-side data initialization & verification
// -----------------------------------------------------------------------------

// Initialize a vector with a simple arithmetic progression: data[i] = offset + i * scale
static void init_vector(float* data, int n, float offset, float scale)
{
    for (int i = 0; i < n; ++i) {
        data[i] = offset + static_cast<float>(i) * scale;
    }
}

// Verify that result[i] == a[i] + b[i] for all i in [0, n).
static bool verify_vector_add(const float* a, const float* b, const float* result, int n)
{
    const float eps = 1e-5f;
    for (int i = 0; i < n; ++i) {
        float expected = a[i] + b[i];
        float diff = std::fabs(result[i] - expected);
        if (diff > eps) {
            std::fprintf(stderr,
                         "VectorAdd mismatch at i=%d: got=%f, expected=%f, diff=%f\n",
                         i, result[i], expected, diff);
            return false;
        }
    }
    return true;
}

// Verify that result[i] == factor * input[i] for all i in [0, n).
static bool verify_scaling(const float* input, const float* result, float factor, int n)
{
    const float eps = 1e-5f;
    for (int i = 0; i < n; ++i) {
        float expected = factor * input[i];
        float diff = std::fabs(result[i] - expected);
        if (diff > eps) {
            std::fprintf(stderr,
                         "Scaling mismatch at i=%d: got=%f, expected=%f, diff=%f\n",
                         i, result[i], expected, diff);
            return false;
        }
    }
    return true;
}


// -----------------------------------------------------------------------------
// Section 2: Device kernel for explicit host/device separation
// -----------------------------------------------------------------------------
// This kernel performs element-wise vector addition:
//
//   C[i] = A[i] + B[i],  for 0 <= i < n
//
// Memory model (explicit separation):
//   - A, B, C are pointers to DEVICE (GPU) memory (allocated via cudaMalloc).
//   - The host holds its own copies of these vectors in HOST memory (CPU DRAM).
//   - The host is responsible for transferring data between host and device.
// -----------------------------------------------------------------------------

__global__ void vector_add_kernel(const float* __restrict__ A,
                                  const float* __restrict__ B,
                                  float* __restrict__ C,
                                  int n)
{
    // Compute a global thread index in a 1D grid of 1D blocks.
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Guard against extra threads when n is not a multiple of blockDim.x.
    if (idx < n) {
        // A, B, C are all in device global memory.
        // Reads and writes are visible to all threads across the GPU.
        C[idx] = A[idx] + B[idx];
    }
}


// -----------------------------------------------------------------------------
// Section 3: Host demo - Explicit host/device separation with synchronous copies
// -----------------------------------------------------------------------------
// This function demonstrates a *classic* heterogeneous CUDA pattern:
//
//   1. Host allocates and initializes input vectors in HOST memory.
//   2. Host allocates output/vector buffers in DEVICE memory via cudaMalloc.
//   3. Host copies input data Host->Device via cudaMemcpy (synchronous).
//   4. Host launches a kernel on the device to compute the result.
//   5. Host copies result Device->Host via cudaMemcpy (synchronous).
//   6. Host verifies the result and frees all allocations.
//
// Heterogeneous principles illustrated:
//   - Host and device run asynchronously: kernel launches return immediately
//     to the host; cudaDeviceSynchronize or cudaMemcpy impose synchronization.
//   - Host and device pointers are distinct: host cannot dereference device
//     pointers, and vice versa.
//   - Global memory is explicitly managed via CUDA runtime API.
// -----------------------------------------------------------------------------

static void demo_explicit_separate_memory(int n)
{
    std::printf("=== demo_explicit_separate_memory, n=%d ===\n", n);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // 1) Allocate and initialize host memory (CPU DRAM).
    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    float* h_C = static_cast<float*>(std::malloc(bytes));

    if (!h_A || !h_B || !h_C) {
        std::fprintf(stderr, "Host malloc failed in demo_explicit_separate_memory\n");
        std::exit(EXIT_FAILURE);
    }

    init_vector(h_A, n, 1.0f, 0.5f);
    init_vector(h_B, n, -2.0f, 1.0f);

    // 2) Allocate device memory (GPU DRAM) via cudaMalloc.
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // 3) Transfer data from host to device (synchronous).
    //    These calls block until the transfer completes.
    CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice));

    // 4) Launch the kernel on the device.
    //    The host and device execute concurrently: kernel launch is async
    //    w.r.t. host by default.
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    vector_add_kernel<<<grid_size, block_size>>>(d_A, d_B, d_C, n);

    // Always check for kernel launch errors before synchronizing.
    CUDA_CHECK(cudaGetLastError());

    // Optionally, synchronize to ensure kernel completion before copying back.
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5) Transfer results back to host (synchronous).
    CUDA_CHECK(cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost));

    // 6) Verify correctness on the host.
    bool ok = verify_vector_add(h_A, h_B, h_C, n);
    std::printf("  Result verification: %s\n", ok ? "PASSED" : "FAILED");

    // Free device and host memory.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
}


// -----------------------------------------------------------------------------
// Section 4: Asynchronous copies + pinned host memory + streams
// -----------------------------------------------------------------------------
// This section demonstrates more advanced host/device interaction:
//
//   - Pinned (page-locked) host memory allocated with cudaMallocHost.
//       * Enables higher bandwidth and asynchronous DMA transfers.
//   - CUDA streams for overlapping host/device transfers and kernel execution.
//       * Operations in different streams may run concurrently when hardware
//         and dependencies allow.
//
// Experimental pattern:
//
//   - We split the vector into two contiguous halves.
//   - For each half, we use a separate CUDA stream:
//         * Asynchronous HtoD copy for that half.
//         * Kernel launch operating on that half.
//         * Asynchronous DtoH copy for that half.
//   - Finally, we synchronize both streams and verify the final result.
//
// Notes:
//
//   - This pattern reflects heterogeneous programming where the CPU orchestrates
//     multiple GPU work queues (streams) and overlaps computation with data
//     movement.
//   - Correctness MUST be preserved via explicit synchronization when the host
//     needs the final data.
// -----------------------------------------------------------------------------

static void demo_async_pinned_and_streams(int n)
{
    std::printf("=== demo_async_pinned_and_streams, n=%d ===\n", n);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // Ensure n is even for a clean split into two halves.
    if ((n & 1) != 0) {
        std::fprintf(stderr,
                     "  Warning: n is not even; adjusting to n-1 for two-way split.\n");
        --n;
    }

    const int half_n   = n / 2;
    const size_t half_bytes = static_cast<size_t>(half_n) * sizeof(float);

    // Allocate pinned (page-locked) host memory for A, B, C.
    // Pinned memory is required for truly asynchronous cudaMemcpyAsync.
    float* h_A = nullptr;
    float* h_B = nullptr;
    float* h_C = nullptr;
    CUDA_CHECK(cudaMallocHost(&h_A, bytes));
    CUDA_CHECK(cudaMallocHost(&h_B, bytes));
    CUDA_CHECK(cudaMallocHost(&h_C, bytes));

    init_vector(h_A, n, 0.0f, 1.0f);
    init_vector(h_B, n, 5.0f, -0.25f);

    // Allocate device memory (global memory) for the full vectors.
    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, bytes));
    CUDA_CHECK(cudaMalloc(&d_B, bytes));
    CUDA_CHECK(cudaMalloc(&d_C, bytes));

    // Create two CUDA streams.
    cudaStream_t stream0, stream1;
    CUDA_CHECK(cudaStreamCreate(&stream0));
    CUDA_CHECK(cudaStreamCreate(&stream1));

    const int block_size = 256;
    const int grid_size_half = (half_n + block_size - 1) / block_size;

    // Lambda to launch async pipeline for a half [offset, offset + half_n).
    auto process_half = [&](int offset, cudaStream_t stream) {
        const size_t offset_bytes = static_cast<size_t>(offset) * sizeof(float);

        // Async HtoD copies.
        CUDA_CHECK(cudaMemcpyAsync(d_A + offset,
                                   h_A + offset,
                                   half_bytes,
                                   cudaMemcpyHostToDevice,
                                   stream));

        CUDA_CHECK(cudaMemcpyAsync(d_B + offset,
                                   h_B + offset,
                                   half_bytes,
                                   cudaMemcpyHostToDevice,
                                   stream));

        // Kernel launch on this half (using an offset pointer).
        vector_add_kernel<<<grid_size_half, block_size, 0, stream>>>(
            d_A + offset, d_B + offset, d_C + offset, half_n
        );

        // Async DtoH copy of the result.
        CUDA_CHECK(cudaMemcpyAsync(h_C + offset,
                                   d_C + offset,
                                   half_bytes,
                                   cudaMemcpyDeviceToHost,
                                   stream));

        // No immediate synchronization; overlaps possible with other stream.
    };

    // Launch two independent asynchronous pipelines.
    process_half(0,        stream0);
    process_half(half_n,   stream1);

    // Wait for both streams to finish all queued work.
    CUDA_CHECK(cudaStreamSynchronize(stream0));
    CUDA_CHECK(cudaStreamSynchronize(stream1));

    // Verify final result on the host.
    bool ok = verify_vector_add(h_A, h_B, h_C, n);
    std::printf("  Async pinned/streams result verification: %s\n",
                ok ? "PASSED" : "FAILED");

    // Cleanup resources.
    CUDA_CHECK(cudaStreamDestroy(stream0));
    CUDA_CHECK(cudaStreamDestroy(stream1));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    CUDA_CHECK(cudaFreeHost(h_A));
    CUDA_CHECK(cudaFreeHost(h_B));
    CUDA_CHECK(cudaFreeHost(h_C));
}


// -----------------------------------------------------------------------------
// Section 5: Unified Memory (Managed Memory) examples
// -----------------------------------------------------------------------------
// Unified Memory (UM):
//
//   - cudaMallocManaged allocates a single memory region that is accessible
//     from both CPU and GPU with a *single virtual address* (Unified Virtual
//     Addressing).
//   - It provides a *single coherent memory image* across all CPUs and GPUs
//     in the system. The CUDA driver/runtime manages migration of pages
//     between host and device under the hood.
//   - Benefits:
//       * Simplifies programming by eliminating manual cudaMemcpy.
//       * Supports oversubscription of device memory (pages can be kept in
//         system memory and migrated on demand).
//   - Requirements and caveats:
//       * On systems without full managed memory support, performance may be
//         limited or some features may be unavailable.
//       * Explicit synchronization is still needed for correct visibility of
//         results (e.g., cudaDeviceSynchronize before host reads data last
//         written by the device).
//       * For performance, programmers can use cudaMemPrefetchAsync to give
//         hints where data will be accessed next.
//
// The following kernels and demos show:
//
//   - A simple scaling kernel operating on data allocated via cudaMallocManaged.
//   - Host initialization and verification using the same pointer as the GPU.
//   - Optional prefetching hints to the GPU and CPU.
// -----------------------------------------------------------------------------

// Simple scaling kernel: data[i] *= factor, 0 <= i < n
__global__ void scale_in_place_kernel(float* data, float factor, int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] *= factor;
    }
}

// Demo: unified memory without explicit prefetching.
static void demo_unified_memory_basic(int n, float factor)
{
    std::printf("=== demo_unified_memory_basic, n=%d, factor=%f ===\n", n, factor);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // 1) Allocate managed (Unified) memory accessible by both host and device.
    float* data = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));

    // 2) Initialize data directly from the host using the managed pointer.
    //    No cudaMemcpy required here.
    init_vector(data, n, 1.0f, 1.0f);  // data[i] = 1 + i

    // 3) Launch kernel that operates on the same pointer.
    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    scale_in_place_kernel<<<grid_size, block_size>>>(data, factor, n);
    CUDA_CHECK(cudaGetLastError());

    // 4) Synchronize to ensure all device writes are visible to the host.
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5) Verify on the host using the same pointer.
    bool ok = verify_scaling(data, data, factor, n);
    std::printf("  Unified memory basic verification: %s\n",
                ok ? "PASSED" : "FAILED");

    // 6) Free the managed allocation.
    CUDA_CHECK(cudaFree(data));
}

// Demo: unified memory with prefetching hints to GPU and CPU.
static void demo_unified_memory_prefetch(int n, float factor)
{
    std::printf("=== demo_unified_memory_prefetch, n=%d, factor=%f ===\n", n, factor);

    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    // Query device properties to check for managed memory and prefetch support.
    int device = 0;
    cudaDeviceProp prop{};
    CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, device));

    if (!prop.managedMemory) {
        std::printf("  Device does not support managed memory; skipping prefetch demo.\n");
        return;
    }

    float* data = nullptr;
    CUDA_CHECK(cudaMallocManaged(&data, bytes));

    // Initialize on host.
    init_vector(data, n, 2.0f, -0.5f);  // data[i] = 2 - 0.5*i

    // Hint: prefetch to device (GPU) before kernel launch to reduce first-touch
    // page faults during device execution.
    // -1 for deviceId would broadcast, but here we prefer the specific device.
    CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, device, /*stream=*/0));
    CUDA_CHECK(cudaDeviceSynchronize());

    const int block_size = 256;
    const int grid_size = (n + block_size - 1) / block_size;

    scale_in_place_kernel<<<grid_size, block_size>>>(data, factor, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Hint: prefetch back to CPU (cudaCpuDeviceId) before intensive CPU access.
    CUDA_CHECK(cudaMemPrefetchAsync(data, bytes, cudaCpuDeviceId, /*stream=*/0));
    CUDA_CHECK(cudaDeviceSynchronize());

    // Verify result on host.
    bool ok = verify_scaling(data, data, factor, n);
    std::printf("  Unified memory prefetch verification: %s\n",
                ok ? "PASSED" : "FAILED");

    CUDA_CHECK(cudaFree(data));
}


// -----------------------------------------------------------------------------
// Section 6: main() - orchestrate heterogeneous programming demonstrations
// -----------------------------------------------------------------------------
// main() drives the examples:
//
//   - Selects a CUDA device.
//   - Demonstrates explicit host/device memory separation with copies.
//   - Demonstrates asynchronous transfers + streams with pinned host memory.
//   - Demonstrates Unified Memory basic usage and prefetching.
//
// This mirrors heterogeneous programming in practice:
//
//   * The host (CPU) acts as an orchestrator, managing:
//       - Work decomposition and kernel launches on the GPU.
//       - Memory allocation and data movement between host and device.
//   * The device (GPU) executes massively parallel kernels using its own
//     memory hierarchy.
// -----------------------------------------------------------------------------

int main()
{
    // Select device 0 for simplicity; for multi-GPU systems, the application
    // may choose among devices based on capability, free memory, etc.
    CUDA_CHECK(cudaSetDevice(0));

    const int n_small  = 1 << 16;  // 65,536 elements
    const int n_medium = 1 << 18;  // 262,144 elements

    // 1) Explicit host/device separation with synchronous cudaMemcpy.
    demo_explicit_separate_memory(n_small);

    // 2) Pinned host memory + asynchronous copies + two streams.
    demo_async_pinned_and_streams(n_medium);

    // 3) Unified Memory basic usage.
    demo_unified_memory_basic(n_small, 3.5f);

    // 4) Unified Memory with prefetch hints.
    demo_unified_memory_prefetch(n_small, 0.25f);

    // Reset the device to flush all state and release resources cleanly.
    CUDA_CHECK(cudaDeviceReset());
    return 0;
}