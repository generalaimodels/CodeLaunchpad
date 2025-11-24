// ============================================================================
// File: cuda_introduction_and_scalable_model.cu
// Description:
//   Executable, deeply commented tutorial that explains, in technical depth:
//
//   3.1 The Benefits of Using GPUs
//   3.2 CUDA: A General-Purpose Parallel Computing Platform and Programming Model
//   3.3 A Scalable Programming Model
//
//   All explanations are embedded as comments, and all demonstrations are
//   high‑quality CUDA C++ implementations that follow best practices.
//
//   NOTE ON RUNTIME ERROR:
//   ----------------------
//   If you see the runtime error:
//
//       "no kernel image is available for execution on the device"
//
//   it almost always means the binary was compiled for GPU architectures
//   that do not match your actual GPU. Typical causes:
//
//     - Compiled only for a newer architecture (e.g., sm_80) and running on
//       an older GPU (e.g., sm_61).
//     - Using a very old GPU that is no longer supported by the CUDA toolkit
//       you are using (e.g., CC 2.x with a modern CUDA 12 toolkit).
//
//   The fix is to recompile with appropriate -gencode flags for your GPU.
//
//   This file contains a helper that prints your GPU's compute capability
//   and commented example nvcc commands to avoid that error.
// ============================================================================

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>

// ============================================================================
// Utility Macros and Helpers
// ============================================================================

/*
    CUDA_CHECK:
    ----------
    Robust error-checking macro for all CUDA runtime API calls.

    In production-quality CUDA code, every API call must be checked.
    Failing to do so can lead to silent failures, undefined behavior, or
    difficult-to-debug performance issues.
*/
#define CUDA_CHECK(call)                                                      \
    do                                                                        \
    {                                                                         \
        cudaError_t err__ = (call);                                           \
        if (err__ != cudaSuccess)                                             \
        {                                                                     \
            std::fprintf(stderr,                                              \
                         "CUDA error at %s:%d: %s\n",                         \
                         __FILE__, __LINE__, cudaGetErrorString(err__));      \
            std::exit(EXIT_FAILURE);                                          \
        }                                                                     \
    } while (0)

// ============================================================================
// Device/Architecture Introspection and Compile-Guidance
// ============================================================================

/*
    print_device_info_and_compile_guidance:
    ---------------------------------------
    This function prints:
      - Basic device information
      - Compute capability (major.minor)
      - Guidance on how to compile for this architecture

    Its purpose is to help diagnose and avoid the error:
      "no kernel image is available for execution on the device"
*/
void print_device_info_and_compile_guidance()
{
    int deviceCount = 0;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess || deviceCount == 0)
    {
        std::fprintf(stderr,
                     "No CUDA-capable device detected or cudaGetDeviceCount failed (%s).\n",
                     cudaGetErrorString(err));
        std::fprintf(stderr,
                     "This program requires an NVIDIA GPU with CUDA support.\n");
        return;
    }

    std::printf("Detected %d CUDA-capable device(s).\n", deviceCount);

    for (int dev = 0; dev < deviceCount; ++dev)
    {
        cudaDeviceProp props{};
        CUDA_CHECK(cudaGetDeviceProperties(&props, dev));

        int ccMajor = props.major;
        int ccMinor = props.minor;
        int cc = ccMajor * 10 + ccMinor;

        std::printf("  Device %d: %s\n", dev, props.name);
        std::printf("    Compute Capability          : %d.%d (sm_%d)\n",
                    ccMajor, ccMinor, cc);
        std::printf("    Global Memory (MB)          : %.1f\n",
                    props.totalGlobalMem / (1024.0 * 1024.0));
        std::printf("    MultiProcessor Count (SMs)  : %d\n",
                    props.multiProcessorCount);
        std::printf("    Max Threads per Block       : %d\n",
                    props.maxThreadsPerBlock);

        /*
            Compilation guidance (comment only):

            For this device, a safe nvcc command looks like:

                nvcc -O3 -std=c++14 \
                     -gencode arch=compute_%d,code=sm_%d \
                     -gencode arch=compute_%d,code=compute_%d \
                     cuda_introduction_and_scalable_model.cu -o cuda_intro

            Where %d is replaced by the computed 'cc' above (e.g., 61, 75, 86).

            Explanation:
              - arch=compute_cc,code=sm_cc
                  → produces native SASS code for your GPU.
              - arch=compute_cc,code=compute_cc
                  → includes PTX for JIT on slightly newer GPUs.

            If your GPU is very old (e.g., compute capability < 5.0) and the
            CUDA toolkit you use no longer supports that architecture, you
            must:
              - Install an older CUDA toolkit that still supports your GPU, or
              - Use a newer GPU that is supported by your toolkit.
        */
    }

    std::printf("\n");
}

// ============================================================================
// SECTION 3.1 — THE BENEFITS OF USING GPUs
// ============================================================================

/*
    Conceptual Overview:
    --------------------
    A modern discrete NVIDIA GPU has:
        - Hundreds to tens of thousands of arithmetic units (CUDA cores)
        - Very high memory bandwidth (hundreds of GB/s)
        - Very high instruction throughput for data-parallel workloads

    A CPU, in contrast, devotes a much larger fraction of its transistor budget to:
        - Complex control logic (branch prediction, out-of-order execution)
        - Large, multi-level caches (L1/L2/L3)
        - Sophisticated speculation & prefetching mechanisms

    The GPU devotes more transistors to:
        - ALUs (floating-point & integer units)
        - Simple, massively parallel control for thousands of threads
        - Hardware schedulers that switch between warps to hide memory latency

    Result:
        - For workloads with abundant data parallelism (e.g., vector operations,
          image processing, deep learning, numerical simulation), the GPU can
          process many more operations per unit time and per unit energy than a CPU.

    Important nuance:
        - GPUs are not always faster:
            * For small data sizes (kernel launch + PCIe overhead dominate)
            * For workloads with strong sequential dependencies
            * For strongly branch-divergent control flow
            * For algorithms with low arithmetic intensity (too memory-bound)
        - But when a problem has enough parallelism and is implemented with
          proper memory access patterns and occupancy, the GPU can outperform
          CPUs by large factors.
*/

/*
    Example Workload: Vector Addition
    ---------------------------------
    We use vector addition as a canonical example to illustrate GPU benefits:

        C[i] = A[i] + B[i]  for i in [0, N)

    This is:
        - Embarrassingly parallel: each output element is independent
        - Memory bandwidth bound: performance depends on how quickly we can move
          data from global memory through the GPU

    On CPU:
        - A typical implementation processes elements in a small number of threads
          (equal to CPU core count, usually <= 64 on mainstream systems).
    On GPU:
        - Tens of thousands of threads can operate concurrently, each processing
          one or more elements, fully utilizing the memory bandwidth.
*/

// ----------------------------------------------------------------------------
// CPU implementation for conceptual comparison (sequential or lightly parallel)
// ----------------------------------------------------------------------------

/*
    cpu_vector_add:
    ---------------
    Simple CPU implementation of vector addition, using a single thread.

    This function is intentionally sequential to highlight the contrast with
    the massively parallel GPU kernel that follows.

    NOTE:
        In real CPU code you might use SIMD intrinsics or multi-threading,
        but even then the core-level parallelism is far smaller than what a GPU
        can expose, and the memory bandwidth is usually lower.
*/
void cpu_vector_add(const float* a, const float* b, float* c, int n)
{
    for (int i = 0; i < n; ++i)
    {
        c[i] = a[i] + b[i];
    }
}

// ----------------------------------------------------------------------------
// GPU implementation — massively parallel vector addition
// ----------------------------------------------------------------------------

/*
    gpu_vector_add_kernel:
    ----------------------
    Each thread is responsible for computing one (or more) elements of C.

    Key design choices:
        - Use __restrict__ to help the compiler assume non-aliasing pointers,
          allowing more aggressive optimizations.
        - Use a grid-stride loop: each thread processes multiple elements
          spaced by totalThreads, guaranteeing scalability for any problem size
          and any grid configuration.
*/
__global__ void gpu_vector_add_kernel(const float* __restrict__ a,
                                      const float* __restrict__ b,
                                      float* __restrict__ c,
                                      int n)
{
    // Compute a unique global thread ID
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    // Total number of threads in the grid
    int totalThreads = blockDim.x * gridDim.x;

    // Grid-stride loop:
    //   - Ensures full coverage of [0, n)
    //   - Automatically scales with gridDim.x and blockDim.x
    for (int i = globalThreadId; i < n; i += totalThreads)
    {
        c[i] = a[i] + b[i];
    }
}

/*
    run_gpu_benefits_demo:
    ----------------------
    Demonstrates the use of the GPU for a simple, yet representative,
    data-parallel workload, reflecting the benefits described in section 3.1.

    Steps:
        1. Allocate & initialize host data.
        2. Allocate device (GPU) memory.
        3. Copy data to GPU.
        4. Launch massively parallel kernel.
        5. Copy result back to CPU.
        6. Verify correctness.

    This pipeline is representative of a wide class of GPU-accelerated workloads.
*/
void run_gpu_benefits_demo(int n)
{
    std::printf("=== 3.1 Demonstration: GPU Benefits via Vector Addition (N = %d) ===\n", n);

    // 1. Allocate and initialize host buffers
    float* h_a     = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* h_b     = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* h_c_cpu = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* h_c_gpu = static_cast<float*>(std::malloc(n * sizeof(float)));

    if (!h_a || !h_b || !h_c_cpu || !h_c_gpu)
    {
        std::fprintf(stderr, "Host memory allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    // 2. Allocate device buffers
    float *d_a = nullptr, *d_b = nullptr, *d_c = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_c, n * sizeof(float)));

    // 3. Copy input data from host (CPU) to device (GPU)
    CUDA_CHECK(cudaMemcpy(d_a, h_a, n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, n * sizeof(float), cudaMemcpyHostToDevice));

    // 4. Launch GPU kernel with a configuration chosen to expose massive parallelism
    int blockSize = 256;
    int gridSize  = (n + blockSize - 1) / blockSize;

    gpu_vector_add_kernel<<<gridSize, blockSize>>>(d_a, d_b, d_c, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // 5. Copy result back to host
    CUDA_CHECK(cudaMemcpy(h_c_gpu, d_c, n * sizeof(float), cudaMemcpyDeviceToHost));

    // 6. Compute CPU reference result and verify correctness
    cpu_vector_add(h_a, h_b, h_c_cpu, n);

    int errorCount = 0;
    for (int i = 0; i < n; ++i)
    {
        if (std::fabs(h_c_cpu[i] - h_c_gpu[i]) > 1e-5f)
        {
            ++errorCount;
            if (errorCount < 10)
            {
                std::fprintf(stderr,
                             "Mismatch at index %d: CPU=%f, GPU=%f\n",
                             i, h_c_cpu[i], h_c_gpu[i]);
            }
        }
    }

    if (errorCount == 0)
    {
        std::printf("Vector addition: CPU and GPU results match (N = %d).\n", n);
    }
    else
    {
        std::fprintf(stderr, "Vector addition verification failed (%d mismatches).\n", errorCount);
    }

    // Clean up
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    std::free(h_a);
    std::free(h_b);
    std::free(h_c_cpu);
    std::free(h_c_gpu);

    std::printf("GPU benefits demo complete.\n\n");
}

/*
    Notes on Other Devices (FPGAs):
    -------------------------------
    FPGAs (Field Programmable Gate Arrays) can provide extremely high energy
    efficiency by customizing hardware exactly to an application. However:

        - They require hardware description languages (VHDL/Verilog) or
          high-level synthesis tools, which are harder and slower to use
          than CUDA C++.
        - The design/compile cycle is significantly longer.
        - Flexibility to adapt to rapidly changing algorithms (like deep learning
          models) is lower than with GPUs.

    GPUs, in contrast:
        - Support high-level languages (C/C++, Fortran, Python via bindings, etc.).
        - Offer a stable, well-documented programming model (CUDA).
        - Provide excellent performance while retaining reasonable programmability.
*/

// ============================================================================
// SECTION 3.2 — CUDA: A GENERAL-PURPOSE PARALLEL COMPUTING PLATFORM
// ============================================================================

/*
    CUDA as a Platform:
    -------------------
    CUDA is both:
        - A programming model: extensions to C/C++ (and bindings for other
          languages) that let you express massive parallelism naturally.
        - A parallel computing platform: the underlying hardware + driver +
          runtime + compiler stack that executes those parallel programs on
          NVIDIA GPUs.

    Key Concepts:
        - Host (CPU) and Device (GPU) memory spaces and execution domains.
        - Kernel functions (__global__) that run on the GPU.
        - Language extensions:
            * __global__  : GPU kernel, callable from host
            * __device__  : GPU-only helper function
            * __host__    : CPU-only function (implicit default)
            * __shared__  : On-chip shared memory for thread blocks
            * __constant__: Read-only cached memory, broadcast-friendly
        - Many language ecosystems:
            * CUDA C/C++
            * CUDA Fortran (via PGI/NVIDIA compilers)
            * High-level APIs / directives: OpenACC, OpenMP target offload
            * Graphics APIs: DirectCompute, CUDA interop with OpenGL/Vulkan
*/

/*
    Device function (helper) — only callable from other device or global funcs.
*/
__device__ float device_scale(float x, float factor)
{
    return x * factor;
}

/*
    kernel_scale_and_bias:
    ----------------------
    General-purpose example of a CUDA kernel implementing a simple
    element-wise transformation:

        out[i] = in[i] * scale + bias

    Demonstrates:
        - __global__ kernel definition
        - __device__ helper usage
        - Standard CUDA indexing pattern
*/
__global__ void kernel_scale_and_bias(const float* __restrict__ in,
                                      float* __restrict__ out,
                                      float scale,
                                      float bias,
                                      int n)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < n)
    {
        float val = device_scale(in[idx], scale);
        out[idx] = val + bias;
    }
}

/*
    run_cuda_platform_demo:
    -----------------------
    Demonstrates CUDA as a general-purpose platform with a custom kernel.
    The algorithm itself (scale + bias) is trivial, but the focus is on how
    the host and device interact in the CUDA programming model.
*/
void run_cuda_platform_demo(int n)
{
    std::printf("=== 3.2 Demonstration: CUDA as a General-Purpose Platform (N = %d) ===\n", n);

    // Host allocations
    float* h_in  = static_cast<float*>(std::malloc(n * sizeof(float)));
    float* h_out = static_cast<float*>(std::malloc(n * sizeof(float)));

    if (!h_in || !h_out)
    {
        std::fprintf(stderr, "Host memory allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; ++i)
    {
        h_in[i] = static_cast<float>(i);
    }

    // Device allocations
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out, n * sizeof(float)));

    // Transfer to device
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    // Kernel launch
    float scale    = 2.5f;
    float bias     = -1.0f;
    int   blockSize = 256;
    int   gridSize  = (n + blockSize - 1) / blockSize;

    kernel_scale_and_bias<<<gridSize, blockSize>>>(d_in, d_out, scale, bias, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Transfer result back
    CUDA_CHECK(cudaMemcpy(h_out, d_out, n * sizeof(float), cudaMemcpyDeviceToHost));

    // Quick sanity check
    int errorCount = 0;
    for (int i = 0; i < n; ++i)
    {
        float expected = h_in[i] * scale + bias;
        if (std::fabs(h_out[i] - expected) > 1e-5f)
        {
            ++errorCount;
            if (errorCount < 5)
            {
                std::fprintf(stderr,
                             "Mismatch at %d: got=%f, expected=%f\n",
                             i, h_out[i], expected);
            }
        }
    }

    if (errorCount == 0)
    {
        std::printf("CUDA platform demo: all results correct.\n\n");
    }
    else
    {
        std::fprintf(stderr,
                     "CUDA platform demo: %d mismatches detected.\n\n",
                     errorCount);
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    std::free(h_in);
    std::free(h_out);

    /*
        Note:
        -----
        In real applications, the same GPU kernels can be invoked from:
            - C/C++ code directly (as here),
            - Fortran (via CUDA Fortran),
            - Python (via libraries like PyCUDA, CuPy, PyTorch),
            - Directive-based models (OpenACC / OpenMP) that generate CUDA under
              the hood for NVIDIA GPUs.

        This ecosystem flexibility is a major reason CUDA is called a
        general-purpose parallel computing platform.
    */
}

// ============================================================================
// SECTION 3.3 — A SCALABLE PROGRAMMING MODEL
// ============================================================================

/*
    Scalability Challenges:
    -----------------------
    Modern CPUs are multicore; modern GPUs are manycore (tens to hundreds of
    Streaming Multiprocessors, SMs). Mainstream processors are inherently
    parallel systems.

    The key challenge:
        - Write applications that automatically scale with:
            * Different GPU models (varying SM counts, clock rates, memory sizes)
            * Future hardware generations (more SMs, different cache sizes)
        - Without rewriting algorithmic logic for each hardware variant.

    CUDA's Solution — Three Core Abstractions:
    -----------------------------------------
    1) Hierarchy of Thread Groups:
        - Thread       : smallest execution unit
        - Block (CTA)  : cooperative group of threads that share:
                          * Fast on-chip shared memory (__shared__)
                          * Synchronization primitives (__syncthreads)
        - Grid         : all blocks launched for a kernel

    2) Shared Memories:
        - Per-block on-chip memory, with low latency and high bandwidth.
        - Enables data reuse and cooperation within a block.

    3) Barrier Synchronization:
        - __syncthreads() provides a barrier at block scope.
        - All threads in a block must reach the barrier before any proceed.

    Automatic Scalability:
    ----------------------
    - A kernel is written assuming:
        * Blocks are independent units of work that may run in any order,
          on any SM, concurrently or sequentially.
    - That means:
        * The same binary kernel can run efficiently on GPUs with different
          numbers of SMs.
        * Only the runtime launch configuration (gridDim, blockDim) may differ.
*/

// ----------------------------------------------------------------------------
// Example: Parallel Block-Level Reduction (sum of an array)
// ----------------------------------------------------------------------------

/*
    block_reduce_sum_kernel:
    ------------------------
    Each block computes a partial sum of a subset of elements from 'in' and
    writes its partial result to 'blockSums[blockIdx.x]'.

    This demonstrates:
        - Thread hierarchy: threads within a block cooperate to reduce.
        - Shared memory: used to store intermediate results efficiently.
        - Barrier synchronization: __syncthreads to coordinate threads.
        - Scalability: any number of blocks can be launched; each is independent.

    Global sum is usually obtained by:
        - Launching a second reduction kernel on blockSums, or
        - Performing a final reduction on the CPU if the number of blocks is small.
*/
__global__ void block_reduce_sum_kernel(const float* __restrict__ in,
                                        float* __restrict__ blockSums,
                                        int n)
{
    extern __shared__ float sdata[];  // Shared memory allocation per block

    int tid        = threadIdx.x;
    int blockStart = blockIdx.x * blockDim.x * 2;
    int globalIdx  = blockStart + tid;

    // Each thread loads up to two elements (to improve utilization)
    float sum = 0.0f;
    if (globalIdx < n)
    {
        sum += in[globalIdx];
    }
    if (globalIdx + blockDim.x < n)
    {
        sum += in[globalIdx + blockDim.x];
    }

    // Store partial sum in shared memory
    sdata[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory (power-of-two sized blockDim assumed)
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (tid < stride)
        {
            sdata[tid] += sdata[tid + stride];
        }
        __syncthreads();
    }

    // The first thread in the block writes the block's partial result
    if (tid == 0)
    {
        blockSums[blockIdx.x] = sdata[0];
    }
}

/*
    run_scalable_model_demo:
    ------------------------
    Demonstrates the scalable CUDA programming model by:

        1. Querying GPU device properties (especially SM count).
        2. Using occupancy calculations (or simple heuristics) to assign
           a number of blocks that scales with hardware capabilities.
        3. Launching a reduction kernel where each block is an independent
           unit of work that can run on any SM in any order.

    Key takeaway:
        - Kernel code doesn't depend on number of SMs.
        - Only host-side launch parameters adapt to the hardware.
*/
void run_scalable_model_demo(int n)
{
    std::printf("=== 3.3 Demonstration: Scalable Programming Model (N = %d) ===\n", n);

    // Step 1: Query device properties for the currently selected device
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));

    cudaDeviceProp props{};
    CUDA_CHECK(cudaGetDeviceProperties(&props, device));

    std::printf("Running on device %d: %s\n", device, props.name);
    std::printf("  SM count                         : %d\n", props.multiProcessorCount);
    std::printf("  Max threads per block            : %d\n", props.maxThreadsPerBlock);
    std::printf("  Shared memory per block (bytes)  : %zu\n", props.sharedMemPerBlock);

    // Step 2: Allocate and initialize input
    float* h_in = static_cast<float*>(std::malloc(n * sizeof(float)));
    if (!h_in)
    {
        std::fprintf(stderr, "Host memory allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    for (int i = 0; i < n; ++i)
    {
        h_in[i] = 1.0f;  // Chosen so true sum = n (easy verification)
    }

    float* d_in = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, h_in, n * sizeof(float), cudaMemcpyHostToDevice));

    // Step 3: Decide block size and grid size
    int blockSize        = 256;  // power of two required by this reduction pattern
    int elementsPerBlock = blockSize * 2;
    int gridSize         = (n + elementsPerBlock - 1) / elementsPerBlock;

    /*
        Scalability note:
        -----------------
        We could also use cudaOccupancyMaxActiveBlocksPerMultiprocessor to
        compute a grid size that saturates the GPU, e.g.:

            int maxBlocksPerSM = 0;
            cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &maxBlocksPerSM,
                block_reduce_sum_kernel,
                blockSize,
                blockSize * sizeof(float));  // shared memory size

            gridSize = maxBlocksPerSM * props.multiProcessorCount;

        Here we use a simpler formula for clarity, but the principle is the same.
    */

    float* d_blockSums = nullptr;
    CUDA_CHECK(cudaMalloc(&d_blockSums, gridSize * sizeof(float)));

    size_t sharedMemSize = blockSize * sizeof(float);

    block_reduce_sum_kernel<<<gridSize, blockSize, sharedMemSize>>>(d_in, d_blockSums, n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Step 4: Final reduction on host (gridSize is usually small after first pass)
    float* h_blockSums = static_cast<float*>(std::malloc(gridSize * sizeof(float)));
    if (!h_blockSums)
    {
        std::fprintf(stderr, "Host memory allocation failed\n");
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(h_blockSums,
                          d_blockSums,
                          gridSize * sizeof(float),
                          cudaMemcpyDeviceToHost));

    double finalSum = 0.0;
    for (int i = 0; i < gridSize; ++i)
    {
        finalSum += static_cast<double>(h_blockSums[i]);
    }

    std::printf("Computed total sum on GPU+CPU: %.2f  (expected: %d)\n",
                finalSum, n);

    if (std::fabs(finalSum - static_cast<double>(n)) < 1e-3)
    {
        std::printf("Scalable reduction: result correct.\n");
    }
    else
    {
        std::fprintf(stderr,
                     "Scalable reduction: result incorrect (difference = %f).\n",
                     std::fabs(finalSum - static_cast<double>(n)));
    }

    // Cleanup
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_blockSums));
    std::free(h_in);
    std::free(h_blockSums);

    std::printf("Scalable model demo complete.\n\n");

    /*
        Summary of Scalability:
        -----------------------
        - The reduction kernel doesn't know how many SMs the GPU has.
        - Each thread block operates independently on its own subset of data.
        - The runtime can schedule blocks across SMs in any order:
              * Concurrently, if enough hardware resources are available.
              * Sequentially, if the GPU has few SMs or is under heavy load.
        - The same binary program runs efficiently on:
              * Small mobile GPUs (few SMs)
              * Datacenter GPUs (80+ SMs, multiple GPU boards)
        - Only launch configuration parameters (gridSize, blockSize) need to be
          tuned to the particular hardware; the algorithm remains unchanged.

        This is the essence of CUDA's scalable programming model.
    */
}

// ============================================================================
// main – Entry point tying all sections together
// ============================================================================

int main()
{
    // Print device information and compile guidance before anything else.
    // This is especially useful if the user encounters:
    //   "no kernel image is available for execution on the device"
    print_device_info_and_compile_guidance();

    // Select device 0 by default; in multi-GPU systems, you may want to select
    // a specific device based on properties or environment variables.
    CUDA_CHECK(cudaSetDevice(0));

    // 3.1 — Benefits of Using GPUs
    run_gpu_benefits_demo(1 << 20);  // 1,048,576 elements

    // 3.2 — CUDA as a General-Purpose Parallel Computing Platform
    run_cuda_platform_demo(1 << 20);

    // 3.3 — Scalable Programming Model
    run_scalable_model_demo(1 << 20);

    CUDA_CHECK(cudaDeviceReset());
    return 0;
}

/*
    Compilation (examples):
    -----------------------

    1) Single-architecture build (for exactly one known GPU):

        nvcc -O3 -std=c++14 \
             -gencode arch=compute_75,code=sm_75 \
             cuda_introduction_and_scalable_model.cu -o cuda_intro

       (replace 75 with your GPU's compute capability * 10, e.g., 61, 86, 89)

    2) More portable build including PTX for JIT on close future GPUs:

        nvcc -O3 -std=c++14 \
             -gencode arch=compute_75,code=sm_75 \
             -gencode arch=compute_75,code=compute_75 \
             cuda_introduction_and_scalable_model.cu -o cuda_intro

    If you still see the error:
        "no kernel image is available for execution on the device"

    after compiling with a matching arch=compute_X,code=sm_X for your GPU:

        - Your GPU may be older than the minimum architecture supported by
          your installed CUDA toolkit. In that case you must install an older
          toolkit which supports your GPU's compute capability, or upgrade
          your GPU hardware.
*/