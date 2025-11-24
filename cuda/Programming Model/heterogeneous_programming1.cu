/**
 * ==================================================================================================
 * FILE: heterogeneous_programming.cu
 * AUTHOR: Best Coder in the World (AI Tutor)
 * PLATFORM: NVIDIA A100 (Compute Capability 8.0)
 * TOPIC: 5.4 Heterogeneous Programming & Memory Models
 * ==================================================================================================
 * 
 * INTRODUCTION
 * --------------------------------------------------------------------------------------------------
 * CUDA is a Heterogeneous Programming Model. This means the system consists of two distinct parts:
 * 
 * 1. THE HOST (CPU): 
 *    - The "Commander".
 *    - Runs the main C++ application.
 *    - Manages resources, memory allocation, and schedules tasks (kernels) for the GPU.
 *    - Has its own DRAM (System RAM), referred to as "Host Memory".
 * 
 * 2. THE DEVICE (GPU - A100):
 *    - The "Co-processor" / "Worker".
 *    - Executes parallel code (Kernels).
 *    - Has its own High-Bandwidth Memory (HBM2e on A100), referred to as "Device Memory".
 * 
 * THE CHALLENGE:
 * Because Host and Device have physically separate memory spaces connected by a bus (PCIe or NVLink),
 * the programmer must explicitly manage data movement. The CPU cannot read VRAM directly (fast enough),
 * and the GPU cannot read System RAM directly (without significant latency), in the traditional model.
 * 
 * WE WILL COVER:
 * Part A: Traditional Heterogeneous Model (Manual Memory Management & Data Transfer).
 * Part B: Unified Memory Model (Automatic Memory Management / Managed Memory).
 * 
 * ==================================================================================================
 */

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <cassert>
#include <cmath>

// ==================================================================================================
// ERROR HANDLING MACROS
// ==================================================================================================
// As the best coder, we never ignore return codes. We catch errors immediately.
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (Line: %d)\n", cudaGetErrorString(err), __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

// ==================================================================================================
// CONSTANTS
// ==================================================================================================
// Using a large vector size to justify GPU usage.
// 2^24 elements * 4 bytes = 64MB per vector.
const int N = 1 << 24; 

// ==================================================================================================
// KERNEL DEFINITION (The Code Running on the Device)
// ==================================================================================================
/**
 * Vector Addition Kernel.
 * 
 * @param A  Input vector A (Device Pointer)
 * @param B  Input vector B (Device Pointer)
 * @param C  Output vector C (Device Pointer)
 * @param n  Number of elements
 * 
 * note: We use __restrict__ to tell the compiler that pointers do not alias.
 * This allows the A100 compiler to optimize memory load/store instructions.
 */
__global__ void vectorAdd(const float* __restrict__ A, 
                          const float* __restrict__ B, 
                          float* __restrict__ C, 
                          int n) 
{
    // Calculate global index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary check: Ensure we don't access memory outside the array
    if (idx < n) {
        // The core computation
        C[idx] = A[idx] + B[idx];
    }
}

// ==================================================================================================
// PART A: TRADITIONAL HETEROGENEOUS PROGRAMMING
// ==================================================================================================
/**
 * This function demonstrates the classic workflow:
 * 1. Allocate Host Memory (CPU RAM).
 * 2. Allocate Device Memory (GPU VRAM).
 * 3. Copy Data Host -> Device (PCIe Bus).
 * 4. Launch Kernel.
 * 5. Copy Data Device -> Host (PCIe Bus).
 * 6. Free Memory.
 */
void runTraditionalModel() {
    std::cout << "\n[Part A] Running Traditional Heterogeneous Model..." << std::endl;

    size_t size = N * sizeof(float);

    // 1. Allocate Host Memory
    // Standard C++ allocation. This resides in System RAM.
    float *h_A = (float*)malloc(size);
    float *h_B = (float*)malloc(size);
    float *h_C = (float*)malloc(size);

    // Initialize Host Data
    for (int i = 0; i < N; ++i) {
        h_A[i] = 1.0f;
        h_B[i] = 2.0f;
    }

    // 2. Allocate Device Memory
    // We use cudaMalloc. These pointers (d_A, d_B, d_C) point to addresses in A100 HBM.
    // The Host (CPU) cannot dereference these pointers directly!
    float *d_A, *d_B, *d_C;
    CUDA_CHECK(cudaMalloc((void**)&d_A, size));
    CUDA_CHECK(cudaMalloc((void**)&d_B, size));
    CUDA_CHECK(cudaMalloc((void**)&d_C, size));

    // 3. Transfer Data: Host -> Device
    // This is often the bottleneck in heterogeneous computing.
    // We move data across the PCIe bus (or NVLink if available).
    // cudaMemcpy is a synchronous call (blocks CPU until copy finishes).
    std::cout << "  - Copying data to Device (H->D)..." << std::endl;
    CUDA_CHECK(cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice));

    // 4. Launch Kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "  - Launching Kernel..." << std::endl;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // Check for launch errors
    CUDA_CHECK(cudaGetLastError());

    // 5. Transfer Data: Device -> Host
    // Get the results back to CPU RAM so we can read them.
    std::cout << "  - Copying results to Host (D->H)..." << std::endl;
    CUDA_CHECK(cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost));

    // Verify
    bool correct = true;
    // Check first few elements to save time
    for (int i = 0; i < 100; i++) {
        if (fabs(h_C[i] - 3.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "  - Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;

    // 6. Free Memory
    // Critical: Must free Device memory with cudaFree and Host memory with free.
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    free(h_A);
    free(h_B);
    free(h_C);
}

// ==================================================================================================
// PART B: UNIFIED MEMORY (MANAGED MEMORY)
// ==================================================================================================
/**
 * Unified Memory (UM) creates a single address space accessible from both CPU and GPU.
 * Under the hood, the driver migrates pages of memory to where they are being accessed.
 * 
 * Advantages:
 * - Simplifies code (No explicit cudaMemcpy needed).
 * - Enables "oversubscription" (Allocating more memory than GPU VRAM, spilling to System RAM).
 * - Deep Copy simplification.
 * 
 * A100 Optimization:
 * On A100, we have hardware support for Page Faults, but for maximum performance, 
 * we use `cudaMemPrefetchAsync` to move data proactively, avoiding the latency of faults.
 */
void runUnifiedMemoryModel() {
    std::cout << "\n[Part B] Running Unified Memory Model..." << std::endl;

    int deviceId = 0;
    CUDA_CHECK(cudaGetDevice(&deviceId));
    size_t size = N * sizeof(float);

    // 1. Allocate Managed Memory
    // accessible by both CPU and GPU.
    float *m_A, *m_B, *m_C;
    CUDA_CHECK(cudaMallocManaged(&m_A, size));
    CUDA_CHECK(cudaMallocManaged(&m_B, size));
    CUDA_CHECK(cudaMallocManaged(&m_C, size));

    // 2. CPU Access (Initialization)
    // When the CPU writes to this memory, the pages are migrated to CPU RAM (if not already there).
    std::cout << "  - Initializing data on Host..." << std::endl;
    for (int i = 0; i < N; ++i) {
        m_A[i] = 1.0f;
        m_B[i] = 2.0f;
    }

    // 3. Data Migration (Prefetching) - BEST PRACTICE FOR A100
    // Although UM handles migration automatically via page faults, "Best Coder" style
    // dictates we give the driver a hint to move data to the GPU *before* the kernel starts.
    // This maximizes bandwidth utilization.
    std::cout << "  - Prefetching data to A100 Device..." << std::endl;
    CUDA_CHECK(cudaMemPrefetchAsync(m_A, size, deviceId, NULL)); // Stream NULL
    CUDA_CHECK(cudaMemPrefetchAsync(m_B, size, deviceId, NULL));
    // We also prefetch C so the GPU doesn't fault when writing to it
    CUDA_CHECK(cudaMemPrefetchAsync(m_C, size, deviceId, NULL));

    // 4. Launch Kernel
    // Notice: We pass the same pointers we used on the CPU!
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    std::cout << "  - Launching Kernel..." << std::endl;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(m_A, m_B, m_C, N);

    // 5. Synchronization
    // In the traditional model, cudaMemcpy acted as a sync point.
    // In UM, the kernel launch is async. We MUST synchronize before the CPU reads the data.
    CUDA_CHECK(cudaDeviceSynchronize());

    // 6. CPU Access (Verification)
    // Accessing m_C on CPU will trigger page migration back to Host RAM.
    // Alternatively, we could prefetch back to cudaCpuDeviceId.
    std::cout << "  - Verifying on Host..." << std::endl;
    
    bool correct = true;
    for (int i = 0; i < 100; i++) {
        if (fabs(m_C[i] - 3.0f) > 1e-5) {
            correct = false;
            break;
        }
    }
    std::cout << "  - Verification: " << (correct ? "PASSED" : "FAILED") << std::endl;

    // 7. Free Memory
    CUDA_CHECK(cudaFree(m_A));
    CUDA_CHECK(cudaFree(m_B));
    CUDA_CHECK(cudaFree(m_C));
}

// ==================================================================================================
// MAIN EXECUTION
// ==================================================================================================
int main() {
    std::cout << "=============================================================" << std::endl;
    std::cout << "   HETEROGENEOUS PROGRAMMING MASTERCLASS (A100)              " << std::endl;
    std::cout << "=============================================================" << std::endl;

    // Query Device properties to confirm A100 environment
    int dev;
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDevice(&dev));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
    
    std::cout << "Device: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    
    if (prop.major < 8) {
        std::cout << "WARNING: This code is optimized for A100 (Ampere) or newer." << std::endl;
    }

    // Run Part A
    runTraditionalModel();

    // Run Part B
    runUnifiedMemoryModel();

    std::cout << "\nSuccess. End of Heterogeneous Programming Demonstration." << std::endl;

    return 0;
}