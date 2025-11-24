/**
 * @file heterogeneous_programming.cu
 * @brief Masterclass on CUDA Heterogeneous Programming Models.
 *
 * @details

 * THEORETICAL FRAMEWORK:
 *
 * 1. THE HETEROGENEOUS COMPUTING PARADIGM:
 *    CUDA architecture treats the system as two distinct entities:
 *    - THE HOST (CPU): The commander. Manages environment, data, and logic.
 *    - THE DEVICE (GPU): The coprocessor. massive parallel throughput engine.
 *    
 *    Crucially, these two entities sit on physically separate hardware (connected via PCIe or NVLink).
 *    Consequently, they maintain physically separate DRAM memory spaces.
 *
 * 2. THE CLASSICAL MEMORY MODEL (DISCRETE):
 *    - Host RAM vs Device VRAM.
 *    - Data must be explicitly allocated on both sides.
 *    - Data must be explicitly copied across the bus (High Latency/Bandwidth Bottleneck).
 *    - API: cudaMalloc, cudaMemcpy, cudaFree.
 *
 * 3. THE UNIFIED MEMORY MODEL (MANAGED):
 *    - Virtualized bridging of Host and Device memory.
 *    - Single pointer accessible by both CPU and GPU.
 *    - The Driver/OS handles page faults and migrates pages to the processor accessing the data.
 *    - API: cudaMallocManaged.
 *
 * --------------------------------------------------------------------------------------------
 */

#include <cuda_runtime.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>

// ============================================================================================
// ERROR HANDLING UTILITY (Best Practice)
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

// Data size configuration
const int N = 1 << 20; // 1 Million elements (~4MB float)
const int BLOCK_SIZE = 256;

// ============================================================================================
// KERNEL DEFINITION
// ============================================================================================
/**
 * @brief Simple SAXPY-like kernel: Output = Input * Scale
 * 
 * @note
 * This kernel is agnostic to the memory model (Discrete vs Unified).
 * It simply expects pointers to accessible memory addresses.
 * 
 * @param input  Pointer to input data (Device accessible)
 * @param output Pointer to output data (Device accessible)
 * @param scale  Scalar multiplier
 * @param n      Number of elements
 */
__global__ void vectorScaleKernel(const float* __restrict__ input, 
                                  float* __restrict__ output, 
                                  float scale, 
                                  int n)
{
    // Global Thread Index Calculation
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Boundary Guard
    if (idx < n)
    {
        // Perform computation
        output[idx] = input[idx] * scale;
    }
}

// ============================================================================================
// APPROACH 1: CLASSICAL HETEROGENEOUS PROGRAMMING (Explicit Memory Management)
// ============================================================================================
/**
 * @brief Demonstrates the manual management of separate memory spaces.
 * 
 * STEPS:
 * 1. Allocate Host Memory (System RAM).
 * 2. Allocate Device Memory (GPU VRAM).
 * 3. Initialize Host Data.
 * 4. COPY Host -> Device (PCIe transfer).
 * 5. Launch Kernel.
 * 6. COPY Device -> Host (PCIe transfer).
 * 7. Verify & Free.
 */
void demonstrateDiscreteMemoryModel()
{
    printf("\n=== [1] Classical Heterogeneous Model (Discrete Memory) ===\n");

    size_t bytes = N * sizeof(float);

    // 1. Host Allocation
    // We use standard C malloc. For higher performance, one would use cudaMallocHost (Pinned Memory).
    float *h_input  = (float*)malloc(bytes);
    float *h_output = (float*)malloc(bytes);

    // 2. Device Allocation
    // These pointers are ONLY valid on the GPU. Dereferencing them on CPU causes segfault.
    float *d_input, *d_output;
    gpuErrchk(cudaMalloc((void**)&d_input, bytes));
    gpuErrchk(cudaMalloc((void**)&d_output, bytes));

    // 3. Initialization (Host Side)
    for (int i = 0; i < N; ++i) h_input[i] = 1.0f;

    // 4. Data Transfer: Host -> Device
    // This is a synchronous blocking call (unless using Async streams).
    // The CPU waits here until data is fully on the GPU.
    gpuErrchk(cudaMemcpy(d_input, h_input, bytes, cudaMemcpyHostToDevice));

    // 5. Kernel Launch
    // The CPU simply queues the launch and moves on (Asynchronous).
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);
    
    vectorScaleKernel<<<grid, block>>>(d_input, d_output, 2.0f, N);
    
    // Check for launch errors
    gpuErrchk(cudaPeekAtLastError());

    // 6. Data Transfer: Device -> Host
    // This implicitly synchronizes the CPU with the GPU, as we need the result before proceeding.
    gpuErrchk(cudaMemcpy(h_output, d_output, bytes, cudaMemcpyDeviceToHost));

    // 7. Verification
    // Verify a few elements
    bool success = true;
    for (int i = 0; i < 10; ++i) { // Check first 10
        if (h_output[i] != 2.0f) {
            success = false;
            printf("Mismatch at index %d: %f\n", i, h_output[i]);
            break;
        }
    }
    if(success) printf("Verification Success: Discrete Memory logic valid.\n");

    // 8. Cleanup
    // Separate free calls for separate memory spaces.
    free(h_input);
    free(h_output);
    gpuErrchk(cudaFree(d_input));
    gpuErrchk(cudaFree(d_output));
}

// ============================================================================================
// APPROACH 2: UNIFIED MEMORY PROGRAMMING (Managed Memory)
// ============================================================================================
/**
 * @brief Demonstrates Unified Memory (UM).
 * 
 * CONCEPTS:
 * - Managed Memory creates a "Single Pointer" illusion.
 * - The data exists physically in either Host RAM or Device VRAM.
 * - The CUDA Driver + OS handles "Page Migration" on demand (Page Faulting).
 * 
 * STEPS:
 * 1. Allocate Managed Memory (visible to CPU & GPU).
 * 2. Access on CPU (Driver moves pages to CPU).
 * 3. Launch Kernel.
 * 4. Access on GPU (Driver moves pages to GPU).
 * 5. Synchronize (Crucial!).
 * 6. Access on CPU (Driver moves pages back to CPU).
 */
void demonstrateUnifiedMemoryModel()
{
    printf("\n=== [2] Unified Memory Model (Managed Memory) ===\n");

    size_t bytes = N * sizeof(float);

    // 1. Allocate Managed Memory
    // Note: No distinct host vs device pointers. Just one pointer.
    float *m_input, *m_output;
    gpuErrchk(cudaMallocManaged(&m_input, bytes));
    gpuErrchk(cudaMallocManaged(&m_output, bytes));

    // 2. Initialization (CPU Access)
    // The CPU writes to the pointer. This triggers page faults, populating Host RAM.
    for (int i = 0; i < N; ++i) m_input[i] = 1.0f;

    // Optimization Hint (IQ 300+):
    // By default, UM migrates on demand (faults). For max performance, we can 
    // "Prefetch" data to the GPU before the kernel starts to avoid runtime stall faults.
    // int deviceId; cudaGetDevice(&deviceId);
    // cudaMemPrefetchAsync(m_input, bytes, deviceId);

    // 3. Kernel Launch
    dim3 block(BLOCK_SIZE);
    dim3 grid((N + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // We pass the SAME pointers used by the CPU directly to the GPU kernel.
    vectorScaleKernel<<<grid, block>>>(m_input, m_output, 2.0f, N);
    gpuErrchk(cudaPeekAtLastError());

    // 4. Synchronization (CRITICAL)
    // In Discrete model, cudaMemcpy acted as a sync point.
    // In Unified model, the kernel runs async. If CPU accesses data immediately, 
    // it might read stale data or cause a race condition (segfault on some systems).
    // We MUST explicitly wait for GPU to finish.
    gpuErrchk(cudaDeviceSynchronize());

    // 5. Verification (CPU Access)
    // Accessing m_output on CPU triggers migration of result pages from GPU back to CPU.
    bool success = true;
    for (int i = 0; i < 10; ++i) {
        if (m_output[i] != 2.0f) {
            success = false;
            printf("Mismatch at index %d: %f\n", i, m_output[i]);
            break;
        }
    }
    if(success) printf("Verification Success: Unified Memory logic valid.\n");

    // 6. Cleanup
    gpuErrchk(cudaFree(m_input));
    gpuErrchk(cudaFree(m_output));
}

// ============================================================================================
// MAIN ENTRY POINT
// ============================================================================================
int main()
{
    printf("Heterogeneous Programming Demonstration\n");
    printf("---------------------------------------\n");

    // Verify we have a CUDA capable device
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    if (deviceCount == 0) {
        fprintf(stderr, "Error: No CUDA capable devices found.\n");
        return 1;
    }

    // Run The Classical Approach
    demonstrateDiscreteMemoryModel();

    // Run The Modern Unified Approach
    demonstrateUnifiedMemoryModel();

    printf("\nDemonstration Complete.\n");
    return 0;
}