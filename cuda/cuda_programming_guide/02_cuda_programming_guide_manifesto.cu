/**
 * @file 02_cuda_programming_guide_manifesto.cu
 * @brief The Living Implementation of the CUDA C Programming Guide.

 *
 * =================================================================================================
 *                  TOPIC: WHAT IS THE CUDA C PROGRAMMING GUIDE?
 * =================================================================================================
 *
 * 1. DEFINITION:
 *    The CUDA C Programming Guide is the bible of GPU development. It is the definitive document
 *    that describes the heterogenous programming model, the memory hierarchy, and the C/C++ 
 *    language extensions required to instruct the GPU.
 *
 * 2. CORE PILLARS COVERED BY THE GUIDE (AND THIS CODE):
 *    
 *    A. HETEROGENEOUS COMPUTING:
 *       The Guide defines a system composed of a Host (CPU) and a Device (GPU). The Host controls
 *       application flow, while the Device executes parallel compute kernels.
 *
 *    B. THE KERNEL & THREAD HIERARCHY:
 *       It introduces the concept of a "Kernel"—a function running on the GPU.
 *       - Grid: The entire problem space.
 *       - Block: A distinct chunk of work, executed on a Streaming Multiprocessor (SM).
 *       - Thread: The smallest execution unit.
 *       
 *    C. MEMORY HIERARCHY:
 *       The Guide details the distinct memory spaces, crucial for performance:
 *       - Global Memory: Large, slow, off-chip (DRAM).
 *       - Shared Memory: Small, fast, on-chip (L1-like user-managed cache).
 *       - Constant/Texture Memory: Read-only specialized caches.
 *
 *    D. SYNCHRONIZATION:
 *       How threads coordinate safely (e.g., __syncthreads(), Atomic Operations).
 *
 * =================================================================================================
 *                                      COMPILER INSTRUCTIONS
 * =================================================================================================
 * The Guide explains that source files are compiled by `nvcc`.
 * `nvcc` separates device code (PTX/SASS) from host code (x86/ARM assembly).
 *
 * Command: nvcc -O3 -arch=sm_80 02_cuda_programming_guide_manifesto.cu -o cuda_guide_demo
 */

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

/*
 * =================================================================================================
 *                              BEST PRACTICE: ERROR HANDLING
 * =================================================================================================
 * The Programming Guide explicitly warns that most CUDA API calls return an error code.
 * A robust application must check these codes. Unchecked errors lead to "sticky" failure states.
 */
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA Error: %s (Code: %d) at %s:%d\n", \
                    cudaGetErrorString(err), err, __FILE__, __LINE__); \
            exit(EXIT_FAILURE); \
        } \
    } while (0)

/*
 * =================================================================================================
 *                          LANGUAGE EXTENSIONS: FUNCTION QUALIFIERS
 * =================================================================================================
 * The Guide introduces specific qualifiers to designate where functions execute.
 * 
 * __global__: Called from Host, Executes on Device (The entry point).
 * __device__: Called from Device, Executes on Device (Helper functions).
 * __host__:   Called from Host, Executes on Host (Standard C functions).
 */

// Size of the static shared memory buffer (defined at compile time for this example)
#define BLOCK_SIZE 256

/**
 * @brief Device helper function to compute a value.
 *        Demonstrates `__device__` qualifier.
 */
__device__ float compute_transformation(float val) {
    // Intrinsic function from the guide: fast math operations
    return __fmaf_rn(val, 2.0f, 1.0f); // (val * 2.0) + 1.0
}

/**
 * @brief A Kernel demonstrating the Memory Hierarchy and Synchronization.
 * 
 * CONCEPT: 1D Stencil / Sliding Window.
 * We load data from Global Memory into Shared Memory to minimize high-latency DRAM accesses.
 * We verify the integrity of the thread block using synchronization.
 *
 * @param input  Pointer to input data in Global Memory.
 * @param output Pointer to output data in Global Memory.
 * @param n      Total number of elements.
 */
__global__ void shared_mem_stencil_kernel(const float* __restrict__ input, 
                                          float* __restrict__ output, 
                                          int n) {
    
    /*
     * 1. THREAD ADDRESSING (The Coordinate System)
     * The Guide explains that unique thread identification is derived from built-in variables.
     */
    int tid = threadIdx.x;                       // Local ID within the block
    int gid = blockIdx.x * blockDim.x + tid;     // Global ID across the grid

    /*
     * 2. SHARED MEMORY DECLARATION
     * The `__shared__` qualifier allocates memory on the SM (Streaming Multiprocessor).
     * This memory is visible to all threads in the SAME block.
     * It acts as a user-managed L1 cache.
     */
    __shared__ float s_data[BLOCK_SIZE];

    /*
     * 3. LOAD PHASE (Global -> Shared)
     * Each thread loads one element from Global Memory into Shared Memory.
     * This is a classic coalesced memory access pattern recommended by the Guide.
     */
    if (gid < n) {
        s_data[tid] = input[gid];
    } else {
        s_data[tid] = 0.0f; // Padding for out-of-bounds threads
    }

    /*
     * 4. SYNCHRONIZATION BARRIER
     * `__syncthreads()` is a barrier. No thread in this block can proceed past this point
     * until ALL threads in the block have reached it.
     * This ensures `s_data` is fully populated before we read neighbors.
     */
    __syncthreads();

    /*
     * 5. COMPUTATION PHASE
     * Perform calculation using data from fast Shared Memory.
     * Note: This simple example just transforms the value. In a real stencil, 
     * we would access s_data[tid-1] or s_data[tid+1].
     */
    if (gid < n) {
        float val = s_data[tid];
        output[gid] = compute_transformation(val);
    }
    
    // No __syncthreads() needed at the end if we don't write to shared memory again.
}

/*
 * =================================================================================================
 *                                          HOST CODE
 * =================================================================================================
 * The Host code initializes the environment, manages memory, and launches the kernels.
 */
int main(void) {
    printf("=== CUDA C PROGRAMMING GUIDE: IMPLEMENTATION DEMO ===\n");

    // --- 1. DEVICE MANAGEMENT ---
    // The Guide recommends querying device properties to optimize grid dimensions.
    int device_id = 0;
    cudaDeviceProp props;
    CUDA_CHECK(cudaSetDevice(device_id));
    CUDA_CHECK(cudaGetDeviceProperties(&props, device_id));
    
    printf("Target Device: %s\n", props.name);
    printf("SM Count: %d\n", props.multiProcessorCount);
    printf("Max Threads per Block: %d\n", props.maxThreadsPerBlock);

    // --- 2. MEMORY ALLOCATION ---
    // Define problem size
    const int N = 1 << 22; // 4 Million elements
    size_t bytes = N * sizeof(float);

    float *h_in, *h_out;
    float *d_in, *d_out;

    // Host Allocation (Pinned memory for faster PCIe transfer is preferred over malloc)
    CUDA_CHECK(cudaMallocHost((void**)&h_in, bytes));
    CUDA_CHECK(cudaMallocHost((void**)&h_out, bytes));

    // Initialize Host Data
    for (int i = 0; i < N; i++) {
        h_in[i] = (float)i;
    }

    // Device Allocation (Linear Memory in DRAM)
    CUDA_CHECK(cudaMalloc((void**)&d_in, bytes));
    CUDA_CHECK(cudaMalloc((void**)&d_out, bytes));

    // --- 3. DATA TRANSFER (Host -> Device) ---
    // Moving data from system RAM to GPU VRAM.
    CUDA_CHECK(cudaMemcpy(d_in, h_in, bytes, cudaMemcpyHostToDevice));

    // --- 4. EXECUTION CONFIGURATION ---
    // Calculating Grid and Block dimensions.
    // Ideally, Block Size is a multiple of Warp Size (32). 256 is a sweet spot.
    dim3 blockSize(BLOCK_SIZE);
    dim3 gridSize((N + blockSize.x - 1) / blockSize.x);

    printf("Launching Kernel: Grid(%d), Block(%d)\n", gridSize.x, blockSize.x);

    // --- 5. KERNEL LAUNCH ---
    // The triple chevron syntax <<< >>> is the signature of CUDA C extension.
    shared_mem_stencil_kernel<<<gridSize, blockSize>>>(d_in, d_out, N);

    // --- 6. ERROR CHECKING (Asynchronous) ---
    // Kernel launches return immediately. We check if the launch parameters were valid.
    CUDA_CHECK(cudaPeekAtLastError());
    
    // Wait for GPU to finish (Synchronization between Host and Device)
    CUDA_CHECK(cudaDeviceSynchronize());

    // --- 7. DATA TRANSFER (Device -> Host) ---
    CUDA_CHECK(cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost));

    // --- 8. VERIFICATION ---
    // Always verify the result on the CPU to ensure algorithm correctness.
    bool success = true;
    for (int i = 0; i < N; i++) {
        float input_val = (float)i;
        float expected = (input_val * 2.0f) + 1.0f; // Matches compute_transformation
        
        if (fabs(h_out[i] - expected) > 1e-5) {
            printf("Verification FAILED at index %d: Expected %f, Got %f\n", i, expected, h_out[i]);
            success = false;
            break;
        }
    }

    if (success) {
        printf("SUCCESS: GPU execution matches Programming Guide specifications.\n");
    }

    // --- 9. RESOURCE CLEANUP ---
    // The Guide emphasizes proper resource destruction to prevent memory leaks.
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFreeHost(h_in));
    CUDA_CHECK(cudaFreeHost(h_out));

    return 0;
}