/*
    File: programming_interface_demo.cu

    This single CUDA C++ source file demonstrates the CUDA Programming Interface in depth.

    Topics covered (all explanations are in comments inside this file, as requested):

    1. CUDA C++ language extensions:
       - Function qualifiers: __global__, __device__, __host__
       - Built-in variables: blockIdx, threadIdx, blockDim, gridDim
       - Kernel launch syntax: <<<gridDim, blockDim, sharedMemBytes, stream>>>
       - Using dim3 for multi-dimensional grids and blocks
       - Use of __restrict__ to express pointer aliasing assumptions

    2. CUDA Runtime API (high-level C/C++ API):
       - Device enumeration and selection
       - Device properties query
       - Device memory allocation / deallocation (cudaMalloc / cudaFree)
       - Host <-> device data transfers (cudaMemcpy)
       - Stream-based asynchronous execution and synchronization
       - Error handling and propagation
       - Simple kernel launches using the runtime

    3. CUDA Driver API (low-level C API):
       - Explicit initialization (cuInit)
       - Explicit context management (cuCtxCreate / cuCtxDestroy)
       - Accessing the context implicitly created by the runtime (cuCtxGetCurrent)
       - Identifying the device bound to a context (cuCtxGetDevice, cuDeviceGetName)
       - Notes (via comments) about CUDA modules (cuModule*) and their relationship to kernels

    4. Runtime / Driver interoperability:
       - How the runtime implicitly manages contexts
       - How to access and inspect that context from the driver API
       - Best practices and caveats when mixing the two APIs

    5. Compilation:
       - Any file that uses CUDA C++ extensions (e.g., __global__) must be compiled by `nvcc`
       - Example compilation command (see comment near main())

    The goal of this file is to be self-contained and instructive while respecting good
    C++ style: RAII wrappers for resources, centralized error handling, and clear separation
    of concerns. All "tutorial" explanations are provided as comments, as required.
*/

#include <cuda_runtime.h>  // CUDA Runtime API (high-level C/C++ API)
#include <cuda.h>          // CUDA Driver API (low-level C API)

#include <iostream>        // For host-side logging
#include <vector>          // For host-side containers
#include <stdexcept>       // For exceptions
#include <string>          // For building error messages
#include <cstdlib>         // For EXIT_SUCCESS / EXIT_FAILURE

/*
    Centralized error handling helpers
    ----------------------------------

    Robust CUDA programs must handle both Runtime API (cuda*) and Driver API (cu*) errors.

    The functions below wrap CUDA calls and throw std::runtime_error with detailed messages
    when an error occurs. This style is convenient in C++ code because it allows us to
    use exceptions for control flow instead of manually propagating error codes everywhere.

    Note: Exceptions are used here for clarity and brevity. In performance-critical or
    low-level libraries, you might prefer explicit error-code handling.
*/

inline void checkCuda(cudaError_t result,
                      const char* expression,
                      const char* file,
                      int line)
{
    if (result != cudaSuccess)
    {
        std::string message = "CUDA Runtime API error at ";
        message += file;
        message += ":";
        message += std::to_string(line);
        message += " for expression \"";
        message += expression;
        message += "\"; code=";
        message += std::to_string(static_cast<int>(result));
        message += " (";
        message += cudaGetErrorName(result);
        message += ") : ";
        message += cudaGetErrorString(result);

        throw std::runtime_error(message);
    }
}

inline void checkCu(CUresult result,
                    const char* expression,
                    const char* file,
                    int line)
{
    if (result != CUDA_SUCCESS)
    {
        const char* name = nullptr;
        const char* description = nullptr;
        (void)cuGetErrorName(result, &name);       // Safe even on failure; used for diagnostic only.
        (void)cuGetErrorString(result, &description);

        std::string message = "CUDA Driver API error at ";
        message += file;
        message += ":";
        message += std::to_string(line);
        message += " for expression \"";
        message += expression;
        message += "\"; code=";
        message += std::to_string(static_cast<int>(result));
        message += " (";
        message += (name ? name : "unknown");
        message += ") : ";
        message += (description ? description : "no description");

        throw std::runtime_error(message);
    }
}

/* Convenience macros to capture expression and location. */
#define CHECK_CUDA(expr) checkCuda((expr), #expr, __FILE__, __LINE__)
#define CHECK_CU(expr)   checkCu((expr),   #expr, __FILE__, __LINE__)

/*
    RAII wrapper for device memory
    ------------------------------

    This template wraps cudaMalloc / cudaFree in an owning class, ensuring that
    memory is always freed even if an exception is thrown.

    Features:
    - Non-copyable, movable
    - Typedef: DeviceBuffer<float> for float arrays, etc.
    - Provides raw pointer access (get()) for use in kernel launches and cudaMemcpy.
*/

template <typename T>
class DeviceBuffer
{
public:
    DeviceBuffer() = default;

    explicit DeviceBuffer(std::size_t elementCount)
    {
        allocate(elementCount);
    }

    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    DeviceBuffer(DeviceBuffer&& other) noexcept
        : ptr_(other.ptr_)
        , size_(other.size_)
    {
        other.ptr_ = nullptr;
        other.size_ = 0;
    }

    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept
    {
        if (this != &other)
        {
            releaseNoexcept();
            ptr_ = other.ptr_;
            size_ = other.size_;
            other.ptr_ = nullptr;
            other.size_ = 0;
        }
        return *this;
    }

    ~DeviceBuffer()
    {
        /*
            Destructors must not throw. We therefore call cudaFree without CHECK_CUDA
            and ignore any potential error here. In production code, logging from
            destructors is sometimes useful if cudaFree fails.
        */
        releaseNoexcept();
    }

    void allocate(std::size_t elementCount)
    {
        /*
            If allocate() is called multiple times, previously allocated memory
            is released first. This mimics the semantics of re-allocation.
        */
        release();
        if (elementCount == 0)
        {
            return;
        }

        size_ = elementCount;
        CHECK_CUDA(cudaMalloc(reinterpret_cast<void**>(&ptr_),
                              size_ * sizeof(T)));
    }

    T* get() noexcept
    {
        return ptr_;
    }

    const T* get() const noexcept
    {
        return ptr_;
    }

    std::size_t size() const noexcept
    {
        return size_;
    }

    bool empty() const noexcept
    {
        return size_ == 0;
    }

private:
    void release()
    {
        if (ptr_ != nullptr)
        {
            CHECK_CUDA(cudaFree(ptr_));
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    void releaseNoexcept() noexcept
    {
        if (ptr_ != nullptr)
        {
            (void)cudaFree(ptr_);
            ptr_ = nullptr;
            size_ = 0;
        }
    }

    T*           ptr_  = nullptr;
    std::size_t  size_ = 0;
};

/*
    RAII wrapper for CUDA streams
    -----------------------------

    Streams are the fundamental mechanism for controlling concurrency and
    asynchronous execution in the CUDA Runtime API.

    This wrapper automatically creates and destroys a stream.
*/

class CudaStream
{
public:
    CudaStream()
    {
        CHECK_CUDA(cudaStreamCreate(&stream_));
    }

    CudaStream(const CudaStream&) = delete;
    CudaStream& operator=(const CudaStream&) = delete;

    CudaStream(CudaStream&& other) noexcept
        : stream_(other.stream_)
    {
        other.stream_ = nullptr;
    }

    CudaStream& operator=(CudaStream&& other) noexcept
    {
        if (this != &other)
        {
            destroyNoexcept();
            stream_ = other.stream_;
            other.stream_ = nullptr;
        }
        return *this;
    }

    ~CudaStream()
    {
        destroyNoexcept();
    }

    cudaStream_t get() const noexcept
    {
        return stream_;
    }

private:
    void destroyNoexcept() noexcept
    {
        if (stream_ != nullptr)
        {
            (void)cudaStreamDestroy(stream_);
            stream_ = nullptr;
        }
    }

    cudaStream_t stream_ = nullptr;
};

/*
    CUDA C++ language extension: __host__ __device__
    -----------------------------------------------

    Functions marked __host__ __device__ can be called from both host and device
    code. This is useful for small mathematical utilities that are shared between
    CPU and GPU implementations.

    Note:
    - Such functions must be defined in headers or in the same translation unit
      that uses them, because device compilation is performed by nvcc.
*/

__host__ __device__ inline float affine_transform(float x,
                                                  float scale,
                                                  float bias)
{
    return scale * x + bias;
}

/*
    CUDA C++ language extension: __global__ kernel
    ---------------------------------------------

    - __global__ marks a function as a kernel that runs on the device (GPU)
      and is callable from the host (CPU).
    - Kernels must have void return type.
    - The special <<<...>>> launch syntax is used to specify the execution
      configuration (grid and block size, shared memory, stream).
    - This example demonstrates a simple vector addition.

    Built-in variables used:
    - blockIdx.x   : index of the current block in the grid (in the x dimension)
    - blockDim.x   : number of threads per block (in the x dimension)
    - threadIdx.x  : index of the current thread within its block (in the x dimension)
*/

__global__ void vector_add_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float*       __restrict__ c,
                                  int n)
{
    /*
        Compute the global 1D thread index. This is the standard idiom for mapping
        a 1D grid of 1D blocks onto a 1D problem domain.
    */
    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;

    /*
        Always guard against out-of-bounds accesses. A typical launch configuration
        uses ceil(n / blockDim.x) blocks, which may produce more threads than needed.
    */
    if (globalThreadId < n)
    {
        c[globalThreadId] = a[globalThreadId] + b[globalThreadId];
    }
}

/*
    Example of a slightly more complex kernel with dynamic shared memory
    --------------------------------------------------------------------

    This kernel is not central to the Programming Interface concept, but it demonstrates
    the fourth parameter of the kernel launch configuration: the number of bytes of
    dynamic shared memory requested per block.

    The kernel computes a block-level partial sum. Only the thread with threadIdx.x == 0
    writes the block's partial sum to the output array.

    Launch signature:
        reduce_sum_kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(...);

    where:
        sharedMemBytes = blockDim.x * sizeof(float)
*/

__global__ void reduce_sum_kernel(const float* __restrict__ input,
                                  float*       __restrict__ blockSums,
                                  int n)
{
    extern __shared__ float shared[];  // Dynamic shared memory.

    int globalThreadId = blockIdx.x * blockDim.x + threadIdx.x;
    int localThreadId  = threadIdx.x;

    /*
        Each thread reads one element into shared memory. If there are more
        threads than elements, extra threads read 0.0f.
    */
    float value = 0.0f;
    if (globalThreadId < n)
    {
        value = input[globalThreadId];
    }

    shared[localThreadId] = value;
    __syncthreads();

    /*
        Parallel reduction in shared memory (simple sequential halving algorithm).
        This is not the most optimized form, but it is clear and illustrates how
        shared memory interacts with per-block threads.
    */
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1)
    {
        if (localThreadId < stride)
        {
            shared[localThreadId] += shared[localThreadId + stride];
        }
        __syncthreads();
    }

    /*
        After the loop, shared[0] holds the block's partial sum. We store this to
        the output array, one element per block.
    */
    if (localThreadId == 0)
    {
        blockSums[blockIdx.x] = shared[0];
    }
}

/*
    Device enumeration and capability query using the Runtime API
    -------------------------------------------------------------

    This function queries the number of CUDA devices visible to the runtime, iterates
    over them, and prints basic properties (name, compute capability, multiprocessors,
    global memory size).

    This illustrates:
    - cudaGetDeviceCount
    - cudaGetDeviceProperties
*/

void print_device_info_via_runtime_api()
{
    int deviceCount = 0;
    CHECK_CUDA(cudaGetDeviceCount(&deviceCount));

    std::cout << "CUDA Runtime API sees " << deviceCount << " device(s).\n";

    for (int device = 0; device < deviceCount; ++device)
    {
        cudaDeviceProp properties{};
        CHECK_CUDA(cudaGetDeviceProperties(&properties, device));

        std::cout << "  Device " << device << " : " << properties.name << "\n"
                  << "    Compute capability : " << properties.major << "." << properties.minor << "\n"
                  << "    Multiprocessors    : " << properties.multiProcessorCount << "\n"
                  << "    Global memory (MiB): " << (properties.totalGlobalMem / (1024 * 1024)) << "\n";
    }

    std::cout << std::endl;
}

/*
    Vector addition using the CUDA Runtime API
    -----------------------------------------

    This function demonstrates the full path from host C++ code to GPU execution
    using the runtime API:

    1. Device selection with cudaSetDevice.
    2. Host-side data initialization (std::vector).
    3. Device memory allocation (cudaMalloc via DeviceBuffer<T>).
    4. Host-to-device memcpy (cudaMemcpy).
    5. Kernel launch with execution configuration:
           vector_add_kernel<<<gridDim, blockDim, sharedMemBytes, stream>>>(...)
    6. Asynchronous execution on a stream, followed by synchronization
       (cudaStreamSynchronize).
    7. Device-to-host memcpy (cudaMemcpy).
    8. Result verification on the host.

    It also demonstrates:
    - Use of dim3 to express a 1D grid / block.
    - Launch-time error checking with cudaGetLastError.
*/

void run_vector_add_with_runtime_api()
{
    /*
        Step 1: Select a device.

        In a multi-GPU system, more sophisticated logic might be used to select
        a device (e.g., based on free memory or compute capability). For this
        example, we simply use device 0.
    */
    int device = 0;
    CHECK_CUDA(cudaSetDevice(device));

    /*
        Step 2: Prepare host-side input data.

        We use std::vector<float> for convenience. The data is contiguous in memory,
        which aligns well with cudaMemcpy and kernel access patterns.
    */
    const int    n     = 1 << 20;  // 1M elements (approximate size for the example)
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_a(n);
    std::vector<float> h_b(n);
    std::vector<float> h_c(n);

    for (int i = 0; i < n; ++i)
    {
        h_a[i] = static_cast<float>(i);
        h_b[i] = static_cast<float>(2 * i);
    }

    /*
        Step 3: Allocate device buffers (RAII managed).

        DeviceBuffer<T> is a small wrapper around cudaMalloc / cudaFree.
    */
    DeviceBuffer<float> d_a(n);
    DeviceBuffer<float> d_b(n);
    DeviceBuffer<float> d_c(n);

    /*
        Step 4: Copy input data from host to device.

        cudaMemcpy is synchronous with respect to the host by default when used
        without streams (or with the default stream). When a non-default stream
        is supplied and pageable host memory is used (as here), the copies still
        behave synchronously at the API level, but may internally overlap with
        kernel execution on devices that support concurrent copy and compute.

        For true asynchronous transfers, one should use pinned (page-locked)
        host memory via cudaHostAlloc or cudaMallocHost.
    */
    CHECK_CUDA(cudaMemcpy(d_a.get(), h_a.data(), bytes, cudaMemcpyHostToDevice));
    CHECK_CUDA(cudaMemcpy(d_b.get(), h_b.data(), bytes, cudaMemcpyHostToDevice));

    /*
        Step 5: Configure and launch the kernel.

        The execution configuration is specified by:
           <<<gridDim, blockDim, sharedMemBytes, stream>>>

        - gridDim / blockDim can be 1D, 2D, or 3D (dim3 type).
        - sharedMemBytes is the number of bytes of dynamic shared memory per block.
          For vector_add_kernel we do not require dynamic shared memory, so we pass 0.
        - stream is a cudaStream_t; we use a user-created stream to demonstrate
          non-default streams.

        The mapping from (n, blockDim) to gridDim is the standard ceiling division:
            gridDim.x = (n + blockDim.x - 1) / blockDim.x
    */
    const int  threadsPerBlock = 256;
    const dim3 blockDim(threadsPerBlock, 1, 1);
    const dim3 gridDim((n + threadsPerBlock - 1) / threadsPerBlock, 1, 1);

    CudaStream stream;  // User-defined stream, not the default stream (0).

    vector_add_kernel<<<gridDim, blockDim, 0, stream.get()>>>(
        d_a.get(),
        d_b.get(),
        d_c.get(),
        n);

    /*
        Step 6: Check for launch errors and synchronize.

        cudaGetLastError detects errors in the kernel launch itself (e.g., invalid
        configuration). Actual execution errors are usually reported when the
        device is synchronized (here via cudaStreamSynchronize).
    */
    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaStreamSynchronize(stream.get()));

    /*
        Step 7: Copy results back to host memory.
    */
    CHECK_CUDA(cudaMemcpy(h_c.data(), d_c.get(), bytes, cudaMemcpyDeviceToHost));

    /*
        Step 8: Verify results.

        This is purely host-side C++ code. Small numerical tolerances are often used
        due to floating-point rounding, but since the operations here are simple and
        use only integer-like values, exact comparison is acceptable.
    */
    bool allOk = true;
    for (int i = 0; i < n; ++i)
    {
        const float expected = h_a[i] + h_b[i];
        if (h_c[i] != expected)
        {
            std::cerr << "Mismatch at index " << i
                      << ": got " << h_c[i]
                      << ", expected " << expected << "\n";
            allOk = false;
            break;
        }
    }

    if (allOk)
    {
        std::cout << "Vector addition via CUDA Runtime API completed successfully.\n";
    }

    std::cout << std::endl;
}

/*
    Demonstrating dynamic shared memory and reduction launch
    -------------------------------------------------------

    This helper function launches reduce_sum_kernel with a 1D execution configuration
    and uses dynamic shared memory equal to blockDim.x * sizeof(float).

    It also illustrates:
    - The third kernel launch parameter (sharedMemBytes).
    - A simple two-pass reduction (block-level partial sums on the device, final
      reduction on the host).
*/

void run_block_reduce_with_runtime_api()
{
    const int    n     = 1 << 16;  // 65,536 elements
    const size_t bytes = static_cast<size_t>(n) * sizeof(float);

    std::vector<float> h_input(n);
    for (int i = 0; i < n; ++i)
    {
        // Using affine_transform to demonstrate __host__ __device__ usage
        h_input[i] = affine_transform(static_cast<float>(i), 1.0f, 0.0f);
    }

    DeviceBuffer<float> d_input(n);

    CHECK_CUDA(cudaMemcpy(d_input.get(), h_input.data(), bytes, cudaMemcpyHostToDevice));

    /*
        Choose a block size that is a power of two, which simplifies the reduction kernel.
    */
    const int  threadsPerBlock = 256;
    const dim3 blockDim(threadsPerBlock, 1, 1);
    const dim3 gridDim((n + threadsPerBlock - 1) / threadsPerBlock, 1, 1);

    /*
        Allocate one output element per block for partial sums.
    */
    DeviceBuffer<float> d_blockSums(gridDim.x);

    const size_t sharedMemBytes = static_cast<size_t>(threadsPerBlock) * sizeof(float);

    reduce_sum_kernel<<<gridDim, blockDim, sharedMemBytes>>>(
        d_input.get(),
        d_blockSums.get(),
        n);

    CHECK_CUDA(cudaGetLastError());
    CHECK_CUDA(cudaDeviceSynchronize());

    /*
        Copy block-level partial sums back to the host and finish the reduction.
    */
    std::vector<float> h_blockSums(gridDim.x);
    CHECK_CUDA(cudaMemcpy(h_blockSums.data(),
                          d_blockSums.get(),
                          static_cast<size_t>(gridDim.x) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float gpuSum = 0.0f;
    for (int i = 0; i < gridDim.x; ++i)
    {
        gpuSum += h_blockSums[i];
    }

    /*
        Compute the reference sum on the host for validation.
    */
    float cpuSum = 0.0f;
    for (int i = 0; i < n; ++i)
    {
        cpuSum += h_input[i];
    }

    const float diff = std::abs(cpuSum - gpuSum);
    const float tol  = 1e-3f;

    if (diff <= tol)
    {
        std::cout << "Block reduction via CUDA Runtime API completed successfully.\n"
                  << "  CPU sum: " << cpuSum << "\n"
                  << "  GPU sum: " << gpuSum << "\n";
    }
    else
    {
        std::cerr << "Block reduction mismatch.\n"
                  << "  CPU sum: " << cpuSum << "\n"
                  << "  GPU sum: " << gpuSum << "\n"
                  << "  Difference: " << diff << "\n";
    }

    std::cout << std::endl;
}

/*
    Runtime / Driver interoperability demonstration
    ----------------------------------------------

    The CUDA Runtime API manages CUDA contexts implicitly:

    - The first call to a runtime function that requires a context (e.g., cudaSetDevice,
      cudaFree, cudaMalloc) will create and "activate" a primary context on the chosen
      device for the current host thread.
    - The Driver API can query and manipulate this context (e.g., cuCtxGetCurrent).

    This function shows how to:
    - Trigger implicit context creation via the runtime.
    - Query the current context with cuCtxGetCurrent.
    - Retrieve the underlying device associated with that context and print its name.

    This demonstrates that:
    - The runtime is built on top of the driver.
    - Both APIs can interoperate correctly when used with care.
*/

void demonstrate_runtime_driver_interoperability()
{
    /*
        Explicitly initialize the driver API.

        cuInit(0) is typically called once at program startup when using the
        driver API directly. When using only the runtime, this is performed
        implicitly. Calling cuInit(0) multiple times is safe; subsequent calls
        are no-ops.
    */
    CHECK_CU(cuInit(0));

    /*
        Force the CUDA Runtime API to create a context if it has not already.

        cudaFree(nullptr) is a common idiom for context creation. It performs
        no memory operation when the pointer is null, but still ensures that
        a context is created on the currently selected device.
    */
    CHECK_CUDA(cudaSetDevice(0));
    CHECK_CUDA(cudaFree(nullptr));

    /*
        Use the Driver API to query the current context.
    */
    CUcontext currentContext = nullptr;
    CHECK_CU(cuCtxGetCurrent(&currentContext));

    if (currentContext == nullptr)
    {
        throw std::runtime_error("Expected a current CUDA context created by the runtime.");
    }

    /*
        Retrieve the device associated with this context and print its name.
    */
    CUdevice cuDevice;
    CHECK_CU(cuCtxGetDevice(&cuDevice));

    char deviceName[256] = {};
    CHECK_CU(cuDeviceGetName(deviceName, static_cast<int>(sizeof(deviceName)), cuDevice));

    std::cout << "Runtime / Driver interoperability:\n"
              << "  Current context (queried via Driver API) is bound to device: "
              << deviceName << "\n\n";
}

/*
    Explicit context management with the CUDA Driver API
    ---------------------------------------------------

    While the Runtime API hides contexts from the user, the Driver API exposes them
    explicitly. This allows advanced control scenarios such as:

    - Creating multiple contexts per device (for isolation).
    - Sharing a context across host threads.
    - Managing separate contexts for separate CUDA modules.

    This function demonstrates:
    - cuInit (driver initialization)
    - cuDeviceGetCount / cuDeviceGet (device enumeration)
    - cuCtxCreate / cuCtxDestroy (explicit context lifetime)
    - cuDeviceGetName (device identification)

    Note:
    - The context created here is separate from the implicit primary context
      managed by the runtime. They can coexist, but care must be taken when
      mixing them in complex applications.
*/

void demonstrate_explicit_context_management_via_driver_api()
{
    CHECK_CU(cuInit(0));

    int deviceCount = 0;
    CHECK_CU(cuDeviceGetCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cout << "No CUDA devices available for explicit driver context demonstration.\n\n";
        return;
    }

    CUdevice device;
    CHECK_CU(cuDeviceGet(&device, 0));

    char deviceName[256] = {};
    CHECK_CU(cuDeviceGetName(deviceName, static_cast<int>(sizeof(deviceName)), device));

    /*
        Create an explicit context on device 0.

        The flags argument can be used to specify scheduling behavior, among other
        things. Here we pass 0, which selects default behavior.
    */
    CUcontext context = nullptr;
    CHECK_CU(cuCtxCreate(&context, 0, device));

    std::cout << "Explicit CUDA context created via Driver API on device: "
              << deviceName << "\n";

    /*
        At this point, kernels could be launched using the Driver API by:
        - Loading a CUDA module (e.g., a PTX or CUBIN file) via cuModuleLoad.
        - Obtaining a CUfunction via cuModuleGetFunction.
        - Launching the kernel via cuLaunchKernel.

        Example (conceptual, not executed here):

            CUmodule  module;
            CUfunction func;

            CHECK_CU(cuModuleLoad(&module, "my_kernels.ptx"));
            CHECK_CU(cuModuleGetFunction(&func, module, "my_kernel"));

            void* kernelParams[] = { &devicePtrA, &devicePtrB, &n };

            CHECK_CU(cuLaunchKernel(func,
                                     gridDimX, 1, 1,
                                     blockDimX, 1, 1,
                                     sharedMemBytes,
                                     /* stream */ 0,
                                     kernelParams,
                                     /* extra */ nullptr));

            CHECK_CU(cuModuleUnload(module));

        This illustrates the lower-level control exposed by the driver API:
        - Modules and functions are first-class objects.
        - The kernel parameters are passed via generic pointer arrays.
        - Context selection and stream management are explicit.
    */

    /*
        Destroy the explicit context. This releases all resources owned by the
        context (e.g., device memory allocated via the driver API within it).
    */
    CHECK_CU(cuCtxDestroy(context));

    std::cout << "Explicit CUDA context destroyed.\n\n";
}

/*
    main()
    ------

    Orchestrates the demonstrations above.

    Compilation:

        Any source file that uses CUDA C++ language extensions (e.g., __global__ kernels)
        must be compiled with nvcc. For example:

            nvcc -std=c++17 -O2 -arch=sm_70 programming_interface_demo.cu -o programming_interface_demo

        Adjust -arch=sm_XX to match your GPU's compute capability.

    Note on exceptions:
        main() catches std::exception and prints its message; in production code,
        more sophisticated error handling and logging might be appropriate.
*/

int main()
{
    try
    {
        print_device_info_via_runtime_api();

        run_vector_add_with_runtime_api();

        run_block_reduce_with_runtime_api();

        demonstrate_runtime_driver_interoperability();

        demonstrate_explicit_context_management_via_driver_api();

        return EXIT_SUCCESS;
    }
    catch (const std::exception& ex)
    {
        std::cerr << "Unhandled exception: " << ex.what() << std::endl;
        return EXIT_FAILURE;
    }
}