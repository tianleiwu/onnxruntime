# Stage 1: CUDA Plugin EP Shell — Implementation Plan

Build the plugin EP shell so CUDA EP can load via the public `OrtEpFactory`/`OrtEp` API and run simple kernels.

## Proposed Changes

### DLL Entry Points

#### [NEW] [cuda_plugin_ep.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_ep.cc)

Main entry point file exporting [CreateEpFactories()](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/example_plugin_ep.cc#18-47) and [ReleaseEpFactory()](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/example_plugin_ep.cc#48-52). Pattern follows [example_plugin_ep.cc](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/example_plugin_ep.cc).

- `#define ORT_API_MANUAL_INIT` + `Ort::InitApi()` pattern
- Store `OrtApi`, `OrtEpApi` pointers
- Create a `CudaEpFactory` instance
- Export via `extern "C"` with `EXPORT_SYMBOL` macro

---

### CudaEpFactory

#### [NEW] [cuda_ep_factory.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_ep_factory.h)
#### [NEW] [cuda_ep_factory.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_ep_factory.cc)

Class `CudaEpFactory : public OrtEpFactory` following [ExampleKernelEpFactory](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry/ep_factory.cc) pattern:

- **Constructor**: Set `ort_version_supported = ORT_API_VERSION` and assign all static callback function pointers (`GetName = GetNameImpl`, etc.)
- **[GetSupportedDevices](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_internal.h#31-38)**: Enumerate CUDA devices via [cudaGetDeviceCount()](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_provider_factory.cc#158-163) / `cudaGetDeviceProperties()`. For each GPU `OrtHardwareDevice`, create an `OrtEpDevice` and add `DEFAULT` + `HOST_ACCESSIBLE` allocator infos.
- **[CreateEp](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_internal.h#39-48)**: Parse options from `OrtSessionOptions`, create `CudaEp` instance
- **[CreateAllocator](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_internal_impl.h#43-51)**: Dispatch to `CudaDeviceAllocator` or [CudaPinnedAllocator](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#206-222) based on `OrtMemoryInfo`
- **[CreateDataTransfer](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_webgpu.cc#60-73)**: Return `CudaDataTransfer` instance
- **[IsStreamAware](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_internal.h#73-76)**: Return [true](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_provider_factory.cc#114-115) (CUDA EP is stream-aware)
- **[CreateSyncStreamForDevice](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_internal_impl.h#78-85)**: Create `CudaSyncStream` with cuBLAS/cuDNN handles

Memory info definitions:
```cpp
// GPU device memory
Ort::MemoryInfo default_memory_info_{"Cuda", OrtMemoryInfoDeviceType_GPU,
                                      OrtDevice::VendorIds::NVIDIA, device_id,
                                      OrtDeviceMemoryType_DEFAULT, 0,
                                      OrtAllocatorType::OrtDeviceAllocator};
// CPU pinned memory
Ort::MemoryInfo pinned_memory_info_{"CudaPinned", OrtMemoryInfoDeviceType_CPU,
                                     OrtDevice::VendorIds::NVIDIA, 0,
                                     OrtDeviceMemoryType_HOST_ACCESSIBLE, 0,
                                     OrtAllocatorType::OrtDeviceAllocator};
```

---

### CudaEp

#### [NEW] [cuda_ep.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_ep.h)
#### [NEW] [cuda_ep.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_ep.cc)

Class `CudaEp : public OrtEp` following [ExampleKernelEp](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry/ep.cc) pattern:

- **Constructor**: Set function pointers for [GetName](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_internal_impl.h#24-25), [GetCapability](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/ep.h#90-92), [GetKernelRegistry](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry/ep_factory.cc#74-98), `OnRunStart`, [OnRunEnd](file:///home/tlwu/onnxruntime/onnxruntime/core/framework/plugin_ep_stream.h#59-63), `GetPreferredDataLayout`. Set `Compile = nullptr` (kernel-registry-based, not compile-based).
- **[GetCapability](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/ep.h#90-92)**: Iterate graph nodes, check each against kernel registry via `EpGraphSupportInfo_LookUpKernel`, call `EpGraphSupportInfo_AddSingleNode` for supported nodes.
- **[GetKernelRegistry](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry/ep_factory.cc#74-98)**: Delegate to factory's cached kernel registry (same pattern as example EP).
- **`OnRunStart`/[OnRunEnd](file:///home/tlwu/onnxruntime/onnxruntime/core/framework/plugin_ep_stream.h#59-63)**: Stubs now. Design note: these will later manage CUDA Graph capture state.
- **`GetPreferredDataLayout`**: Return `OrtDataLayout_NHWC` if configured, else `OrtDataLayout_NCHW`.

---

### CudaAllocator

#### [NEW] [cuda_allocator_plugin.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_allocator_plugin.h)
#### [NEW] [cuda_allocator_plugin.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_allocator_plugin.cc)

Two allocator classes inheriting [OrtAllocator](file:///home/tlwu/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h#355-422):

**`CudaDeviceAllocator`**:
- [Alloc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#134-140): `cudaSetDevice(device_id)` + `cudaMalloc(&p, size)`, return `p`
- `Free`: `cudaFree(p)`
- [Info](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry/kernels/utils.h#57-66): return `OrtMemoryInfo` for CUDA device memory

**[CudaPinnedAllocator](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc#206-222)**:
- [Alloc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#134-140): `cudaHostAlloc(&p, size, cudaHostAllocDefault)`
- `Free`: `cudaFreeHost(p)`
- [Info](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry/kernels/utils.h#57-66): return `OrtMemoryInfo` for pinned host memory

---

### CudaDataTransfer

#### [NEW] [cuda_data_transfer_plugin.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_data_transfer_plugin.h)
#### [NEW] [cuda_data_transfer_plugin.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_data_transfer_plugin.cc)

Class `CudaDataTransfer : public OrtDataTransferImpl`:

- **`CanCopy(src, dst)`**: Return true for CPU→GPU, GPU→CPU, GPU→GPU transfers. Check via `OrtMemoryDevice` device type comparison.
- **`CopyTensors(srcs, dsts, streams, count)`**: For each tensor pair, determine `cudaMemcpyKind`, extract data pointer/size via `OrtEpApi`, use `cudaMemcpyAsync` with stream handle if provided, else [cudaMemcpy](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_provider_factory.cc#138-150).

---

### CudaSyncStream & Notification

#### [NEW] [cuda_stream_plugin.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h)
#### [NEW] [cuda_stream_plugin.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.cc)

**`CudaSyncStream : public OrtSyncStreamImpl`**:
- Owns a `cudaStream_t` (created via `cudaStreamCreateWithFlags`)
- Owns `cublasHandle_t`, `cudnnHandle_t`, `cublasLtHandle_t` (created and bound to this stream)
- [GetHandle](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/ep_stream_support.h#35-36): return `cudaStream_t`
- [CreateNotification](file:///home/tlwu/onnxruntime/onnxruntime/core/framework/plugin_ep_stream.h#45-56): create `CudaSyncNotification`
- [Flush](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/ep_stream_support.h#36-37): `cudaStreamSynchronize`
- [OnSessionRunEnd](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/ep_stream_support.h#37-38): cleanup deferred CPU buffers (port from existing `CudaStream::CleanUpOnRunEnd`)
- [Release](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_internal.h#100-104): destroy stream, handles

**`CudaSyncNotification : public OrtSyncNotificationImpl`**:
- Owns a `cudaEvent_t` (created via `cudaEventCreateWithFlags`)
- [Activate](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/ep_stream_support.h#59-60): `cudaEventRecord(event_, stream)`
- [WaitOnDevice](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/ep_stream_support.h#60-62): `cudaStreamWaitEvent(stream, event_)`
- [WaitOnHost](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/ep_stream_support.h#62-63): `cudaEventSynchronize(event_)`
- [Release](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_internal.h#100-104): `cudaEventDestroy`

---

### Simple Kernel Registration (Validation Kernels)

#### [NEW] [cuda_plugin_kernels.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_kernels.h)
#### [NEW] [cuda_plugin_kernels.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_kernels.cc)

Register a minimal set of kernels to validate the plugin EP works end-to-end:
- **Relu** (simple element-wise, validates GPU compute)
- **Add** (binary element-wise, validates multi-input)
- **MemcpyFromHost** / **MemcpyToHost** (validates data transfer)

Use the `ONNX_OPERATOR_KERNEL_EX` macro pattern from [utils.h](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep_kernel_registry/kernels/utils.h) adapted for CUDA:
- `OrtKernelDef` built with `Ort::KernelDefBuilder` 
- `OrtKernelCreateFunc` lambda that creates kernel implementation
- Kernel impl wraps CUDA kernel launches

For Stage 1, these can directly call CUDA runtime without the full [CudaKernel](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#18-209) adapter (which comes in Stage 2/3).

---

### CMake Integration

#### [NEW] [onnxruntime_providers_cuda_plugin.cmake](file:///home/tlwu/onnxruntime/cmake/onnxruntime_providers_cuda_plugin.cmake)

- Add `BUILD_CUDA_EP_AS_PLUGIN` CMake option (default OFF)
- When ON, compile `onnxruntime/core/providers/cuda/plugin/*.cc` + CUDA kernels into a shared library `onnxruntime_providers_cuda_plugin`
- Link against CUDA runtime, cuBLAS, cuDNN
- Add symbol export via `CUDA_PLUGIN_EP_SYMBOLS.def` (Windows) / link script (Linux)
- Include public headers only: [onnxruntime_c_api.h](file:///home/tlwu/onnxruntime/include/onnxruntime/core/session/onnxruntime_c_api.h), [onnxruntime_cxx_api.h](file:///home/tlwu/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_api.h), [onnxruntime_ep_c_api.h](file:///home/tlwu/onnxruntime/include/onnxruntime/core/session/onnxruntime_ep_c_api.h)

#### [MODIFY] [CMakeLists.txt](file:///home/tlwu/onnxruntime/CMakeLists.txt)

- Add `include(onnxruntime_providers_cuda_plugin.cmake)` when `BUILD_CUDA_EP_AS_PLUGIN` is ON

---

### Internal Adapter (Static Link Mode)

> [!NOTE]
> This component is **deferred** to Stage 2. For Stage 1, we focus on the plugin-only mode. The internal adapter (`ep_factory_cuda.h/cc`) following the [WebGpuEpFactory](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_webgpu.h#15-17) pattern will bridge the static-link build to use the same CUDA EP factory.

---

## File Layout

```
onnxruntime/core/providers/cuda/plugin/
├── cuda_plugin_ep.cc          # DLL entry points
├── cuda_ep_factory.h/cc       # OrtEpFactory implementation
├── cuda_ep.h/cc               # OrtEp implementation
├── cuda_allocator_plugin.h/cc # OrtAllocator implementations
├── cuda_data_transfer_plugin.h/cc  # OrtDataTransferImpl
├── cuda_stream_plugin.h/cc    # OrtSyncStreamImpl + OrtSyncNotificationImpl
├── cuda_plugin_kernels.h/cc   # Initial kernel registrations (Relu, Add, Memcpy)
└── cuda_plugin_utils.h        # Shared helpers, error macros
```

---

## Verification Plan

### Automated Tests

We will create a new test file following the existing [test_execution.cc](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/test_execution.cc) pattern:

#### [NEW] test_cuda_plugin_ep.cc

Test cases:
1. **`CudaPluginEp_LoadFactory`** — Load plugin DLL, verify [CreateEpFactories](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/example_plugin_ep.cc#18-47) returns valid factory
2. **`CudaPluginEp_EnumerateDevices`** — Call [GetSupportedDevices](file:///home/tlwu/onnxruntime/onnxruntime/core/session/plugin_ep/ep_factory_internal.h#31-38), verify CUDA GPUs detected
3. **`CudaPluginEp_CreateEp`** — Create EP, verify name and stream-awareness
4. **`CudaPluginEp_ReluInference`** — Run a simple Relu ONNX model through the plugin EP, verify output
5. **`CudaPluginEp_DataTransfer`** — Verify CPU→GPU→CPU data transfer via allocator + data transfer

> [!IMPORTANT]
> These tests require a CUDA-capable GPU. They should be gated behind `TEST_CUDA_EP_PLUGIN` or similar CMake flag.

### Build Verification

```bash
# Build with plugin mode
./build.sh --use_cuda --cuda_home /usr/local/cuda --cudnn_home /usr \
  --cmake_extra_defines BUILD_CUDA_EP_AS_PLUGIN=ON \
  --build_dir build/plugin_cuda --config Debug --parallel

# Verify the plugin DLL is built
ls build/plugin_cuda/Debug/libonnxruntime_providers_cuda_plugin.so

# Run the tests
cd build/plugin_cuda/Debug
./onnxruntime_test_all --gtest_filter="CudaPluginEp*"
```

### Manual Verification

The implementation can be manually verified by the user:
1. Build with `BUILD_CUDA_EP_AS_PLUGIN=ON` (see command above)
2. Verify the shared library exists in the build output
3. Run the automated test suite — all `CudaPluginEp*` tests should pass
4. Verify `nm -D libonnxruntime_providers_cuda_plugin.so` shows only [CreateEpFactories](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/example_plugin_ep.cc#18-47) and [ReleaseEpFactory](file:///home/tlwu/onnxruntime/onnxruntime/test/autoep/library/example_plugin_ep/example_plugin_ep.cc#48-52) as public symbols
