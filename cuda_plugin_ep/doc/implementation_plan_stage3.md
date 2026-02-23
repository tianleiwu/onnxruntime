# Stage 3: Kernel Adapter Completion & Registration Migration

**Status**: Draft done (commit `33bff13`). Build passes. Remaining work is completeness and validation.

**Objective**: Finish the adapter layer so any CUDA kernel can compile with `BUILD_CUDA_EP_AS_PLUGIN`. Extend registration coverage beyond the five pilot ops.

---

## What Was Done in the Stage 3 Draft

| Item | Status | Notes |
|------|--------|-------|
| [cuda_kernel_adapter.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h) — [Tensor](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/shared_library/provider_wrappedtypes.h#1512-1513), [OpKernelInfo](file:///home/tlwu/onnxruntime/include/onnxruntime/core/framework/op_kernel_info.h#25-33), [OpKernelContext](file:///home/tlwu/onnxruntime/include/onnxruntime/core/framework/op_kernel_context.h#11-45) | ✅ | Good shape; `GetAttr<T>(name, *out)` uses [Status](file:///home/tlwu/onnxruntime/include/onnxruntime/core/session/onnxruntime_cxx_inline.h#60-63) return |
| [CudaKernel](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#20-25) base class — [Compute()](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h#138-149), [Stream()](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#76-80), [GetCudnnHandle()](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#85-88), [GetCublasHandle()](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h#28-29) | ✅ | Routes through [CudaSyncStream](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h#20-58) |
| [migrate_cuda_registrations.py](file:///home/tlwu/onnxruntime/tools/python/migrate_cuda_registrations.py) | ✅ | Parses [function_table](file:///home/tlwu/onnxruntime/tools/python/migrate_cuda_registrations.py#152-162) in [cuda_execution_provider.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc); emits [.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) |
| [cuda_plugin_generated_registrations.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) | ✅ | 21 entries for Relu/Add/MatMul/Gemm/Conv (float only) |
| Registry loop in `CreateCudaKernelRegistry()` | ✅ | Drives off [.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc); dispatches `GetCreateFnForOp()` |
| Build passing (no [provider_api.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/shared_library/provider_api.h) for plugin sources) | ✅ | [activations.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/activation/activations.cc) removed from sources; plugin compiles clean |

---

## Remaining Tasks

### Task 3.1 — Complete [CudaKernel](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#20-25) Missing Methods

The adapter needs these methods for the broader kernel population:

| Method | Impl Strategy |
|--------|---------------|
| `GetScratchBuffer<T>(n, stream)` | `cudaMalloc` + [AddDeferredReleaseCPUPtr](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#62-67) via [CudaSyncStream](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h#20-58) |
| [AddDeferredReleaseCPUPtr(ptr, stream)](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#62-67) | Delegate to `CudaSyncStream::EnqueueDeferredCPUBuffer` |
| `GetDeviceProp()` | Cache `cudaDeviceProp` in [CudaKernel](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#20-25) constructor (from `cudaGetDeviceProperties`) |
| [UseTF32()](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#97-100) | Return from an EP option stored in [CudaKernel](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#20-25) (pass in from `OrtKernelInfo`) |
| `IsArchAvailable(arch)` | Compare `device_prop_.major` |
| `GetConstOnes<T>(n, stream)` | Lazy-init a GPU ones-buffer per type; store in [CudaKernel](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#20-25) |
| `AllocateBufferOnCPUPinned<T>(n)` | `cudaMallocHost` + RAII wrapper |

> [!IMPORTANT]
> [GetScratchBuffer](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#42-51) is used by the vast majority of kernels. This is the most blocking item.

#### Files to Modify
- [MODIFY] [cuda_kernel_adapter.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h) — add the above methods

---

### Task 3.2 — Extend Registration Script for Multiple Types & Domains

**Current gap**: The script generates float-only registrations because `--types float` is the default. Real migration needs all types and contrib ops.

#### Script Enhancements ([tools/python/migrate_cuda_registrations.py](file:///home/tlwu/onnxruntime/tools/python/migrate_cuda_registrations.py))

1. **Multi-type output**: When `--types` is not filtered, emit one row per type, grouping into the struct. Or, switch the [.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) format to emit a `type_constraints` array per entry.
2. **Contrib domain support**: Add `--domain kMSDomain` flag + run a second pass.
3. **`TWO_TYPED` / `THREE_TYPED` macros**: Currently skipped. Parse `ONNX_OPERATOR_TWO_TYPED_KERNEL_CLASS_NAME` (6 args) and `_THREE_TYPED` (7 args).

**Suggested output change for multi-type constraints**:
```cpp
// .inc entry
{"Relu", 6, 12, {ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT, ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16}},
```

Or keep it one-row-per-type and expand `GetCreateFnForOp` to handle typed dispatch.

- [MODIFY] [migrate_cuda_registrations.py](file:///home/tlwu/onnxruntime/tools/python/migrate_cuda_registrations.py)
- [MODIFY] [cuda_plugin_generated_registrations.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) — regenerate

---

### Task 3.3 — Add Memory-Type Annotations to Registry

Currently, ops with CPU-input requirements (e.g., `Gather`, `GridSample`) don't get `SetInputMemType(i, OrtMemTypeCPUInput)`. This causes incorrect placement.

**Approach**: Encode memory type annotations in the [.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) format or as a separate override map in `CreateCudaKernelRegistry()`.

---

### Task 3.4 — CMake Build-Time Script Integration

Add a CMake `add_custom_command` so the [.inc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc) is regenerated if [cuda_execution_provider.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_execution_provider.cc) changes.

```cmake
# cmake/onnxruntime_providers_cuda_plugin.cmake
add_custom_command(
  OUTPUT  ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_generated_registrations.inc
  COMMAND ${Python3_EXECUTABLE} tools/python/migrate_cuda_registrations.py
          --input  onnxruntime/core/providers/cuda/cuda_execution_provider.cc
          --output ${CUDA_PLUGIN_EP_DIR}/cuda_plugin_generated_registrations.inc
  DEPENDS ${ONNXRUNTIME_ROOT}/core/providers/cuda/cuda_execution_provider.cc
          tools/python/migrate_cuda_registrations.py
  WORKING_DIRECTORY ${REPO_ROOT}
)
```

- [MODIFY] [onnxruntime_providers_cuda_plugin.cmake](file:///home/tlwu/onnxruntime/cmake/onnxruntime_providers_cuda_plugin.cmake)

---

### Task 3.5 — Validate Adapter Portability with Real Kernel Files

Test that actual CUDA kernel source files compile under the adapter:

1. Try adding [cuda/math/unary_elementwise_ops.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/unary_elementwise_ops.cc) + [activations.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/activation/activations.cc) back to plugin sources.
2. Fix any missing [CudaKernel](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/cuda_kernel.h#20-25) method stubs (Task 3.1).
3. Route [activations.cc](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/activation/activations.cc) [ComputeInternal](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/math/binary_elementwise_ops.cc#575-581) calls through the adapter.
4. Run [test_cuda_plugin_ep.py](file:///home/tlwu/onnxruntime/onnxruntime/test/python/transformers/test_cuda_plugin_ep.py) Relu test using adapter-compiled kernel (not the standalone `ReluKernelImpl`).

This is the proof-of-concept that justifies the whole adapter strategy.

- [MODIFY] [onnxruntime_providers_cuda_plugin.cmake](file:///home/tlwu/onnxruntime/cmake/onnxruntime_providers_cuda_plugin.cmake) — add sources back
- [MODIFY] [cuda_plugin_kernels.cu](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/cuda/plugin/cuda_plugin_kernels.cu) — switch `ReluKernelImpl` to delegate to `cuda::Relu<float>`

---

## Verification Plan

### Automated Tests
```bash
python onnxruntime/test/python/transformers/test_cuda_plugin_ep.py
```
All 5 ops (Relu, Add, MatMul, Gemm, Conv) must pass.

### Build Validation
```bash
make -j8 onnxruntime_providers_cuda_plugin
```
Zero errors, zero [provider_api.h](file:///home/tlwu/onnxruntime/onnxruntime/core/providers/shared_library/provider_api.h) includes in plugin-exclusive sources.

### Script Verification
```bash
python tools/python/migrate_cuda_registrations.py --ops --types  # all ops, all types
wc -l onnxruntime/core/providers/cuda/plugin/cuda_plugin_generated_registrations.inc
# Expect ~1000+ entries
```
