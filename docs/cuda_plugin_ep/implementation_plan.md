# Recommended CUDA Plugin EP Design and Implementation Plan

## Objective

Refactor the CUDA plugin EP so it follows the useful parts of the WebGPU plugin pattern while staying compatible with the current CUDA kernel code and the plugin build constraints.

The target end state is:

1. Kernel code obtains a plugin-side `CUDAExecutionProvider*` from `info.GetExecutionProvider()`.
2. `CudaEp` owns that provider through `adapter::Ep` instead of relying on process-wide static configuration.
3. The plugin-side provider exposes the framework CUDA EP API surface needed for all CUDA operators over time.
4. Existing CUDA kernel source changes stay minimal and limited to clear cleanup or unavoidable compatibility fixes.

This plan assumes current cmake exclusions are temporary. The long-term goal remains support for all operators currently supported by the framework CUDA EP. Tunable support stays lower priority.

## Recommended Design

### 1. Keep the plugin-side class name `CUDAExecutionProvider`

The plugin should continue defining its own `CUDAExecutionProvider` in `cuda_kernel_adapter.h`, but that class should stop being a hollow static-config shim.

Instead, it should become an instance-backed compatibility provider that:

- inherits from `IExecutionProvider`
- owns CUDA configuration as normal instance fields
- caches device properties per instance
- delegates handle and stream access to plugin-owned stream objects
- exposes the provider methods that existing CUDA kernels already call

This is better than introducing a new provider type plus aliasing because it preserves all existing kernel casts without adding naming indirection or migration work.

### 2. Change `CudaEp` to wrap the provider through `adapter::Ep`

The WebGPU plugin gets one important design choice right: the plugin `OrtEp` is only a wrapper around a concrete execution provider. CUDA should adopt that same ownership pattern even though it cannot directly instantiate the framework `CUDAExecutionProvider`.

`CudaEp` should therefore inherit from `onnxruntime::ep::adapter::Ep` and own a `std::unique_ptr<CUDAExecutionProvider>`.

That gives the plugin:

- provider lifetime tied to EP lifetime
- `EpImpl()` support so `info.GetExecutionProvider()` returns the provider instance
- a path away from static atomics
- closer structural alignment with the framework and WebGPU plugin

### 3. Treat the plugin-side provider as a compatibility layer

The plugin-side `CUDAExecutionProvider` should not claim more than it can currently provide.

Recommended contract:

- API compatibility first: implement the methods kernels need so code builds and can be re-enabled incrementally.
- Semantic parity where already supported: config getters, device properties, stream-backed handle lookup.
- Explicit approximation where needed: `PerThreadDefaultCublasHandle()`, `PerThreadDefaultCudnnHandle()`, `PerThreadCublasLtHandle()`, and `ComputeStream()` can initially return values derived from the factory's current compute stream, which is compatibility-correct for the plugin design today even if it is not full framework per-thread behavior.
- Explicit stubs where low priority is acceptable: attention options and tuning context.

This framing is important because the current plugin factory owns one compute stream path, not the framework CUDA EP's full per-thread context model.

### 4. Keep the global stream reverse-lookup for now

The plugin should keep the existing `CudaSyncStream::FromCudaStream()` global reverse-lookup map in the first refactor.

Reason:

- existing static helper call sites depend on `CudaKernel::GetCudnnHandle(cudaStream_t)` and `CudaKernel::GetCublasHandle(cudaStream_t)`
- those call sites do not carry provider or EP state
- moving stream ownership entirely into provider state would force wider kernel or EP API changes than the current effort needs

This is not ideal, but it is the right short-term tradeoff. Remove static atomics now; leave global stream lookup as future work.

### 5. Move to one allocator path for `adapter::Ep`

`adapter::Ep` requires `AllocatorPtr` allocators, so the plugin should add `IAllocator`-based wrappers and make them the primary allocator path for the refactor.

Recommended approach:

- add plugin `IAllocator` implementations for device memory and pinned host memory as needed
- store them on `CudaEpFactory` as `AllocatorPtr`
- pass them into `adapter::Ep`
- expose C API allocators through `adapter::Allocator` wrappers

The old direct `OrtAllocator` plugin implementations should not be expanded further. They can remain temporarily only if required for a smooth transition, but the new design should treat the `IAllocator` path as the destination, not as an optional side path.

## Required Provider Surface

The plugin-side `CUDAExecutionProvider` should expose at least the following groups of methods.

### Config and device state

- `GetDeviceId()`
- `GetDeviceProp()`
- `UseTF32()`
- `GetCudnnConvAlgo()`
- `GetCudnnConv1dPadToNc1d()`
- `GetCudnnConvUseMaxWorkspace()`
- `IsSkipLayerNormInStrictMode()`
- `IsNHWCPreferred()`
- `IsFuseConvBias()`
- `DoCopyOnDefaultStream()`

### Stream and handle access

- `PerThreadDefaultCublasHandle()`
- `PerThreadDefaultCudnnHandle()`
- `PerThreadCublasLtHandle()`
- `ComputeStream()`

Initial implementation note:

These methods may delegate to the plugin factory's compute stream and its associated handles. The document and code should describe this as current plugin behavior compatibility, not full per-thread parity with the framework EP.

### Data transfer

- `GetDataTransfer()` — **required by `adapter::Ep`**

`adapter::Ep`'s constructor calls `impl_->GetDataTransfer()` and stores the result in a `DataTransferManager` that dereferences the pointer without a null check. The default `IExecutionProvider::GetDataTransfer()` returns `nullptr`, so the plugin-side `CUDAExecutionProvider` **must** override it.

The existing `CudaDataTransfer` class (`cuda_data_transfer_plugin.h`) implements `OrtDataTransferImpl`, not `IDataTransfer`. The simplest approach for the first pass is to create a thin `CudaPluginDataTransfer : public IDataTransfer` in a separate internal file such as `cuda_idata_transfer_plugin.h`, wrap `cudaMemcpyAsync` for GPU↔CPU copies, and return it from `GetDataTransfer()`.

Important separation of concerns:

- Keep `CreateDataTransferImpl` and `CudaDataTransfer` for the public plugin factory callback surface.
- Add a separate internal `IDataTransfer` implementation used only by `CUDAExecutionProvider::GetDataTransfer()` and `adapter::Ep`.

### Compatibility helpers and low-priority stubs

- `GetAttentionKernelOptions()`
- `GetTuningContext()`
- `GetConstOnes<T>()`

Recommended choice for `GetConstOnes<T>()`:

Put it on the provider so the plugin-side API matches the framework CUDA EP shape. The implementation should still reuse the current static `std::once_flag` pattern internally (as seen in the current `CudaKernel::GetConstOnes<T>`) to ensure the constant buffer is safely allocated across multiple threads in the compatibility layer.

## Why This Design Is Better Than The Current One

### Better than the current static-atomic design

- configuration becomes EP-scoped instead of process-scoped
- multiple EP instances become structurally safer
- kernels use the same provider-access pattern as the framework CUDA kernels
- provider capabilities stop being split between static globals and thin shims

### Better than a separate `CudaEpProvider` type

- no kernel cast churn
- no alias-based ambiguity
- easier code review because the compatibility provider clearly occupies the existing adaptation point
- simpler long-term migration if more framework CUDA kernels are enabled in the plugin build

### Better than trying to copy WebGPU literally

- preserves the good ownership pattern from WebGPU
- avoids pretending the framework `CUDAExecutionProvider` can be linked into the plugin today
- respects the real CUDA stream and handle constraints already present in this repo

## Implementation Plan

### Phase 1: Convert the provider shim into an instance-backed compatibility provider

Modify `onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h` so the plugin-defined `CUDAExecutionProvider` is no longer a shim that reads state from static globals, but instead an instance-backed class:

```cpp
class CUDAExecutionProvider : public IExecutionProvider {
public:
  CUDAExecutionProvider(CudaEpFactory& factory, int device_id, bool use_tf32,
                        int cudnn_conv_algo, bool cudnn_conv1d_pad_to_nc1d,
                        bool cudnn_conv_use_max_workspace,
                        bool skip_layer_norm_strict_mode, bool prefer_nhwc);

  // --- Config accessors (instance-backed) ---
  const cudaDeviceProp& GetDeviceProp() const { return device_prop_; }
  int GetDeviceId() const override { return device_id_; }
  bool UseTF32() const { return use_tf32_; }
  // ... other config getters ...

  // --- Handle accessors (delegate to factory's compute stream) ---
  cublasHandle_t PerThreadDefaultCublasHandle() const {
    auto* stream = factory_.GetComputeStream();
    return stream ? stream->GetCublasHandle() : nullptr;
  }
  // ... same for cudnn, cublasLt, and ComputeStream ...

  // --- Data transfer (required by adapter::Ep) ---
  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;

private:
  CudaEpFactory& factory_;
  int device_id_;
  cudaDeviceProp device_prop_{};
  bool use_tf32_;
  // ... other fields
};
```

At the same time, remove the static configuration path:

- delete `CudaKernelAdapterRuntimeConfig`
- delete `GetCudaKernelAdapterRuntimeConfig()`
- delete `SetCudaKernelAdapterRuntimeConfig()`
- remove the call site in `cuda_ep.cc`

Phase 1 should also wire the missing convolution workspace option end to end:

- add `cudnn_conv_use_max_workspace` to `CudaEp::Config`
- parse it from session options in the factory before constructing the provider
- store it on the plugin-side `CUDAExecutionProvider` and expose `GetCudnnConvUseMaxWorkspace()`

Implementation note:

Do not attempt to override `Type()`. Set the provider type through the `IExecutionProvider` base constructor only.

### Phase 2: Rewrite `CudaKernel` around the provider instance

Update the plugin's `CudaKernel` in `onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h` to cache:

```cpp
provider_ = static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider());
```

Then route provider-backed methods through that pointer (e.g. `provider_->PerThreadDefaultCublasHandle()`).

**Important**: Do not modify the framework's `core/providers/cuda/cuda_kernel.h`. This change applies only to the plugin adapter.

Keep the static stream-handle helper overloads unchanged:

- `GetCudnnHandle(cudaStream_t)`
- `GetCudnnHandle(CudaStream*)`
- `GetCudnnHandle(Stream*)`
- `GetCublasHandle(cudaStream_t)`
- `GetCublasHandle(CudaStream*)`
- `GetCublasHandle(Stream*)`

Those helpers still need the global reverse-lookup map in the first iteration.

### Phase 3: Add allocators and move `CudaEp` to `adapter::Ep`

These two pieces must land together because the `adapter::Ep` constructor requires `AllocatorPtr` arguments.

**Allocator setup:**

- Create `onnxruntime/core/providers/cuda/plugin/cuda_iallocator_plugin.h` wrapping `cudaMalloc`/`cudaFree` in a `CudaDeviceIAllocator` class.
- `cuda_ep_factory.h` and `cuda_ep_factory.cc` updated to own `AllocatorPtr` instances for device and pinned host memory.
- For the CPU temp allocator required by `adapter::Ep`, recycle `onnxruntime::CPUAllocator::DefaultInstance()` (just as the WebGPU plugin does), avoiding the need to invent a new CPU allocator.
- C API allocator creation routed through `adapter::Allocator`.
- Extend `CudaEp::Config` and factory parsing to include `cudnn_conv_use_max_workspace`, since `conv_8.h` requires `GetCudnnConvUseMaxWorkspace()` parity.

**`CudaEp` transition:**

Modify `onnxruntime/core/providers/cuda/plugin/cuda_ep.h` and `onnxruntime/core/providers/cuda/plugin/cuda_ep.cc` so `CudaEp` inherits from `adapter::Ep` and owns the plugin-side `CUDAExecutionProvider`.

Factory flow (`CreateEpImpl`) should become:

```cpp
// 1. Parse config
// 2. Construct the plugin-side CUDAExecutionProvider
auto provider = std::make_unique<CUDAExecutionProvider>(*factory, config.device_id, ...);

// 3. Construct CudaEp wrapper with provider and allocators
auto actual_ep = std::make_unique<CudaEp>(
    std::move(provider), *factory, config, ep_logger,
    factory->cpu_allocator_, factory->device_allocator_);

// 4. Return as the plugin EP
*ep = actual_ep.release();
```

This is the structural change that lets `info.GetExecutionProvider()` return the correct provider instance (`EpImpl()`) instead of leaving kernels dependent on a disconnected shim.

### Phase 4: Make the minimum kernel source changes that are actually worthwhile

Keep kernel changes narrow.

Recommended first cleanup:

- Delete the dead `const CUDAExecutionProvider* cuda_ep_;` member from `onnxruntime/core/providers/cuda/reduction/reduction_ops.h`. Since `CudaKernel` now caches the `provider_` correctly, this old member is obsolete.

Beyond that, only change kernel sources where the provider refactor exposes a real incompatibility. Avoid broad mechanical edits.

### Phase 5: Re-enable excluded CUDA areas incrementally

Do not treat temporary build exclusions as permanent architecture boundaries.

Re-enable excluded operator areas in stages once the provider surface is present and tested:

1. conv-related gaps already identified by current kernels
2. controlflow and einsum paths that call provider methods directly
3. remaining excluded CUDA areas such as rnn, llm, and attention-dependent code

Tunable support can remain stubbed until the basic provider migration is stable.

## Verification Plan

Run both non-plugin and plugin validation after each major phase:

```bash
./cuda.sh --build --test
./cuda_plugin.sh --build --test --test_plugin
```

Also verify the following invariants:

1. `CudaKernelAdapterRuntimeConfig` no longer exists.
2. `CudaEp` now derives from `adapter::Ep`.
3. `info.GetExecutionProvider()` reaches the plugin-side `CUDAExecutionProvider` instance.
4. Static stream-handle helper overloads still compile and still use the current reverse-lookup map.
5. The plugin provider now defines `GetCudnnConvUseMaxWorkspace()`.
6. No broad CUDA kernel churn was introduced outside targeted cleanup.

## File-Level Worklist

Primary files expected to change:

- `onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h`
- `onnxruntime/core/providers/cuda/plugin/cuda_ep.h`
- `onnxruntime/core/providers/cuda/plugin/cuda_ep.cc`
- `onnxruntime/core/providers/cuda/plugin/cuda_ep_factory.h`
- `onnxruntime/core/providers/cuda/plugin/cuda_ep_factory.cc`
- new plugin allocator file(s)
- new plugin `IDataTransfer` implementation (e.g. `cuda_idata_transfer_plugin.h`)
- `onnxruntime/core/providers/cuda/reduction/reduction_ops.h`
- `cmake/onnxruntime_providers_cuda_plugin.cmake`

Files that should stay unchanged in the first pass unless a concrete build issue forces a touch:

- `onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.h`
- `onnxruntime/core/providers/cuda/plugin/cuda_stream_plugin.cc`
- most CUDA kernel source files outside targeted cleanup

## Risk Assessment

| Risk | Likelihood | Mitigation |
|------|------------|------------|
| `GetDataTransfer()` returns null, crashing `DataTransferManager` | **High** if missed | Override `GetDataTransfer()` on the plugin-side provider (see Phase 1 and Required Provider Surface) |
| Handle methods (`PerThreadDefaultCublasHandle`) return null before stream creation | Medium | Document that these are only valid after `CreateSyncStreamForDevice` has been called; guard with null checks in delegation |
| CUDA graph capture breaks | Medium | `CudaEp` retains all graph logic; explicitly test with `enable_cuda_graph=true` |
| `adapter::Ep` header not available in plugin build | Low | Already used by WebGPU plugin |
| `CUDAExecutionProvider` name collision with framework class | None | Plugin DLL has its own copy; framework header is excluded from plugin build |

## Non-Goals For The First Refactor

- replacing the global stream reverse-lookup map
- implementing full framework per-thread handle semantics
- wiring a real tuning context
- linking the real framework `CUDAExecutionProvider` into the plugin build

These are valid future improvements, but they should not block the provider ownership and compatibility cleanup.

## Final Recommendation

Proceed with the instance-backed plugin-side `CUDAExecutionProvider` plus `adapter::Ep` refactor.

That is the best balance of correctness, compatibility, and implementation cost in the current repo:

- it fixes the root structural problem, which is the lack of an EP-owned provider instance
- it preserves existing CUDA kernel integration points
- it borrows the right pattern from WebGPU without forcing a false equivalence between the two plugins
- it creates a credible path to full operator coverage instead of hard-coding the current temporary exclusions into the design
