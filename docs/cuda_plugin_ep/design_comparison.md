# CUDA Plugin EP Design: WebGPU vs CUDA Comparison & Proposed Improvements

## 1. Overview

This document compares the WebGPU plugin EP and CUDA plugin EP architectures, identifies design gaps in the CUDA plugin, and proposes borrowing the WebGPU pattern to improve the CUDA plugin EP.

**End goal:** The CUDA plugin EP must support all operators the current framework CUDA EP supports (tunable ops are low priority). Current cmake exclusions (einsum, controlflow, rnn, llm, attention) are temporary.

**Commits analyzed:**
- `05be55ec` — WebGPU plugin EP prototype
- `4379b054` — CUDA plugin EP prototype

---

## 2. WebGPU Plugin EP Architecture

### 2.1 Class Hierarchy

```
OrtEpFactory                             OrtEp
    ↑                                      ↑
webgpu::ep::Factory               adapter::Ep (holds unique_ptr<IExecutionProvider>)
                                           ↑
                                   webgpu::ep::Ep (wraps WebGpuExecutionProvider)
```

### 2.2 Key Design Pattern: Wrapping the Real EP

The WebGPU plugin creates the **real, non-plugin `WebGpuExecutionProvider`** inside the plugin factory, then wraps it in a thin adapter:

```cpp
// factory.cc — CreateEpImpl
auto webgpu_ep_factory = WebGpuProviderFactoryCreator::Create(config_options);
auto webgpu_ep = webgpu_ep_factory->CreateProvider(*session_options, *logger);
static_cast<WebGpuExecutionProvider*>(webgpu_ep.get())->SetEpLogger(logger);
auto factory = static_cast<Factory*>(this_ptr);
*ep = new Ep(std::move(webgpu_ep), *factory, *logger, factory->config_);
```

The `Ep` class inherits from `adapter::Ep`, which stores the underlying EP in a `unique_ptr<IExecutionProvider> impl_` member. The helper `EpImpl()` returns it:

```cpp
// include/onnxruntime/ep/adapter/ep.h
class Ep : public OrtEp {
protected:
  explicit Ep(std::unique_ptr<IExecutionProvider> impl, AllocatorPtr cpu_alloc, AllocatorPtr device_alloc);
public:
  inline IExecutionProvider* EpImpl() const noexcept { return impl_.get(); }
};
```

### 2.3 Kernel Access Pattern

WebGPU kernels access the full `WebGpuExecutionProvider` via `info.GetExecutionProvider()`:

```cpp
WebGpuKernel::WebGpuKernel(const OpKernelInfo& info)
    : OpKernel(info),
      ep_(*static_cast<const WebGpuExecutionProvider*>(info.GetExecutionProvider())),
      webgpu_context_(WebGpuContextFactory::GetContext(ep_.GetDeviceId())) {}
```

### 2.4 Benefits

- **Zero global state** — all config/resources live in the provider instance
- **Zero shim classes** — kernels cast to the real EP type
- **Full API parity** — plugin kernels have the exact same access as framework kernels
- **Clean ownership** — `adapter::Ep` owns the provider; lifetime is well-defined

### 2.5 Limitations

WebGPU is **not stream-aware** (`CreateSyncStreamForDevice` returns `ORT_NOT_IMPLEMENTED`), so it does not face the stream-ownership problem described in §3.3.

---

## 3. CUDA Plugin EP Architecture (Current)

### 3.1 Class Hierarchy

```
OrtEpFactory                   OrtEp
    ↑                            ↑
CudaEpFactory              CudaEp (: OrtEp, NO underlying IExecutionProvider)
```

### 3.2 Configuration via Static Atomics

Config is pushed into **global static atomics** because there is no provider instance:

```cpp
struct CudaKernelAdapterRuntimeConfig {
  std::atomic<bool> use_tf32{true};
  std::atomic<bool> skip_layer_norm_strict_mode{false};
  std::atomic<int> device_id{0};
  std::atomic<int> cudnn_conv_algo{0};
  std::atomic<bool> cudnn_conv1d_pad_to_nc1d{false};
};
```

### 3.3 Stream and Handle Access via Global Map

cuBLAS/cuDNN handles are owned by `CudaSyncStream` objects and looked up via a global map:

```cpp
static std::unordered_map<cudaStream_t, CudaSyncStream*>* g_stream_map;
CudaSyncStream* CudaSyncStream::FromCudaStream(cudaStream_t stream) { /* mutex + map lookup */ }
```

**Stream ownership constraint:** `CreateSyncStreamForDeviceImpl` receives only `(OrtEpFactory*, ...)` — no `OrtEp*` parameter. Stream creation is factory-scoped.

### 3.4 Shim CUDAExecutionProvider

A minimal shim exists in `cuda_kernel_adapter.h` to satisfy kernel code that casts `info.GetExecutionProvider()`:

```cpp
class CUDAExecutionProvider : public IExecutionProvider {
  int GetCudnnConvAlgo() const;         // reads static atomic
  bool GetCudnnConv1dPadToNc1d() const; // reads static atomic
  bool UseTF32() const;                 // reads static atomic
  bool IsFuseConvBias() const;          // returns false
  const cudaDeviceProp& GetDeviceProp() const; // lazy-cached
};
```

**Missing from current shim** (needed by `conv_8.h`): `GetCudnnConvUseMaxWorkspace()`.

---

## 4. Provider Method Compatibility Audit

### 4.1 Kernel access patterns

CUDA kernels access the provider in two ways:

1. **Via `CudaKernel::provider_`** — every method on the framework `CudaKernel` delegates to `provider_`:
   - `GetDeviceProp()`, `UseTF32()`, `GetDeviceId()`
   - `GetAttentionKernelOptions()`, `GetTuningContext()`
   - `PerThreadDefaultCublasHandle()`, `PerThreadDefaultCudnnHandle()`, `PerThreadCublasLtHandle()`
   - `ComputeStream()`, `GetConstOnes<T>()`

2. **Via direct cast** — kernels cast `info.GetExecutionProvider()` to `CUDAExecutionProvider*`:
   - `conv_8.h`: `GetCudnnConv1dPadToNc1d()`, `GetCudnnConvAlgo()`, `GetCudnnConvUseMaxWorkspace()`
   - `conv.cc`, `conv_transpose.cc`: above + `UseTF32()`, `IsFuseConvBias()`
   - `conv_transpose_8.h`: `GetCudnnConv1dPadToNc1d()`
   - `loop.cc` (currently excluded): `DoCopyOnDefaultStream()`
   - `reduction_ops.h`: stores `cuda_ep_` but never dereferences (dead member)
   - `einsum.h` (currently excluded): stores `cuda_ep_` for `EinsumCudaAssets`

### 4.2 Full required provider surface

Since all operators must eventually be supported, the plugin-side provider must expose:

| Method | Source | Plugin Implementation |
|--------|--------|-----------------------|
| `GetDeviceProp()` | config | Instance member `cudaDeviceProp` |
| `GetDeviceId()` | config | Instance member |
| `UseTF32()` | config | Instance member |
| `GetCudnnConvAlgo()` | config | Instance member |
| `GetCudnnConv1dPadToNc1d()` | config | Instance member |
| `GetCudnnConvUseMaxWorkspace()` | config | Instance member (currently missing!) |
| `DoCopyOnDefaultStream()` | config | Always `true` (plugin does not support dedicated copy stream) |
| `IsNHWCPreferred()` | config | Instance member |
| `IsFuseConvBias()` | config | Always `false` (no fused conv bias support) |
| `IsSkipLayerNormInStrictMode()` | config | Instance member |
| `GetAttentionKernelOptions()` | stub | Return static default instance |
| `GetTuningContext()` | stub | Return stub with `IsTunableOpEnabled() = false` |
| `PerThreadDefaultCublasHandle()` | stream | Delegate to factory's compute stream |
| `PerThreadDefaultCudnnHandle()` | stream | Delegate to factory's compute stream |
| `PerThreadCublasLtHandle()` | stream | Delegate to factory's compute stream |
| `ComputeStream()` | stream | Delegate to factory's compute stream |
| `GetConstOnes<T>()` | buffer | Static once_flag + `IConstantBuffer` (same as current) |

---

## 5. Comparison: Framework vs Plugin CudaKernel

| Aspect | Framework `CudaKernel` | Plugin (current) | Plugin (proposed) |
|--------|------------------------|-------------------|-------------------|
| **Provider access** | `provider_` = `CUDAExecutionProvider*` via `info.GetExecutionProvider()` | Static atomics + shim class | Instance-backed `CUDAExecutionProvider` via `info.GetExecutionProvider()` |
| **`Stream(ctx)`** | `ctx->GetComputeStream()->GetHandle()` | `ctx->GetGPUComputeStream()` | Same as current |
| **`GetCudnnHandle()`** | `stream->cudnn_handle_` — O(1) | Global map lookup — O(1) avg + mutex | Same global map (unchanged) |
| **`GetDeviceProp()`** | `provider_->GetDeviceProp()` | Cached from static atomic | `provider_->GetDeviceProp()` — instance member |
| **`UseTF32()`** | `provider_->UseTF32()` | `use_tf32_` from static atomic | `provider_->UseTF32()` — instance member |
| **`DefaultCublasHandle()`** | `provider_->PerThread...()` | Not available | `provider_->PerThread...()` → factory compute stream |
| **`GetConstOnes()`** | `provider_->GetConstOnes()` | Static once_flag on `CudaKernel` | Same static once_flag (stays on `CudaKernel` for now) |
| **`GetTuningContext()`** | Real `ITuningContext*` | Stub | Same stub (low priority) |
| **Config source** | EP instance members | Global static atomics | EP instance members |

---

## 6. Why the WebGPU Pattern Cannot Be Directly Copied

WebGPU wraps the **real** `WebGpuExecutionProvider` because it has lightweight dependencies. CUDA cannot include `CUDAExecutionProvider` because it depends on `BFCArena`, `PerThreadContext`, and provider bridge types excluded from the plugin build.

**What we borrow:** The `adapter::Ep` wrapping pattern and `info.GetExecutionProvider()` kernel access pattern. Instead of the real `CUDAExecutionProvider`, we keep the **class name** `CUDAExecutionProvider` and make it an instance-backed compatibility class with the full API surface. This is the minimal-change approach: zero kernel cast changes, zero type-alias confusion.

---

## 7. Architecture Diagrams

### Current

```
┌─────────────────────────────────────────────────────┐
│                    Plugin DLL                        │
│                                                     │
│  CudaEpFactory ──creates──→ CudaEp (: OrtEp)       │
│       └──creates──→ CudaSyncStream (owns handles)   │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ Global State                                 │   │
│  │ • CudaKernelAdapterRuntimeConfig (atomics)   │   │
│  │ • g_stream_map (mutex-locked map)            │   │
│  │ • Shim CUDAExecutionProvider (hollow)        │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```

### Proposed

```
┌─────────────────────────────────────────────────────┐
│                    Plugin DLL                        │
│                                                     │
│  CudaEpFactory ──creates──→ CudaEp (: adapter::Ep) │
│       │                           │                 │
│       │                           └──owns──→        │
│       │                     CUDAExecutionProvider    │
│       │                     (: IExecutionProvider)   │
│       │                     ├─ all config members    │
│       │                     ├─ device properties     │
│       │                     └─ factory& (handles)    │
│       │                                             │
│       └──creates──→ CudaSyncStream (owns handles)   │
│                                                     │
│  CudaKernel gets CUDAExecutionProvider* via         │
│    info.GetExecutionProvider() → EpImpl()           │
│    (exact same cast as framework kernels)           │
│                                                     │
│  ┌──────────────────────────────────────────────┐   │
│  │ Static atomics ELIMINATED                    │   │
│  │ Stream map stays global (future work)        │   │
│  └──────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┘
```
