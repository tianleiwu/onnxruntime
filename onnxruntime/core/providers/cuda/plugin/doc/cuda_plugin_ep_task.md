# CUDA EP Plugin Migration — Implementation Tasks

> **Plan**: [cuda_plugin_ep_plan.md](cuda_plugin_ep_plan.md)
> **Prototype commit**: `31a6e1d2b96801033053f559bbd20d817ea8642e`
> **EP Adapter commit**: `72a4cd7025a8740f2f8996f9899f898223c34941`
> **Build & Test**: `./cuda_plugin.sh --build --test --test_plugin`

---

## Stage 1: Plugin Shell & Infrastructure

> **Goal**: EP loads via plugin API, creates streams/allocators, runs Memcpy + simple ops.
> Items 1.1–1.9 are already done in the prototype. Work focuses on 1.10–1.12.

### 1.10 Remove `SHARED_PROVIDER` Bridge

Remove all `provider_api.h` / `g_host` / `ProviderHost_impl.h` dependencies from the plugin build.
The `SHARED_PROVIDER` bridge's namespace-level type stubs **conflict** with the adapter's `using` declarations and must be severed **before** the adapter forced include is enabled.

- [ ] **1.10.1** Delete [provider_host_bridge.cc](../provider_host_bridge.cc) from plugin sources
  - This file initializes `g_host = Provider_GetHost()` under `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER`
  - Currently 19 LOC, fully guarded — safe to delete since `ORT_CUDA_PLUGIN_USE_ADAPTER=1` is always set
  - Also remove from CMake source list if explicitly listed

- [ ] **1.10.2** Refactor [provider_api_shims.cc](../provider_api_shims.cc) to remove dual-path pattern
  - Currently has `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER` / `#else` blocks
  - Remove the `#ifndef` (legacy) path entirely — keep only the adapter implementations:
    - `GetEnvironmentVar()` → `std::getenv()`
    - `math::floatToHalf()` → `MLFloat16(f).val`
    - `math::halfToFloat()` → `MLFloat16::FromBits(h).ToFloat()`
  - Remove `#include "core/providers/shared_library/provider_api.h"` include

- [ ] **1.10.3** Audit and remove all `provider_api.h` includes from plugin-compiled files
  - Run: `grep -rn 'provider_api\.h\|SHARED_PROVIDER\|g_host' onnxruntime/core/providers/cuda/plugin/`
  - Address each hit — the force-included [cuda_kernel_adapter.h](../cuda_kernel_adapter.h) currently includes `provider_api.h` under `#if SHARED_PROVIDER` (line ~76)
  - The `SHARED_PROVIDER` define/include block in `cuda_kernel_adapter.h` (lines 72–76) should be removed on the adapter path

- [ ] **1.10.4** Remove `core/providers/shared/common.cc` from plugin build if present
  - This provides `Provider_GetHost()` which is only needed for the legacy bridge
  - Verify in CMake that this file is not collected by the glob patterns

- [ ] **1.10.5** Verify: `grep -rn 'provider_api\.h\|SHARED_PROVIDER\|g_host' plugin/` returns zero matches

### 1.11 EP Adapter Forced-Include Integration

Refactor `cuda_kernel_adapter.h` to inherit from `adapter::OpKernel` (from `ep/adapters.h`) instead of the `SHARED_PROVIDER` bridge types.

- [ ] **1.11.1** Remove `AdapterKernelImpl` class from `cuda_kernel_adapter.h` (lines ~225–247)
  - This is replaced by `ep::adapter::KernelImpl` in [adapter/op_kernel.h](../../../../../../include/onnxruntime/ep/adapter/op_kernel.h)
  - The adapter framework's `KernelImpl` already wraps `OpKernel` → `OrtKernelImpl` with `Compute`/`Release`/`PrePackWeight`

- [ ] **1.11.2** Remove `PluginRegistry` class from `cuda_kernel_adapter.h` (if present)
  - Replaced by `ep::adapter::KernelRegistry` in [adapter/kernel_registry.h](../../../../../../include/onnxruntime/ep/adapter/kernel_registry.h)

- [ ] **1.11.3** Refactor `CudaKernel` base class in `cuda_kernel_adapter.h`
  - Currently provides CUDA-specific methods: `Stream()`, `GetCublasHandle()`, `GetCudnnHandle()`, `GetCublasLtHandle()`, `GetScratchBuffer<T>()`, `GetDeviceProp()`, `UseTF32()`
  - Must inherit from `ep::adapter::OpKernel` instead of current base
  - Keep all CUDA-specific accessor methods — these are what `adapter::OpKernel` does *not* provide
  - Preserve the runtime config pattern (`CudaKernelAdapterRuntimeConfig` with atomics for `use_tf32`, `device_id`, etc.)

- [ ] **1.11.4** Switch forced include from `cuda_kernel_adapter.h` to `ep/adapters.h`
  - In [onnxruntime_providers_cuda_plugin.cmake](../../../../../../cmake/onnxruntime_providers_cuda_plugin.cmake) line 126:
    ```cmake
    # Before:
    "$<$<COMPILE_LANGUAGE:CXX>:-include;${CUDA_PLUGIN_EP_DIR}/cuda_kernel_adapter.h>"
    # After:
    "$<$<COMPILE_LANGUAGE:CXX>:-include;${REPO_ROOT}/include/onnxruntime/ep/adapters.h>"
    ```
  - `cuda_kernel_adapter.h` is **not removed** — it continues to provide the `CudaKernel` base class
  - It will be included explicitly by files that need CUDA-specific accessors (e.g., `cuda_plugin_kernels.cu`)

- [ ] **1.11.5** Resolve namespace conflicts
  - `ep/adapters.h` defines `EP_SPECIFIC_USING_DECLARATIONS` in `onnxruntime::cuda` and `onnxruntime::contrib::cuda`
  - These remap `OpKernel`, `OpKernelContext`, `KernelDefBuilder`, etc. to adapter types
  - Verify that `cuda_kernel_adapter.h`'s own `using` declarations (lines ~96–101, 128–150) do not conflict
  - Remove duplicate aliases from `cuda_kernel_adapter.h` that are now provided by `adapters.h`
  - Domain constant shadowing (`#define kOnnxDomain __kOnnxDomain_ignore`, lines 77–85) should also be reviewed — the adapter framework may handle this differently

- [ ] **1.11.6** Remove the `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER` legacy path from `cuda_kernel_adapter.h`
  - The file currently has two major branches: SHARED_PROVIDER path (lines ~70–105) and adapter path (lines ~107+)
  - Since `ORT_CUDA_PLUGIN_USE_ADAPTER=1` is always set, remove the SHARED_PROVIDER branch entirely
  - Simplify the file to only contain:
    - The `CudaKernelAdapterRuntimeConfig` struct
    - The `CudaKernel` base class (inheriting from `ep::adapter::OpKernel`)
    - The `CUDAExecutionProvider` shim class
    - Error-return macros (`CUDA_RETURN_IF_ERROR`, `CUBLAS_RETURN_IF_ERROR`, `CUDNN_RETURN_IF_ERROR`)

- [ ] **1.11.7** Remove the `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER` legacy path from `cuda_plugin_kernels.cu`
  - Lines 48–97 define `AdapterKernelImpl` template under `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER`
  - Remove this block entirely — the adapter path uses `ep::adapter::KernelImpl` instead

### 1.12 Validate Stage 1

- [ ] **1.12.1** Build plugin: `./cuda_plugin.sh --build`
  - Verify clean compilation with zero warnings related to adapter/SHARED_PROVIDER conflicts
- [ ] **1.12.2** Run C++ tests: `./cuda_plugin.sh --build --test`
  - `onnxruntime_test_all` with plugin mode should pass
- [ ] **1.12.3** Run plugin Python tests: `./cuda_plugin.sh --build --test --test_plugin`
  - `test_cuda_plugin_ep.py`: Add, MatMul, Gemm, Conv should all pass
- [ ] **1.12.4** Verify no SHARED_PROVIDER references:
  ```bash
  grep -rn 'provider_api\.h\|SHARED_PROVIDER\|g_host' \
    onnxruntime/core/providers/cuda/plugin/
  ```
  Should return zero matches.

---

## Stage 2: Kernel Registration Migration

> **Goal**: All existing CUDA kernel registrations work through the EP adapter's `KernelRegistry`.

### 2.1 Standard Op Registrations

- [ ] **2.1.1** Integrate existing registration tables into adapter-based registry
  - Files: [cuda_execution_provider.cc](../../../cuda/cuda_execution_provider.cc) contains `GetCudaKernelList()`
  - These `ONNX_OPERATOR_*_KERNEL_EX` macros produce `BuildKernelCreateInfo<>()` specializations
  - With the adapter forced-include, these macros automatically produce `adapter::KernelCreateInfo`
  - Wire up: in `CudaEpFactory::GetKernelRegistryForEp()`, call `RegisterCudaKernels()` which iterates the kernel list and calls `adapter::KernelRegistry::Register(KernelCreateInfo&&)` for each

- [ ] **2.1.2** Create a new registration function that uses the existing kernel tables
  - Function: `RegisterCudaKernels(adapter::KernelRegistry& registry)`
  - Iterate `BuildKernelCreateInfo` function pointers from the existing registration tables
  - Call `registry.Register(build_fn())` for each
  - This replaces the manual `kAdapterRegistrations[]` table in `cuda_plugin_adapter_registry.cc`

- [ ] **2.1.3** Test: verify basic ops (Add, Relu, MatMul, Gemm, Conv, Softmax, etc.) still work

### 2.2 NHWC Registrations

- [ ] **2.2.1** Port [cuda_nhwc_kernels.cc](../../cuda/cuda_nhwc_kernels.cc) registrations to plugin
  - Currently excluded by CMake filter (line 76): `.*/cuda_nhwc_kernels\.cc$`
  - The file registers kernels under `kMSInternalNHWCDomain`
  - With adapter forced-include, the same macros should work
  - Remove the CMake exclusion and verify compilation
  - Alternative: Create a new `RegisterCudaNhwcKernels()` adapted function if the existing file has EP infrastructure deps

### 2.3 Contrib Op Registrations

- [ ] **2.3.1** Wire up [cuda_contrib_kernels.cc](../../../../contrib_ops/cuda/cuda_contrib_kernels.cc) registration table
  - This file defines `GetCudaContribKernelList()` with `com.microsoft` domain ops
  - Same adapter approach: iterate list, call `registry.Register()` for each
  - Function: `RegisterCudaContribKernels(adapter::KernelRegistry& registry)`

### 2.4 Resolve Excluded Ops

- [ ] **2.4.1** Control flow ops (`If`/`Loop`/`Scan`)
  - These inherit from CPU base classes — cannot compile directly
  - Use `OrtEpApi::CreateIfKernel`/`CreateLoopKernel`/`CreateScanKernel`
  - Override `OpKernel::CreateControlFlowKernelImpl()` (already supported by `adapter::KernelRegistry::CreateKernel()`)
  - CMake currently excludes entire `controlflow/` dir — create wrapper classes
  - See task 5.1 for full implementation

- [ ] **2.4.2** Document remaining excluded op categories for Stage 5
  - RNN ops: `rnn/` directory excluded (dynamic_cast issue)
  - Tunable ops: `tunable/` directory excluded (CudaTuningContext dep)
  - Einsum: `math/einsum.cc` excluded (cuda_execution_provider.h dep)
  - Object detection: `object_detection/` excluded (CPU base class dep)
  - Identity/Sequence: `identity_op.cc`, `sequence_op.cc` excluded (TensorSeq incomplete type)
  - ScatterND: `scatter_nd.cc` excluded (CPU validation)
  - Size: `size.cc` excluded (CPU op)
  - IntegerGemm: `integer_gemm.cc` excluded (CudaStream dep)

### 2.5 Remove Manual Registry

- [ ] **2.5.1** Delete [cuda_plugin_adapter_registry.cc](../cuda_plugin_adapter_registry.cc) (~339 LOC)
  - Contains `kAdapterRegistrations[]` static table and `CreateCudaKernelRegistryFromOrtTables()`
  - Fully replaced by the adapter-based registry path

- [ ] **2.5.2** Remove the legacy `Create*Kernel` functions and `AdapterKernelImpl` template from [cuda_plugin_kernels.cu](../cuda_plugin_kernels.cu)
  - Lines 48–97: `AdapterKernelImpl` template under `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER`
  - Already addressed in 1.11.7 — verify it's clean

- [ ] **2.5.3** Update CMake to remove references to deleted files
  - `cuda_plugin_adapter_registry.cc` is explicitly added in CMake line 112 — remove it

### 2.6 Validate Stage 2

- [ ] **2.6.1** Registration parity verification
  - Dump `(domain, op_type, since_version, type_constraints)` tuples from both plugin and bundled EP registries
  - Plugin count must equal bundled count minus tracked exclusions
  - Create a diagnostic tool or test to automate this comparison

- [ ] **2.6.2** Build and test
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ```
  All existing tests should pass.

---

## Stage 3: NHWC & GetCapability Integration

> **Goal**: NHWC layout transformation and CPU-preferred-node logic work correctly.

### 3.1 `ShouldConvertDataLayoutForOp`

- [ ] **3.1.1** Implement `ShouldConvertDataLayoutForOpImpl` callback on `CudaEp`
  - Add static method to [cuda_ep.h](../cuda_ep.h) / [cuda_ep.cc](../cuda_ep.cc)
  - Same op list as `CUDAExecutionProvider::ShouldConvertDataLayoutForOp()`:
    - ONNX ops: `BatchNormalization`, `Conv`, `ConvTranspose`, `GlobalMaxPool`, `MaxPool`, `GlobalAveragePool`, `AveragePool`, `GridSample`, `DepthToSpace`, `SpaceToDepth`, `LRN`
    - MS domain: `GridSample`
  - Wire up in `CudaEp` constructor's `OrtEp` callback table
  - Returns `>0` (convert), `0` (don't), or `<0` (let ORT decide)

### 3.2 `GetCapability` with CPU-Preferred Nodes

- [ ] **3.2.1** Integrate `GetCpuPreferredNodes` in `CudaEp::GetCapabilityImpl()`
  - Use [get_capability_utils.h](../../../../../../include/onnxruntime/ep/get_capability_utils.h)
  - Flow:
    1. Iterate all graph nodes via `Ort::ConstGraph::GetNodes()`
    2. Call `ep_api.EpGraphSupportInfo_LookUpKernel()` for each node
    3. Collect tentative nodes (ones we have kernels for)
    4. Call `ep::GetCpuPreferredNodes()` to filter out CPU-preferred nodes (e.g., `Shape`, `NonZero`)
    5. Add final nodes via `ep_api.EpGraphSupportInfo_AddSingleNode()`
  - Currently `GetCapabilityImpl` may be a simple all-or-nothing — needs the CPU-preferred filtering

### 3.3 NHWC Kernel Validation

- [ ] **3.3.1** Add NHWC test cases to `test_cuda_plugin_ep.py`
  - Test Conv, BatchNorm, Pool ops with `prefer_nhwc=true` session option
  - Set session option: `ep.cuda.prefer_nhwc_layout = "1"`
  - Verify correctness against PyTorch reference

- [ ] **3.3.2** Test with reference models
  - ResNet-50 with NHWC enabled — verify outputs match bundled EP within tolerance
  - If EfficientNet-B0 model is available, test that as well

### 3.4 Validate Stage 3

- [ ] **3.4.1** Verify `GetCapability` claims same nodes as bundled EP for a reference model
- [ ] **3.4.2** Verify CPU-fallback nodes (Shape, NonZero, etc.) correctly left on CPU
- [ ] **3.4.3** Build and test:
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ```

---

## Stage 4: CUDA Graph Integration

> **Goal**: Full CUDA Graph capture/replay via plugin EP.

### 4.1 Port `CUDAGraphManager`

- [ ] **4.1.1** Copy/adapt [cuda_graph.h](../../cuda/cuda_graph.h) / [cuda_graph.cc](../../cuda/cuda_graph.cc) into plugin
  - Currently excluded by CMake: `.*/cuda_graph\.cc$`
  - Create `plugin/cuda_graph_plugin.h` and `plugin/cuda_graph_plugin.cc`
  - Remove dependencies on internal EP types (`CUDAExecutionProvider`, `CudaStream`)
  - Use `CudaSyncStream` and `CudaEp` instead
  - Key class: `CUDAGraphManager` — stores `cudaGraphExec_t` per annotation ID, manages capture/end/replay lifecycle

### 4.2 `OnRunStart` Implementation

- [ ] **4.2.1** Implement `CudaEp::OnRunStartImpl()` in [cuda_ep.cc](../cuda_ep.cc)
  - Parse `OrtRunOptions` config entries:
    - `ep.cuda.enable_cuda_graph` (bool)
    - `ep.cuda.cuda_graph_annotation_id` (int)
    - `ep.cuda.min_num_runs_before_cuda_graph_capture` (int, default: 1)
  - State machine:
    - If graph capture not enabled → no-op
    - If warm-up count not reached → increment regular run count
    - If `IsGraphCaptureAllowed()` → call `CaptureBegin(annotation_id)`
    - If `IsGraphCaptured()` → call `Replay(annotation_id)`

### 4.3 `OnRunEnd` Implementation

- [ ] **4.3.1** Implement `CudaEp::OnRunEndImpl()` in [cuda_ep.cc](../cuda_ep.cc)
  - If capturing: call `CaptureEnd()` then `Replay()` (first run needs actual execution)
  - Handle `sync_stream` flag
  - Clear deferred CPU buffers only when not capturing

### 4.4 `SetDynamicOptions`

- [ ] **4.4.1** Implement `SetDynamicOptionsImpl()` callback on `CudaEp`
  - Support `enable_cuda_graph` toggle at runtime
  - Wire up to `OrtEp` callback table

### 4.5 Memory Stability

- [ ] **4.5.1** Ensure allocator holds allocations stable during graph capture
  - Configure `OrtAllocator` (arena-backed via `BFCArena`) to not free/reallocate between capture and replay
  - The plugin's `CreateAllocator` in [cuda_ep_factory.cc](../cuda_ep_factory.cc) may need arena config options

### 4.6 Validate Stage 4

- [ ] **4.6.1** Add CUDA graph tests to `test_cuda_plugin_ep.py`
  - Warm-up runs (N runs before capture)
  - Capture + first replay
  - Subsequent replays (10+ runs) — verify bit-exact output
  - Multi-annotation (2+ graphs, different seq lengths)
  - Graph capture disabled mid-session

- [ ] **4.6.2** Build and test:
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ```

---

## Stage 5: Remove Private Bridge & Excluded Ops

> **Goal**: Sever all `provider_api.h` / `ProviderHost_impl.h` dependencies. Bring excluded ops into plugin.

### 5.1 Control Flow Ops

- [ ] **5.1.1** Implement `If` kernel wrapper for plugin EP
  - Create a kernel class that overrides `CreateControlFlowKernelImpl()`
  - Calls `OrtEpApi::CreateIfKernel(info, out)` to create the implementation
  - Register with the adapter `KernelRegistry`

- [ ] **5.1.2** Implement `Loop` kernel wrapper (same pattern as `If`)

- [ ] **5.1.3** Implement `Scan` kernel wrapper (same pattern as `If`)

- [ ] **5.1.4** Remove CMake exclusion for `controlflow/` directory
  - Or create new wrapper files in `plugin/` rather than modifying original controlflow code

### 5.2 RNN Ops

- [ ] **5.2.1** Update RNN ops to remove `dynamic_cast<CudaStream*>`
  - Files in `rnn/` directory (`cudnn_rnn_base.cc`, `rnn_impl.cu`, etc.)
  - Replace with `CudaKernel::GetCudnnHandle(ctx)` which routes through `CudaSyncStream::GetCudnnHandle()`
  - cuDNN RNN handle already held by `CudaSyncStream`

- [ ] **5.2.2** Remove CMake exclusion for `rnn/` directory

### 5.3 Tunable Ops

- [ ] **5.3.1** Assess tunable op infrastructure requirements
  - Depends on `CudaTuningContext` and `CUDAExecutionProvider`
  - Options: port, stub, or document deferral with rationale

### 5.4 Einsum

- [ ] **5.4.1** Remove `cuda_execution_provider.h` dependency from `math/einsum.cc`
  - Factor out the compute logic from EP-specific code
  - Remove CMake exclusion

### 5.5 Remaining Excluded Ops

- [ ] **5.5.1** `identity_op.cc` / `sequence_op.cc` — provide `TensorSeq` adapter or exclude
- [ ] **5.5.2** `scatter_nd.cc` — port CPU validation logic inline
- [ ] **5.5.3** `size.cc` — port CPU compute logic or exclude
- [ ] **5.5.4** `integer_gemm.cc` — update to use `CudaSyncStream` instead of `CudaStream`
- [ ] **5.5.5** `object_detection/` — port CPU base class logic or document deferral
- [ ] **5.5.6** `cuda_common.cc` — resolve `HalfGemmOptions` conflict

### 5.6 Audit Remaining Internal Includes

- [ ] **5.6.1** Run audit:
  ```bash
  grep -rn 'provider_api\.h\|ProviderHost_impl\|g_host' \
    onnxruntime/core/providers/cuda/
  ```
  Create hit list of remaining references in plugin-compiled files.

### 5.7 Validate Stage 5

- [ ] **5.7.1** Verify zero `provider_api.h`/`ProviderHost_impl` references in plugin code
- [ ] **5.7.2** Verify all excluded ops are included or have documented deferral
- [ ] **5.7.3** Full CI suite with plugin-only CUDA EP:
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ```

---

## Stage 6: Advanced Features & Polish

> **Goal**: Profiling, perf validation, packaging.

### 6.1 NVTX Profiling

- [ ] **6.1.1** Add NVTX range markers around kernel execution in plugin EP
  - Use NVTX directly (no internal profiler API needed)
  - Tag with kernel name and op type

### 6.2 External Resource Import

- [ ] **6.2.1** Implement `OrtExternalResourceImporterImpl` for CUDA memory/semaphore import
  - Wire up to factory or EP callbacks

### 6.3 Performance Regression Testing

- [ ] **6.3.1** Create benchmark suite: plugin EP vs bundled EP
  - Models: BERT-base, ResNet-50, GPT-2, Stable Diffusion
  - Measure latency (target: within 2% of bundled EP)
  - Measure GPU memory (target: within 5% of bundled EP)

### 6.4 Python Packaging

- [ ] **6.4.1** `onnxruntime-gpu` pip package includes plugin DLL
  - Update setup.py / packaging scripts to include `libonnxruntime_providers_cuda_plugin.so`

### 6.5 CI Pipeline

- [ ] **6.5.1** Create dedicated CI jobs for plugin mode
  - Build + test with `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON`
  - Run both `onnxruntime_test_all` and `test_cuda_plugin_ep.py`

### 6.6 Documentation

- [ ] **6.6.1** Write API migration guide for third-party CUDA kernels
  - How to port existing `ONNX_OPERATOR_*_KERNEL_EX` based kernels
  - How to use the adapter framework

---

## Cross-Cutting Concerns

### Files to Delete (cumulative across stages)

| File | Stage | Replaced By |
|------|-------|-------------|
| `plugin/provider_host_bridge.cc` | 1.10 | N/A (legacy bridge) |
| `plugin/provider_api_shims.cc` | 1.10 | Standalone implementations (may keep simplified version) |
| `plugin/cuda_plugin_adapter_registry.cc` | 2.5 | Adapter-based `RegisterCudaKernels()` |
| Legacy `AdapterKernelImpl` in `cuda_plugin_kernels.cu` | 1.11 | `ep::adapter::KernelImpl` |

### Files to Modify Significantly

| File | Stage | Changes |
|------|-------|---------|
| `plugin/cuda_kernel_adapter.h` | 1.11 | Remove SHARED_PROVIDER path; `CudaKernel` inherits from `adapter::OpKernel`; remove `AdapterKernelImpl`/`PluginRegistry` |
| `cmake/onnxruntime_providers_cuda_plugin.cmake` | 1.11, 2.5 | Switch forced-include to `ep/adapters.h`; remove manual registry entry; progressively remove exclusion filters |
| `plugin/cuda_ep.h` / `cuda_ep.cc` | 3.1, 4.2, 4.3 | Add `ShouldConvertDataLayoutForOp`, CUDA graph callbacks |
| `plugin/cuda_ep_factory.cc` | 2.1 | Use `adapter::KernelRegistry` for registration |

### New Files to Create

| File | Stage | Purpose |
|------|-------|---------|
| `plugin/cuda_graph_plugin.h/.cc` | 4.1 | Plugin-compatible CUDA graph manager |
| Control flow wrappers | 5.1 | `If`/`Loop`/`Scan` kernel wrappers using `OrtEpApi` |

### Verification Command

```bash
# Full build + test cycle
./cuda_plugin.sh --build --test --test_plugin

# Quick rebuild + plugin test only
./cuda_plugin.sh --build --test_plugin

# Non plugin build and test to ensure backward compatibility
./cuda.sh --build --test

# Audit for legacy references
grep -rn 'provider_api\.h\|SHARED_PROVIDER\|g_host' \
  onnxruntime/core/providers/cuda/plugin/
```
