# CUDA EP Plugin Migration — Implementation Tasks

> **Plan**: [cuda_plugin_ep_plan.md](cuda_plugin_ep_plan.md)
> **Prototype commit**: `31a6e1d2b96801033053f559bbd20d817ea8642e`
> **EP Adapter commit**: `72a4cd7025a8740f2f8996f9899f898223c34941`
> **1.10 commit**: `98dea517f3` — Remove SHARED_PROVIDER bridge
> **1.11 commit**: `4f18312537` — EP Adapter forced-include integration
> **Build & Test**: `./cuda_plugin.sh --build --test --test_plugin`

---

## Stage 1: Plugin Shell & Infrastructure

> **Goal**: EP loads via plugin API, creates streams/allocators, runs Memcpy + simple ops.
> Items 1.1–1.9 are already done in the prototype. Work focuses on 1.10–1.12.

### 1.10 Remove `SHARED_PROVIDER` Bridge

Remove all `provider_api.h` / `g_host` / `ProviderHost_impl.h` dependencies from the plugin build.
The `SHARED_PROVIDER` bridge's namespace-level type stubs **conflict** with the adapter's `using` declarations and must be severed **before** the adapter forced include is enabled.

- [x] **1.10.1** Delete `provider_host_bridge.cc` from plugin sources *(commit `98dea517f3`)*
  - File deleted entirely (was 18 LOC). `g_host = Provider_GetHost()` dependency removed.

- [x] **1.10.2** Refactor [provider_api_shims.cc](../provider_api_shims.cc) to remove dual-path pattern *(commit `98dea517f3`)*
  - Removed all `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER` / `#else` blocks
  - Kept only direct implementations: `GetEnvironmentVar()` → `std::getenv()`, `math::floatToHalf()` → `MLFloat16(f).val`, `math::halfToFloat()` → `MLFloat16::FromBits(h).ToFloat()`
  - Removed `#include "core/providers/shared_library/provider_api.h"`

- [x] **1.10.3** Audit and remove all `provider_api.h` includes from plugin-compiled files *(commit `98dea517f3`)*
  - Removed the `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER` block from `cuda_kernel_adapter.h` (Section 1) which contained `#define SHARED_PROVIDER` and `#include "core/providers/shared_library/provider_api.h"`
  - Removed legacy `OpKernel`/`OpKernelContext`/`OpKernelInfo` type aliases and `static_assert` blocks from the `#ifndef` path in `cuda_kernel_adapter.h` (Section 6a)
  - Removed legacy `AdapterKernelImpl` template from `cuda_plugin_kernels.cu` (57 lines under `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER`)

- [x] **1.10.4** Remove `core/providers/shared/common.cc` from plugin build *(commit `98dea517f3`)*
  - `Provider_GetHost()` dependency eliminated by deleting `provider_host_bridge.cc`
  - `common.cc` not collected by CMake glob patterns (confirmed)

- [x] **1.10.5** Verify: `grep -rn 'provider_api\.h\|SHARED_PROVIDER\|g_host' plugin/` returns zero matches *(verified)*
  - Only documentation files match — no code references remain

### 1.11 EP Adapter Forced-Include Integration

Refactor `cuda_kernel_adapter.h` to inherit from `adapter::OpKernel` (from `ep/adapters.h`) instead of the `SHARED_PROVIDER` bridge types.

- [x] **1.11.1** Remove `AdapterKernelImpl` class from `cuda_kernel_adapter.h` *(commit `4f18312537`)*
  - Removed `AdapterKernelImpl` struct, `GenericCreateKernel`, `KernelFactory` typedef, and all `ONNX_OPERATOR_*_KERNEL_EX` macro overrides (~200 lines)
  - Now provided by `ep::adapter::KernelImpl` in [adapter/op_kernel.h](../../../../../../include/onnxruntime/ep/adapter/op_kernel.h)

- [x] **1.11.2** Remove `PluginRegistry` class from `cuda_kernel_adapter.h` *(commit `4f18312537`)*
  - Removed `PluginRegistry` class and legacy `BUILD_CUDA_EP_AS_PLUGIN` no-op macro definitions
  - `cuda_plugin_adapter_registry.cc` updated to use `ResolvePluginKernelCreateFn()` instead of `PluginRegistry::Instance().AllEntries()`

- [x] **1.11.3** Refactor `CudaKernel` base class in `cuda_kernel_adapter.h` *(commit `4f18312537`)*
  - `CudaKernel` now inherits from `OpKernel` (resolved to `ep::adapter::OpKernel` via `adapters.h`)
  - All CUDA-specific accessors preserved: `Stream()`, `GetCublasHandle()`, `GetCudnnHandle()`, etc.
  - `Stream()` simplified: uses `ctx->GetGPUComputeStream()` directly instead of `Ort::KernelContext` reinterpretation
  - Added `GetScratchStream()` method returning `void*` (plugin) to match `GetScratchBuffer` parameter type
  - Corresponding `GetScratchStream()` added to framework [cuda_kernel.h](../../cuda_kernel.h) returning `onnxruntime::Stream*`
  - Runtime config pattern (`CudaKernelAdapterRuntimeConfig` with atomics) preserved

- [x] **1.11.4** Switch forced include from `cuda_kernel_adapter.h` to `ep/adapters.h` *(commit `4f18312537`)*
  - CMake changed: `-include;${CUDA_PLUGIN_EP_DIR}/cuda_kernel_adapter.h` → `-include;${REPO_ROOT}/include/onnxruntime/ep/adapters.h`
  - `cuda_kernel_adapter.h` now explicitly `#include "ep/adapters.h"` at line 57
  - `cuda_kernel_adapter.h` continues to provide `CudaKernel` base class (included by `cuda_plugin_kernels.cu`)

- [x] **1.11.5** Resolve namespace conflicts *(commit `4f18312537`)*
  - Removed all duplicate `using` aliases from `cuda_kernel_adapter.h` in both `onnxruntime::cuda` and `onnxruntime::contrib::cuda` namespaces (~30 aliases total)
  - These are now provided by `EP_SPECIFIC_USING_DECLARATIONS` in `adapters.h`
  - Domain constant shadowing (`#define kOnnxDomain __kOnnxDomain_ignore`) removed — handled by adapter framework
  - Kept only `BuildKernelCreateInfo` forward declaration and `HandleNegativeAxis` using-declaration

- [x] **1.11.6** Remove legacy path from `cuda_kernel_adapter.h` *(commit `4f18312537`)*
  - All `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER` / `#ifdef ORT_CUDA_PLUGIN_USE_ADAPTER` guards removed
  - File reduced from ~900 LOC to ~665 LOC, containing only:
    - `CudaKernelAdapterRuntimeConfig` struct
    - `CudaKernel` base class (inheriting from `OpKernel` → `ep::adapter::OpKernel`)
    - `CUDAExecutionProvider` shim class
    - Error-return macros, type mapping helpers, CPU provider shims

- [x] **1.11.7** Remove legacy path from `cuda_plugin_kernels.cu` *(done in 1.10 commit `98dea517f3`)*
  - Removed `AdapterKernelImpl` template and `GetCudaSyncStream` helper (57 lines under `#ifndef ORT_CUDA_PLUGIN_USE_ADAPTER`)

### 1.11b Kernel Compatibility Fixes *(commit `4f18312537`)*

Additional modifications needed to compile existing kernel files with the adapter forced-include.

- [x] **1.11b.1** Introduce `GetScratchStream` abstraction for `GetScratchBuffer` calls
  - Framework `CudaKernel::GetScratchBuffer` expects `onnxruntime::Stream*`; plugin version expects `void*` (from `GetGPUComputeStream()`)
  - Added `CudaKernel::GetScratchStream(OpKernelContext*)` to both [cuda_kernel.h](../../cuda_kernel.h) and [cuda_kernel_adapter.h](../cuda_kernel_adapter.h)
  - Updated callers: `batch_norm.cc`, `conv.cc`, `conv_transpose.cc`, `dropout.cc`, `instance_norm.cc`, `pool.cc`, `compress.cc`, `nonzero_op.cc`, `upsample.cc`
  - For standalone files (`compress.cc`, `nonzero_op.cc`, `upsample.cc`, `reduction_ops.cc`), added local `GetScratchStream` inline helpers with `#ifdef BUILD_CUDA_EP_AS_PLUGIN` guards

- [x] **1.11b.2** Abstract stream type for softmax, topk, reduction compute functions
  - Introduced `SoftmaxComputeStreamT` (`Stream*` or `cudaStream_t`) in [softmax.h](../../math/softmax.h)
  - Introduced `ReduceComputeStreamT` (`Stream*` or `cudaStream_t`) in [reduction_ops.h](../../reduction/reduction_ops.h)
  - Updated `TopKImpl` to accept `cudaStream_t` directly in plugin build ([topk_impl.h](../../math/topk_impl.h), [topk_impl.cuh](../../math/topk_impl.cuh))
  - Template instantiation macros updated to match new stream types
  - `ReduceComputeCore` takes additional `const CudaKernel*` parameter for scratch buffer allocation in plugin path

- [x] **1.11b.3** Remove CPU base class dependencies from specific kernels
  - `Clip_6`: Inlined `Clip_6Base` attribute reading (`min`/`max` via `GetAttrOrDefault`) directly in [clip.h](../../math/clip.h), removing inheritance from CPU `Clip_6Base`
  - `ConvTranspose`: Replaced `OpKernel::Node().InputDefs().size()` with `context->Input<Tensor>(idx) != nullptr` for bias detection in [conv_transpose.cc](../../nn/conv_transpose.cc) and [conv_transpose_8.h](../../nn/conv_transpose_8.h)
  - `Conv`/`ConvTranspose`: Added `#ifdef BUILD_CUDA_EP_AS_PLUGIN` overloads for `GetWorkSpace()` accepting `void*` instead of `onnxruntime::Stream*` in [conv.h](../../nn/conv.h) and [conv_transpose.h](../../nn/conv_transpose.h)

- [x] **1.11b.4** EP Adapter framework fixes (in `include/onnxruntime/ep/adapter/`)
  - `adapter::OpKernel::Node()` return type fixed: `Node` → `adapter::Node` (resolved ambiguity)
  - `adapter::Node::Domain()` method added in [node.h](../../../../../../include/onnxruntime/ep/adapter/node.h)
  - Added `(void)` casts for unused-result warnings in `KernelImpl::ComputeImpl`, `KernelImpl::PrePackImpl`, `KernelRegistry::CreateKernel`, `KernelRegistry::Register`

- [x] **1.11b.5** New CMake exclusions for ops not yet compatible with adapter
  - Added 18 new exclusion filters in [onnxruntime_providers_cuda_plugin.cmake](../../../../../../cmake/onnxruntime_providers_cuda_plugin.cmake):
    - `llm/*` — uses `onnxruntime::Stream*` in QkvToContext
    - `generator/constant_of_shape.cc` — inherits CPU `ConstantOfShapeBase`
    - `math/matmul_integer.cc` — uses `GetComputeStream()` with `GemmInt8`
    - `math/matmul.cc` — uses `GetComputeStream()` in `FuncCallAdapter`
    - `math/variadic_elementwise_ops.cc` — uses `InputArgCount`/`RequiredInput`/`RequiredOutput`
    - `tensor/slice.cc` — inherits CPU `SliceBase`
    - `tensor/space_depth_ops.cc` — inherits CPU `SpaceDepthBase`
    - `tensor/concat.cc` — uses `InputArgCount` and `GetComputeStream()`
    - `tensor/gather.cc` — passes adapter `OpKernelContext*` to framework `PrepareForCompute`
    - `tensor/gather_nd.cc` — uses `GetComputeStream()` with `PrepareCompute`
    - `tensor/pad.cc` — passes adapter context to framework `PadBase::HandleDimension`
    - `tensor/reshape.cc` — uses `GetComputeStream()` and `CopyTensor`
    - `tensor/split.cc` — uses `GetComputeStream()` with `CopyToGpu`
    - `tensor/upsample.cc` — uses `InputDefs()` and `OpKernelInfo::GetAllocator()`
    - `tensor/unsqueeze.cc` — passes adapter context to framework `FlattenHelper`/`CopyTensor`
    - `tensor/shape_op.cc` — inherits from framework `onnxruntime::OpKernel`
  - Commented out `matmul_nbits.h` include in `cuda_plugin_kernels.cu` (uses `InputDefs()`)

- [x] **1.11b.6** Misc cleanup
  - Removed `RETURN_IF_ERROR` and `RETURN_IF` macros from [cuda_plugin_utils.h](../cuda_plugin_utils.h) (now provided by `adapters.h`)
  - Added `-Wno-maybe-uninitialized` to suppress false GCC warnings with adapter `GetAttr` output parameters
  - Guarded `ReductionOps::ReduceCompute` helper with `#ifndef BUILD_CUDA_EP_AS_PLUGIN` (uses `AllocatorPtr` + `Stream*` pattern)

### 1.12 Validate Stage 1

- [x] **1.12.1** Build plugin: `./cuda_plugin.sh --build`
  - Verify clean compilation with zero warnings related to adapter/SHARED_PROVIDER conflicts
- [x] **1.12.2** Run C++ tests: `./cuda_plugin.sh --build --test`
  - `onnxruntime_test_all` with plugin mode should pass
- [x] **1.12.3** Run plugin Python tests: `./cuda_plugin.sh --build --test --test_plugin`
  - `test_cuda_plugin_ep.py`: Add, MatMul, Gemm, Conv should all pass
- [x] **1.12.4** Run non-plugin build and test: `./cuda.sh --build --test` to make sure no regression.
- [x] **1.12.5** Verify no SHARED_PROVIDER references *(verified)*
  ```bash
  grep -rn 'provider_api\.h\|SHARED_PROVIDER\|g_host' \
    onnxruntime/core/providers/cuda/plugin/
  ```
  Returns only documentation matches — zero code matches.

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
  - **Added in 1.11** — additional exclusions discovered during adapter integration:
    - LLM ops: `llm/*` excluded (uses `onnxruntime::Stream*` in QkvToContext)
    - ConstantOfShape: `generator/constant_of_shape.cc` excluded (CPU `ConstantOfShapeBase`)
    - MatMulInteger: `math/matmul_integer.cc` excluded (`GetComputeStream()` with `GemmInt8`)
    - MatMul: `math/matmul.cc` excluded (`GetComputeStream()` in `FuncCallAdapter`)
    - VariadicElementwise: `math/variadic_elementwise_ops.cc` excluded (`InputArgCount`/`RequiredInput`/`RequiredOutput`)
    - Slice: `tensor/slice.cc` excluded (CPU `SliceBase`)
    - SpaceDepthOps: `tensor/space_depth_ops.cc` excluded (CPU `SpaceDepthBase`)
    - Concat: `tensor/concat.cc` excluded (`InputArgCount`, `GetComputeStream()`)
    - Gather: `tensor/gather.cc` excluded (adapter/framework `OpKernelContext*` mismatch)
    - GatherND: `tensor/gather_nd.cc` excluded (`GetComputeStream()`)
    - Pad: `tensor/pad.cc` excluded (framework `PadBase::HandleDimension`)
    - Reshape: `tensor/reshape.cc` excluded (`GetComputeStream()`, `CopyTensor`)
    - Split: `tensor/split.cc` excluded (`GetComputeStream()` with `CopyToGpu`)
    - Upsample: `tensor/upsample.cc` excluded (`InputDefs()`, `OpKernelInfo::GetAllocator()`)
    - Unsqueeze: `tensor/unsqueeze.cc` excluded (framework `FlattenHelper`/`CopyTensor`)
    - Shape: `tensor/shape_op.cc` excluded (inherits framework `onnxruntime::OpKernel`)
    - MatMulNBits: `matmul_nbits.h` include commented out (uses `InputDefs()`)

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
  ./cuda.sh --build --test
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
  ./cuda.sh --build --test
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
  ./cuda.sh --build --test
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
  ./cuda.sh --build --test
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

| File | Stage | Status | Replaced By |
|------|-------|--------|-------------|
| `plugin/provider_host_bridge.cc` | 1.10 | ✅ Deleted | N/A (legacy bridge) |
| `plugin/provider_api_shims.cc` | 1.10 | ✅ Simplified | Standalone implementations (kept; `g_host` calls removed) |
| `plugin/cuda_plugin_adapter_registry.cc` | 2.5 | Pending | Adapter-based `RegisterCudaKernels()` |
| Legacy `AdapterKernelImpl` in `cuda_plugin_kernels.cu` | 1.10 | ✅ Removed | `ep::adapter::KernelImpl` |
| `AdapterKernelImpl`/`PluginRegistry`/macro overrides in `cuda_kernel_adapter.h` | 1.11 | ✅ Removed | `ep::adapter::KernelImpl`/`KernelRegistry` |

### Files Modified (completed)

| File | Stage | Changes |
|------|-------|---------|
| `plugin/cuda_kernel_adapter.h` | 1.10, 1.11 | Removed SHARED_PROVIDER path, legacy guards, `AdapterKernelImpl`, `PluginRegistry`, macro overrides, duplicate type aliases. `CudaKernel` now inherits from `adapter::OpKernel`. Added `GetScratchStream()`. Reduced from ~900 to ~665 LOC. |
| `cmake/onnxruntime_providers_cuda_plugin.cmake` | 1.11 | Switched forced-include to `ep/adapters.h`; added 18 new op exclusions; added `-Wno-maybe-uninitialized` |
| `include/onnxruntime/ep/adapter/op_kernel.h` | 1.11 | Fixed `Node()` return type; added `(void)` casts for warnings |
| `include/onnxruntime/ep/adapter/node.h` | 1.11 | Added `Domain()` method |
| `include/onnxruntime/ep/adapter/kernel_registry.h` | 1.11 | Added `(void)` casts for `AddKernel` and `CreateControlFlowKernelImpl` |
| `cuda/cuda_kernel.h` | 1.11 | Added `GetScratchStream()` for framework build |
| `cuda/math/clip.h` | 1.11 | Inlined `Clip_6Base` attributes to remove CPU base class dep |
| `cuda/math/softmax.h/.cc/.cu` | 1.11 | Introduced `SoftmaxComputeStreamT` abstraction |
| `cuda/math/topk.cc/.cuh/.h` | 1.11 | `TopKImpl` stream parameter uses `cudaStream_t` in plugin build |
| `cuda/nn/batch_norm.cc` | 1.11 | `GetScratchBuffer` → `GetScratchStream` |
| `cuda/nn/conv.cc/.h` | 1.11 | `GetWorkSpace` overload for `void*`; `GetScratchStream` |
| `cuda/nn/conv_transpose.cc/.h` | 1.11 | Bias detection via `Input!=nullptr`; `GetWorkSpace` overload; `GetScratchStream` |
| `cuda/nn/conv_transpose_8.h` | 1.11 | Same as conv_transpose; `Stream(context)` for scratch |
| `cuda/nn/dropout.cc` | 1.11 | `GetScratchStream` |
| `cuda/nn/instance_norm.cc` | 1.11 | `GetScratchStream` |
| `cuda/nn/pool.cc` | 1.11 | `GetScratchStream` |
| `cuda/reduction/reduction_ops.cc/.h` | 1.11 | `ReduceComputeStreamT`; `AllocateScratchBuffer` helper; extra `CudaKernel*` param |
| `cuda/tensor/compress.cc` | 1.11 | Local `GetScratchStream` helper |
| `cuda/tensor/nonzero_op.cc` | 1.11 | Local `GetScratchStream` helper |
| `cuda/tensor/upsample.cc` | 1.11 | Local `GetScratchStream` helper |
| `plugin/cuda_plugin_adapter_registry.cc` | 1.11 | Uses `ResolvePluginKernelCreateFn` instead of `PluginRegistry` |
| `plugin/cuda_plugin_kernels.cu` | 1.11 | Commented out `matmul_nbits.h` include |
| `plugin/cuda_plugin_utils.h` | 1.11 | Removed `RETURN_IF_ERROR`/`RETURN_IF` macros |

### Files to Modify (remaining stages)

| File | Stage | Changes |
|------|-------|---------|
| `cmake/onnxruntime_providers_cuda_plugin.cmake` | 2.5 | Remove manual registry entry; progressively remove exclusion filters |
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

# If you change files except those under onnxruntime/core/providers/cuda/plugin (This directory are excluded from non-plugin build),
# you need run non plugin build and test to ensure backward compatibility
./cuda.sh --build --test

# Audit for legacy references
grep -rn 'provider_api\.h\|SHARED_PROVIDER\|g_host' \
  onnxruntime/core/providers/cuda/plugin/
```
