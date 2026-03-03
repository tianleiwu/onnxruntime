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

- [x] **2.1.1** Replace `PluginRegistry` with `PluginKernelCollector` self-registration
  - Removed `PluginRegistry` class and `GenericCreateKernel` function from `cuda_kernel_adapter.h`
  - Added `PluginKernelCollector` singleton class: stores `BuildKernelCreateInfoFn` pointers
  - Rewrote `ONNX_OPERATOR_*_KERNEL_EX` macro overrides to BOTH produce `BuildKernelCreateInfo<>()` template specializations AND auto-register in `PluginKernelCollector::Instance().Add()` via static bool initializers
  - Each compiled kernel `.cc` file's macros now self-register at static init time — no manual tables needed

- [x] **2.1.2** Rewrite `CreateCudaKernelRegistry` in `cuda_plugin_kernels.cu`
  - Removed all old includes and `CreateCudaKernelRegistryFromOrtTables()` delegation
  - New implementation: creates `adapter::KernelRegistry`, iterates `PluginKernelCollector::Instance().Entries()`, registers each `BuildKernelCreateInfo`
  - Added `BuildKernelCreateInfo<void>()` sentinel definition in `onnxruntime::cuda` namespace (normally in excluded `cuda_execution_provider.cc`)
  - Simplified `cuda_plugin_kernels.h`: only declares `CreateCudaKernelRegistry()`

- [x] **2.1.3** Test: Add, Relu, MatMul, Gemm, Conv, Softmax all pass via `test_cuda_plugin_ep.py`

### 2.2 NHWC Registrations

- [x] **2.2.1** NHWC ops auto-register via macro overrides in individual kernel files
  - The `ONNX_OPERATOR_*_KERNEL_EX` macros in `batch_norm.cc`, `conv.cc`, `conv_transpose.cc`, `pool.cc` register NHWC variants (with `kMSInternalNHWCDomain`) via `PluginKernelCollector`
  - `cuda_nhwc_kernels.cc` is excluded from plugin build: its centralized `RegisterCudaNhwcKernels()` table references ALL NHWC ops including those in excluded source files (e.g., `space_depth_ops.cc`), causing undefined symbol errors
  - Fixed `cuda_nhwc_kernels.h`: changed `onnxruntime::KernelRegistry&` to unqualified `KernelRegistry&` (resolves to `adapter::KernelRegistry` in plugin, framework `KernelRegistry` in normal build)
  - NHWC ops from non-excluded source files (BatchNorm, Conv, ConvTranspose, Pool, LRN) are available in plugin

### 2.3 Contrib Op Registrations

- [x] **2.3.1** Contrib ops auto-register via macro overrides in individual kernel files
  - Same `PluginKernelCollector` self-registration pattern as standard ops
  - `cuda_contrib_kernels.cc` is excluded from plugin build: its centralized registration table references ALL contrib ops including those in excluded files, causing link errors
  - Guarded `provider_api.h` includes: moved `#ifdef BUILD_CUDA_EP_AS_PLUGIN` guard into `provider_api.h` itself (replacing the old `ORT_CUDA_PLUGIN_USE_ADAPTER` guard), so individual files include it unconditionally
  - Contrib ops from non-excluded source files (LayerNorm, BiasGelu, FlashAttention, etc.) are available in plugin

### 2.4 Resolve Excluded Ops

- [ ] **2.4.1** Control flow ops (`If`/`Loop`/`Scan`) — deferred to Stage 5
  - Inherit from CPU base classes — need `OrtEpApi::CreateIfKernel`/`CreateLoopKernel`/`CreateScanKernel`

- [x] **2.4.2** Document excluded op categories
  - **Standard ops excluded** (in `core/providers/cuda/`):
    - `cuda_execution_provider.cc`, `cuda_provider_factory.cc`, `cuda_provider_interface.cc` — replaced by plugin equivalents
    - `cuda_stream_handle.cc`, `cuda_execution_provider_info.cc`, `cuda_graph.cc`, `cuda_mempool_arena.cc`, `cuda_common.cc` — EP infrastructure replaced by plugin
    - `cuda_nhwc_kernels.cc` — centralized table excluded; NHWC ops self-register from individual files
    - `controlflow/*` — CPU base class deps (Stage 5)
    - `tunable/*` — CudaTuningContext dep
    - `rnn/*` — CudaStream dynamic_cast
    - `llm/*` — uses `onnxruntime::Stream*`
    - `math/einsum.cc`, `math/einsum_utils/*` — cuda_execution_provider.h dep
    - `math/matmul.cc` — `GetComputeStream()` in `FuncCallAdapter`
    - `math/matmul_integer.cc` — `GetComputeStream()` with `GemmInt8`
    - `math/variadic_elementwise_ops.cc` — `InputArgCount`/`RequiredInput`/`RequiredOutput`
    - `math/cumsum.cc` — CPU provider dep
    - `generator/constant_of_shape.cc` — CPU `ConstantOfShapeBase`
    - `integer_gemm.cc` — CudaStream dep
    - `tensor/slice.cc` — CPU `SliceBase`
    - `tensor/space_depth_ops.cc` — CPU `SpaceDepthBase`
    - `tensor/concat.cc` — `InputArgCount`, `GetComputeStream()`
    - `tensor/gather.cc` — adapter/framework `OpKernelContext*` mismatch
    - `tensor/gather_nd.cc` — `GetComputeStream()`
    - `tensor/pad.cc` — framework `PadBase::HandleDimension`
    - `tensor/reshape.cc` — `GetComputeStream()`, `CopyTensor`
    - `tensor/split.cc` — `GetComputeStream()` with `CopyToGpu`
    - `tensor/upsample.cc`, `tensor/resize.cc` — `InputDefs()`, `OpKernelInfo::GetAllocator()`
    - `tensor/unsqueeze.cc` — framework `FlattenHelper`/`CopyTensor`
    - `tensor/shape_op.cc` — inherits framework `onnxruntime::OpKernel`
    - `tensor/identity_op.cc`, `tensor/sequence_op.cc` — TensorSeq incomplete type
    - `tensor/scatter_nd.cc` — CPU validation dep
    - `tensor/size.cc` — CPU op
    - `tensor/tile.cc` — CPU `TileOp::IsTileMemcpy`
    - `object_detection/*` — CPU base class dep
  - **Contrib ops excluded** (in `contrib_ops/cuda/`):
    - `cuda_contrib_kernels.cc` — centralized table excluded; contrib ops self-register from individual files
    - `aten_ops/*`, `collective/*` — not applicable to plugin
    - `llm/*` — `onnxruntime::Stream*` dep
    - `transformers/*` — beam search, greedy search, sampling (complex deps)
    - `bert/attention.cc`, `bert/decoder_attention.cc`, `bert/decoder_masked_self_attention.cc`, `bert/embed_layer_norm.cc`, `bert/fast_gelu.cc`, `bert/group_query_attention.cc`, `bert/longformer_attention.cc`, `bert/multihead_attention.cc`, `bert/packed_attention.cc`, `bert/packed_multihead_attention.cc`, `bert/paged_attention.cc`, `bert/relative_attn_bias.cc`, `bert/remove_padding.cc` — `GetComputeStream()` or framework `OpKernelContext`
    - `diffusion/group_norm.cc`, `fused_conv.cc`, `inverse.cc`, `math/bias_dropout.cc`, `math/fft_ops.cc`, `moe/moe.cc`, `sparse/sparse_attention.cc` — `GetComputeStream()` or framework deps
    - `tensor/crop.cc`, `tensor/dynamic_time_warping.cc`, `tensor/dynamicslice.cc`, `tensor/shrunken_gather.cc` — various deps
    - `quantization/attention_quantization.cc`, `quantization/matmul_bnb4.cc`, `quantization/matmul_nbits.cc`, `quantization/moe_quantization.cc`, `quantization/qordered_ops/*` — `GetComputeStream()` deps
    - `math/gemm_float8.cc/.cu` — `GetComputeStream()` in `.cu`
    - `math/fused_matmul.cc` — registers `MatMul<T>` class from excluded `matmul.cc`

### 2.5 Remove Manual Registry

- [x] **2.5.1** Deleted `cuda_plugin_adapter_registry.cc` (~339 LOC)
  - File removed from disk; CMake exclusion line also removed
  - Fully replaced by `PluginKernelCollector` self-registration

- [x] **2.5.2** Legacy `Create*Kernel` and `AdapterKernelImpl` already removed (1.11.7)
  - `cuda_plugin_kernels.cu` is clean — only contains `CreateCudaKernelRegistry()` using `PluginKernelCollector`

- [x] **2.5.3** CMake cleaned up
  - Removed `cuda_plugin_adapter_registry.cc` exclusion and stale comments
  - Removed `ORT_CUDA_PLUGIN_USE_ADAPTER=1` compile definition (replaced by `BUILD_CUDA_EP_AS_PLUGIN` in `provider_api.h`)

### 2.6 Validate Stage 2

- [ ] **2.6.1** Registration parity verification (deferred — needs diagnostic tooling)
  - Need to dump `(domain, op_type, since_version, type_constraints)` tuples from both registries
  - Plugin count must equal bundled count minus tracked exclusions

- [x] **2.6.2** Build and test — all pass
  ```
  ./cuda.sh --build --test          → 1170 tests PASSED
  ./cuda_plugin.sh --build --test --test_plugin → 1170 tests PASSED + plugin tests PASSED
  ```

---

## Stage 3: NHWC & GetCapability Integration

> **Goal**: NHWC layout transformation and CPU-preferred-node logic work correctly.

### 3.1 `ShouldConvertDataLayoutForOp`

- [x] **3.1.1** Implement `ShouldConvertDataLayoutForOpImpl` callback on `CudaEp`
  - Added static method `ShouldConvertDataLayoutForOpImpl` to [cuda_ep.h](../cuda_ep.h) / [cuda_ep.cc](../cuda_ep.cc)
  - Same op list as `CUDAExecutionProvider::ShouldConvertDataLayoutForOp()`:
    - ONNX ops: `BatchNormalization`, `Conv`, `ConvTranspose`, `GlobalMaxPool`, `MaxPool`, `GlobalAveragePool`, `AveragePool`, `GridSample`, `DepthToSpace`, `SpaceToDepth`, `LRN`
    - MS domain: `GridSample`
  - Wired up in `CudaEp` constructor: `ShouldConvertDataLayoutForOp = ShouldConvertDataLayoutForOpImpl;`
  - Returns `1` (convert) for NHWC-compatible ops, `-1` (let ORT decide) for others
  - Uses `std::unordered_set<std::string_view>` for O(1) op lookup

### 3.2 `GetCapability` with CPU-Preferred Nodes

- [x] **3.2.1** Integrate `GetCpuPreferredNodes` in `CudaEp::GetCapabilityImpl()`
  - Uses [get_capability_utils.h](../../../../../../include/onnxruntime/ep/get_capability_utils.h)
  - Three-phase flow in `GetCapabilityImpl`:
    1. Phase 1: Iterate all graph nodes via `Ort::ConstGraph::GetNodes()`, skip already-assigned nodes, call `EpGraphSupportInfo_LookUpKernel()` to collect tentative nodes
    2. Phase 2: Call `ep::GetCpuPreferredNodes()` to identify CPU-preferred nodes (e.g., `Shape`, `NonZero`, small compute ops)
    3. Phase 3: Add final nodes (tentative minus CPU-preferred) via `EpGraphSupportInfo_AddSingleNode()`
  - Includes `ep/get_capability_utils.h` and `<unordered_set>` for `cpu_preferred_nodes` set

### 3.3 NHWC Kernel Validation

- [x] **3.3.1** Add NHWC test cases to `test_cuda_plugin_ep.py`
  - Added `create_batch_norm_model`, `create_maxpool_model`, `create_avgpool_model` model builders
  - Added NHWC tests: Conv, BatchNormalization, MaxPool, AveragePool with `ep.cuda.prefer_nhwc_layout = "1"`
  - Updated `test_operator()` to accept `session_config` dict for session config entries
  - All NHWC tests verify correctness against PyTorch reference (rtol=1e-3, atol=1e-3)

- [ ] **3.3.2** Test with reference models (deferred — needs model download infrastructure)
  - ResNet-50 with NHWC enabled — verify outputs match bundled EP within tolerance
  - If EfficientNet-B0 model is available, test that as well

### 3.4 Validate Stage 3

- [ ] **3.4.1** Verify `GetCapability` claims same nodes as bundled EP for a reference model (deferred — needs reference model)
- [ ] **3.4.2** Verify CPU-fallback nodes (Shape, NonZero, etc.) correctly left on CPU (deferred — needs model with Shape/NonZero nodes)
- [x] **3.4.3** Build and test:
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ```
  - Build: 1535/1535 compiled, `libonnxruntime_providers_cuda_plugin.so` linked successfully
  - C++ tests: 1170 tests PASSED
  - Plugin Python tests: All Stage 2 + Stage 3 NHWC tests PASSED
    - Conv (NHWC), BatchNormalization (NHWC), MaxPool (NHWC), AveragePool (NHWC) all pass

---

## Stage 4: CUDA Graph Integration

> **Goal**: CUDA Graph capture/replay lifecycle via plugin EP's `OnRunStart`/`OnRunEnd` callbacks.
> Matches the bundled CUDA EP's `gpu_graph_id` run option and `enable_cuda_graph` provider option patterns.

### 4.1 Port `CUDAGraphManager`

- [x] **4.1.1** Created [cuda_graph_plugin.h](../cuda_graph_plugin.h) and [cuda_graph_plugin.cc](../cuda_graph_plugin.cc)
  - Adapted from [cuda_graph.h](../../cuda_graph.h) / [cuda_graph.cc](../../cuda_graph.cc)
  - Removed dependencies on internal EP types (`CUDAExecutionProvider`, `CudaStream`)
  - Key types:
    - `CudaGraphAnnotation_t = int` — annotation ID type (matches bundled EP)
    - `kCudaGraphAnnotationSkip = -1` — sentinel for disabling capture/replay per-run
    - `kCudaGraphAnnotationDefault = 0` — default annotation when `gpu_graph_id` not specified
    - `CudaGraphSet` — stores `cudaGraphExec_t` per annotation ID in `unordered_map`
    - `CUDAGraphManager` — manages capture/instantiation/replay lifecycle:
      - `SetStream(cudaStream_t)` — lazy stream binding (doesn't own the stream)
      - `CaptureBegin(annotation_id)` — syncs stream, then `cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal)`
      - `CaptureEnd(annotation_id)` — `cudaStreamEndCapture` → `cudaGraphInstantiate` → stores in `CudaGraphSet`
      - `Replay(annotation_id, sync)` — `cudaGraphLaunch` + optional `cudaStreamSynchronize`
      - `IsGraphCaptureAllowedOnRun(id)` — returns `id != kCudaGraphAnnotationSkip`
      - `IsGraphCaptured(id)` — checks `CudaGraphSet::Contains(id)`
      - `Reset()` — destroys all `cudaGraphExec_t` handles
  - Auto-collected by CMake `GLOB_RECURSE` for `*.cc` under `core/providers/cuda/`
  - Existing exclusion `.*/cuda_graph\.cc$` only excludes the bundled `cuda_graph.cc`, not `cuda_graph_plugin.cc`
  - All errors use `std::runtime_error` (caught by `EXCEPTION_TO_STATUS_BEGIN/END` in OnRunStart/OnRunEnd)

### 4.2 `OnRunStart` Implementation

- [x] **4.2.1** Implemented `CudaEp::OnRunStartImpl()` in [cuda_ep.cc](../cuda_ep.cc)
  - State machine:
    1. If `cuda_graph_enabled_` is false → no-op (return nullptr)
    2. Parse `gpu_graph_id` from `OrtRunOptions` via `GetAnnotationId()` — uses `OrtApi::GetRunConfigEntry(run_options, "gpu_graph_id")`, matching the bundled EP's `kOrtRunOptionsConfigCudaGraphAnnotation`
    3. If `annotation_id == kCudaGraphAnnotationSkip` (-1) → no-op (per-run disable)
    4. Lazily set graph manager's stream from `factory_.GetComputeStream()` — returns nullptr gracefully if stream not yet created
    5. If `IsGraphCaptured(annotation_id)` → no-op (replay happens in OnRunEnd)
    6. If `IsGraphCaptureAllowed(annotation_id)` (warm-up count met) → `CaptureBegin(annotation_id)`, set `is_capturing_ = true`
  - `GetAnnotationId()`: reads `"gpu_graph_id"` key (same as bundled EP), defaults to `kCudaGraphAnnotationDefault` (0)
  - `IsGraphCaptureAllowed()`: checks `IsGraphCaptureAllowedOnRun(id)` (not -1) AND `graph_id_to_run_count_[id] >= min_runs_before_capture_`

### 4.3 `OnRunEnd` Implementation

- [x] **4.3.1** Implemented `CudaEp::OnRunEndImpl()` in [cuda_ep.cc](../cuda_ep.cc)
  - Same guards as OnRunStart: check `cuda_graph_enabled_`, parse annotation, check skip
  - If not yet captured AND was capturing:
    - Call `CaptureEnd(annotation_id)` → `CaptureBegin` stream capture ends, graph instantiated
    - Call `Replay(annotation_id, sync_stream)` — first execution (capture stream doesn't run on GPU)
  - If not yet captured AND not capturing:
    - Increment warm-up run count: `graph_id_to_run_count_[annotation_id]++`
  - **Note**: For subsequent runs after capture, the plugin EP does NOT replay the graph in OnRunEnd. The ORT framework dispatches kernels normally. Full graph-only replay (bypassing kernel dispatch) requires stream executor support not yet available in the plugin EP API. This is a known limitation — capture infrastructure is in place for when the framework adds replay support.

### 4.4 `SetDynamicOptions`

- [x] **4.4.1** `SetDynamicOptions` callback NOT implemented (deliberately)
  - `SetDynamicOptions` is called via `OrtApi::SetEpDynamicOptions()` — a post-session-creation API explicitly called on an `InferenceSession`
  - It is NOT connected to `add_session_config_entry()` — those are read at EP construction time
  - The bundled CUDA EP does not use `SetDynamicOptions` for `enable_cuda_graph` either — it's a provider option set at session creation
  - `enable_cuda_graph` and `min_num_runs_before_cuda_graph_capture` are read from session config in `CudaEpFactory::CreateEpImpl()`, stored in `CudaEp::Config`, and used to initialize `cuda_graph_enabled_` and `min_runs_before_capture_`
  - Can be added in the future if there's a need for runtime toggling

### 4.5 Memory Stability

- [x] **4.5.1** Arena-backed allocator provides memory stability
  - Plugin's `CreateAllocator` in [cuda_ep_factory.cc](../cuda_ep_factory.cc) creates ORT-managed arena allocators
  - Arena allocators maintain stable virtual addresses across runs — addresses captured in the graph remain valid for replay
  - No special arena configuration needed beyond the default setup
  - **Known limitation**: `update_inplace` + graph replay doesn't work because the framework still dispatches kernels normally after capture (no stream executor bypass). The captured graph and normal kernel dispatch both write to the output buffer, resulting in double execution with stale captured-graph results.

### 4.6 Validate Stage 4

- [x] **4.6.1** Added CUDA graph tests in `test_cuda_plugin_cuda_graph()` in [test_cuda_plugin_ep.py](../../../../test/python/transformers/test_cuda_plugin_ep.py)
  - Tests moved to a **separate function** from Stage 2/3 tests (was previously in `test_cuda_plugin_registration()`)
  - All tests use **IO binding + OrtValue** for memory-stable GPU buffers (matching the bundled EP test pattern in [onnxruntime_test_python_cudagraph.py](../../../../test/python/onnxruntime_test_python_cudagraph.py))
  - Uses `gpu_graph_id` run config key (same as bundled EP)
  - Test cases:
    1. **Add model** — IO binding, warmup (1 run) + capture + 5 replay runs, verify correctness
    2. **MatMul model** — IO binding, warmup + capture + 5 replay runs, verify correctness
    3. **Disable via `gpu_graph_id=-1`** — IO binding, 3 runs with `gpu_graph_id=-1`, verify normal execution
  - Deferred test cases:
    - `update_inplace` + replay (needs stream executor bypass — see 4.3 Note)
    - Multi-annotation (2+ graphs, different seq lengths) — needs more complex model
    - Concurrent sessions — needs threading infrastructure

- [x] **4.6.2** Build and test:
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ```
  - Build: clean (only plugin `.cc` files recompiled)
  - C++ tests: 1170 tests PASSED
  - Plugin Python tests: All Stage 2 + Stage 3 + Stage 4 tests PASS

### Session Config Keys (Plugin EP)

| Key | Where Set | Where Read | Type | Default |
|-----|-----------|------------|------|---------|
| `ep.cuda.enable_cuda_graph` | `add_session_config_entry()` | `CudaEpFactory::CreateEpImpl()` | bool | `false` |
| `ep.cuda.min_num_runs_before_cuda_graph_capture` | `add_session_config_entry()` | `CudaEpFactory::CreateEpImpl()` | int | `1` |

### Run Option Keys (Plugin EP)

| Key | Where Set | Where Read | Type | Notes |
|-----|-----------|------------|------|-------|
| `gpu_graph_id` | `add_run_config_entry()` | `CudaEp::GetAnnotationId()` | int | Same key as bundled EP. Default 0. Set to -1 to disable capture/replay for that run. |

---

## Stage 5: Remove Private Bridge & Excluded Ops

> **Goal**: Bring excluded ops into the plugin build. Sever remaining `provider_api.h` / `ProviderHost_impl.h` dependencies for plugin-compiled files.
>
> Stage 5 is split into **6 sub-stages** ordered by dependency pattern — from easiest stream fixes to hardest refactors. Each sub-stage is independently buildable and testable.
>
> **Two recurring incompatibility patterns** block most excluded ops:
> 1. `ctx->GetComputeStream()` — the framework `OpKernelContext::GetComputeStream()` returns `onnxruntime::Stream*`. The adapter has `GetGPUComputeStream()` → `void*`. Fix: use `CudaKernel::GetComputeStream(ctx)` which returns `void*`, or `Stream(ctx)` for `cudaStream_t`.
> 2. **CPU base class inheritance** — many CUDA tensor ops inherit from CPU base classes (`SliceBase`, `PadBase`, `ConcatBase`, etc.) for shape validation/attribute parsing. Fix: inline the base class logic under `#ifdef BUILD_CUDA_EP_AS_PLUGIN`.

### Current Excluded Ops Inventory (CMake exclusions)

| Category | Files | Root Cause |
|----------|-------|-----------|
| Infrastructure (replaced) | `cuda_execution_provider.cc`, `cuda_provider_factory.cc`, `cuda_provider_interface.cc`, `cuda_stream_handle.cc`, `cuda_execution_provider_info.cc`, `cuda_graph.cc`, `cuda_mempool_arena.cc`, `cuda_common.cc` | Plugin equivalents exist |
| Registration tables | `cuda_nhwc_kernels.cc`, `cuda_contrib_kernels.cc` | Self-registration via `PluginKernelCollector` |
| Stream fix only | `reshape.cc`, `split.cc`, `concat.cc` | `ctx->GetComputeStream()` |
| Single CPU utility | `cumsum.cc`, `tile.cc`, `gather.cc`, `unsqueeze.cc` | One function from CPU provider |
| CPU base class | `pad.cc`, `slice.cc`, `space_depth_ops.cc`, `constant_of_shape.cc`, `upsample.cc`, `resize.cc` | Multiple inheritance from CPU base |
| Missing adapter API | `variadic_elementwise_ops.cc` | `RequiredInput/RequiredOutput`, `InputArgCount` |
| API gap / complex | `rnn/*`, `tunable/*`, `einsum.cc`, `einsum_utils/*`, `object_detection/*`, `identity_op.cc`, `sequence_op.cc` | Deferred |
| CPU ops (permanent) | `shape_op.cc`, `size.cc` | Pure CPU class; handled by `GetCpuPreferredNodes` |
| Control flow | `controlflow/*` | ✅ Already replaced by `plugin/cuda_controlflow_plugin.cc` |
| Contrib: GetComputeStream | 13 bert ops, `group_norm.cc`, `fused_conv.cc`, `inverse.cc`, `bias_dropout.cc`, `fft_ops.cc`, `moe.cc`, `sparse_attention.cc`, `crop.cc`, `dynamic_time_warping.cc`, `dynamicslice.cc` | `ctx->GetComputeStream()` pattern |
| Contrib: quantization | `attention_quantization.cc`, `matmul_bnb4.cc`, `matmul_nbits.cc`, `moe_quantization.cc`, `qordered_ops/*` | `GetComputeStream` / `GetScratchBuffer(Stream*)` |
| Contrib: complex deps | `transformers/*`, `llm/*`, `aten_ops/*`, `collective/*`, `gemm_float8.cc/.cu`, `shrunken_gather.cc` | Deferred |

---

### Stage 5A: Stream Fix Ops (~1 day, 4 tasks)

These ops have **only** `ctx->GetComputeStream()` issues — no CPU base class problems.

- [x] **5A.0** Add `CudaAsyncBuffer::CopyToGpu(void*)` overload (if not present)
  - Search for `CudaAsyncBuffer` class definition (likely in `cuda_utils.h` or `cuda_common.h`)
  - Add `#ifdef BUILD_CUDA_EP_AS_PLUGIN` overload: `Status CopyToGpu(void* stream)` that casts `stream` to `cudaStream_t` and calls `cudaMemcpyAsync`
  - This unblocks all `CopyToGpu(ctx->GetComputeStream())` fixes below
  - `CopyToGpu(void*)` already exists in both framework [`cuda_kernel.h`](../../cuda_kernel.h) and plugin [`cuda_kernel_adapter.h`](../cuda_kernel_adapter.h), so no code change needed

- [x] **5A.1** Fix [tensor/reshape.cc](../../tensor/reshape.cc) — 2 lines at L50–L51
  - L50: `ORT_ENFORCE(ctx->GetComputeStream())` → `ORT_ENFORCE(GetComputeStream(ctx))`
  - L51: `cuda_kernel->CopyTensor(*X, *Y, *ctx->GetComputeStream())` → replace with `cudaMemcpyAsync(Y->MutableDataRaw(), X->DataRaw(), X->SizeInBytes(), cudaMemcpyDeviceToDevice, Stream(ctx))` or add `CopyTensor(Tensor&, Tensor&, cudaStream_t)` overload
  - Also fixed matching legacy path in [`tensor/reshape.h`](../../tensor/reshape.h) (`Reshape_1::ComputeInternal`) to avoid adapter `GetComputeStream()` compile failure
  - Remove CMake exclusion: `.*/tensor/reshape\\.cc$` (line ~L137)

- [x] **5A.2** Fix [tensor/split.cc](../../tensor/split.cc) — 5 lines at L132, L138, L140, L146, L148
  - All are `buf.CopyToGpu(ctx->GetComputeStream())` → `buf.CopyToGpu(GetComputeStream(ctx))`
  - `SplitKernel` inherits `SplitBase` — verify `SplitBase` compiles in plugin (it stores `split_sizes_` attribute parsed from `OpKernelInfo`; adapter's `GetAttrs<int64_t>` should work). If not, inline `split_sizes_` attribute reading under `#ifdef BUILD_CUDA_EP_AS_PLUGIN`.
  - Remove CMake exclusion: `.*/tensor/split\\.cc$` (line ~L140)

- [x] **5A.3** Fix [tensor/concat.cc](../../tensor/concat.cc) — stream + `InputArgCount` + `PrepareForCompute`
  - L36: `Node().InputArgCount().front()` → `ctx->InputCount()` (adapter provides `InputCount()`)
  - L46: `PrepareForCompute(ctx, ...)` — `ConcatBase::PrepareForCompute` expects framework `OpKernelContext*`. Check if template variant exists; if not, inline (~30 LOC: iterate inputs, validate axis, collect shapes) under `#ifdef BUILD_CUDA_EP_AS_PLUGIN`
  - L79, L92–L95: 5× `CopyToGpu(ctx->GetComputeStream())` → `CopyToGpu(GetComputeStream(ctx))`
  - `Concat` inherits `ConcatBase` — verify base class compiles with adapter (constructor parses `axis` attribute)
  - Remove CMake exclusion: `.*/tensor/concat\\.cc$` (line ~L124)

- [ ] **5A.4** Validate Stage 5A
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ./cuda.sh --build --test  # non-plugin regression
  ```
  - Progress:
    - `./cuda_plugin.sh --build` passed after 5A.1–5A.3 changes
    - `./cuda_plugin.sh --test_plugin` passed (Stage 2/3/4 plugin tests)

---

### Stage 5B: Single-Function CPU Dependency Ops (~2 days, 7 tasks)

These ops call a **single** utility function defined in the CPU provider. Fix by inlining the function body.

- [x] **5B.1** Fix [math/cumsum.cc](../../math/cumsum.cc) — 1 call at L56
  - `cumsum_op::GetAxis(axis_tensor, rank, axis)` — inline this function (~5 LOC: reads scalar from 1-element tensor, validates `-rank ≤ axis < rank`, wraps negative)
  - Add `#ifdef BUILD_CUDA_EP_AS_PLUGIN` block with inlined helper, else include original header
  - Remove CMake exclusion: `.*/math/cumsum\\.cc$` (line ~L166)

- [x] **5B.2** Fix [tensor/tile.cc](../../tensor/tile.cc) — 1 call at L106
  - `TileOp::IsTileMemcpy(input_shape, repeats, rank, ...)` — inline this static method (~30 LOC: iterates dims, checks if all repeats==1 except one)
  - Add `#ifdef BUILD_CUDA_EP_AS_PLUGIN` block with function duplicate, or extract into a shared header
  - Remove CMake exclusion: `.*/tensor/tile\\.cc$` (line ~L169)

- [x] **5B.3** Fix [tensor/gather.cc](../../tensor/gather.cc) — 1 call at L49
  - `PrepareForCompute(context, p)` from `GatherBase` — [gatherbase.h](../../../cpu/tensor/gatherbase.h) has a **template method** `PrepareForComputeImpl<KernelContextType>` at L21 that uses only `context->Input<Tensor>()` and shapes
  - Replace `PrepareForCompute(context, p)` → `PrepareForComputeImpl(context, p)` (context-type agnostic)
  - Verify `GatherBase` template constructor `GatherBase(const KernelInfoType& info)` works with adapter `OpKernelInfo`
  - Remove CMake exclusion: `.*/tensor/gather\\.cc$` (line ~L128)

- [x] **5B.4** Fix [tensor/unsqueeze.cc](../../tensor/unsqueeze.cc) — 1 call at L66
  - `PrepareCompute(ctx, p)` from `UnsqueezeBase` — body is ~10 LOC: reads input shape, computes output axes, calls `ctx->Output()`
  - Inline `PrepareCompute` under `#ifdef BUILD_CUDA_EP_AS_PLUGIN`
  - `UnsqueezeBase` constructor parses `axes` attribute — inline that too (~5 LOC)
  - Remove CMake exclusion: `.*/tensor/unsqueeze\\.cc$` (line ~L159)

- [x] **5B.5** Permanently exclude [tensor/shape_op.cc](../../tensor/shape_op.cc) and [tensor/size.cc](../../tensor/size.cc)
  - Both reuse CPU classes directly (no CUDA compute — they just read tensor metadata)
  - `Shape` and `Size` ops land on CPU via `GetCpuPreferredNodes` anyway
  - Added permanent exclusion comment in CMake:
    ```cmake
    # Permanently excluded — pure CPU ops, handled by GetCpuPreferredNodes.
    ```
  - Rationale documented in this task doc and reflected in `onnxruntime_providers_cuda_plugin.cmake`

- [x] **5B.6** Add adapter `RequiredInput<T>()` and `RequiredOutput()` to [op_kernel.h](../../../../../../include/onnxruntime/ep/adapter/op_kernel.h)
  - In `struct OpKernelContext`:
    ```cpp
    template <typename T, typename = std::enable_if_t<std::is_same_v<T, Tensor>>>
    const T& RequiredInput(int index) const {
      auto* p = Input<T>(index);
      ORT_ENFORCE(p != nullptr, "Required input ", index, " is null");
      return *p;
    }
    Tensor& RequiredOutput(int index, const TensorShape& shape) {
      auto* p = Output(index, shape);
      ORT_ENFORCE(p != nullptr, "Required output ", index, " is null");
      return *p;
    }
    ```
  - These are convenience wrappers needed by `variadic_elementwise_ops.cc` (Stage 5C.1)

- [x] **5B.7** Validate Stage 5B
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ./cuda.sh --build --test  # non-plugin regression (adapter changes affect both builds)
  ```
  - Progress:
    - `./cuda_plugin.sh --build` passed with Stage 5B inclusions (`gather`, `cumsum`, `tile`, `unsqueeze`)
    - `./cuda_plugin.sh --test_plugin` passed
    - `./cuda.sh --build` passed
    - `./cuda.sh --test` passed (`onnxruntime_test_all`: 1180 passed)

---

### Stage 5C: CPU Base Class & Adapter API Gap Ops (~3 days, 8 tasks)

These ops inherit from CPU base classes or use missing adapter APIs. Strategy: inline base class logic for the plugin build using `#ifdef BUILD_CUDA_EP_AS_PLUGIN`.

- [x] **5C.1** Fix [math/variadic_elementwise_ops.cc](../../math/variadic_elementwise_ops.cc) — 3 missing APIs
  - L161: `Node().InputArgCount().front()` → `context->InputCount()`
  - L169: `context->RequiredInput<Tensor>(i)` uses adapter `RequiredInput<T>()` added in 5B.6
  - L179, L197, L217: `context->RequiredOutput(0, shape)` uses adapter `RequiredOutput()` added in 5B.6
  - Remove CMake exclusion: `.*/math/variadic_elementwise_ops\\.cc$` (line ~L115)

- [x] **5C.2** Fix [tensor/pad.cc](../../tensor/pad.cc) — 3 `PadBase` static calls at L114, L117, L164
  - `PadBase::ComputePads(*ctx, ...)` replaced with plugin-safe `ComputePadsLocal(*ctx, ...)`
    - Plugin build path uses `PadBase::ComputePadsImpl(ctx, ...)` (templated context support)
    - Non-plugin path keeps `PadBase::ComputePads(ctx, ...)`
  - `PadBase::SeparateNegativeToSlices(pads, slices)` kept as-is (header-inline, context-free)
  - `PadBase::HandleDimValueZero(...)` replaced with plugin-safe `HandleDimValueZeroLocal(...)`
    - Plugin build path inlines CPU-equivalent validation for `Constant`/`Edge`/`Reflect`
    - Non-plugin path keeps `PadBase::HandleDimValueZero(...)`
  - Removed CMake exclusion: `.*/tensor/pad\\.cc$` (line ~L134)

- [ ] **5C.3** Fix [tensor/slice.cc](../../tensor/slice.cc) — `SliceBase` calls at L175, L180, L182, L264
  - `SliceBase::PrepareForCompute(starts, ends, axes, steps, compute_metadata)` at L180 — shape validation ~60 LOC
  - `SliceBase::FlattenOutputDims(...)` at L264 — dimension optimization ~30 LOC
  - Both are pure shape computation (no CUDA, no context). Create plugin-local inline versions under `#ifdef BUILD_CUDA_EP_AS_PLUGIN`.
  - `Slice` inherits `SliceBase` for `starts_/ends_/axes_` attributes: inline attribute parsing (~10 LOC).
  - `SliceOp::PrepareForComputeMetadata` at L175 is a POD struct — should compile as-is.
  - Remove CMake exclusion: `.*/tensor/slice\\.cc$` (line ~L118)

- [ ] **5C.4** Fix [tensor/space_depth_ops.cc](../../tensor/space_depth_ops.cc)
  - Inherited `SpaceDepthBase` provides `blocksize_` (int64) and `is_dcr_` (bool) members: used at L188–L198, L239, L243
  - `SpaceDepthBase` is small: constructor reads `blocksize` attribute + `mode` attribute for DCR/CRD (~5 LOC)
  - Inline the constructor logic and member fields under `#ifdef BUILD_CUDA_EP_AS_PLUGIN` by adding a plugin-local `SpaceDepthPlugin` mixin or by adding the fields directly to the classes:
    ```cpp
    #ifdef BUILD_CUDA_EP_AS_PLUGIN
    int64_t blocksize_;  bool is_dcr_;   // parsed from OpKernelInfo in ctor
    #endif
    ```
  - Remove CMake exclusion: `.*/tensor/space_depth_ops\\.cc$` (line ~L121)

- [ ] **5C.5** Fix [generator/constant_of_shape.cc](../../generator/constant_of_shape.cc) — L24, L26
  - `PrepareCompute(ctx, &output_tensor)` from `ConstantOfShapeBase<>` — reads shape input, creates output tensor (~15 LOC)
  - `GetValuePtr()` from `ConstantOfShapeBase<>` — returns pointer to stored `value` attribute (~5 LOC)
  - `ConstantOfShapeBase` constructor parses `value` TensorProto attribute (~20 LOC)
  - Inline all under `#ifdef BUILD_CUDA_EP_AS_PLUGIN`: store parsed value tensor directly in the class
  - Remove CMake exclusion: `.*/generator/constant_of_shape\\.cc$` (line ~L106)

- [ ] **5C.6** Fix [tensor/upsample.cc](../../tensor/upsample.cc) — L48, L108, L277 + `UpsampleBase`
  - L48: `info.GetAllocator(OrtMemTypeDefault)` — not in adapter. Replace with `GetScratchBuffer<T>()` at compute time (already used at L108, L277)
  - `UpsampleBase` is a **large** base class (~200 LOC): parses `mode`, `coordinate_transform_mode`, `scales`, `roi`, `exclude_outside`, etc.
  - **Approach**: Try to include the `UpsampleBase` header directly (it's in `core/providers/cpu/tensor/upsample.h`). If the constructor compiles with adapter `OpKernelInfo` (it uses `GetAttr`/`GetAttrs`), this may just work.
  - If `UpsampleBase` doesn't compile: **defer** to a later iteration — document rationale.
  - Remove CMake exclusion: `.*/tensor/upsample\\.cc$` (line ~L148) — or keep if deferred

- [ ] **5C.7** Fix [tensor/resize.cc](../../tensor/resize.cc)
  - Inherits from `Upsample<T>` which inherits `UpsampleBase + CudaKernel`
  - Blocked until 5C.6 (upsample) is resolved
  - Remove CMake exclusion: `.*/tensor/resize\\.cc$` (line ~L151) — same timeline as 5C.6

- [ ] **5C.8** Validate Stage 5C
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ./cuda.sh --build --test  # non-plugin regression
  ```
  - Progress:
    - `./cuda_plugin.sh --build` passed after 5C.1 inclusion (`variadic_elementwise_ops.cc`)
    - `./cuda_plugin.sh --test_plugin` passed
    - `./cuda.sh --build` passed
    - `./cuda_plugin.sh --build` passed after 5C.2 inclusion (`pad.cc`)
    - `./cuda_plugin.sh --test_plugin` passed after 5C.2 inclusion
    - `./cuda.sh --build` passed after 5C.2 inclusion

---

### Stage 5D: Contrib Ops — GetComputeStream Batch (~3–5 days, 5 tasks)

All excluded contrib ops share the same `ctx->GetComputeStream()` pattern. Fix systematically.

> **Note**: Some excluded contrib ops (e.g., `fast_gelu.cc`, `embed_layer_norm.cc`) may only use `Stream(context)` (the CudaKernel member), not `ctx->GetComputeStream()`. Try removing their CMake exclusions first — they may compile without changes.

- [ ] **5D.1** Create systematic `GetComputeStream` fix infrastructure
  - The root pattern is: `context->GetComputeStream()` returns `onnxruntime::Stream*` in the framework, but `adapter::OpKernelContext` doesn't have this method (it has `GetGPUComputeStream()` returning `void*`)
  - Three sub-patterns to fix:
    1. `buf.CopyToGpu(ctx->GetComputeStream())` — fix via `CopyToGpu(void*)` overload (task 5A.0)
    2. `GetScratchBuffer<T>(bytes, ctx->GetComputeStream())` — fix: use `GetScratchBuffer<T>(bytes, GetComputeStream(ctx))` where `CudaKernel::GetComputeStream(ctx)` returns `void*`
    3. `QkvToContext(... context->GetComputeStream() ...)` — function signature takes `Stream*`: add `#ifdef BUILD_CUDA_EP_AS_PLUGIN` overload or change to accept `cudaStream_t`
  - Add a macro/helper in [cuda_kernel_adapter.h](../cuda_kernel_adapter.h):
    ```cpp
    // Helper for contrib ops that pass stream to downstream functions
    #define CUDA_STREAM_FROM_CTX(ctx) static_cast<cudaStream_t>(GetComputeStream(ctx))
    ```

- [ ] **5D.2** Fix contrib bert ops (13 files)
  - **Try compile first** — remove exclusion, build, see what fails. Some ops may only use `Stream(context)`:
    - [bert/fast_gelu.cc](../../../../contrib_ops/cuda/bert/fast_gelu.cc) — uses `Stream(context)` at L55, **likely compiles**
    - [bert/embed_layer_norm.cc](../../../../contrib_ops/cuda/bert/embed_layer_norm.cc) — uses `Stream(context)` at L67, **likely compiles**
  - Ops that use `context->GetComputeStream()`:
    - [bert/attention.cc](../../../../contrib_ops/cuda/bert/attention.cc) — also inherits `AttentionBase` (contrib base, should compile). ComputeInternal uses stream via attention helpers.
    - [bert/decoder_attention.cc](../../../../contrib_ops/cuda/bert/decoder_attention.cc)
    - [bert/decoder_masked_self_attention.cc](../../../../contrib_ops/cuda/bert/decoder_masked_self_attention.cc)
    - [bert/group_query_attention.cc](../../../../contrib_ops/cuda/bert/group_query_attention.cc)
    - [bert/longformer_attention.cc](../../../../contrib_ops/cuda/bert/longformer_attention.cc)
    - [bert/multihead_attention.cc](../../../../contrib_ops/cuda/bert/multihead_attention.cc)
    - [bert/packed_attention.cc](../../../../contrib_ops/cuda/bert/packed_attention.cc)
    - [bert/packed_multihead_attention.cc](../../../../contrib_ops/cuda/bert/packed_multihead_attention.cc)
    - [bert/paged_attention.cc](../../../../contrib_ops/cuda/bert/paged_attention.cc)
    - [bert/relative_attn_bias.cc](../../../../contrib_ops/cuda/bert/relative_attn_bias.cc)
    - [bert/remove_padding.cc](../../../../contrib_ops/cuda/bert/remove_padding.cc)
  - For each: replace `ctx->GetComputeStream()` with `GetComputeStream(ctx)` or `Stream(ctx)` depending on what the callee expects (`void*` vs `cudaStream_t`)
  - If attention helpers take `Stream*`: add `#ifdef BUILD_CUDA_EP_AS_PLUGIN` overloads or change to `cudaStream_t`
  - Remove corresponding CMake exclusions (13 lines)

- [ ] **5D.3** Fix other contrib ops (10 files)
  - [diffusion/group_norm.cc](../../../../contrib_ops/cuda/diffusion/group_norm.cc) — L211, L215: `context->GetComputeStream()`
  - [fused_conv.cc](../../../../contrib_ops/cuda/fused_conv.cc)
  - [inverse.cc](../../../../contrib_ops/cuda/inverse.cc)
  - [math/bias_dropout.cc](../../../../contrib_ops/cuda/math/bias_dropout.cc) — L127: `context->GetComputeStream()`, L139: `Stream(context)` (mixed)
  - [math/fft_ops.cc](../../../../contrib_ops/cuda/math/fft_ops.cc)
  - [moe/moe.cc](../../../../contrib_ops/cuda/moe/moe.cc) — L52: `context->GetComputeStream()`, L107/L119: `Stream(context)` (mixed)
  - [sparse/sparse_attention.cc](../../../../contrib_ops/cuda/sparse/sparse_attention.cc)
  - [tensor/crop.cc](../../../../contrib_ops/cuda/tensor/crop.cc)
  - [tensor/dynamic_time_warping.cc](../../../../contrib_ops/cuda/tensor/dynamic_time_warping.cc)
  - [tensor/dynamicslice.cc](../../../../contrib_ops/cuda/tensor/dynamicslice.cc)
  - Same fix pattern: `ctx->GetComputeStream()` → `GetComputeStream(ctx)` with `#ifdef` guards
  - Remove corresponding CMake exclusions (10+ lines)

- [ ] **5D.4** Fix contrib quantization ops (5 entries)
  - [quantization/attention_quantization.cc](../../../../contrib_ops/cuda/quantization/attention_quantization.cc)
  - [quantization/matmul_bnb4.cc](../../../../contrib_ops/cuda/quantization/matmul_bnb4.cc)
  - [quantization/matmul_nbits.cc](../../../../contrib_ops/cuda/quantization/matmul_nbits.cc) — L307: `static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())`, L355/L397: `GetScratchBuffer<T>(bytes, ctx->GetComputeStream())`
  - [quantization/moe_quantization.cc](../../../../contrib_ops/cuda/quantization/moe_quantization.cc)
  - [quantization/qordered_ops/\*](../../../../contrib_ops/cuda/quantization/qordered_ops/)
  - Fix pattern: replace `ctx->GetComputeStream()` and `ctx->GetComputeStream()->GetHandle()` with adapter equivalents
  - For `matmul_nbits.cc` L307: `static_cast<cudaStream_t>(ctx->GetComputeStream()->GetHandle())` → `Stream(ctx)` (CudaKernel member)
  - Remove corresponding CMake exclusions (~5 lines)

- [ ] **5D.5** Validate Stage 5D
  ```bash
  ./cuda_plugin.sh --build --test --test_plugin
  ./cuda.sh --build --test  # non-plugin regression
  ```

---

### Stage 5E: Deferred Ops (no code changes — documented exclusion)

> These ops have dependencies that cannot be resolved with simple fixes. Each has a documented rationale for deferral.

- [x] **5E.1** Control flow ops (`controlflow/*`) — **Already resolved**
  - Plugin has its own wrappers in `plugin/cuda_controlflow_plugin.cc` using `OrtEpApi::CreateIfKernel`/`CreateLoopKernel`/`CreateScanKernel`
  - CMake exclusion for `controlflow/*` is **deliberate** — plugin files replace originals

- [ ] **5E.2** RNN ops (`rnn/*`) — **Deferred: missing ORT C API**
  - Blocker: [rnn.h L21](../../rnn/rnn.h) calls `info.GetAttrs("activations", activations_)` — this requires `KernelInfoGetAttributeArray_string` in the ORT C API, which is **not implemented**
  - All other RNN deps are resolved: `GetComputeStream(ctx)`, `GetCudnnHandle(ctx)`, `Stream(ctx)`, `GetScratchBuffer` — all route through `CudaKernel` → `CudaSyncStream`
  - No `dynamic_cast<CudaStream*>` exists in any RNN file
  - **Action**: File ORT C API extension request for `KernelInfoGetAttributeArray_string`
  - **Files**: `cudnn_rnn_base.cc/.h`, `rnn.cc/.h`, `gru.cc/.h`, `lstm.cc/.h`, `rnn_impl.cu/.h`

- [ ] **5E.3** Tunable ops (`tunable/*`) — **Deferred: non-critical**
  - Depends on `CudaTuningContext` + `CUDAExecutionProvider*`
  - Plugin has `PluginTuningContextStub` returning `IsTunableOpEnabled() → false`
  - Tuning is a performance optimization, not functional — safe to defer

- [ ] **5E.4** Einsum (`math/einsum.cc` + `einsum_utils/*`) — **Deferred: hard-coupled**
  - [einsum.h L10](../../math/einsum.h): `#include "core/providers/cuda/cuda_execution_provider.h"`
  - [einsum.h L21](../../math/einsum.h): `cuda_ep_ = static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider())`
  - The class directly stores `CUDAExecutionProvider*` and uses it for stream/handle access
  - Also, `einsum_auxiliary_ops.cc` calls `ReductionOps::ReduceCompute` (framework-only path)
  - Requires major refactoring to decouple from concrete EP class — not worth the effort now

- [ ] **5E.5** Object detection (`object_detection/*`) — **Deferred: complex CPU base classes**
  - `NonMaxSuppression` inherits `NonMaxSuppressionBase`, `RoiAlign` inherits `RoiAlignBase`
  - Both base classes have significant CPU compute logic (~100+ LOC each)
  - 6 files: `non_max_suppression.cc/.h`, `non_max_suppression_impl.cu/.h`, `roialign.cc/.h`, `roialign_impl.cu/.h`
  - Low model coverage priority — these ops rarely appear in modern inference models

- [ ] **5E.6** Transformers (`contrib_ops/cuda/transformers/*`) — **Deferred: massive deps**
  - Beam search, greedy search, sampling — depend on subgraph execution, generator state, and deep framework integration
  - Not portable via simple `GetComputeStream` fixes

- [ ] **5E.7** LLM ops (`core/providers/cuda/llm/*` + `contrib_ops/cuda/llm/*`) — **Deferred: deep Stream\* usage**
  - `QkvToContext` and related functions pass `onnxruntime::Stream*` through multiple call layers
  - Requires function signature refactoring across many files

- [ ] **5E.8** TensorSeq ops (`tensor/identity_op.cc`, `tensor/sequence_op.cc`) — **Deferred: incomplete type**
  - `TensorSeq` is an incomplete type in the plugin build
  - Adapter doesn't provide `Input<TensorSeq>()` / `Output<TensorSeq>()`
  - Sequence ops typically land on CPU via `GetCpuPreferredNodes`

- [ ] **5E.9** Training / out-of-scope (`tensor/shrunken_gather.cc`, `aten_ops/*`, `collective/*`) — **Deferred: not applicable**
  - Training-specific or multi-GPU ops, explicitly out of scope

- [ ] **5E.10** Contrib gemm_float8 (`math/gemm_float8.cc/.cu`) — **Deferred: .cu stream dep**
  - `GetComputeStream()` used in `.cu` file (NVCC-compiled)
  - NVCC doesn't get the forced include; needs `cudaStream_t` parameter change in compute kernel signatures

---

### Stage 5F: Audit & Validate (~1 day, 4 tasks)

- [ ] **5F.1** Run audit for remaining `provider_api.h` references
  ```bash
  grep -rn 'provider_api\.h\|ProviderHost_impl\|g_host' \
    onnxruntime/core/providers/cuda/plugin/
  ```
  Verify zero code matches (docs excluded).

- [ ] **5F.2** Create registration parity report
  - Dump `(domain, op_type, since_version)` tuples from plugin `adapter::KernelRegistry`
  - Compare against bundled EP registry
  - Plugin count must equal bundled count minus tracked deferred exclusions
  - Add diagnostic script or test to `test_cuda_plugin_ep.py`

- [ ] **5F.3** Expand test coverage in [test_cuda_plugin_ep.py](../../../../test/python/transformers/test_cuda_plugin_ep.py)
  - **Standard ops newly included**: Reshape, Split, Concat, Gather, Pad, Slice, Unsqueeze, Tile, CumSum, ConstantOfShape, SpaceToDepth/DepthToSpace
  - **Variadic ops**: Sum (variadic add), Max (variadic max), Min (variadic min)
  - **Control flow ops**: If, Loop, Scan (already implemented in plugin but untested)
  - **Contrib ops**: attention, fast_gelu, bias_dropout, group_norm (as available from 5D)
  - Each test: build model, run with plugin EP, compare against CPU/PyTorch reference

- [ ] **5F.4** Full CI validation
  ```bash
  ./cuda.sh --build --test           # Non-plugin regression (1170+ tests)
  ./cuda_plugin.sh --build --test --test_plugin  # Plugin build + tests
  ```

---

## Stage 6: Advanced Features & Polish

> **Goal**: Profiling, perf validation, packaging, CI, documentation.

### 6.1 NVTX Profiling

- [ ] **6.1.1** Add NVTX range markers around kernel execution in plugin EP
  - Include `nvtx3/nvToolsExt.h` in [cuda_kernel_adapter.h](../cuda_kernel_adapter.h) or in `adapter::KernelImpl::ComputeImpl()`
  - Wrap `Compute()` calls with `nvtxRangePushA(op_type)` / `nvtxRangePop()`
  - Kernel name available via `OpKernel::Node().OpType()`
  - Link against `nvToolsExt` library in CMake
  - Make profiling opt-in via compile flag or session config

### 6.2 External Resource Import

- [ ] **6.2.1** Implement `OrtExternalResourceImporterImpl` for CUDA memory/semaphore import
  - For CUDA memory import (`cudaIpcMemHandle_t` based sharing)
  - For CUDA semaphore/event import
  - Wire up to `CudaEpFactory` callbacks
  - Test with multi-process inference scenario

### 6.3 Performance Regression Testing

- [ ] **6.3.1** Create benchmark harness: [test/python/transformers/test_cuda_plugin_perf.py](../../../../test/python/transformers/test_cuda_plugin_perf.py)
  - Framework: `pytest-benchmark` or custom timing loop
  - Models to benchmark:
    | Model | Size | Notes |
    |-------|------|-------|
    | BERT-base | ~110M params | Attention + LayerNorm heavy |
    | ResNet-50 | ~25M params | Conv + BatchNorm heavy; tests NHWC path |
    | GPT-2 | ~117M params | MatMul + Softmax heavy |
    | Stable Diffusion (UNet) | ~860M params | Conv + Attention + GroupNorm |
  - Metric targets:
    | Metric | Target | Measurement |
    |--------|--------|-------------|
    | Latency | < 2% regression vs bundled EP | Median of 100 runs after 10 warmup |
    | GPU memory peak | < 5% increase vs bundled EP | `torch.cuda.max_memory_allocated()` equivalent |
    | First-run latency | < 10% regression | Includes plugin DLL load + warm-up |
  - Output: CSV report + pass/fail gate for CI

- [ ] **6.3.2** Run initial performance comparison
  - Identify any latency regressions and root-cause them
  - Likely sources: adapter layer overhead (virtual dispatch), missed optimizations, memory layout

### 6.4 Python Packaging

- [ ] **6.4.1** Update [setup.py](../../../../../setup.py) to include plugin DLL
  - Add `libonnxruntime_providers_cuda_plugin.so` to package data
  - Gate on `BUILD_CUDA_EP_AS_PLUGIN=ON` build flag
  - Update `onnxruntime-gpu` wheel spec
- [ ] **6.4.2** Verify end-to-end pip install + plugin load
  ```python
  pip install onnxruntime-gpu
  import onnxruntime as ort
  ort.register_execution_provider_library("CUDAExecutionProvider", "/path/to/plugin.so")
  sess = ort.InferenceSession("model.onnx", providers=["CUDAExecutionProvider"])
  ```

### 6.5 CI Pipeline

- [ ] **6.5.1** Create dedicated CI job for plugin mode
  - Build with `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON`
  - Run `onnxruntime_test_all` + `test_cuda_plugin_ep.py`
  - Add to PR gate checks (at least nightly initially)
- [ ] **6.5.2** Add performance regression CI (nightly)
  - Run `test_cuda_plugin_perf.py` on dedicated GPU machine
  - Alert on > 2% regression

### 6.6 Documentation

- [ ] **6.6.1** Write API migration guide: [plugin/doc/migration_guide.md](doc/migration_guide.md)
  - How to port existing `ONNX_OPERATOR_*_KERNEL_EX` based kernels
  - How the `ep/adapters.h` forced-include works
  - How the `PluginKernelCollector` self-registration pattern works
  - Step-by-step example: porting a single kernel from bundled to plugin
- [ ] **6.6.2** Document deferred ops and exclusion rationale
  - Reference 5E.1–5E.10 with permanent vs temporary classification
  - Include migration path for each deferred category when blockers are resolved
- [ ] **6.6.3** Update [cuda_plugin_ep_plan.md](cuda_plugin_ep_plan.md)
  - Mark Stages 5 & 6 as complete
  - Update timeline with actual durations
  - Add lessons learned section

---

## Cross-Cutting Concerns

### Files to Delete (cumulative across stages)

| File | Stage | Status | Replaced By |
|------|-------|--------|-------------|
| `plugin/provider_host_bridge.cc` | 1.10 | ✅ Deleted | N/A (legacy bridge) |
| `plugin/provider_api_shims.cc` | 1.10 | ✅ Simplified | Standalone implementations (kept; `g_host` calls removed) |
| `plugin/cuda_plugin_adapter_registry.cc` | 2.5 | ✅ Deleted | `PluginKernelCollector` self-registration |
| Legacy `AdapterKernelImpl` in `cuda_plugin_kernels.cu` | 1.10 | ✅ Removed | `ep::adapter::KernelImpl` |
| `AdapterKernelImpl`/`PluginRegistry`/macro overrides in `cuda_kernel_adapter.h` | 1.11 | ✅ Removed | `ep::adapter::KernelImpl`/`KernelRegistry` |

### Files Modified (completed — Stages 1–4)

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
| `plugin/cuda_ep.h/.cc` | 3.1, 4.1–4.3 | Added `ShouldConvertDataLayoutForOp`; CUDA graph state; `OnRunStartImpl`/`OnRunEndImpl` with capture state machine; `GetAnnotationId()` using `gpu_graph_id` key |
| `plugin/cuda_ep_factory.h/.cc` | 2.1, 4.1 | Use `adapter::KernelRegistry`; added `GetComputeStream()` and `compute_stream_` member; reads CUDA graph session config |
| `test/python/transformers/test_cuda_plugin_ep.py` | 3.3, 4.6 | NHWC tests; CUDA graph tests with IO binding; `gpu_graph_id` run config key |

### Files to Modify (remaining — Stages 5–6)

| File | Stage | Changes |
|------|-------|---------|
| `cmake/onnxruntime_providers_cuda_plugin.cmake` | 5A–5D | Progressively remove CMake exclusion filters as ops are fixed |
| `cuda/tensor/reshape.cc` | 5A.1 | `#ifdef BUILD_CUDA_EP_AS_PLUGIN` block for `GetComputeStream` |
| `cuda/tensor/split.cc` | 5A.2 | Replace 5× `CopyToGpu(ctx->GetComputeStream())` with `CopyToGpu(GetComputeStream(ctx))` |
| `cuda/tensor/concat.cc` | 5A.3 | Replace `InputArgCount`, `PrepareForCompute`, `CopyToGpu` stream calls |
| `cuda/math/cumsum.cc` | 5B.1 | Inline `cumsum_op::GetAxis` under `#ifdef` |
| `cuda/tensor/tile.cc` | 5B.2 | Inline `TileOp::IsTileMemcpy` under `#ifdef` |
| `cuda/tensor/gather.cc` | 5B.3 | `PrepareForCompute` → `PrepareForComputeImpl` (template) |
| `cuda/tensor/unsqueeze.cc` | 5B.4 | Inline `UnsqueezeBase::PrepareCompute` under `#ifdef` |
| `include/onnxruntime/ep/adapter/op_kernel.h` | 5B.6 | Add `RequiredInput<T>()` and `RequiredOutput()` |
| `cuda/math/variadic_elementwise_ops.cc` | 5C.1 | `InputArgCount()` → `InputCount()`; use `RequiredInput`/`RequiredOutput` |
| `cuda/tensor/pad.cc` | 5C.2 | Inline 3× `PadBase::` methods + `mode_` attribute |
| `cuda/tensor/slice.cc` | 5C.3 | Inline `SliceBase::PrepareForCompute` + `FlattenOutputDims` |
| `cuda/tensor/space_depth_ops.cc` | 5C.4 | Inline `SpaceDepthBase` members |
| `cuda/generator/constant_of_shape.cc` | 5C.5 | Inline `ConstantOfShapeBase` methods |
| `cuda/tensor/upsample.cc` | 5C.6 | Fix `GetAllocator`; attempt `UpsampleBase` inclusion |
| `cuda/tensor/resize.cc` | 5C.7 | Dependent on upsample fix |
| `plugin/cuda_kernel_adapter.h` | 5D.1 | Add `GetComputeStream` macro/helper for contrib ops |
| ~25 contrib ops `.cc` files | 5D.2–5D.4 | Stream fix `#ifdef` blocks |
| `test/python/transformers/test_cuda_plugin_ep.py` | 5F.3, 6.3.1 | Add tests for newly-included ops + perf benchmarks |

### New Files to Create

| File | Stage | Status | Purpose |
|------|-------|--------|---------|
| `plugin/cuda_graph_plugin.h/.cc` | 4.1 | ✅ Created | Plugin-compatible CUDA graph manager |
| `plugin/cuda_controlflow_plugin.h/.cc/.cu` | 5E.1 | ✅ Created | Plugin control flow wrappers |
| `test/python/transformers/test_cuda_plugin_perf.py` | 6.3.1 | Pending | Performance regression benchmarks |
| `plugin/doc/migration_guide.md` | 6.6.1 | Pending | Third-party kernel migration guide |

### Verification Commands

```bash
# Full build + test cycle
./cuda_plugin.sh --build --test --test_plugin

# Quick rebuild + plugin test only
./cuda_plugin.sh --build --test_plugin

# If you change files except those under onnxruntime/core/providers/cuda/plugin
# (this directory is excluded from non-plugin build),
# you need to run non-plugin build and test to ensure backward compatibility
./cuda.sh --build --test

# Audit for legacy references
grep -rn 'provider_api\.h\|SHARED_PROVIDER\|g_host' \
  onnxruntime/core/providers/cuda/plugin/

# Count registered kernels (diagnostic)
python -c "
import onnxruntime as ort
ort.register_execution_provider_library('CUDAExecutionProvider', '/path/to/plugin.so')
# ... load a model and check GetCapability output
"
```

### Excluded Op Tracking Summary

| Category | Files | Status | Blocker |
|----------|-------|--------|---------|
| Stream fix (standard) | reshape, split, concat | 5A: **Pending** | `ctx->GetComputeStream()` |
| CPU utility (standard) | cumsum, tile, gather, unsqueeze | 5B: **Pending** | Single function from CPU provider |
| CPU base class (standard) | pad, slice, space_depth_ops, constant_of_shape, variadic_elementwise_ops | 5C: **Pending** | Multiple inheritance / missing adapter API |
| CPU base class (complex) | upsample, resize | 5C: **Pending** | Large `UpsampleBase` class (~200 LOC) |
| Stream fix (contrib) | 13 bert + 10 other + 5 quant | 5D: **Pending** | `ctx->GetComputeStream()` |
| Permanent CPU exclusion | shape_op, size | 5B: **Permanent** | CPU ops handled by `GetCpuPreferredNodes` |
| Control flow | If, Loop, Scan | 5E: ✅ **Resolved** | Plugin wrappers using `OrtEpApi` |
| RNN | 6 files | 5E: **Deferred** | Missing `KernelInfoGetAttributeArray_string` C API |
| Tunable | tunable/* | 5E: **Deferred** | Non-critical; stub returns disabled |
| Einsum | einsum.cc + einsum_utils/* | 5E: **Deferred** | Hard-coupled to `CUDAExecutionProvider*` |
| Object detection | 8 files | 5E: **Deferred** | Complex CPU base classes (~100+ LOC each) |
| Transformers | transformers/* | 5E: **Deferred** | Subgraph execution / generator deps |
| LLM | llm/* | 5E: **Deferred** | Deep `onnxruntime::Stream*` usage |
| TensorSeq | identity_op, sequence_op | 5E: **Deferred** | Incomplete type; lands on CPU anyway |
| Training/scope | shrunken_gather, aten_ops, collective | 5E: **Deferred** | Out of scope |
| gemm_float8 | gemm_float8.cc/.cu | 5E: **Deferred** | `GetComputeStream` in .cu file |
