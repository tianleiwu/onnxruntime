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
| `plugin/cuda_ep.h` / `cuda_ep.cc` | 3.1, 4.1–4.3 | Added `ShouldConvertDataLayoutForOp`; CUDA graph state (`cuda_graph_enabled_`, `cuda_graph_manager_`, `graph_id_to_run_count_`, `is_capturing_`); `OnRunStartImpl`/`OnRunEndImpl` with capture state machine; `GetAnnotationId()` using `gpu_graph_id` key; `IsGraphCaptureAllowed()` |
| `plugin/cuda_ep_factory.h` / `cuda_ep_factory.cc` | 2.1, 4.1 | Use `adapter::KernelRegistry`; added `GetComputeStream()` and `compute_stream_` member; reads `enable_cuda_graph` and `min_num_runs_before_cuda_graph_capture` from session config |
| `test/python/transformers/test_cuda_plugin_ep.py` | 4.6 | Moved CUDA graph tests to `test_cuda_plugin_cuda_graph()`; IO binding + OrtValue pattern; `gpu_graph_id` run config key |

### New Files to Create

| File | Stage | Purpose |
|------|-------|---------|
| `plugin/cuda_graph_plugin.h/.cc` | 4.1 | ✅ Created | Plugin-compatible CUDA graph manager |
| Control flow wrappers | 5.1 | Pending | `If`/`Loop`/`Scan` kernel wrappers using `OrtEpApi` |

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
