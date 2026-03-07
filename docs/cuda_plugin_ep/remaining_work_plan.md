# Plan: CUDA Plugin EP — Remaining Work

## TL;DR

The CUDA Plugin EP core infrastructure (Stages 1-5) is substantially complete: plugin loads, ~80% of CUDA kernels compile and pass tests, CUDA graph capture works (replay pending API extension), NHWC layout transformation works, and the architecture has been refactored to follow the WebGPU adapter::Ep pattern. The remaining work focuses on three tracks: (1) migrating the remaining ~20% of excluded kernels (priority), (2) filling ORT core API gaps to unblock blocked op categories, and (3) infrastructure work (CI, packaging, profiling, performance testing).

---

## Phase 1: High-Priority Kernel Migrations (Low Effort) — **COMPLETED**

9 ops enabled and verified (build passes, `ConstantOfShape` tested in Stage 5).

**Completed ops:**

1. ✅ `generator/constant_of_shape.cc` — already had `#ifdef BUILD_CUDA_EP_AS_PLUGIN` path; CMake exclusion removed
2. ✅ `contrib/bert/embed_layer_norm.cc` — added `#pragma push_macro("SHARED_PROVIDER")` shim for inline template `CheckInputs`
3. ✅ `contrib/bert/relative_attn_bias.cc` — all patterns adapter-compatible; CMake exclusion removed
4. ✅ `contrib/bert/remove_padding.cc` — `GetScratchBuffer`/`Stream(ctx)` adapter-compatible; CMake exclusion removed
5. ✅ `contrib/tensor/crop.cc` — added plugin-local `CropBase` in `crop.h` (avoids framework `OpKernelInfo` type mismatch)
6. ✅ `contrib/tensor/dynamic_time_warping.cc` — guarded `using onnxruntime::OpKernelContext/OpKernelInfo` in header
7. ✅ `contrib/tensor/dynamicslice.cc` — pure registration delegating to `cuda::Slice<true>`; CMake exclusion removed
8. ✅ `contrib/inverse.cc` — `MLTypeCallDispatcher`/`Stream`/`GetScratchBuffer` all adapter-compatible; CMake exclusion removed
9. ✅ `contrib/quantization/matmul_bnb4.cc` — `ctx->GetComputeStream()->GetHandle()` works via adapter shim; CMake exclusion removed

**Deferred to Phase 2:**
- `matmul_nbits.cc` — uses `info.node().InputDefs()[idx]->Exists()` / `TypeAsProto()` which adapter `Node` doesn't support
- `gemm_float8.cc/.cu` — `.cu` uses `ToCudaDataType()` defined in `cuda_common.cc` (excluded from plugin); needs shim

**Changed files:**
- `cmake/onnxruntime_providers_cuda_plugin.cmake` — 9 exclusions commented out
- `onnxruntime/contrib_ops/cuda/tensor/crop.h` — plugin `CropBase` shim
- `onnxruntime/contrib_ops/cuda/bert/embed_layer_norm.cc` — `SHARED_PROVIDER` push/pop for inline `CheckInputs`
- `onnxruntime/contrib_ops/cuda/tensor/dynamic_time_warping.h` — guarded `using` declarations

---

## Phase 2: Medium-Priority Kernel Migrations (Moderate Refactoring) — **COMPLETED**

20 ops enabled across non-attention and attention categories. Build passes for both plugin and bundled EP.

**Non-attention ops (Steps 1-5, 7-10):**

1. ✅ `contrib/diffusion/group_norm.cc` — `CudaTuningContext*` cast guard under `#ifdef BUILD_CUDA_EP_AS_PLUGIN`
2. ✅ `contrib/moe/moe.cc` — stream/allocator patterns already adapter-compatible; CMake exclusion removed
3. **DEFERRED** `contrib/fused_conv.cc` — casts to `CUDAExecutionProvider*` directly, requires deeper EP interface refactoring
4. ✅ `contrib/math/fft_ops.cc` — `CUFFT_RETURN_IF_ERROR` macro shim added to `cuda_kernel_adapter.h`; `CUDA::cufft` linked
5. ✅ `contrib/math/bias_dropout.cc` — `PhiloxGenerator` is a plain framework class, no guards needed
6. ✅ `contrib/bert/fast_gelu.cc` — inline `CheckInputs` shim replacing `bias_gelu_helper::CheckInputs`
7. ✅ `contrib/quantization/matmul_nbits.cc` — `InputDefs()`/`TypeAsProto()` replaced with KernelInfo C API
8. ✅ `contrib/math/gemm_float8.cc/.cu` — `ToCudaDataType()`, `cublasGetErrorEnum()`, `CudaDataTypeToString()`, `CublasComputeTypeToString()` shims added
9. ✅ `contrib/quantization/moe_quantization.cc` — same adapter patterns as `moe.cc`

**Attention ops (Step 6):**

10. ✅ `bert/attention.cc` — `GetComputeStream()`/`GetScratchBuffer()`/`Stream*` all adapter-compatible
11. ✅ `bert/multihead_attention.cc` — same adapter-compatible patterns
12. ✅ `bert/decoder_attention.cc` — same adapter-compatible patterns
13. ✅ `bert/decoder_masked_self_attention.cc` — same adapter-compatible patterns
14. ✅ `bert/group_query_attention.cc` — fixed `Output<Tensor>(index)` → saved `present_key/value_tensor` pointers from prior `Output(index, shape)` call
15. ✅ `bert/packed_attention.cc` — already adapter-compatible
16. ✅ `bert/packed_multihead_attention.cc` — already adapter-compatible
17. ✅ `bert/paged_attention.cc` — already adapter-compatible
18. ✅ `sparse/sparse_attention.cc` — `Stream*` variable usage fully compatible
19. ✅ `quantization/attention_quantization.cc` — same adapter-compatible patterns

**Deferred from Phase 2:**

- `bert/longformer_attention.cc` — `LongformerAttentionBase` ctor takes framework `OpKernelInfo` + `CheckInputs` defined in CPU `.cc` not linked into plugin. Low model demand; deferred.
- `fused_conv.cc` — casts to `CUDAExecutionProvider*` directly. Requires deeper refactoring.

**Changed files (Phase 2):**
- `cmake/onnxruntime_providers_cuda_plugin.cmake` — 19 exclusions removed, `CUDA::cufft` linked
- `onnxruntime/contrib_ops/cuda/bert/fast_gelu.cc` — inline CheckInputs under `BUILD_CUDA_EP_AS_PLUGIN`
- `onnxruntime/contrib_ops/cuda/bert/group_query_attention.cc` — save output tensor pointers to avoid `Output<Tensor>(index)` template
- `onnxruntime/contrib_ops/cuda/diffusion/group_norm.cc` — `CudaTuningContext*` cast guard
- `onnxruntime/contrib_ops/cuda/quantization/matmul_nbits.h` — KernelInfo C API path for plugin build
- `onnxruntime/core/providers/cuda/plugin/cuda_kernel_adapter.h` — 5 shims (CUFFT macro, cublas/CUDA type helpers, `ToCudaDataType`)
- `tools/ci_build/cuda_plugin_parity_report.py` — whitespace cleanup, `os` import, `IOError` handling

**Verification:**
1. ✅ Plugin build compiles with all 19 newly-included ops
2. ✅ Standard CUDA EP build is unaffected (bundled build green)
3. `test_cuda_plugin_ep.py` should be run to check runtime correctness

---

## Phase 3: ORT Core API Extensions (Unblocking Blocked Categories)

These require changes to the ORT core C API (`onnxruntime_ep_c_api.h`) and framework code. Each unlocks a category of deferred ops.

**Step 3.1** — Add `KernelInfoGetAttributeArray_string` to ORT C API (*unblocks RNN ops*) (*deferred*)
- Add function to `OrtEpApi` struct in `onnxruntime_ep_c_api.h`
- Implement in framework-side `ep_api_impl.cc`
- Wire in adapter's `OpKernelInfo::GetAttrs<std::string>()` in `op_kernel_info.h`
- Enable `rnn/rnn.cc`, `rnn/gru.cc`, `rnn/lstm.cc` in plugin CMake
- *depends on no prior steps*

**Step 3.2** — Extend `OrtEp` C API for CUDA Graph replay (*unblocks efficient graph replay*) (*deferred*)
- Add three callbacks to `OrtEp` struct:
  - `IsGraphCaptureEnabled(OrtEp*, bool*)`
  - `IsGraphCaptured(OrtEp*, int annotation_id, bool*)`
  - `ReplayGraph(OrtEp*, int annotation_id)`
- Version-gate with `ort_version_supported >= next_version`
- Wire framework's `InferenceSession::Run()` replay shortcut to call these when EP is plugin-based
- Implement callbacks in `CudaEp` connecting to existing `CUDAGraphManager`
- *depends on no prior steps; existing capture/replay infrastructure already works*

**Step 3.3** — Expose `TensorSeq` through EP API (*unblocks identity_op, sequence_op*) (*deferred*)
- Add `TensorSeq`-aware methods to `OrtEpApi` or adapter `OpKernelContext`
- Enable `tensor/identity_op.cc`, `tensor/sequence_op.cc`
- *lower priority — these ops have limited model coverage*

**Relevant files:**
- `include/onnxruntime/core/session/onnxruntime_ep_c_api.h` — API struct extensions
- `onnxruntime/core/session/ep_api/ep_api_impl.cc` — implementations
- `include/onnxruntime/ep/adapter/op_kernel_info.h` — adapter wiring
- `onnxruntime/core/providers/cuda/plugin/cuda_ep.h/.cc` — graph replay callback implementations
- `onnxruntime/core/session/inference_session.cc` — graph replay shortcut integration

**Verification:**
1. EP adapter tests pass
2. RNN model inference via plugin EP (Step 3.1)
3. CUDA graph replay test shows skip of kernel dispatch on subsequent runs (Step 3.2)

---

## Phase 4: Low-Priority Kernel Migrations (High Effort)

These ops have deep framework dependencies and require significant refactoring. Tackle individually based on model demand.

**Step 4.1** — Einsum (`math/einsum.cc`, `math/einsum_utils/*`)
- Decouple compute helpers from concrete `CUDAExecutionProvider` type
- Route stream/allocator access through `CudaKernel` adapter helpers
- Extensive testing required due to reduction code complexity

**Step 4.2** — Object detection (`object_detection/non_max_suppression.cc`, `roi_align.cc`)
- Port `NonMaxSuppressionBase` and `RoiAlignBase` CPU logic (~100+ LOC each) into plugin-safe wrappers
- Low model coverage; defer unless model demand justifies

**Step 4.3** — LLM/Attention deep pipeline (`llm/*`, remaining `bert/attention*.cc`)
- Requires systematic `onnxruntime::Stream*` → `cudaStream_t` conversion across all helper call chains
- Large surface area (~33 files in attention family alone)
- *Depends on Phase 2 attention work establishing the pattern*

**Step 4.4** — Transformers (`transformers/*` — beam search, greedy search, sampling) (*deferred*)
- Deep subgraph execution dependencies on framework `IExecutionProvider` and `SessionState`
- Likely requires new `OrtEpApi` extensions for subgraph dispatch
- *Lowest priority; models using these can fall back to bundled EP*

**Relevant files:**
- Per operation, see [cuda_ops_for_plugin_ep.md](cuda_ops_for_plugin_ep.md) for exact file lists and blocking patterns

**Verification:**
1. Each newly-enabled op has dedicated test in `test_cuda_plugin_ep.py`
2. Parity report shows increased registration count

---

## Phase 5: Infrastructure & Quality

**Step 5.1** — CI Pipeline (*parallel with all phases*)
- Add CI job with `onnxruntime_BUILD_CUDA_EP_AS_PLUGIN=ON` to CUDA CI matrix
- Run `test_cuda_plugin_ep.py` and `cuda_plugin_parity_report.py` as CI gates
- Run standard CUDA EP tests to verify no regression

**Step 5.2** — NVTX Profiling (Task 6.1.1) (*parallel with Phase 1-2*)
- Confirm framework `KernelScope` NVTX markers already work for plugin EP (expected: yes)
- Add `CUDA::nvtx3` linkage to `onnxruntime_providers_cuda_plugin.cmake` when `ENABLE_NVTX_PROFILE` is set
- If finer granularity needed: add plugin-side ranges in `cuda_kernel_adapter.h` (not shared `op_kernel.h`)
- Document profiling setup in design doc

**Step 5.3** — Performance Regression Testing (*after Phase 2*)
- Establish benchmark suite: compare plugin EP vs bundled EP latency on representative models
- Target: <5% overhead (expected: plugin overhead is in stream/handle lookup)
- Track: matmul, conv, attention, softmax, end-to-end model inference

**Step 5.4** — Python Packaging & Distribution (*after Phase 1*)
- Ensure `libonnxruntime_providers_cuda_plugin.so` is included in pip wheel when `BUILD_CUDA_EP_AS_PLUGIN=ON`
- Add `cuda_plugin_ep_helper.py` auto-discovery to the test infrastructure
- Update `setup.py` / `pyproject.toml` data files if needed

**Step 5.5** — GPU Profiling / CUPTI (Separate RFC) (*deferred*)
- Draft RFC for `OrtEp::GetProfiler` C API extension
- Port `CudaProfiler` + `CUPTIManager` to plugin build
- Out of scope until ORT core API change is approved

**Relevant files:**
- CI configuration files (Azure DevOps / GitHub Actions YAML)
- `cmake/onnxruntime_providers_cuda_plugin.cmake` — NVTX linkage
- `setup.py`, `pyproject.toml` — wheel packaging
- `test_cuda_plugin_ep.py` — add perf benchmark variants

**Verification:**
1. CI green on plugin build + test
2. Nsight Systems shows NVTX ranges for plugin EP kernels
3. Performance benchmarks tracked per commit

---

## Phase 6: Documentation Updates

**Step 6.1** — Update design document `docs/cuda_plugin_ep/cuda_plugin_ep_design.md`:
- Update Section 13 (Future Work) to reflect that architecture refactoring is complete
- Add Phase status tracking (which phases are done, in-progress, planned)
- Document CudaEpProvider pattern (adapter::Ep inheritance, provider-scoped state)
- Update CUDA Graph section with current recommendation and timeline for API extension

**Step 6.2** — Update ops document `docs/cuda_plugin_ep/cuda_ops_for_plugin_ep.md`:
- Refresh priority tables with current exclusion count and newly-enabled ops
- Add per-op status column (enabled / in-progress / deferred / blocked)

**Step 6.3** — Archive old docs
- Move `cuda_plugin_ep_old_docs/prototype/` docs that are superseded by current design to an archive folder or add deprecation headers

---

## Summary: Priority Order

| Phase | Focus | Effort | Dependency |
|-------|-------|--------|------------|
| 1 | High-priority kernel migrations (~10 ops) | Low | None | ✅ **COMPLETED** |
| 2 | Medium-priority kernel migrations (attention, MoE, ~20 ops) | Medium | None | ✅ **COMPLETED** |
| 5.1 | CI Pipeline | Low | Phase 1 | Not started |
| 5.2 | NVTX Profiling | Low | None | Not started |
| 5.4 | Python packaging | Low | Phase 1 | Not started |
| 3 | ORT Core API extensions (RNN, CUDA Graph, TensorSeq) | Medium | ORT core approval | Not started |
| 4 | Low-priority kernels (Einsum, LLM) | High | Phase 2-3 | Not started |
| 5.3 | Performance testing | Medium | Phase 2 | Not started |
| 6 | Documentation | Low | Ongoing | In progress |
| 7 | Low-priority kernels (Transformers) | High | Phase 2-3 | Not started |
**Phases 1 and 2 are complete.** Phase 5.1, 5.2, 5.4 can start now. Phase 4 depends on Phase 3 API extensions.

---

## Decisions

- Architecture refactoring (WebGPU pattern) is complete — plan focuses on remaining work
- Kernel coverage is the top priority over infrastructure
- CUDA Graph replay deferred until ORT core API extension (Step 3.2)
- CUPTI/GPU profiling deferred to separate RFC
- Training ops, ATen ops, and NCCL collective ops are permanently out of scope
- Transformers (beam search, sampling) are lowest priority given subgraph execution complexity. We can defer them to the lowest priority.
