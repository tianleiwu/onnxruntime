# QMoE FP4 (MXFP4) Support — Design, Implementation & Verification

## 1. Overview

### 1.1 Goal
Extend the existing QMoE CUDA operator to support **MXFP4 quantized weights** with FP16/BF16 activations (**W4A16**). This enables inference for models like **GPT-OSS** that use MXFP4 quantization for MoE layers.

### 1.2 Scope

| In Scope | Out of Scope |
|----------|-------------|
| W4A16: MXFP4 weights + FP16/BF16 activations | W4A8: FP8 activations (future work) |
| SM90+ (Hopper and Blackwell via mixed-input path) | W4A4: FP4 activations |
| Extend existing QMoE op schema | New standalone FP4MoE op |
| MXFP4 (group_size=32) block scaling | NVFP4 (group_size=16) in W4A16 mode |

### 1.3 Key Architecture Insight

**W4A16 uses the groupwise/mixed-input TMA warp-specialized CUTLASS kernel path**, NOT the block-scaled tensor op path used by W4A4 and W4A8. This is because:
- Block-scaled tensor ops (`OpClassBlockScaledTensorOp`) require both operands to use block scaling (e.g., FP4×FP4 or FP8×FP4)
- W4A16 has full-precision activations (FP16/BF16) paired with narrow FP4 weights — this is a **mixed-input** configuration
- TRT-LLM controls this via a `use_wfp4a16` flag that routes to `CollectiveBuilderMixedInput` with MXFP4 group size and `float_ue8m0_t` scale elements

### 1.4 Background

- **GPT-OSS model**: hidden_size=2880, top_k=4, num_experts=128, quantization=W4A8_MXFP4_MXFP8
- **MXFP4 format**: `__nv_fp4_e2m1` (2-bit exponent, 1-bit mantissa), 2 values packed per byte
- **Block scaling**: MXFP4 group_size=32, scale factors as `float_ue8m0_t` (CUDA: `uint8_t`), per-expert `float` global scale
- **Build gating**: `ENABLE_FP4` defined when CUDA ≥ 12.8

---

## 2. Implementation Status

### Phase Summary

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | FP4 MoE GEMM template instantiations | ✅ Done |
| 2 | CUTLASS mixed-input kernel generalization | ✅ Done |
| 3 | QMoE ONNX operator schema extension | ✅ Done |
| 4 | QMoE operator implementation (constructor, ComputeInternal, PrePack) | ✅ Done |
| 5 | FP4 weight packing utility | ✅ Done |
| 6 | Python tests (`test_qmoe_fp4_cuda.py`) | ✅ Done |
| 7 | End-to-end verification & GPT-OSS smoke test | 🔄 CUDA provider build verified; runtime parity and GPT-OSS smoke pending |

### 2.1 Native SM90 Launcher Status

**Status:** The native SM90 mixed-input PtrArray FP4 GEMM path has been
restored and aligned with TensorRT-LLM's WFP4A16 implementation. The CUDA
provider target builds successfully with CUDA 13.0 after enabling the generated
launcher set. The fallback stub remains in tree as an emergency compile-time
safety net, but the active build now uses generated launchers instead of the
stub.

**What changed from the earlier crash investigation:**

- The mixed-input collective now keeps MXFP4's logical scale group at
  `detail::mxfp4_group_size = 32` instead of collapsing the scale dimension to
  `TileK`.
- For `TileK = 256`, each tile carries eight raw MXFP4 scale lanes through
  `PackedScalesNum = TileK / 32` and
  `ElementScalePacked = Array<float_ue8m0_t, 8>`.
- The INT4 `UseScaleLookupTable` path is INT4-only again. FP4 BF16 conversion
  uses a separate `UseFP4ToBF16LookupTable` path that converts FP4 values to
  BF16 in registers before applying the existing scale flow.
- QMoE FP4 block scales are prepacked with a TRT-LLM-style
  `block_scale_interleave` layout: rows padded to 128, scale columns padded to
  4, then swizzled into the one-byte UE8M0 scale buffer used by the TMA
  mainloop.
- W4A16 scale stride setup now describes the raw scale-group shape
  `[N, K / 32]`; pointer offsets account for the padded interleaved prepack
  layout when constant initializers are available.

**Current verification state:**

- ✅ `onnxruntime_providers_cuda` build passes on the CUDA 13.0 SM90 build.
- 🔄 Python parity tests are implemented and no longer intentionally skip the
  native path, but still need to be run on an SM90/SM100 machine.
- 🔄 GPT-OSS-sized smoke coverage is still pending.

### 2.2 Verification Plan

The implementation is in place; the remaining work is runtime validation and
guard coverage.

| # | Test | Method | Status |
|---|------|--------|--------|
| 1 | CUDA provider build | `cmake --build ... --target onnxruntime_providers_cuda --parallel 2` on CUDA 13.0 / SM90 target | ✅ Passed |
| 2 | FP4 FP16 correctness | `test_qmoe_fp4_cuda.py::TestQMoEFP4::test_fp4_fp16_*` against dequant-then-matmul reference | 🔄 Pending |
| 3 | FP4 BF16 correctness | `test_qmoe_fp4_cuda.py::TestQMoEFP4::test_fp4_bf16_*` against dequant-then-matmul reference | 🔄 Pending |
| 4 | INT4/INT8 regression | Existing `test_qmoe_cuda.py` coverage | 🔄 Pending |
| 5 | Non-FP4 build guard | CUDA build without `ENABLE_FP4` rejects `quant_type="fp4"` clearly | 🔄 Pending |
| 6 | GPT-OSS smoke test | hidden=2880, top_k=4, experts=128, block_size=32 | 🔄 Pending |

Recommended execution order:

1. Run the FP16 and BF16 parity tests on H100/H200 first; these cover the
   restored native W4A16 path directly.
2. Run `compute-sanitizer --tool memcheck` on `test_fp4_bf16_silu_basic` if any
   runtime fault reappears.
3. Run existing INT4/INT8 QMoE tests to confirm the groupwise INT path was not
   regressed by the shared mixed-input collective changes.
4. Validate one non-FP4 CUDA build to confirm the constructor guard and
   `ENABLE_FP4` aliases are sufficient.
5. Add a GPT-OSS-shaped smoke once the small parity cases pass.



### Review Follow-up Notes

The initial review found two blocking integration issues, which are now addressed in the implementation:

- W4A16 routes `QuantParams::FP4()` block scales through `TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams` (`ptr_s_a`/`stride_s_a`) with MXFP4 `float_ue8m0_t` scale pointers and `group_size=32`.
- Non-FP4 builds reject `quant_type="fp4"` at QMoE construction time, and FP4-only type references are guarded by `ENABLE_FP4` or equivalent safe aliases.
- The Python parity tests and GPT-OSS smoke test still need to be run on a CUDA 12.8+ SM90/SM100 build before marking Phase 7 complete.

### Phase 1: FP4 MoE GEMM Template Instantiations — ✅ Done

**Files created:**

| File | Description |
|------|-------------|
| `contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels_fp16_fp4.cu` | Template: `MoeGemmRunner<half, __nv_fp4_e2m1, half>` |
| `contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels_bf16_fp4.cu` | Template: `MoeGemmRunner<__nv_bfloat16, __nv_fp4_e2m1, __nv_bfloat16>` |

**Files modified:**

| File | Changes |
|------|---------|
| `moe_gemm/moe_gemm_kernels.h` | Added `use_wfp4a16` constexpr flag; `weight_fp4` constexpr |
| `moe_gemm/moe_kernels.h` | Updated `mayHaveFinalizeFused()` for `use_wfp4a16` |
| `moe_gemm/moe_tma_warp_specialized_traits.h` | Added FP4 weight type to `isValidHopperMOESpecialisation` |

### Phase 2: CUTLASS Mixed-Input Kernel Generalization — ✅ Done

| File | Changes |
|------|---------|
| `cutlass_extensions/detail/collective/mixed_input_utils.hpp` | Added `int4_group_size=128` and `mxfp4_group_size=32` constants |
| `cutlass_extensions/gemm/collective/sm90_mma_array_..._mixed_input_.hpp` | Added `IsMXFP4`, `ScalingGroupSize`; replaced `#define GROUP_SIZE 128` macro with type-dependent group size |
| `moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl` | Generalized `ElementA`/`ElementB` from template params; conditional scale type (`float_ue8m0_t` for FP4), group size, epilogue alpha |
| `moe_gemm/moe_gemm_template_dispatch_tma_ws_mixed_dtype.h` | Added FP4 tile configs (Ntile=64, Ktile=256); updated workspace calculation |
| `moe_gemm/moe_gemm_template_dispatch.h` | Added `use_wfp4a16` dispatch branch; updated `calcMaxWorkspaceSize` |

### Phase 3: QMoE ONNX Operator Schema Extension — ✅ Done

| File | Changes |
|------|---------|
| `core/graph/contrib_ops/contrib_defs.cc` | Added `quant_type` attr, FP4 inputs (15-20), T3/T4 type constraints |

### Phase 4: QMoE Operator Implementation — ✅ Done

| File | Changes |
|------|---------|
| `contrib_ops/cuda/moe/moe_quantization.h` | Added `quant_type_`, FP4 buffer members |
| `contrib_ops/cuda/moe/moe_quantization.cc` | FP4 runner instantiation, QuantParams::FP4 dispatch, PrePack for FP4 scales, GEMM profiler `kFP4` weight type |

---

## 3. CUTLASS Kernel Path for W4A16

### 3.1 Mixed-Input Dispatch Flow

```
CutlassMoeFCRunner<half, __nv_fp4_e2m1, half>::dispatchToArch()
  └─ use_wfp4a16 == true
    └─ sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<..., PackedScalesNum=8>()
      └─ Ntile=64, Ktile=256 (vs INT4: Ntile=128, Ktile=128*PSN/sizeof(T))
           └─ sm90_generic_mixed_moe_gemm_kernelLauncher()
              ├─ ElementA = cutlass::half_t  (activation)
              ├─ ElementB = cutlass::float_e2m1_t  (weight)
                ├─ raw MXFP4 group size = 32
                ├─ PackedScalesNum = 8 for Ktile=256
                ├─ ElementScalePacked = cutlass::Array<cutlass::float_ue8m0_t, 8>
              └─ CollectiveBuilderMixedInput<..., tuple<ElementB, ElementScalePacked>, ...>
```

### 3.2 Key Differences: W4A16 vs W4A8-INT4

| Property | W4A16 (FP4) | W4A8 (INT4) |
|----------|-------------|-------------|
| ElementA | `half_t` / `bfloat16_t` | `float_e4m3_t` |
| ElementB | `float_e2m1_t` | `int4b_t` |
| Logical scale group size | 32 (MXFP4) | 128 (INT4) |
| Packed scale lanes per Ktile | 8 for `Ktile=256` | 1 for the common INT4 path |
| ElementScalePacked | `Array<float_ue8m0_t, 8>` | `__nv_bfloat16` (SFA) |
| Epilogue alpha | `1` (no per-group scaling) | `0` (uses `alpha_ptr_array`) |
| Ntile | 64 | 128 |
| Ktile | 256 | 128 × PackedScalesNum / sizeof(T) |

### 3.3 Mainloop Modifications

The TRT-LLM-aligned CUTLASS collective mainloop keeps MXFP4's logical group size at 32:

```cpp
static constexpr bool IsMXFP4 = cute::is_same_v<ElementA, cutlass::float_e2m1_t>;
static constexpr int ScalingGroupSize = IsMXFP4 ? detail::mxfp4_group_size : detail::int4_group_size;
```

This affects `scale_k = K / ScalingGroupSize`, `NumMMAsPerChunk`, and `NumChunksPerTileK` calculations throughout the mainloop. ORT now follows the TRT-LLM behavior above for W4A16 MXFP4.

---

## 4. QMoE Operator Schema Extension

### 4.1 New Attribute

| Attribute | Type | Default | Values | Description |
|-----------|------|---------|--------|-------------|
| `quant_type` | string | `"int"` | `"int"`, `"fp4"` | Quantization mode |

### 4.2 New Inputs (appended after index 14)

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 15 | `fp4_fc1_block_scales` | T3 (uint8) | `[E, N_fc1, K_fc1/32]` | MXFP4 block scale factors for FC1, encoded as `float_ue8m0_t` bytes |
| 16 | `fp4_fc1_global_scale` | T4 (float32) | `[E]` | Per-expert global scale for FC1 |
| 17 | `fp4_fc2_block_scales` | T3 (uint8) | `[E, N_fc2, K_fc2/32]` | MXFP4 block scale factors for FC2, encoded as `float_ue8m0_t` bytes |
| 18 | `fp4_fc2_global_scale` | T4 (float32) | `[E]` | Per-expert global scale for FC2 |
| 19 | `fp4_fc3_block_scales` | T3 (uint8) | Optional (SwiGLU) | MXFP4 block scale factors for FC3, encoded as `float_ue8m0_t` bytes |
| 20 | `fp4_fc3_global_scale` | T4 (float32) | Optional (SwiGLU) | Global scale for FC3 |

### 4.3 Type Constraints

| Constraint | Types | Description |
|-----------|-------|-------------|
| T | float, float16, bfloat16 | Input/output |
| T1 | uint8 | Packed weights (FP4: 2 values/byte, INT4: 2 values/byte) |
| T2 | float, float16, bfloat16 | Existing integer-quantization scales |
| T3 | uint8 | FP4 block scales encoded as `float_ue8m0_t` bytes |
| T4 | float | FP4 per-expert global scales |

---

## 5. QMoE Operator Implementation Details

### 5.1 Constructor

When `quant_type="fp4"`:
```cpp
m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, __nv_fp4_e2m1, half>>(...);
// or for BF16:
m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_fp4_e2m1, __nv_bfloat16>>(...);
```

### 5.2 ComputeInternal

FP4 uses `QuantParams::FP4()` with `act_global_scale=nullptr` (no activation quantization for W4A16):

```cpp
quant_params = QuantParams::FP4(
    nullptr,  // fc1_act_global_scale (W4A16: no activation quant)
    fc1_block_scales, fc1_global_scale,
    nullptr,  // fc2_act_global_scale
    fc2_block_scales, fc2_global_scale);
```

### 5.3 GEMM Profiler

Weight type set to `kFP4` for tactic profiling, which triggers the mixed-input dispatch path.

### 5.4 PrePack

FP4 global scales are copied to GPU memory. FP4 block scales are repacked from
the ONNX input layout `[E, N, K/32]` into the TRT-LLM-style interleaved layout
used by the SM90 mixed-input TMA kernel: N rows padded to 128, scale columns
padded to 4, then swizzled as one-byte UE8M0 scale elements. No zero-point or
bias computation is needed because MXFP4 is symmetric quantization.

---

## 6. Key Data Structures

### 6.1 QuantParams::FP4Inputs (`moe_kernels.h`)

```cpp
struct FP4Inputs {
  struct GemmInputs {
    bool use_per_expert_act_scale = false;
    float const* act_global_scale = nullptr;      // nullptr for W4A16
    NVFP4ElementSF const* weight_block_scale;     // (E, N, K/32) uint8 ue8m0 bytes
    float const* global_scale;                    // (E,) float
  };
  GemmInputs fc1, fc2;
};
```

### 6.2 CutlassMoeFCRunner Flags (`moe_gemm_kernels.h`)

```cpp
static constexpr bool use_wfp4a16 = weight_fp4 && (std::is_same_v<T, half> || std::is_same_v<T, __nv_bfloat16>);
// Routes to mixed-input dispatch (not block-scaled)
// mayHaveFinalizeFused() returns false for use_wfp4a16
```

### 6.3 FpXBlockScalingType (`moe_gemm_kernels.h`)

```cpp
enum class FpXBlockScalingType {
  MXFPX,    // block_size = 32 (MXFP4)
  NVFP4,    // block_size = 16 (NVFP4)
  NONE
};
```

Note: the W4A16 mixed-input path uses MXFP4 weight scales with group size 32. The `NVFP4ElementSF` alias in `QuantParams::FP4Inputs` is currently a `uint8_t` storage alias; it should not imply the W4A16 path uses NVFP4 group size 16.

### 6.4 Block Scale Alignment Constants (`moe_gemm_kernels.h`)

```cpp
constexpr static int NVFP4BlockScaleVectorSize = 16;   // 16 FP4 elements per block
constexpr static int MXFPXBlockScaleVectorSize = 32;   // 32 FP4 elements per block
constexpr static int MinNDimAlignmentNVFP4 = 128;
constexpr static int MinNDimAlignmentMXFPX = 128;
constexpr static int MinKDimAlignmentNVFP4 = 64;
constexpr static int MinKDimAlignmentMXFPX = 128;
```

---

## 7. Data Flow Diagram

```
                    QMoE Operator (quant_type="fp4")
                    ================================

Input Tensors:
  [0] input              (num_tokens, hidden_size) T=fp16/bf16
  [1] router_probs        (num_tokens, num_experts) T
  [2] fc1_weights         (E, hidden, inter/2) T1=uint8   <- FP4 packed, 2 per byte
  [5] fc2_weights         (E, inter, hidden/2) T1=uint8
  [15] fc1_block_scales   (E, inter, hidden/32) T3=uint8 <- ue8m0 encoded
  [16] fc1_global_scale   (E,) T4=float32
  [17] fc2_block_scales   (E, hidden, inter/32) T3=uint8 <- ue8m0 encoded
  [18] fc2_global_scale   (E,) T4=float32

           +----------------+
  input -->|  SoftmaxTopK   |--> expert_indices, expert_scales
           +----------------+
                  |
                  v
           +------------------------------------------------------+
           |  Construct QuantParams::FP4(                         |
           |    act_global_scale=nullptr,  // W4A16: no quant     |
           |    fc1_block_scales, fc1_global_scale,               |
           |    fc2_block_scales, fc2_global_scale                |
           |  )                                                   |
           +------------------------------------------------------+
                  |
                  v
           +------------------------------------------------------+
           |  CutlassMoeFCRunner<half, __nv_fp4_e2m1, half>      |
           |  ::runMoe(input, weights, quant_params, ...)         |
           |                                                      |
           |  GEMM1: input x fc1_weights (with block dequant)    |
           |    -> val = fc1_weight_fp4 * block_scale * global    |
           |  Activation: SiLU / SwiGLU / ReLU                    |
           |  GEMM2: inter x fc2_weights (with block dequant)    |
           |  Weighted sum by expert_scales                       |
           +------------------------------------------------------+
                  |
                  v
           output (num_tokens, hidden_size) T=fp16/bf16
```

---

## 8. Remaining Work

### 8.1 Phase 5: FP4 Weight Packing Utility — ✅ Done

SM90+ TMA-based FP4 kernels expect a simpler column-major packed layout (no Ampere-style interleaving):

1. **Input**: `[N, K/2]` FP4 weights (2 per byte along K, row-major per expert)
2. **Transpose**: nibble-level transpose `[N, K]` → `[K, N]`
3. **Output**: `[K, N/2]` bytes (2 per byte along N, column-major packed)

**Files modified:**

| File | Changes |
|------|---------|
| `python/onnxruntime_pybind_quant.cc` | Added `PackFP4WeightsForMoE()` C++ function and `pack_fp4_weights_for_cuda_moe_gemm` Python binding |
| `core/graph/contrib_ops/contrib_defs.cc` | Changed FP4 global scale inputs (16, 18, 20) from T2 to T4 (float32); added T4 type constraint |
| `contrib_ops/cuda/moe/moe_quantization.cc` | Added T4 type constraint in kernel registration |

### 8.2 Phase 6: Python Tests — ✅ Done

Created `onnxruntime/test/python/transformers/test_qmoe_fp4_cuda.py`.

**Test matrix (implemented):**

| Config | Hidden | Inter | Experts | TopK | Block Size | Activation | Dtype |
|--------|--------|-------|---------|------|-----------|------------|-------|
| Basic FP16 | 256 | 256 | 4 | 2 | 32 (MXFP4) | silu | fp16 |
| Basic BF16 | 256 | 256 | 4 | 2 | 32 (MXFP4) | silu | bf16 |
| SwiGLU FP16 | 256 | 256 | 4 | 2 | 32 | swiglu | fp16 |
| SwiGLU BF16 | 256 | 256 | 4 | 2 | 32 | swiglu | bf16 |
| Token counts | 256 | 256 | 4 | 2 | 32 | silu | fp16 |
| More experts | 256 | 256 | 8 | 2 | 32 | silu | fp16 |
| Top-4 | 256 | 256 | 8 | 4 | 32 | silu | fp16 |
| Larger dims | 512 | 512 | 4 | 2 | 32 | silu | fp16 |

**Test components:**
- `quantize_weight_to_mxfp4()`: MXFP4 quantization with ue8m0 block scales
- `create_fp4_moe_onnx_graph()`: ONNX model builder with QMoE `quant_type="fp4"`
- `TestQMoEFP4`: End-to-end parity tests (ORT vs dequant-then-matmul reference)
- `TestFP4PackingUtility`: Unit tests for packing, quantization, ue8m0 encoding

### 8.3 Phase 7: Verification & Smoke Test — 🔄 In Progress

The CUDA provider build has passed with the native generated FP4 launchers
enabled. Runtime verification remains open.

| # | Test | Method | Status |
|---|------|--------|--------|
| 1 | Build compiles | CUDA 12.8+ with SM90/SM100 targets | ✅ CUDA 13.0 SM90-target build passed |
| 2 | FP4 FP16 correctness | Dequant-then-matmul reference, atol ~1e-1 | 🔄 Pending |
| 3 | FP4 BF16 correctness | Same with BF16 dtype | 🔄 Pending |
| 4 | INT4/INT8 regression | Existing `test_qmoe_cuda.py` unchanged | 🔄 Pending |
| 5 | GPT-OSS smoke test | hidden=2880, top_k=4, experts=128, block_size=32 | 🔄 Pending |
| 6 | Architecture/build guard | FP4 path raises a clear error on SM < 90 or when `ENABLE_FP4` is not defined | 🔄 Pending |

---

## 9. Existing Infrastructure Reference

The following FP4-aware infrastructure was already present in ORT (ported from TRT-LLM) and leveraged by this work:

| Component | Location | Status |
|-----------|----------|--------|
| `QuantParams::FP4()` factory | `moe_gemm/moe_kernels.h` | ✅ Pre-existing |
| `FP4Inputs` struct | `moe_gemm/moe_kernels.h` | ✅ Pre-existing |
| `FpXBlockScalingType::NVFP4` / `MXFPX` | `moe_gemm/moe_gemm_kernels.h` | ✅ Pre-existing |
| NVFP4/MXFPX alignment constants | `moe_gemm/moe_gemm_kernels.h` | ✅ Pre-existing |
| FP4 TMA warp-specialized dispatch | `moe_gemm/moe_gemm_template_dispatch_tma_ws.h` | ✅ Pre-existing |
| FP4 block-scaled traits | `moe_gemm/moe_tma_warp_specialized_traits.h` | ✅ Pre-existing |
| FP4 activation kernel support | `moe_gemm/moe_gemm_activation_kernels.cuh` | ✅ Pre-existing |
| Standalone FP4 GEMM runner | `llm/fp4_gemm/fp4_gemm.h` | ✅ Pre-existing |
| `ENABLE_FP4` cmake gate | `cmake/CMakeLists.txt` (CUDA ≥ 12.8) | ✅ Pre-existing |

---

## 10. Build Configuration

- **CUDA 12.8+**: default architectures = `60;70;75;80;86;89;90;100;120`
- **CUDA 13.x**: architectures = `75;80;86;89;90;100;120`
- SM90+ gets accelerated `-a` suffix (enables WGMMA, TMA, setmaxnreg)
- `ENABLE_FP4` is defined when `CMAKE_CUDA_COMPILER_VERSION >= 12.8`
- Architecture exclusion defines: `EXCLUDE_SM_100`, `EXCLUDE_SM_120`

## 11. Design Decisions

- **Extend QMoE op** (not a new op) — minimizes code duplication, leverages existing routing/permutation logic
- **New optional inputs** for FP4 scales (not reusing existing scale inputs) — avoids type constraint conflicts (FP4 block scales are uint8 `float_ue8m0_t` bytes and global scales are T4 float32 vs existing T2 float/fp16/bf16 scales)
- **W4A16 only** for initial implementation — W4A8 (FP8 activations) can be added later by enabling `moe_gemm_kernels_fp8_fp4.cu`
- **SM90+ via mixed-input path** — not block-scaled tensor ops (which require both operands block-scaled)

---

## 12. Extending to W4A8 (FP4 Weights + FP8 Activations)

### 12.1 Overview

W4A8 pairs MXFP4 weights with FP8 (e4m3) activations. Unlike W4A16, which uses the **mixed-input** CUTLASS path, W4A8 uses the **block-scaled tensor op** path (`OpClassBlockScaledTensorOp`), because both operands now use block scaling and the hardware can apply native FP4×FP8 fused multiply-accumulate.

### 12.2 Operator Schema Changes

The schema change is minimal — only **activation scale inputs** are needed. All weight-side inputs (15–20) are reused as-is.

#### New optional inputs

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 21 | `fp4_fc1_act_scale` | T2 (float) | `[E]` or `[1]` | Activation global scale for FC1 (quantize FP16/BF16→FP8 before GEMM1) |
| 22 | `fp4_fc2_act_scale` | T2 (float) | `[E]` or `[1]` | Activation global scale for FC2 (quantize intermediate→FP8 before GEMM2) |

#### Mode determination

**Implicit (presence-based, proposed)**: When `quant_type="fp4"` and `fp4_fc1_act_scale` is provided, the operator would use W4A8. When absent, it would use W4A16. This is not implemented in the staged W4A16 change; inputs 21-22 and the constructor/dispatch switch still need to be added.

No new `quant_type` value is needed, keeping the schema simple.

### 12.3 CUTLASS Kernel Path Differences

| Property | W4A16 (current) | W4A8 (extension) |
|----------|-----------------|------------------|
| CUTLASS path | **Mixed-input** (`CollectiveBuilderMixedInput`) | **Block-scaled tensor op** (`OpClassBlockScaledTensorOp`) |
| Template | `MoeGemmRunner<half, __nv_fp4_e2m1, half>` | `MoeGemmRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>` |
| ElementA | `half_t` / `bfloat16_t` (full precision) | `float_e4m3_t` (FP8, quantized on-the-fly) |
| ElementB | `float_e2m1_t` | `float_e2m1_t` (same) |
| Dispatch header | `moe_gemm_template_dispatch_tma_ws_mixed_dtype.h` | `moe_gemm_template_dispatch_tma_ws.h` (block-scaled path) |
| HW requirement | SM90+ (mixed-input TMA) | SM100+ (block-scaled tensor ops require Blackwell) |
| Activation quant | None (FP16/BF16 passed directly) | Runtime FP16→FP8 quantization using `act_global_scale` |

### 12.4 Implementation Checklist

#### Prerequisites (from current cleanup)

The standalone FP8 block-scaled GEMM runner (`fp8_blockscale_gemm/`) and `moe_gemm_kernels_fp8_fp4.cu` were removed
in the MoE cleanup pass because they depended on infrastructure incompatible with CUTLASS 4.4.2 and were not wired
to any ONNX op. The block-scaled dispatch plumbing (`moe_gemm_template_dispatch_tma_ws.h` `#ifdef ENABLE_FP4` sections)
and traits (`moe_tma_warp_specialized_traits.h`) remain in tree.

#### Changes needed

1. **Schema** (`contrib_defs.cc`): Add inputs 21–22 for activation scales.

2. **Kernel instantiation**: Re-add `moe_gemm_kernels_fp8_fp4.cu` with `MoeGemmRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>` using CUTLASS 4.4.2 block-scaled tensor op path.

3. **Constructor** (`moe_quantization.cc`): When `quant_type="fp4"` and activation scales are provided, instantiate `CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>` instead of the W4A16 runner.

4. **ComputeInternal** (`moe_quantization.cc`): Pass `act_global_scale` pointers to `QuantParams::FP4()` instead of `nullptr`:
    ```cpp
    quant_params = QuantParams::FP4(
        fc1_act_scale,     // non-null → triggers FP8 activation quantization
        fc1_block_scales, fc1_global_scale,
        fc2_act_scale,
        fc2_block_scales, fc2_global_scale);
    ```

5. **PrePack** (`moe_quantization.cc`): Add GPU copy for activation scale inputs (indices 21–22), same pattern as existing global scale handling.

6. **`use_per_expert_act_scale`**: If `act_scale` shape is `[E]` (per-expert), set `FP4Inputs::GemmInputs::use_per_expert_act_scale = true`. If shape is `[1]` (global), set to `false`.

### 12.5 Runtime Activation Quantization

For W4A8, the MoE runner must quantize FP16/BF16 activations to FP8 before each GEMM. This happens inside `CutlassMoeFCRunner::runMoe()`:
- Before GEMM1: quantize permuted input tokens → FP8 using `fc1_act_global_scale`
- Before GEMM2: quantize intermediate (post-activation) → FP8 using `fc2_act_global_scale`

The quantization kernel and workspace for FP8 intermediates are already handled by the existing `FP4Inputs` plumbing in the runner.

### 12.6 Key Insight: Why Two Different CUTLASS Paths

```
W4A16: FP16 activation (full precision) × FP4 weight (narrow)
  → Mixed-input: only weights are block-scaled, activations are not
  → Uses CollectiveBuilderMixedInput with group_size=32, ElementScale=float_ue8m0_t

W4A8: FP8 activation (block-scaled) × FP4 weight (block-scaled)
  → Both operands use block scaling → hardware block-scaled tensor ops
  → Uses OpClassBlockScaledTensorOp (native FP4×FP8 in tensor cores)
  → Higher throughput than mixed-input, but requires activation quantization overhead
```

The block-scaled tensor op path is fundamentally more efficient because the hardware performs the dequantization fused with the matrix multiply, rather than the software dequant-in-registers approach used by the mixed-input path.
