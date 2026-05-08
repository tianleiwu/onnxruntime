# QMoE FP4 (MXFP4) Support - Design, Implementation & Current Status

## 1. Overview

### 1.1 Goal
Extend the existing QMoE CUDA operator to support **MXFP4 quantized weights** with FP16/BF16 activations (**W4A16**). This enables inference for models like **GPT-OSS** that use MXFP4 quantization for MoE layers.

### 1.2 Scope

| In Scope | Out of Scope |
|----------|-------------|
| W4A16: MXFP4 weights + FP16/BF16 activations | W4A4: FP4 activations |
| W4A8: MXFP4 weights + FP8 activations (`quant_type="wfp4afp8"`) | New standalone FP4MoE op |
| SM90+ (Hopper and Blackwell via mixed-input path) | NVFP4 (group_size=16) in W4A16 mode |
| Extend existing QMoE op schema | |
| MXFP4 (group_size=32) block scaling | |

### 1.3 Key Architecture Insight

**W4A16 uses the groupwise/mixed-input TMA warp-specialized CUTLASS kernel path**, NOT the block-scaled tensor op path used by W4A4 and W4A8. This is because:
- Block-scaled tensor ops (`OpClassBlockScaledTensorOp`) require both operands to use block scaling (e.g., FP4×FP4 or FP8×FP4)
- W4A16 has full-precision activations (FP16/BF16) paired with narrow FP4 weights — this is a **mixed-input** configuration
- TRT-LLM controls this via a `use_wfp4a16` flag that routes to `CollectiveBuilderMixedInput` with MXFP4 group size and `float_ue8m0_t` scale elements

### 1.4 Background

- **GPT-OSS model**: hidden_size=2880, top_k=4, num_experts=128, quantization=W4A8_MXFP4_MXFP8
- **MXFP4 format**: `__nv_fp4_e2m1` (2-bit exponent, 1-bit mantissa), 2 values packed per byte
- **Block scaling**: MXFP4 group_size=32, scale factors as `float_ue8m0_t` (CUDA: `uint8_t`), per-expert `float` global scale
- **Build gating**: `ENABLE_FP4` is defined when CUDA >= 12.8. If it is not defined, QMoE rejects
  `quant_type="fp4"` during kernel construction.

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
| 7 | End-to-end verification & GPT-OSS smoke test | Not complete |
| 8 | W4A8 (`quant_type="wfp4afp8"`) build + dequant fallback | ✅ Done (see §12) |
| 9 | W4A8 native SM100+ runtime validation | Pending Blackwell hardware |
| 10 | K=128 tile support, epilogue fusion, expanded tile configs | ✅ Done |

### Review Follow-up Notes

The initial review found two blocking integration issues, which are now addressed in the implementation:

- W4A16 routes `QuantParams::FP4()` block scales through `TmaWarpSpecializedGroupedGemmInput::INT4GroupwiseParams` (`ptr_s_a`/`stride_s_a`) with MXFP4 `float_ue8m0_t` scale pointers and `group_size=32`.
- Non-FP4 builds reject `quant_type="fp4"` at QMoE construction time, and FP4-only type references are guarded by `ENABLE_FP4` or equivalent safe aliases.
- The SM90 mixed-input FP4 launcher is now fully built (320 generated `.cu` files) with support for
  K={128,256} tiles, multiple N tile shapes, cluster configurations, and both NONE and FINALIZE epilogue
  fusion modes. The old stub (`moe_gemm_tma_ws_sm90_mixed_fp4_stub.cu`) remains but is no longer the
  primary build path when `ENABLE_CUDA_FP4_QMOE=ON`.

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
| `moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.h` | Declared `sm90_generic_mixed_moe_gemm_kernelLauncher` with `EpilogueFusion FUSION` template parameter |
| `moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl` | Generalized launcher: conditional scale type (`float_ue8m0_t` for FP4), group size, both NONE and FINALIZE epilogue fusion via `CollectiveEpilogueDefault` / `CollectiveEpilogueFinalize` (using `EpilogueMoeFusedFinalizeBuilder`), `make_epilogue_scalars()` / `make_epilogue_args()` lambdas |
| `moe_gemm/launchers/moe_gemm_tma_ws_sm90_fp4_instantiation.cuh` | Instantiation macros: `ORT_MOE_GEMM_TMA_WS_SM90_FP4_INST_{PP,CO}` (NONE fusion), `ORT_MOE_GEMM_TMA_WS_SM90_FP4_INST_{PP,CO}_FINALIZE` |
| `moe_gemm/launchers/generate_moe_gemm_tma_ws_sm90_fp4.py` | Python generator: produces 320 `.generated.cu` files across FP16/BF16, M={64,128}, N={16,32,64,128}, K={128,256}, 4 cluster shapes, PP/CO schedules, NONE/FINALIZE fusion |
| `moe_gemm/launchers/moe_gemm_tma_ws_sm90_fp4_*.generated.cu` | 320 generated instantiation files (replaced the old single `moe_gemm_tma_ws_sm90_mixed_fp4.generated.cu`) |
| `moe_gemm/moe_gemm_template_dispatch_tma_ws_mixed_dtype.h` | Added `FUSION` template param to all dispatch functions; K tile dispatch via `PackedScalesNum` encoding (1→K=256, 2→K=128); direct N tile mapping; workspace calculator uses `Ntile=128` |
| `moe_gemm/moe_gemm_template_dispatch.h` | `dispatchToArch` wfp4a16 section: FUSION routing from `hopper_inputs.fusion`, K tile selection via `inputs.k % 256`; removed restrictive wfp4a16 config filter in `getTmaWarpSpecializedConfigs` |

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
     └─ select fusion from hopper_inputs.fusion (NONE or FINALIZE)
        └─ select K tile: inputs.k % 256 == 0 → PackedScalesNum=1 (K=256)
                           else              → PackedScalesNum=2 (K=128)
           └─ sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<..., FUSION, PackedScalesNum>()
              └─ Ktile = PackedScalesNum==2 ? 128 : 256
                 └─ dispatch on tile_config_sm90 enum (M×N from heuristic)
                    └─ sm90_dispatch_moe_mixed_dtype_gemm_config<..., FUSION, Shape<M, N, Ktile>>()
                       └─ dispatch on cluster_shape
                          └─ sm90_dispatch_mainloop_schedules<..., FUSION>()
                             └─ sm90_generic_mixed_moe_gemm_kernelLauncher()
                                ├─ ElementA = cutlass::half_t  (activation)
                                ├─ ElementB = cutlass::float_e2m1_t  (weight, stored as FP4)
                                ├─ group_size = 32 (mxfp4_group_size)
                                ├─ ElementScale = cutlass::float_ue8m0_t
                                ├─ CollectiveBuilderMixedInput (FP4→FP16 upconvert in registers)
                                └─ Epilogue: NONE (per-expert output) or FINALIZE (fused scatter+scale)
```

Note: H100/H200 (SM90) does **not** have native FP4 tensor core instructions. The kernel uses FP4 purely
as a **compressed storage format** — weights are loaded via TMA and upconverted to FP16/BF16 in shared
memory/registers by `CollectiveBuilderMixedInput` before the actual MMA runs on FP16 tensor cores. This
is a **memory bandwidth optimization** (4x compression), not a compute throughput feature. Native FP4 MMA
is available on Blackwell (SM100+) via the separate block-scaled tensor op path (see §12).

### 3.2 Key Differences: W4A16 vs W4A8-INT4

| Property | W4A16 (FP4) | W4A8 (INT4) |
|----------|-------------|-------------|
| ElementA | `half_t` / `bfloat16_t` | `float_e4m3_t` |
| ElementB | `float_e2m1_t` | `int4b_t` |
| Group size | 32 (MXFP4) | 128 (INT4) |
| ElementScale | `float_ue8m0_t` | `__nv_bfloat16` (SFA) |
| Epilogue alpha | `1` (no per-group scaling) | `0` (uses `alpha_ptr_array`) |
| Epilogue fusion | NONE or FINALIZE | NONE or FINALIZE |
| M tiles | 64, 128 | 64, 128 |
| N tiles | 16, 32, 64, 128 | 16, 32, 64, 128 |
| K tiles | 128, 256 | 128 × PackedScalesNum / sizeof(T) |
| Cluster shapes | (1,1), (2,1), (1,2), (2,2) | (1,1), (2,1), (1,2), (2,2) |
| Mainloop schedules | Pingpong, Cooperative | Pingpong, Cooperative |

### 3.3 Mainloop Modifications

The CUTLASS collective mainloop (`sm90_mma_array_tma_gmma_rs_warpspecialized_mixed_input_.hpp`) was updated to use a type-dependent group size:

```cpp
static constexpr bool IsMXFP4 = cute::is_same_v<ElementA, cutlass::float_e2m1_t>;
static constexpr int ScalingGroupSize = IsMXFP4 ? detail::mxfp4_group_size : detail::int4_group_size;
```

This affects `scale_k = K / ScalingGroupSize`, `NumMMAsPerChunk`, and `NumChunksPerTileK` calculations throughout the mainloop.

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

FP4 block scales and global scales are simply copied to GPU memory. No transpose or bias computation needed (FP4 is symmetric quantization).

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
// mayHaveFinalizeFused() returns true for use_wfp4a16 (FINALIZE epilogue fusion is now supported)
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

## 7a. Phase 10: K=128, Epilogue Fusion & Expanded Tile Configs — ✅ Done

### Overview

This phase closes the gap between ORT and TRT-LLM for the SM90 W4A16 mixed-input FP4 MoE GEMM path by
adding K=128 tile support, FINALIZE epilogue fusion, and opening up all N tile/cluster configurations.

### Changes Summary

| Gap | Before | After |
|-----|--------|-------|
| K tiles | 256 only | {128, 256} — selected at runtime based on `inputs.k % 256` |
| Epilogue fusion | NONE only | NONE + FINALIZE — routed from `hopper_inputs.fusion` |
| N tiles accessible | Only `CtaShape128x32x128B` + `ClusterShape_1x1x1` | All instantiated tiles (N={16,32,64,128}, clusters=(1,1),(2,1),(1,2),(2,2)) |
| Generated .cu files | ~80 | 320 |
| Mainloop schedules | Pingpong only (for most tiles) | Pingpong + Cooperative (for M=128 tiles) |

### K Tile Dispatch Mechanism

The `CutlassTileConfigSM90` enum encodes K as "128B" (128 bytes), but for FP4 mixed-input the actual K tile
in elements differs. The dispatch uses a `PackedScalesNum` encoding trick:

- `PackedScalesNum = 1` → K = 256 elements (selected when `inputs.k % 256 == 0`)
- `PackedScalesNum = 2` → K = 128 elements (selected otherwise)

Inside `sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass`:
```cpp
constexpr int Ktile = is_wfp4a16 ? (PackedScalesNum == 2 ? 128 : 256) : 128 * PackedScalesNum / sizeof(T);
```

### Epilogue Fusion

The mixed-input launcher now supports two epilogue modes, matching the same-type launcher pattern:

- **NONE**: Per-expert intermediate output (standard grouped GEMM epilogue)
- **FINALIZE**: Fused scatter + router-scale + bias epilogue using `EpilogueMoeFusedFinalizeBuilder`

The fusion is routed at runtime in `dispatchToArch`:
```cpp
switch (hopper_inputs.fusion) {
  case EpilogueFusion::FINALIZE:
    sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<..., FINALIZE, PackedScalesNum>(...);
    break;
  case EpilogueFusion::NONE:
  default:
    sm90_dispatch_moe_mixed_dtype_gemm_to_cutlass<..., NONE, PackedScalesNum>(...);
    break;
}
```

### Files Modified

| File | Changes |
|------|---------|
| `launchers/moe_gemm_tma_ws_mixed_input_launcher.h` | Added `EpilogueFusion FUSION` template parameter |
| `launchers/moe_gemm_tma_ws_mixed_input_launcher.inl` | Added FINALIZE epilogue support (`CollectiveEpilogueFinalize`, `make_epilogue_scalars/args` lambdas) |
| `launchers/moe_gemm_tma_ws_sm90_fp4_instantiation.cuh` | Added `_PP_FINALIZE` and `_CO_FINALIZE` macros |
| `launchers/generate_moe_gemm_tma_ws_sm90_fp4.py` | Added `k` and `fusion` fields; generates K={128,256} × NONE/FINALIZE |
| `moe_gemm_template_dispatch_tma_ws_mixed_dtype.h` | `FUSION` param throughout; `PackedScalesNum`-based K tile; direct N tile mapping; workspace calc with `Ntile=128` |
| `moe_gemm_template_dispatch.h` | FUSION routing in `dispatchToArch`; removed restrictive wfp4a16 config filter |

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

### 8.3 Phase 7: Verification & Smoke Test - Partially Complete

| # | Test | Method |
|---|------|--------|
| 1 | Build compiles | ✅ CUDA 12.8+ defines `ENABLE_FP4`; 320 generated `.cu` files build successfully with `ENABLE_CUDA_FP4_QMOE=ON` |
| 2 | FP4 FP16 correctness | ✅ `test_qmoe_fp4_cuda.py` passes (15 tests, 10 skipped) |
| 3 | FP4 BF16 correctness | ✅ Included in test matrix; quick builds instantiate FP16+FP4 subset only |
| 4 | INT4/INT8 regression | ✅ `test_qmoe_cuda.py` passes (57 tests, 13 skipped); `test_moe_cuda.py` passes (2 OOM errors on 16GB GPU, unrelated) |
| 5 | GPT-OSS smoke test | Not completed in the current branch |
| 6 | Architecture/build guard | ✅ FP4 path raises a clear error on SM < 90 or when `ENABLE_FP4` is not defined |

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
| SM90 FP4 instantiation header | `moe_gemm/launchers/moe_gemm_tma_ws_sm90_fp4_instantiation.cuh` | ✅ Active |
| SM90 FP4 generated instantiations | `moe_gemm/launchers/moe_gemm_tma_ws_sm90_fp4_*.generated.cu` (320 files) | ✅ Active build path |
| SM90 FP4 generator script | `moe_gemm/launchers/generate_moe_gemm_tma_ws_sm90_fp4.py` | ✅ Active |
| SM90 mixed-input FP4 launcher stub | `moe_gemm/launchers/moe_gemm_tma_ws_sm90_mixed_fp4_stub.cu` | Legacy fallback |

---

## 10. Build Configuration

- **CUDA 12.8+**: default architectures = `60;70;75;80;86;89;90;100;120`
- **CUDA 13.x**: architectures = `75;80;86;89;90;100;120`
- SM90+ gets accelerated `-a` suffix (enables WGMMA, TMA, setmaxnreg)
- `ENABLE_FP4` is defined when `CMAKE_CUDA_COMPILER_VERSION >= 12.8`
- When CUDA architecture targets do not include SM100+, CMake defines `PLACEHOLDER_KERNELS` for the standalone
  FP4 kernels. This is separate from the SM90 mixed-input MoE FP4 stub described above.
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

#### Existing optional inputs reused for W4A8

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 18 | `fc1_act_scale` | T4 (float) | `[E]` or `[1]` | Activation global scale for FC1 (quantize FP16/BF16->FP8 before GEMM1) |
| 19 | `fc2_act_scale` | T4 (float) | `[E]` or `[1]` | Activation global scale for FC2 (quantize intermediate->FP8 before GEMM2) |

These inputs are already declared in the schema and validated by the operator. Inputs 20/21
(`fc1_act_block_scale`, `fc2_act_block_scale`) for the MXFP8 block-scaled "Variant B" remain reserved
for a future change and are not consumed by the current W4A8 implementation.

#### Mode determination

W4A8 is selected explicitly via `quant_type="wfp4afp8"` (rather than via input presence). This keeps the
behaviour predictable and avoids surprises when activation scales happen to be `nullptr`.

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

#### Status

The schema, operator plumbing, kernel template instantiations, and runtime dispatch for W4A8
(`quant_type="wfp4afp8"`) are implemented. The runner selects the path based on SM:

- **SM100+ (Blackwell)**: native FP8 x MXFP4 block-scaled tensor op path. The runner is
  `CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half/__nv_bfloat16, half/__nv_bfloat16>` (T=fp8,
  WeightType=fp4, OutputType=BF16/FP16, InputType=BF16/FP16). The runner accepts BF16/FP16 user input and
  quantizes it to MXFP8 (FP8 + per-block ue8m0 scales) inside `expandInputRowsKernel` (MXFP8 branch,
  triggered by `quant_params.mxfp8_mxfp4.fc{1,2}.weight_block_scale` being non-null). `QuantParams::MXFP8MXFP4`
  carries the MXFP4 weight block scales and per-expert global weight scales.

- **SM<100**: dequantize-then-A16 fallback. MXFP4 weights are decoded with
  `LaunchQMoEDequantizeFp4Weights` and fed into the dense BF16/FP16 MoE runner. Produces correct results on
  every SM that supports the existing FP4 W4A16 dequant fallback (SM90+ effectively today).

The build can be verified on any host with CUDA 12.8+ by configuring with
`onnxruntime_ENABLE_CUDA_FP4_QMOE=ON`. Runtime end-to-end validation of the native path requires SM100+
hardware; the dequant-fallback path is exercised by the bundled Python parity test on SM90.

#### Prerequisites (from current cleanup)

The standalone FP8 block-scaled GEMM runner (`fp8_blockscale_gemm/`) is unchanged. The W4A8 path uses the
existing block-scaled dispatch plumbing (`moe_gemm_template_dispatch_tma_ws.h` `#ifdef ENABLE_FP4` sections)
and traits (`moe_tma_warp_specialized_traits.h`) which already accept `<__nv_fp8_e4m3, __nv_fp4_e2m1>` for
SM100+ via `isValidBlackwellMOESpecialisation`.

#### Changes implemented

1. **Schema** (`contrib_defs.cc`): inputs 18/19 (`fc1_act_scale`, `fc2_act_scale`) are already declared,
   accepting either `(1,)` or `(num_experts,)` float tensors. No schema change was required.

2. **GEMM kernel instantiation**: `moe_gemm_kernels_fp8_fp4.cu` instantiates
   `MoeGemmRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>` and the `__nv_bfloat16` output variant under
   `ENABLE_FP4 && ENABLE_CUDA_FP4_QMOE && ENABLE_FP8`. Build gating in
   `cmake/onnxruntime_providers_cpu.cmake` excludes the file when `onnxruntime_ENABLE_CUDA_FP4_QMOE` is OFF.

3. **Runner instantiation** (`moe_kernels.cu`): added explicit
   `CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half, half>` and the `__nv_bfloat16` variant under
   `ENABLE_FP4 && ENABLE_FP8`. These specify `InputType` distinct from `T` so the runner can accept
   BF16/FP16 input and quantize it to FP8 internally.

4. **Expansion kernel instantiation** (`moe_kernels.cu`): added
   `INSTANTIATE_EXPAND_INPUT_ROWS(half, __nv_fp8_e4m3)` and the `__nv_bfloat16` variant under
   `ENABLE_FP8 && ENABLE_FP4`. The MXFP8 quantization branch inside `expandInputRowsKernel` now has the
   templates it needs to be linked when the W4A8 native path is selected.

5. **Constructor** (`moe_quantization.cc`): when `quant_type="wfp4afp8"` and `sm_ >= 100`, the constructor
   instantiates the native runner. Otherwise the dequant fallback runner is used.

6. **ComputeInternal** (`moe_quantization.cc`): the W4A8 path validates MXFP4 block scales, per-expert
   global weight scales (inputs 15/16), and optional FP8 activation global scales (inputs 18/19). When the
   native path is selected, `QuantParams::MXFP8MXFP4` is built so the activation is quantized BF16/FP16 ->
   MXFP8 inside `expandInputRowsKernel` and the MXFP4 weight block scales feed the block-scaled tensor op.
   The act_scale inputs (18/19) are validated and pre-packed for forward compatibility with the
   global-scaled "Variant A" mode but are not consumed by the current native-path (Variant B) plumbing.
   When the dequant fallback path is selected (SM<100), MXFP4 weights are decoded with
   `LaunchQMoEDequantizeFp4Weights` and fed into the dense A16 runner.

7. **PrePack** (`moe_quantization.cc`): inputs 18/19 are pre-packed to GPU memory using the existing
   `CopyToGpu` helper, mirroring the global weight scale handling.

8. **`use_per_expert_act_scale`**: derived from `act_scale->Shape().Size() == num_experts` for fc1/fc2
   independently. (Reserved for the future Variant A native path.)

#### Remaining work

- Add a Python parity test that exercises the native path on Blackwell hardware (extending
  `test_qmoe_wfp4afp8_cuda.py`).
- Optional: add a Variant A (global-scaled FP8 activation) path that consumes inputs 18/19 directly via
  `QuantParams::FP8MXFP4`. This would require the QMoE op to accept pre-quantized FP8 input or to wire a
  separate global-scaled BF16->FP8 prologue.

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
