# QMoE FP8 Support — Design & Implementation Plan

## 1. Motivation

The existing QMoE FP4 (MXFP4) implementation crashes on H200 (SM90) because the
SM90 CUTLASS mixed-input FP4 kernel requires native FP4 tensor ops, which only
Blackwell (SM100+) hardware provides. On H200 the build falls back to a
software dequantize-then-recompute path that is still broken due to CUTLASS
version incompatibility.

FP8 is the practical alternative: every GPU from Ada Lovelace (SM89, RTX 4090)
onward has hardware FP8 GEMM support, and Hopper (SM90, H100/H200) has Tensor
Core FP8 fast-accumulate mode. This document specifies how to add three FP8-related
modes to the QMoE operator, in priority order, and why W8A16-fp8 is implemented first.

A fourth future mode — **WFP4AFP8** (FP4 weight + FP8 activation) — targets
Blackwell (SM100+, RTX 5090) where both FP4 and FP8 tensor ops are native and
can be combined in a single block-scaled GEMM. The schema and dispatch design
are specified here so the operator interface does not need to change later.

---

## 2. Supported Modes

| Mode | Notation | Activation | Weight | SM support |
|------|----------|-----------|--------|------------|
| **Phase 1** | W8A16-fp8 | BF16 / FP16 | FP8 e4m3 | SM80+ (dequant), SM89 (Ada), SM90 (Hopper), SM100 (Blackwell) |
| **Phase 2** | WFP4AFP8 | FP8 e4m3 (MXFP8) | FP4 e2m1 (MXFP4) | SM100+ (Blackwell) only — requires block-scaled tensor ops |
| **Future**  | W4AFP8 | FP8 e4m3 | INT4 (uint4b_t) | SM89 (Ada) fast path; SM80+ via dequant fallback |

### Why W8A16-fp8 First, WFP4AFP8 Second

1. **Immediate H100/H200 unblocking.** W8A16-fp8 uses the existing SM80 Ampere
   GEMM path on SM90 (weight is dequantized to BF16 before the GEMM). This is
   functional on H200 today with minimal new code.

2. **Almost no new dispatch logic.** The SM80 specialization
   `genericMoeGemmKernelLauncher<__nv_bfloat16, __nv_fp8_e4m3, ...>` already
   exists in `moe_gemm_template_dispatch.h:302`. Only kernel instantiation
   `.cu` files and the op-schema extension are missing.

3. **WFP4AFP8 before W4AFP8.** WFP4AFP8 targets Blackwell (SM100+, RTX 5090)
   where FP4+FP8 block-scaled ops are native. It reuses the existing MXFP4
   weight infrastructure (inputs 15–20) and only adds activation scales, making
   the incremental work modest. W4AFP8 by contrast has a narrower hardware
   target (SM89 fast path only) and requires a new runtime activation
   quantization kernel with no reuse from existing paths.

4. **W4AFP8 is deferred.** Its fast path is gated to `sm == 89` only
   (`moe_gemm_template_dispatch.h` line 596). On SM90 it falls back to the
   Ampere dequant path, which offers no advantage over W4A16-int. A proper SM90
   W4AFP8 TMA WS kernel would be needed to make it useful on H100/H200.

---

## 3. Existing FP8 Infrastructure in ORT

Most of the plumbing already exists under `ENABLE_FP8` (defined automatically
for CUDA ≥ 11.8 by `cmake/CMakeLists.txt:1467`).

### 3.1 `moe_gemm_kernels.h`
```cpp
// MoeGemmRunner type-level flags (lines 258–284)
static constexpr bool use_fp8     = (T == fp8_e4m3 || T == fp8_e5m2) && WeightType != uint4b_t && WeightType != fp4_e2m1;
static constexpr bool use_w4afp8  = T == fp8_e4m3 && WeightType == uint4b_t;
// Note: confusingly named "use_wfp4afp4" in the source but actually means W=FP4, A=FP8 (WFP4AFP8):
static constexpr bool use_wfp4afp4 = T == fp8_e4m3 && WeightType == fp4_e2m1;  // = WFP4AFP8
```
`OutputTypeAdaptor_t<T>` maps `fp8_e4m3` → `nv_bfloat16` so the GEMM output
is always widened to BF16.

### 3.2 `moe_kernels.h` — `QuantParams` FP8 variants

Three `QuantParams` factory functions cover the FP8-related modes:

```cpp
struct QuantParams {
  // W8A16-fp8 / W4AFP8 — global per-expert float scales
  struct {
    bool  fc2_use_per_expert_act_scale = false;
    float const* dequant_fc1   = nullptr;  // (num_experts,)
    float const* quant_fc2     = nullptr;  // (1,) or (num_experts,) — W4AFP8 activation quant
    float const* dequant_fc2   = nullptr;  // (num_experts,)
    float const* quant_final   = nullptr;  // (1,)   — unused in Phases 1/2
    float const* dequant_input = nullptr;  // (1,)   — unused in Phases 1/2
  } fp8;

  // WFP4AFP8 variant A — global-scaled FP8 activation + MXFP4 block-scaled weight
  struct FP8MXFP4Inputs {
    struct GemmInputs {
      bool  use_per_expert_act_scale = false;
      float const* act_global_scale  = nullptr;          // (1,) or (num_experts,)
      MXFPXElementSF const* weight_block_scale = nullptr; // (num_experts, N, K/32)
      float const* global_scale      = nullptr;          // (num_experts,) weight global
    };
    GemmInputs fc1, fc2;
  } fp8_mxfp4;

  // WFP4AFP8 variant B — MXFP8 block-scaled activation + MXFP4 block-scaled weight
  struct MXFP8MXFP4Inputs {
    struct GemmInputs {
      MXFPXElementSF const* weight_block_scale = nullptr; // (num_experts, N, K/32)
      float const* global_scale                = nullptr; // (num_experts,)
    };
    GemmInputs fc1, fc2;
  } mxfp8_mxfp4;

  static QuantParams FP8(float const* dequant_fc1, float const* quant_fc2,
                         float const* dequant_fc2, ...);        // Phase 1, 2
  static QuantParams FP8MXFP4(...);   // Phase 3 variant A
  static QuantParams MXFP8MXFP4(...); // Phase 3 variant B
};
```

### 3.3 `moe_gemm_template_dispatch.h`
- **Line 302:** Full SM80 specialization for `<__nv_bfloat16, __nv_fp8_e4m3>`.
  Handles W8A16-fp8 by dequantizing FP8 weights during the GEMM via
  `alpha_scales`.
- **Line 346 / 367:** SM89 W4AFP8 Ampere path (FP8 act + INT4 weight).
- **Line 596:** `use_w4afp8 && sm != 89` → skips Ampere configs for W4AFP8
  on SM90+.

### 3.4 `moe_gemm_tma_ws_launcher.inl` and SM validity checks

The SM90 TMA WS launcher (`moe_gemm_tma_ws_launcher.inl` line 237–243) asserts
`T == WeightType || IsWFP4AFP8`. `IsWFP4AFP8 = (WeightType==FP4 && T==FP8)`.

**What this means per mode:**

| Mode | T | WeightType | SM90 TMA WS? | SM100 TMA WS? | Ampere fallback? |
|------|---|-----------|:---:|:---:|:---:|
| W8A16-fp8 | bf16 | fp8 | ✗ (mixed, not WFP4AFP8) | ✗ | ✓ |
| W4AFP8 | fp8 | uint4b_t | ✗ (`isValidHopperMOESpecialisation` excludes it) | ✗ | ✓ (SM89 only) |
| WFP4AFP8 | fp8 | fp4 | ✗ (`isValidHopperMOESpecialisation` excludes fp8+fp4) | ✓ (`isValidBlackwellMOESpecialisation` includes it) | ✗ (fp4 excluded from Ampere) |

`isValidBlackwellMOESpecialisation<fp8_e4m3, fp4_e2m1>()` = **true** (SM100
block-scaled tensor op).
`isValidHopperMOESpecialisation<fp8_e4m3, fp4_e2m1>()` = **false** (the Hopper
path explicitly excludes FP4 weight when T==FP8 because the MXFPX scaling
required by that combination is a Blackwell-only primitive).

### 3.5 `moe_kernels.cu`
- `dequantFP8Kernel` / `dequantFP8()` (line 1771): post-GEMM scale-and-cast.
- `computeFP8DequantScaleKernel` / `computeFP8DequantScale()` (line 789):
  builds the `alpha_scale_ptr_array` needed by the CUTLASS FP8 epilogue.

### 3.6 Missing (the work to do)
| File | What's needed | Phase |
|------|--------------|-------|
| `moe_gemm_kernels_bf16_fp8.cu` | `MoeGemmRunner<bf16, fp8_e4m3, bf16>` instantiation | 1 |
| `moe_gemm_kernels_fp16_fp8.cu` | `MoeGemmRunner<half, fp8_e4m3, half>` instantiation | 1 |
| `moe_gemm_kernels_fp8_uint4.cu` | `MoeGemmRunner<fp8_e4m3, uint4b_t, half/bf16>` | Future |
| `moe_gemm_kernels_fp8_fp8.cu` | `MoeGemmRunner<fp8_e4m3, fp8_e4m3, bf16>` (W8A8-fp8, future) | — |
| `moe_gemm_kernels_fp8_fp4.cu` | `MoeGemmRunner<fp8_e4m3, fp4_e2m1, half/bf16>` (WFP4AFP8) | 2 |
| `contrib_defs.cc` | Schema inputs 21–28; new `quant_type` values | 1–3 |
| `moe_quantization.cc` / `.h` | Runner selection, ComputeInternal wiring, PrePack | 1–3 |
| `test_qmoe_fp8_cuda.py` | Python correctness tests | 1–3 |

---

## 4. QMoE Operator Spec Update

### 4.1 Attribute: `quant_type`

Current allowed values: `"int"` (default), `"fp4"`.

**Add:**

```
quant_type (string, default "int"):
  "int"      – integer weight quantization (INT4 or INT8 based on expert_weight_bits)
  "fp4"      – MXFP4 weight quantization (requires CUDA >= 12.8, SM100+ for native)
  "fp8"      – FP8 e4m3 weight-only quantization, BF16/FP16 activation  [Phase 1]
               (requires CUDA >= 11.8, SM80+ functional, SM89/SM90 recommended)
  "w4afp8"   – INT4 weight + FP8 e4m3 activation                        [Future]
               (requires CUDA >= 11.8, SM89 fast path)
  "wfp4afp8" – MXFP4 weight + FP8 e4m3 activation                      [Phase 2]
               (requires CUDA >= 12.8, SM100+ only)
```

### 4.2 Attribute: `expert_weight_bits`

For `quant_type='fp8'`, `expert_weight_bits` must be `8`. The validator in
`moe_quantization.cc` already accepts `8`; add a check that rejects `4` when
`quant_type='fp8'`.

For `quant_type='w4afp8'`, `expert_weight_bits` must be `4` (INT4 weights).

For `quant_type='wfp4afp8'`, `expert_weight_bits` must be `4` (MXFP4 weights,
same storage format as `"fp4"`). The FP4 weight block scales (inputs 15–20)
are reused directly; only the activation scales are new.

### 4.3 New Inputs (W8A16-fp8 — Phase 1)

Append after the existing 21 inputs (indices 0–20). All three are **optional**
at the schema level but **required** at runtime when `quant_type='fp8'`.

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 21 | `fp8_fc1_dequant_scale` | T4 (float32) | `(num_experts,)` | Per-expert weight dequant scale for FC1. `weight_fp32 ≈ weight_fp8 × scale`. |
| 22 | `fp8_fc2_dequant_scale` | T4 (float32) | `(num_experts,)` | Per-expert weight dequant scale for FC2. |
| 23 | `fp8_fc3_dequant_scale` | T4 (float32) | `(num_experts,)` | Per-expert weight dequant scale for FC3 (optional; needed only when FC3 is present). |

These map directly to `QuantParams::FP8::dequant_fc1` and
`QuantParams::FP8::dequant_fc2` in `moe_kernels.h`.

### 4.4 Additional Input (W4AFP8 — Future Work)

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 24 | `fp8_activation_quant_scale` | T4 (float32) | `(1,)` or `(num_experts,)` | Scale for on-the-fly quantization of BF16 activations to FP8 before GEMM. Maps to `QuantParams::FP8::quant_fc2`. |

> **Note:** For W4AFP8 the `quant_type` attribute value will be `"w4afp8"`.
> `expert_weight_bits` remains `4` (INT4 weights). Inputs 3/6/9 (INT4 scales)
> are still required. Implementation is deferred — see Section 7.

### 4.5 New Inputs (WFP4AFP8 — Phase 2)

WFP4AFP8 reuses the existing MXFP4 weight inputs (indices 15–20, already
defined for `quant_type='fp4'`). Only the FP8 activation scales are new.

Two sub-variants are supported, controlled at runtime based on which inputs
are provided:

**Variant A — `FP8MXFP4`**: global-scaled FP8 activation (simpler, lower
overhead). Activation is quantized with a single per-expert or per-tensor
float scale.

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 25 | `wfp4afp8_fc1_act_global_scale` | T4 (float32) | `(1,)` or `(num_experts,)` | FP8 activation quantization scale for FC1. Maps to `FP8MXFP4Inputs::act_global_scale`. |
| 26 | `wfp4afp8_fc2_act_global_scale` | T4 (float32) | `(1,)` or `(num_experts,)` | FP8 activation quantization scale for FC2. |

**Variant B — `MXFP8MXFP4`**: MXFP8 block-scaled activation (higher accuracy,
required when activation dynamic range varies significantly across the K
dimension). Activation block scales have the same group size (32) as MXFP4.
The block scales are pre-computed offline or by a quantization prologue kernel.

| Index | Name | Type | Shape | Description |
|-------|------|------|-------|-------------|
| 27 | `wfp4afp8_fc1_act_block_scale` | T3 (uint8) | `(num_experts, M_padded, K/32)` | MXFP8 activation block scales for FC1, stored as `float_ue8m0_t`. Shape uses padded M aligned to `MinMDimAlignmentMXFPX=32`. |
| 28 | `wfp4afp8_fc2_act_block_scale` | T3 (uint8) | `(num_experts, M_padded, inter_size/32)` | MXFP8 activation block scales for FC2. |

When inputs 27–28 are present, `QuantParams::MXFP8MXFP4` is used; otherwise
`QuantParams::FP8MXFP4` is used with inputs 25–26. Both variants also require
the MXFP4 weight inputs 15–20.

> **Note on M dimension:** Unlike weight block scales (static shape), activation
> block scales for MXFP8 depend on the runtime token count M. For offline
> export scenarios (e.g., compiling a fixed-batch model), M is known and shapes
> can be pre-allocated. For dynamic shapes, block scales must be computed by a
> quantization prologue kernel at each inference step and are not model inputs.

### 4.5 Weight Tensor Format for `quant_type='fp8'`

- **`fc1_experts_weights` (index 2), `fc2_experts_weights` (index 5),
  `fc3_experts_weights` (index 8):** stored as `uint8_t`, reinterpreted as
  `__nv_fp8_e4m3` (1 byte per value). No packing (`pack_size = 1`).
- **Shapes** (no change from INT8 format):
  - FC1: `(num_experts, fusion_size × inter_size, hidden_size)`
  - FC2: `(num_experts, hidden_size, inter_size)`
  - FC3: `(num_experts, inter_size, hidden_size)` (if FC3 present)
- **Zero points:** not applicable (FP8 e4m3 is a symmetric format). Inputs
  11–13 (`fc*_zero_points`) must be absent when `quant_type='fp8'`.
- **Block scales (inputs 3, 6, 9, `T2`):** not used for W8A16-fp8. Can be
  omitted. The per-expert global dequant scales are in inputs 21–23 instead.

### 4.6 Full Schema Diff Summary

```
Attribute quant_type: add "fp8", "w4afp8", "wfp4afp8" as valid values.

# Phase 1 — W8A16-fp8
Input 21 (optional):  fp8_fc1_dequant_scale        T4 float32 (num_experts,)
Input 22 (optional):  fp8_fc2_dequant_scale        T4 float32 (num_experts,)
Input 23 (optional):  fp8_fc3_dequant_scale        T4 float32 (num_experts,)

# Phase 2 — W4AFP8
Input 24 (optional):  fp8_activation_quant_scale   T4 float32 (1,)|(num_experts,)

# Phase 3 — WFP4AFP8 (reuses FP4 weight inputs 15–20)
Input 25 (optional):  wfp4afp8_fc1_act_global_scale  T4 float32 (1,)|(num_experts,)  [variant A]
Input 26 (optional):  wfp4afp8_fc2_act_global_scale  T4 float32 (1,)|(num_experts,)  [variant A]
Input 27 (optional):  wfp4afp8_fc1_act_block_scale   T3 uint8   (num_experts, M_pad, K/32)  [variant B]
Input 28 (optional):  wfp4afp8_fc2_act_block_scale   T3 uint8   (num_experts, M_pad, inter/32)  [variant B]

TypeConstraints T3 (uint8) and T4 (float32) unchanged.
```

Files to edit:
- `onnxruntime/core/graph/contrib_ops/contrib_defs.cc` — add `Input(21..24)`,
  extend `quant_type` docs
- `onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc` — validate new
  attribute values, wire scales into `QuantParams::FP8`

---

## 5. Phase 1 Implementation: W8A16-fp8

### 5.1 Kernel Instantiation Files

**`onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels_bf16_fp8.cu`**
```cpp
#include "moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>;
#endif
}
```

**`onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels_fp16_fp8.cu`**
```cpp
#include "moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
template class MoeGemmRunner<half, __nv_fp8_e4m3, half>;
}
```

Both files are auto-picked up by the `GLOB_RECURSE` in
`cmake/onnxruntime_providers_cpu.cmake:23`. No CMake changes are required.

If a minimal/fast build wants to exclude them, add a filter analogous to the
existing FP4 exclusion:
```cmake
if(NOT ENABLE_CUDA_FP8_QMOE)
  list(FILTER onnxruntime_cuda_contrib_ops_cu_srcs EXCLUDE REGEX
    "moe_gemm_kernels_(fp16|bf16)_fp8\\.cu")
endif()
```

### 5.2 `moe_quantization.h` — New Member Variables

```cpp
// FP8 per-expert dequant scales (pre-packed to GPU)
IAllocatorUniquePtr<void> packed_fp8_fc1_dequant_scale_;
IAllocatorUniquePtr<void> packed_fp8_fc2_dequant_scale_;
IAllocatorUniquePtr<void> packed_fp8_fc3_dequant_scale_;
```

### 5.3 `moe_quantization.cc` — Constructor

```cpp
// Add "fp8" to quant_type validation:
ORT_ENFORCE(quant_type_ == "int" || quant_type_ == "fp4" || quant_type_ == "fp8",
            "quant_type must be 'int', 'fp4', or 'fp8'");

#if defined(ENABLE_FP8)
if (quant_type_ == "fp8") {
  ORT_ENFORCE(expert_weight_bits_ == 8, "FP8 quantization requires expert_weight_bits=8");
  if (is_fp16) {
    m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, __nv_fp8_e4m3, half>>(
        sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
  } else {
    m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, __nv_fp8_e4m3, __nv_bfloat16>>(
        sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
  }
} else
#endif
```

### 5.4 `moe_quantization.cc` — `ComputeInternal`

Read the new scale inputs:
```cpp
const Tensor* fp8_fc1_dequant = packed_fp8_fc1_dequant_scale_
    ? nullptr : context->Input<Tensor>(21);
const Tensor* fp8_fc2_dequant = packed_fp8_fc2_dequant_scale_
    ? nullptr : context->Input<Tensor>(22);
```

Wire into `QuantParams`:
```cpp
if (quant_type_ == "fp8") {
  const float* p1 = packed_fp8_fc1_dequant_scale_
      ? static_cast<const float*>(packed_fp8_fc1_dequant_scale_.get())
      : (fp8_fc1_dequant ? fp8_fc1_dequant->Data<float>() : nullptr);
  const float* p2 = packed_fp8_fc2_dequant_scale_
      ? static_cast<const float*>(packed_fp8_fc2_dequant_scale_.get())
      : (fp8_fc2_dequant ? fp8_fc2_dequant->Data<float>() : nullptr);
  quant_params = QuantParams::FP8(
      /*dequant_fc1=*/ p1,
      /*quant_fc2=*/   nullptr,   // no activation quantization in W8A16-fp8
      /*dequant_fc2=*/ p2);
}
```

Also: `pack_size = 1` (no weight packing for 8-bit), so the `int64_t pack_size`
local must be set to `1` when `quant_type_ == "fp8"`.

### 5.5 `moe_quantization.cc` — `PrePack`

Add cases for inputs 21, 22, 23 using the same `CopyToGpu` lambda already used
for FP4 global scales (indices 16, 18, 20):

```cpp
} else if (input_idx >= 21 && input_idx <= 23 && quant_type_ == "fp8") {
  switch (input_idx) {
    case 21: CopyToGpu(packed_fp8_fc1_dequant_scale_); break;
    case 22: CopyToGpu(packed_fp8_fc2_dequant_scale_); break;
    case 23: CopyToGpu(packed_fp8_fc3_dequant_scale_); break;
  }
}
```

---

## 6. Phase 2 Implementation: WFP4AFP8

WFP4AFP8 (FP4 weight + FP8 activation) is the highest-density mode, combining
MXFP4 weights (2 bits effective per parameter with block scaling) with FP8
activations on Blackwell's block-scaled tensor ops. This is the same compute
primitive used by the existing FP4×FP4 path but with FP8 activations instead
of FP4 activations.

### 6.1 Hardware Requirement

SM100+ (Blackwell) only. The `isValidBlackwellMOESpecialisation<fp8_e4m3, fp4_e2m1>()`
check in `moe_tma_warp_specialized_traits.h` returns `true`. Both SM90 (Hopper)
and SM80 (Ampere) reject this combination — Hopper explicitly excludes it, and
Ampere excludes all FP4 weights.

### 6.2 Kernel Instantiation File

**`onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels_fp8_fp4.cu`**

This file does not exist in ORT yet but the identical pattern is used by
TRT-LLM:

```cpp
#include "moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
#if defined(ENABLE_FP4) && defined(ENABLE_FP8)
template class MoeGemmRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>;
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16>;
#endif
#endif
}
```

Add a CMake exclusion filter for builds without both flags:
```cmake
if(NOT ENABLE_CUDA_FP4_QMOE)
  list(FILTER onnxruntime_cuda_contrib_ops_cu_srcs EXCLUDE REGEX
    "moe_gemm_kernels_fp8_fp4\\.cu")
endif()
```

### 6.3 `moe_quantization.h` — New Member Variables

```cpp
// WFP4AFP8 activation scales (pre-packed to GPU)
IAllocatorUniquePtr<void> packed_wfp4afp8_fc1_act_global_scale_;
IAllocatorUniquePtr<void> packed_wfp4afp8_fc2_act_global_scale_;
```

MXFP8 activation block scales (inputs 27–28) are dynamic-shape tensors and
are therefore **not** pre-packed — they are read directly from `context->Input`
at runtime each step.

### 6.4 `moe_quantization.cc` — Constructor

```cpp
#if defined(ENABLE_FP4) && defined(ENABLE_FP8)
if (quant_type_ == "wfp4afp8") {
  ORT_ENFORCE(expert_weight_bits_ == 4, "WFP4AFP8 requires expert_weight_bits=4");
  ORT_ENFORCE(sm_ >= 100, "WFP4AFP8 requires SM100+ (Blackwell)");
  if (is_fp16) {
    m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, half>>(
        sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
  } else {
    m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_fp8_e4m3, __nv_fp4_e2m1, __nv_bfloat16>>(
        sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
  }
} else
#endif
```

### 6.5 `moe_quantization.cc` — `ComputeInternal`

WFP4AFP8 reuses the existing MXFP4 weight block scale pointers (inputs 15–20)
plus the new activation scale inputs (25–28). The key decision is which
`QuantParams` factory to use:

```cpp
if (quant_type_ == "wfp4afp8") {
  const Tensor* act_block_scale_fc1 = context->Input<Tensor>(27);
  const Tensor* act_block_scale_fc2 = context->Input<Tensor>(28);

  if (act_block_scale_fc1 != nullptr) {
    // Variant B: MXFP8 block-scaled activations
    quant_params = QuantParams::MXFP8MXFP4(
        /*fc1_weight_block_scale=*/ p_fp4_fc1_block_scales,
        /*fc1_global_scale=*/       p_fp4_fc1_global_scale,
        /*fc2_weight_block_scale=*/ p_fp4_fc2_block_scales,
        /*fc2_global_scale=*/       p_fp4_fc2_global_scale);
    // Set activation block scales on hopper_input separately
    // (via TmaWarpSpecializedGroupedGemmInput::fpX_block_scaling_factors_A)
  } else {
    // Variant A: global-scaled FP8 activations
    const float* p_act1 = packed_wfp4afp8_fc1_act_global_scale_
        ? static_cast<const float*>(packed_wfp4afp8_fc1_act_global_scale_.get())
        : context->Input<Tensor>(25)->Data<float>();
    const float* p_act2 = packed_wfp4afp8_fc2_act_global_scale_
        ? static_cast<const float*>(packed_wfp4afp8_fc2_act_global_scale_.get())
        : context->Input<Tensor>(26)->Data<float>();
    quant_params = QuantParams::FP8MXFP4(
        /*fc1_act_global_scale=*/   p_act1,
        /*fc1_weight_block_scale=*/ p_fp4_fc1_block_scales,
        /*fc1_global_scale=*/       p_fp4_fc1_global_scale,
        /*fc2_act_global_scale=*/   p_act2,
        /*fc2_weight_block_scale=*/ p_fp4_fc2_block_scales,
        /*fc2_global_scale=*/       p_fp4_fc2_global_scale);
  }
}
```

WFP4AFP8 also requires runtime FP8 activation quantization. The BF16 input
tensor must be quantized to FP8 before `runMoe()`, using either a per-tensor
scale (Variant A) or a block-quantization prologue (Variant B). The `input_sf`
parameter of `runMoe()` carries the resulting scale-factor pointer array.

### 6.6 Relationship to Existing FP4 Path

WFP4AFP8 reuses almost all of the existing FP4 weight infrastructure:
- The MXFP4 weight block scale format (group_size=32, `float_ue8m0_t` elements)
  is identical to `quant_type='fp4'`.
- Inputs 15–20 (`fp4_fc*_block_scales`, `fp4_fc*_global_scale`) are reused
  without any change.
- The `PrePack` logic for inputs 15–20 (copy to GPU) applies unchanged.

The only additions are the activation scale inputs (25–28) and switching the
runner type from `<bf16/half, fp4_e2m1>` to `<fp8_e4m3, fp4_e2m1>`.

---

## 7. Future Work: W4AFP8

W4AFP8 (INT4 weight + FP8 activation) has a fast hardware path only on SM89
(Ada Lovelace, RTX 4090). On SM90+ it falls back to a slow dequantize path and
provides no advantage over W8A16-fp8. A dedicated SM90 TMA WS kernel path must
be added before this mode becomes useful on H100/H200.

### 7.1 Kernel Instantiation File

**`onnxruntime/contrib_ops/cuda/llm/moe_gemm/moe_gemm_kernels_fp8_uint4.cu`**
```cpp
#include "moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
#ifdef ENABLE_FP8
template class MoeGemmRunner<__nv_fp8_e4m3, cutlass::uint4b_t, half>;
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_fp8_e4m3, cutlass::uint4b_t, __nv_bfloat16>;
#endif
#endif
}
```

### 7.2 Runtime Activation Quantization

W4AFP8 requires quantizing BF16 activations to FP8 before each GEMM.
This needs a new kernel `LaunchFP8QuantizeActivations` applied to the
expanded (post-permutation) activation buffer before calling `runMoe()`.
The scale (`fp8_activation_quant_scale`, input 24) and resulting FP8 buffer
are passed as `input_sf` and the first argument to `runMoe()`.

The `runMoe` interface already has an `input_sf` parameter (second argument,
currently `nullptr` for all non-FP8 modes).

### 7.3 SM Guard

The W4AFP8 SM89-specific path is controlled in `moe_gemm_template_dispatch.h`
line 596:
```cpp
if (!isValidAmpereMOESpecialisation<T, WeightType>() || (use_w4afp8 && sm != 89)) {
  // skip Ampere configs
}
```
To support SM90+ W4AFP8 properly in the future, a dedicated SM90 TMA WS kernel
path must be added (analogous to what TRT-LLM has for W4AFP8 SM90 support).

---

## 8. Weight Quantization Reference

### W8A16-fp8 Weight Format

```
Storage dtype:   uint8  (same T1 constraint as INT8)
Interpreted as:  __nv_fp8_e4m3 (1 byte per value, E4M3 format, max value 448.0)
Shape:
  fc1_experts_weights: (num_experts, fusion_size * inter_size, hidden_size)
  fc2_experts_weights: (num_experts, hidden_size, inter_size)
Scale dtype:     float32
Scale shape:     (num_experts,)  — one scalar per expert, per GEMM
```

Reference Python quantization:
```python
import numpy as np

def quantize_fp8_weights(weights_bf16: np.ndarray) -> tuple:
    """
    weights_bf16: (num_experts, N, K) in float32/bf16
    Returns: (fp8_weights_uint8, dequant_scale_f32)
    """
    FP8_MAX = 448.0
    num_experts = weights_bf16.shape[0]
    dequant_scale = np.zeros(num_experts, dtype=np.float32)
    fp8_weights = np.zeros_like(weights_bf16, dtype=np.uint8)

    for e in range(num_experts):
        amax = np.max(np.abs(weights_bf16[e]))
        scale = amax / FP8_MAX  # dequant_scale
        dequant_scale[e] = scale
        quant_scale = FP8_MAX / amax  # reciprocal
        # quantize: clip to [-448, 448] then cast to fp8
        w_scaled = np.clip(weights_bf16[e] * quant_scale, -FP8_MAX, FP8_MAX)
        # store bit pattern as uint8
        fp8_weights[e] = w_scaled.astype(np.float8_e4m3fn).view(np.uint8)

    return fp8_weights, dequant_scale
```

### W4AFP8 Weight Format

Same as existing INT4 weight format (see `qmoe_int4_format.md`):
4-bit values packed two-per-byte as `uint8_t`.

### WFP4AFP8 Weight Format

Same as existing MXFP4 weight format (see `qmoe_fp4.md`):
two FP4 values packed per byte, with MXFP4 block scales stored as `uint8_t`
(`float_ue8m0_t` bit patterns). Inputs 15–20 are reused unchanged.

Activation format for WFP4AFP8:
```
Variant A (global-scaled):
  Activation dtype:   __nv_fp8_e4m3 (computed at runtime from BF16 input)
  Scale dtype:        float32
  Scale shape:        (1,) per-tensor  OR  (num_experts,) per-expert

Variant B (MXFP8 block-scaled):
  Activation dtype:   __nv_fp8_e4m3 with MXFP8 block scales
  Block scale dtype:  uint8 (float_ue8m0_t, same format as MXFP4 weight scales)
  Block scale shape:  (num_experts, M_padded, K/32) — M_padded is runtime-dependent
  Block group size:   32 (same as MXFP4 weight)
```

---

## 9. Testing Plan

New test file: `onnxruntime/test/python/transformers/test_qmoe_fp8_cuda.py`

Structure mirrors `test_qmoe_fp4_cuda.py`. Key tests:

| Test | `quant_type` | hidden | inter | experts | top_k | Notes |
|------|-------------|--------|-------|---------|-------|-------|
| `test_fp8_bf16_basic` | `"fp8"` | 256 | 1024 | 4 | 2 | BF16 act, FP8 weight |
| `test_fp8_fp16_basic` | `"fp8"` | 256 | 1024 | 4 | 2 | FP16 act, FP8 weight |
| `test_fp8_bf16_swiglu` | `"fp8"` | 512 | 2048 | 8 | 4 | SwiGLU activation |
| `test_fp8_bf16_top4` | `"fp8"` | 256 | 1024 | 8 | 4 | top-4 selection |
| `test_w4afp8_bf16_basic` | `"w4afp8"` | 256 | 1024 | 4 | 2 | INT4 weight, FP8 act |
| `test_wfp4afp8_bf16_basic` | `"wfp4afp8"` | 256 | 1024 | 4 | 2 | MXFP4 weight, FP8 act |

Skip conditions:
- `sm < 80` — FP8 types require at least SM80
- W4AFP8 fast path: `sm != 89`
- WFP4AFP8: `sm < 100` (Blackwell required)

Reference computation for correctness check:
```python
def fp8_moe_reference(input_bf16, experts_weights_fp8, dequant_scales, ...):
    # dequantize weights per expert
    weights_bf16 = [weights_fp8[e].view(fp8_e4m3).float() * dequant_scales[e]
                    for e in range(num_experts)]
    # run standard BF16 MoE GEMM
    return bf16_moe_reference(input_bf16, weights_bf16, ...)
```

---

## 10. Summary

| Item | W8A16-fp8 (Phase 1) | WFP4AFP8 (Phase 2) | W4AFP8 (Future) |
|------|---------------------|-------------------|-----------------|
| New `.cu` files | `moe_gemm_kernels_{bf16,fp16}_fp8.cu` | `moe_gemm_kernels_fp8_fp4.cu` | `moe_gemm_kernels_fp8_uint4.cu` |
| Schema `quant_type` | `"fp8"` | `"wfp4afp8"` | `"w4afp8"` |
| New schema inputs | 21–23 (weight dequant) | 25–28 (act scales); reuses 15–20 for weights | 24 (act quant scale) |
| New runtime kernels | none | `LaunchFP8QuantizeActivations` (+ optional MXFP8 block quant prologue) | `LaunchFP8QuantizeActivations` |
| SM support | SM80+, SM89/SM90 recommended | SM100+ only | SM89 fast; SM80+ dequant |
| CUTLASS path | SM80 Ampere GEMM (weight dequant) | SM100 block-scaled TMA WS | SM89 Ampere (mixed FP8+INT4) |
| Build flags | `ENABLE_FP8` (CUDA ≥ 11.8) | `ENABLE_FP4 && ENABLE_FP8` | `ENABLE_FP8` |
| `QuantParams` used | `FP8` | `FP8MXFP4` or `MXFP8MXFP4` | `FP8` |
