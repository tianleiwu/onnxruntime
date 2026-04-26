/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Explicit template instantiations for SM90 mixed-input MoE GEMM launcher
 * with FP4 (MXFP4) weights. These instantiations are required because the
 * launcher function template is defined in a .inl file and called from a
 * dispatch header, but never explicitly instantiated.
 */

#ifndef EXCLUDE_SM_90
#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS
#ifdef ENABLE_FP4

#include "contrib_ops/cuda/llm/common/logger.h"
#ifndef LLM_LOG_ERROR
#define LLM_LOG_ERROR(...) ORT_LLM_LOG_ERROR("mixed_input_launcher error")
#endif

#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl"

namespace onnxruntime::llm::kernels::cutlass_kernels {

// Shorthand aliases
using EpiTag = onnxruntime::llm::cutlass_extensions::EpilogueOpDefault;
using EpiSched = cutlass::epilogue::TmaWarpSpecializedCooperative;
using PP = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using COOP = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
static constexpr auto QOP = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

// Helper macros for explicit template instantiation.
// Pingpong schedule (used for all tile sizes)
#define INST_PP(T, M, N, K, CM, CN, CK)                         \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<     \
      T, __nv_fp4_e2m1, T, EpiTag,                              \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,    \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>, \
      PP, EpiSched, QOP>(                                       \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*);

// Cooperative schedule (used only when tile M >= 128 and NOT (M==128 && N==128))
#define INST_CO(T, M, N, K, CM, CN, CK)                         \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<     \
      T, __nv_fp4_e2m1, T, EpiTag,                              \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,    \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>, \
      COOP, EpiSched, QOP>(                                     \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*);

// ============================================================================
// FP16 activations + FP4 weights → FP16 output
// ============================================================================

#ifdef ORT_QUICK_BUILD
// Quick build: FP16 only, M=128 tiles, cluster 1x1x1, Pingpong schedule only.
// BF16+FP4 is excluded to cut CUTLASS compilation time in half.
// This reduces instantiations from 84 to 4 for much faster compilation.
INST_PP(half, 128, 16, 128, 1, 1, 1)
INST_PP(half, 128, 32, 128, 1, 1, 1)
INST_PP(half, 128, 64, 128, 1, 1, 1)
INST_PP(half, 128, 128, 128, 1, 1, 1)

#else  // !ORT_QUICK_BUILD

// M < 128 tiles: Pingpong only (Ntile=64, Ktile=128 for FP4)
INST_PP(half, 64, 16, 128, 1, 1, 1)
INST_PP(half, 64, 16, 128, 2, 1, 1)
INST_PP(half, 64, 16, 128, 1, 2, 1)
INST_PP(half, 64, 16, 128, 2, 2, 1)

INST_PP(half, 64, 32, 128, 1, 1, 1)
INST_PP(half, 64, 32, 128, 2, 1, 1)
INST_PP(half, 64, 32, 128, 1, 2, 1)
INST_PP(half, 64, 32, 128, 2, 2, 1)

INST_PP(half, 64, 64, 128, 1, 1, 1)
INST_PP(half, 64, 64, 128, 2, 1, 1)
INST_PP(half, 64, 64, 128, 1, 2, 1)
INST_PP(half, 64, 64, 128, 2, 2, 1)

// M >= 128 tiles (N != 128): Pingpong + Cooperative
INST_PP(half, 128, 16, 128, 1, 1, 1)
INST_CO(half, 128, 16, 128, 1, 1, 1)
INST_PP(half, 128, 16, 128, 2, 1, 1)
INST_CO(half, 128, 16, 128, 2, 1, 1)
INST_PP(half, 128, 16, 128, 1, 2, 1)
INST_CO(half, 128, 16, 128, 1, 2, 1)
INST_PP(half, 128, 16, 128, 2, 2, 1)
INST_CO(half, 128, 16, 128, 2, 2, 1)

INST_PP(half, 128, 32, 128, 1, 1, 1)
INST_CO(half, 128, 32, 128, 1, 1, 1)
INST_PP(half, 128, 32, 128, 2, 1, 1)
INST_CO(half, 128, 32, 128, 2, 1, 1)
INST_PP(half, 128, 32, 128, 1, 2, 1)
INST_CO(half, 128, 32, 128, 1, 2, 1)
INST_PP(half, 128, 32, 128, 2, 2, 1)
INST_CO(half, 128, 32, 128, 2, 2, 1)

INST_PP(half, 128, 64, 128, 1, 1, 1)
INST_CO(half, 128, 64, 128, 1, 1, 1)
INST_PP(half, 128, 64, 128, 2, 1, 1)
INST_CO(half, 128, 64, 128, 2, 1, 1)
INST_PP(half, 128, 64, 128, 1, 2, 1)
INST_CO(half, 128, 64, 128, 1, 2, 1)
INST_PP(half, 128, 64, 128, 2, 2, 1)
INST_CO(half, 128, 64, 128, 2, 2, 1)

// M=128, N=128: Cooperative uses Pingpong mainloop, so only Pingpong needed
INST_PP(half, 128, 128, 128, 1, 1, 1)
INST_PP(half, 128, 128, 128, 2, 1, 1)
INST_PP(half, 128, 128, 128, 1, 2, 1)
INST_PP(half, 128, 128, 128, 2, 2, 1)

// ============================================================================
// BF16 activations + FP4 weights → BF16 output
// ============================================================================
INST_PP(__nv_bfloat16, 64, 16, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 64, 16, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 64, 16, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 64, 16, 128, 2, 2, 1)

INST_PP(__nv_bfloat16, 64, 32, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 64, 32, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 64, 32, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 64, 32, 128, 2, 2, 1)

INST_PP(__nv_bfloat16, 64, 64, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 64, 64, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 64, 64, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 64, 64, 128, 2, 2, 1)

INST_PP(__nv_bfloat16, 128, 16, 128, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 16, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 16, 128, 2, 1, 1)
INST_CO(__nv_bfloat16, 128, 16, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 16, 128, 1, 2, 1)
INST_CO(__nv_bfloat16, 128, 16, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 16, 128, 2, 2, 1)
INST_CO(__nv_bfloat16, 128, 16, 128, 2, 2, 1)

INST_PP(__nv_bfloat16, 128, 32, 128, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 32, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 32, 128, 2, 1, 1)
INST_CO(__nv_bfloat16, 128, 32, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 32, 128, 1, 2, 1)
INST_CO(__nv_bfloat16, 128, 32, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 32, 128, 2, 2, 1)
INST_CO(__nv_bfloat16, 128, 32, 128, 2, 2, 1)

INST_PP(__nv_bfloat16, 128, 64, 128, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 64, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 64, 128, 2, 1, 1)
INST_CO(__nv_bfloat16, 128, 64, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 64, 128, 1, 2, 1)
INST_CO(__nv_bfloat16, 128, 64, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 64, 128, 2, 2, 1)
INST_CO(__nv_bfloat16, 128, 64, 128, 2, 2, 1)

INST_PP(__nv_bfloat16, 128, 128, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 128, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 128, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 128, 128, 2, 2, 1)

#endif  // ORT_QUICK_BUILD

#undef INST_PP
#undef INST_CO

}  // namespace onnxruntime::llm::kernels::cutlass_kernels

#endif  // ENABLE_FP4
#endif  // COMPILE_HOPPER_TMA_GROUPED_GEMMS
#endif  // EXCLUDE_SM_90
