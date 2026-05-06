/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Explicit FP16 M=128, N=64 cluster-M=2, cluster-N=2 pingpong template
 * instantiation for SM90 mixed-input MoE GEMM launcher with FP4 (MXFP4)
 * weights.
 */

#ifndef EXCLUDE_SM_90
#ifdef COMPILE_HOPPER_TMA_GROUPED_GEMMS
#ifdef ENABLE_FP4
#ifndef ORT_QUICK_BUILD

#include "contrib_ops/cuda/llm/common/logger.h"
#ifndef LLM_LOG_ERROR
#define LLM_LOG_ERROR(...) ORT_LLM_LOG_ERROR("mixed_input_launcher error")
#endif

#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.inl"

namespace onnxruntime::llm::kernels::cutlass_kernels {

using EpiTag = onnxruntime::llm::cutlass_extensions::EpilogueOpDefault;
using EpiSched = cutlass::epilogue::TmaWarpSpecializedCooperative;
using PP = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
static constexpr auto QOP = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

#define INST_PP(T, M, N, K, CM, CN, CK)                         \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<     \
      T, __nv_fp4_e2m1, T, EpiTag,                              \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,    \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>, \
      PP, EpiSched, QOP>(                                       \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*);

INST_PP(half, 128, 64, 256, 2, 2, 1)

#undef INST_PP

}  // namespace onnxruntime::llm::kernels::cutlass_kernels

#endif  // ORT_QUICK_BUILD
#endif  // ENABLE_FP4
#endif  // COMPILE_HOPPER_TMA_GROUPED_GEMMS
#endif  // EXCLUDE_SM_90
