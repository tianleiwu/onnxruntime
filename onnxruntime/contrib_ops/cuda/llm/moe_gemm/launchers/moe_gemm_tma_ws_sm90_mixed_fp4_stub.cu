/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 *
 * Stub instantiations for SM90 mixed-input FP4 MoE GEMM launcher. The full
 * implementation in moe_gemm_tma_ws_sm90_mixed_fp4.generated.cu is incompatible
 * with the bundled CUTLASS 4.4.2 mainloop and is excluded from the build. These
 * stubs satisfy the linker for the dispatch sites in
 * moe_gemm_template_dispatch_tma_ws_mixed_dtype.h. Calling any of these at
 * runtime will throw — the FP4 (MXFP4) QMoE path is not exercised by the
 * current MoE/QMoE tests.
 */

#ifdef ENABLE_FP4

#include "contrib_ops/cuda/llm/moe_gemm/launchers/moe_gemm_tma_ws_mixed_input_launcher.h"
#include "contrib_ops/cuda/llm/cutlass_extensions/epilogue_helpers.h"
#include "core/common/common.h"

#include <cute/tensor.hpp>
#include <cutlass/epilogue/dispatch_policy.hpp>
#include <cutlass/gemm/dispatch_policy.hpp>

namespace onnxruntime::llm::kernels::cutlass_kernels {

template <typename T, typename WeightType, typename GemmOutputType, typename EpilogueTag, typename CTAShape,
          typename ClusterShape, typename MainloopScheduleType, typename EpilogueScheduleType,
          cutlass::WeightOnlyQuantOp QuantOp>
void sm90_generic_mixed_moe_gemm_kernelLauncher(GroupedGemmInput<T, WeightType, GemmOutputType, GemmOutputType> /*inputs*/,
                                                TmaWarpSpecializedGroupedGemmInput /*hopper_inputs*/, int /*sm_count_*/,
                                                size_t* /*workspace_size*/) {
  ORT_THROW(
      "SM90 mixed-input FP4 MoE GEMM launcher is not available in this build "
      "(incompatible with the bundled CUTLASS).");
}

using EpiTag = onnxruntime::llm::cutlass_extensions::EpilogueOpDefault;
using EpiSched = cutlass::epilogue::TmaWarpSpecializedCooperative;
using PP = cutlass::gemm::KernelTmaWarpSpecializedPingpong;
using COOP = cutlass::gemm::KernelTmaWarpSpecializedCooperative;
static constexpr auto QOP = cutlass::WeightOnlyQuantOp::FINEGRAINED_SCALE_ONLY;

#define INST_PP(T, M, N, K, CM, CN, CK)                         \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<     \
      T, __nv_fp4_e2m1, T, EpiTag,                              \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,    \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>, \
      PP, EpiSched, QOP>(                                       \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*);

#define INST_CO(T, M, N, K, CM, CN, CK)                         \
  template void sm90_generic_mixed_moe_gemm_kernelLauncher<     \
      T, __nv_fp4_e2m1, T, EpiTag,                              \
      cute::Shape<cute::Int<M>, cute::Int<N>, cute::Int<K>>,    \
      cute::Shape<cute::Int<CM>, cute::Int<CN>, cute::Int<CK>>, \
      COOP, EpiSched, QOP>(                                     \
      GroupedGemmInput<T, __nv_fp4_e2m1, T, T>, TmaWarpSpecializedGroupedGemmInput, int, size_t*);

// Base instantiations: M=128, cluster 1x1x1, PP schedule (used by both quick and non-quick builds).
// K=128 is the FP4 dispatch tile depth; K=256 is used by calcMaxWorkspaceSizeTmaWarpSpecializedMixedInput.
INST_PP(half, 128, 16, 128, 1, 1, 1)
INST_PP(half, 128, 32, 128, 1, 1, 1)
INST_PP(half, 128, 64, 128, 1, 1, 1)
INST_PP(half, 128, 128, 128, 1, 1, 1)
INST_PP(half, 128, 16, 256, 1, 1, 1)
INST_PP(half, 128, 32, 256, 1, 1, 1)
INST_PP(half, 128, 64, 256, 1, 1, 1)
INST_PP(half, 128, 128, 256, 1, 1, 1)
// calcMaxWorkspaceSizeTmaWarpSpecializedMixedInput uses COOP with M=128, N=64, K=256 for FP4.
INST_CO(half, 128, 64, 256, 1, 1, 1)
#ifdef ENABLE_BF16
INST_PP(__nv_bfloat16, 128, 16, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 32, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 64, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 128, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 16, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 32, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 64, 256, 1, 1, 1)
INST_PP(__nv_bfloat16, 128, 128, 256, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 64, 256, 1, 1, 1)
#endif  // ENABLE_BF16

#ifndef ORT_QUICK_BUILD
// Non-quick builds enable additional CTA shapes (M=64 tiles) and additional cluster shapes
// (2x1x1, 1x2x1, 2x2x1) in the dispatch switch.  Provide stub instantiations for all of them.

// M=64 tiles — PP schedule only (COOP requires M>=128), K=128, all four cluster shapes.
INST_PP(half, 64, 16, 128, 1, 1, 1)
INST_PP(half, 64, 32, 128, 1, 1, 1)
INST_PP(half, 64, 64, 128, 1, 1, 1)
INST_PP(half, 64, 16, 128, 2, 1, 1)
INST_PP(half, 64, 32, 128, 2, 1, 1)
INST_PP(half, 64, 64, 128, 2, 1, 1)
INST_PP(half, 64, 16, 128, 1, 2, 1)
INST_PP(half, 64, 32, 128, 1, 2, 1)
INST_PP(half, 64, 64, 128, 1, 2, 1)
INST_PP(half, 64, 16, 128, 2, 2, 1)
INST_PP(half, 64, 32, 128, 2, 2, 1)
INST_PP(half, 64, 64, 128, 2, 2, 1)

// M=128 tiles — PP schedule, K=128, additional cluster shapes (2x1x1, 1x2x1, 2x2x1).
INST_PP(half, 128, 16, 128, 2, 1, 1)
INST_PP(half, 128, 32, 128, 2, 1, 1)
INST_PP(half, 128, 64, 128, 2, 1, 1)
INST_PP(half, 128, 128, 128, 2, 1, 1)
INST_PP(half, 128, 16, 128, 1, 2, 1)
INST_PP(half, 128, 32, 128, 1, 2, 1)
INST_PP(half, 128, 64, 128, 1, 2, 1)
INST_PP(half, 128, 128, 128, 1, 2, 1)
INST_PP(half, 128, 16, 128, 2, 2, 1)
INST_PP(half, 128, 32, 128, 2, 2, 1)
INST_PP(half, 128, 64, 128, 2, 2, 1)
INST_PP(half, 128, 128, 128, 2, 2, 1)

// M=128 tiles — COOP schedule, K=128, N=16/32/64 (N=128 uses PP even under COOP), all cluster shapes.
INST_CO(half, 128, 16, 128, 1, 1, 1)
INST_CO(half, 128, 32, 128, 1, 1, 1)
INST_CO(half, 128, 64, 128, 1, 1, 1)
INST_CO(half, 128, 16, 128, 2, 1, 1)
INST_CO(half, 128, 32, 128, 2, 1, 1)
INST_CO(half, 128, 64, 128, 2, 1, 1)
INST_CO(half, 128, 16, 128, 1, 2, 1)
INST_CO(half, 128, 32, 128, 1, 2, 1)
INST_CO(half, 128, 64, 128, 1, 2, 1)
INST_CO(half, 128, 16, 128, 2, 2, 1)
INST_CO(half, 128, 32, 128, 2, 2, 1)
INST_CO(half, 128, 64, 128, 2, 2, 1)

#ifdef ENABLE_BF16
// BF16 counterparts of all non-quick additions above.
INST_PP(__nv_bfloat16, 64, 16, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 64, 32, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 64, 64, 128, 1, 1, 1)
INST_PP(__nv_bfloat16, 64, 16, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 64, 32, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 64, 64, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 64, 16, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 64, 32, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 64, 64, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 64, 16, 128, 2, 2, 1)
INST_PP(__nv_bfloat16, 64, 32, 128, 2, 2, 1)
INST_PP(__nv_bfloat16, 64, 64, 128, 2, 2, 1)

INST_PP(__nv_bfloat16, 128, 16, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 32, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 64, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 128, 128, 2, 1, 1)
INST_PP(__nv_bfloat16, 128, 16, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 32, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 64, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 128, 128, 1, 2, 1)
INST_PP(__nv_bfloat16, 128, 16, 128, 2, 2, 1)
INST_PP(__nv_bfloat16, 128, 32, 128, 2, 2, 1)
INST_PP(__nv_bfloat16, 128, 64, 128, 2, 2, 1)
INST_PP(__nv_bfloat16, 128, 128, 128, 2, 2, 1)

INST_CO(__nv_bfloat16, 128, 16, 128, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 32, 128, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 64, 128, 1, 1, 1)
INST_CO(__nv_bfloat16, 128, 16, 128, 2, 1, 1)
INST_CO(__nv_bfloat16, 128, 32, 128, 2, 1, 1)
INST_CO(__nv_bfloat16, 128, 64, 128, 2, 1, 1)
INST_CO(__nv_bfloat16, 128, 16, 128, 1, 2, 1)
INST_CO(__nv_bfloat16, 128, 32, 128, 1, 2, 1)
INST_CO(__nv_bfloat16, 128, 64, 128, 1, 2, 1)
INST_CO(__nv_bfloat16, 128, 16, 128, 2, 2, 1)
INST_CO(__nv_bfloat16, 128, 32, 128, 2, 2, 1)
INST_CO(__nv_bfloat16, 128, 64, 128, 2, 2, 1)
#endif  // ENABLE_BF16
#endif  // !ORT_QUICK_BUILD

#undef INST_PP
#undef INST_CO

}  // namespace onnxruntime::llm::kernels::cutlass_kernels

#endif  // ENABLE_FP4
