/*
 * Copyright (c) Microsoft Corporation. All rights reserved.
 * Licensed under the MIT License.
 */

#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_template_dispatch.h"

namespace onnxruntime::llm::kernels::cutlass_kernels {
// Native NVFP4 (W4A16 block-scaled): FP4 e2m1 activations + FP4 e2m1 weights.
// Routes through the Blackwell SM120 FP4xFP4 block-scaled tensor-op path
// (isValidSM120MOESpecialisation) inside dispatchMoeGemmSelectBiasTmaWarpSpecialized.
// The BF16/FP16 user activation is quantized to NVFP4 inside expandInputRowsKernel.
// Requires ENABLE_FP4 (CUDA >= 12.8) and the SM120 TMA grouped-GEMM launchers.
#if defined(ENABLE_FP4) && defined(USE_FP4_QMOE)
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, half>;
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_fp4_e2m1, __nv_fp4_e2m1, __nv_bfloat16>;
#endif
#endif
}  // namespace onnxruntime::llm::kernels::cutlass_kernels
