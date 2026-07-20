// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

class MatMulBlockScaledFp8 final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulBlockScaledFp8(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  int64_t block_size_;
  int sm_{0};
};

Status LaunchMatMulBlockScaledFp8(const void* input_a,
                                  const void* input_b,
                                  const void* scale_a,
                                  const void* scale_b,
                                  void* output,
                                  int m,
                                  int n,
                                  int k,
                                  int block_size,
                                  bool fp16_io,
                                  bool fp16_scales,
                                  cudaStream_t stream);

// Fused GEMV fast path for the decode phase (small M). Same operands and layout as
// LaunchMatMulBlockScaledFp8, but each warp reduces one output column, streaming the
// packed FP8 weight exactly once (no dequant buffer, no underutilized M==1 tensor-core
// GEMM). Requires k % 16 == 0 and block_size % 16 == 0. Runs on any architecture.
Status LaunchMatMulBlockScaledFp8Gemv(const void* input_a,
                                      const void* input_b,
                                      const void* scale_a,
                                      const void* scale_b,
                                      void* output,
                                      int m,
                                      int n,
                                      int k,
                                      int block_size,
                                      bool fp16_io,
                                      bool fp16_scales,
                                      cudaStream_t stream);

// Converts a buffer of MLFloat16 (fp16) values to fp32. Used to normalize the
// block scales before invoking the CUTLASS fast-path GEMM, which expects fp32 scales.
void LaunchConvertHalfToFloat(const void* src_fp16, float* dst, int64_t count, cudaStream_t stream);

// Fast CUTLASS blockwise-scaled FP8 (E4M3) GEMM for NVIDIA Hopper (SM90).
// A is [M, K] row-major fp8, B is [N, K] row-major fp8 (K-major),
// scale_a is [M, K/block_size] fp32 (K-major), scale_b is [N, K/block_size] fp32 (K-major),
// output is [M, N] row-major bfloat16. Requires block_size == 128.
Status LaunchBlockQuantizedFp8GemmSm90(const void* a_fp8,
                                       const void* b_fp8,
                                       const float* scale_a,
                                       const float* scale_b,
                                       void* output_bf16,
                                       int m,
                                       int n,
                                       int k,
                                       int block_size,
                                       void* workspace,
                                       size_t workspace_size,
                                       cudaStream_t stream);
size_t GetBlockQuantizedFp8GemmSm90WorkspaceSize(int m, int n, int k);

// Fast CUTLASS blockwise-scaled FP8 (E4M3) GEMM for NVIDIA Blackwell (SM100).
// Same operand contract as the SM90 variant above.
Status LaunchBlockQuantizedFp8GemmSm100(const void* a_fp8,
                                        const void* b_fp8,
                                        const float* scale_a,
                                        const float* scale_b,
                                        void* output_bf16,
                                        int m,
                                        int n,
                                        int k,
                                        int block_size,
                                        void* workspace,
                                        size_t workspace_size,
                                        cudaStream_t stream);
size_t GetBlockQuantizedFp8GemmSm100WorkspaceSize(int m, int n, int k);

}  // namespace onnxruntime::contrib::cuda