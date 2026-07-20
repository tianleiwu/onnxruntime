// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

namespace onnxruntime::contrib::cuda {

// Weight-only NVFP4 (E2M1) matrix multiplication.
//
// The weight tensor B is stored as packed NVFP4: two E2M1 values per byte (low nibble first),
// with a per-16-block E4M3 scale (weight_scale) and a single global fp32 scale (weight_scale_2).
// The weight is dequantized to the activation type (FP16/BF16) and multiplied with the FP16/BF16
// activation via cuBLAS. This path works on any CUDA architecture (including Hopper/SM90) because
// it does not rely on native NVFP4 block-scaled tensor cores (SM100/SM120 only).
class MatMulBlockScaledFp4 final : public onnxruntime::cuda::CudaKernel {
 public:
  explicit MatMulBlockScaledFp4(const OpKernelInfo& info);

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  template <typename T>
  Status ComputeImpl(OpKernelContext* context) const;

  int64_t K_;
  int64_t N_;
  int64_t block_size_;
};

// Dequantizes NVFP4 (E2M1) weights with per-block E4M3 scales and a global fp32 scale into
// FP16/BF16. b_packed is [N, K/2] uint8 (two E2M1 values per byte, low nibble first),
// weight_scale is [N, ceil(K/block_size)] uint8 (raw E4M3 bytes), weight_scale_2 is a device
// fp32 scalar. Output b_dequant is [N, K] in the activation type (is_bf16 selects BF16 vs FP16).
Status LaunchDequantizeNvFp4(void* b_dequant,
                             const void* b_packed,
                             const void* weight_scale,
                             const float* weight_scale_2,
                             int n,
                             int k,
                             int block_size,
                             bool is_bf16,
                             cudaStream_t stream);

// Adds a per-column bias of shape [N] to a [M, N] row-major output in place.
Status LaunchAddBiasNvFp4(void* y,
                          const void* bias,
                          int m,
                          int n,
                          bool is_bf16,
                          cudaStream_t stream);

}  // namespace onnxruntime::contrib::cuda
