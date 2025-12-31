// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class QMoE final : public CudaKernel, public MoEBase {
 public:
  explicit QMoE(const OpKernelInfo& op_kernel_info);
  Status ComputeInternal(OpKernelContext* ctx) const override;
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;

 private:
  int64_t expert_weight_bits_;
  int64_t block_size_;
  bool has_fc3_;
  bool is_fp16_;

  std::unique_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface> m_moe_runner;

  // Pre-packed buffers
  // Note: For QMoE, we need both Scales (for dequant) and Bias (derived from ZP/Scale) during inference.
  // PrePack logic:
  // - Copies scales to GPU buffer (if in CPU) or just keeps them. For simplicity, we allocate and copy.
  // - Computes Bias from ZP and Scale using PrePack kernel.
  IAllocatorUniquePtr<void> packed_fc1_scales_;
  IAllocatorUniquePtr<void> packed_fc1_bias_;
  IAllocatorUniquePtr<void> packed_fc2_scales_;
  IAllocatorUniquePtr<void> packed_fc2_bias_;
  IAllocatorUniquePtr<void> packed_fc3_scales_;
  IAllocatorUniquePtr<void> packed_fc3_bias_;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
