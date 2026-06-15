// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/common.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "contrib_ops/cuda/moe/moe_base.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_gemm_profiler.h"

#include <mutex>
#include <unordered_map>

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
  // PrePack helpers - each handles one category of input tensor.
  void PrePackTransposeAndPack(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                               IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  void PrePackComputeBias(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                          const IAllocatorUniquePtr<void>& packed_scale,
                          IAllocatorUniquePtr<void>& packed_bias, bool& is_packed);
  void PrePackCopyToGpu(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                        IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  void PrePackSwizzleBlockScales(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                                 IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  void PrePackRepackFP4Weights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                               IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  // Prepacks int4/int8 expert weights into the CUTLASS fpA_intB layout so the
  // QMoE runner can consume them directly. Mirrors what MatMulNBits.PrePack
  // does, looped over the E expert dimension. ``tensor`` is the 3-D
  // ``[E, N, K / (8 / bits)]`` weight initializer; ``packed_buf`` receives a
  // GPU buffer in the kernel-expected ``[E, K, N / (8 / bits)]`` layout.
  void PrePackIntExpertWeights(const Tensor& tensor, cudaStream_t stream, AllocatorPtr alloc,
                               IAllocatorUniquePtr<void>& packed_buf, bool& is_packed);
  int64_t expert_weight_bits_;
  bool is_fp16_;
  // When true, the int4/int8 fc1/fc2 weight initializers are already in a
  // CUTLASS fpA_intB layout — produced offline e.g. via
  // ``pack_weights_for_cuda_mixed_gemm`` — and the compute path reads them
  // as-is. When false, the raw schema-conformant ``[E, N, K/pack]`` layout
  // (as produced by ``quantize_matmul_{4,8}bits``) is rewritten inside the
  // PrePack hook via ``PrePackIntExpertWeights``, removing the offline
  // prepack dependency. Only meaningful when ``quant_type_ == "int"``.
  // Derived from the optional tri-state ``weights_prepacked`` attribute:
  // -1 (default) and 1 both map to true; 0 maps to false. The concrete
  // prepacked layouts selected by -1 and 1 are determined by the execution
  // provider. For the CUDA EP the int4/int8 MoE GEMM always dispatches to the
  // Ampere (SM80) grouped-GEMM kernel -- even on SM90 -- because mixed
  // int-weight + fp16/bf16 activation is not a valid Hopper TMA warp-specialized
  // specialisation (matches TensorRT-LLM, which also routes W4A16/W8A16 MoE to
  // the SM80 kernel on Hopper). The kernel therefore consumes the SM80 fpA_intB
  // layout on every GPU, so -1 and 1 are currently equivalent for the CUDA EP;
  // 1 is reserved for a possible future Hopper-specific layout (e.g. W4A8).
  bool weights_prepacked_ = true;
  // Cached source weight shapes captured at PrePack time. When the
  // PrePack hook consumed and released the original int4/int8 weight
  // initializers (``is_packed = true``), ``context->Input<Tensor>(2)``
  // and ``(5)`` return nothing, so ``moe_helper::CheckInputs`` can no
  // longer read the shapes from the live tensors. We feed it these
  // cached shapes instead via the ``TensorShape*`` overload, matching
  // how ``MatMulNBits`` caches ``N_`` / ``K_`` in its constructor.
  TensorShape fc1_weights_shape_;
  TensorShape fc2_weights_shape_;
  bool use_fp4_dequant_fallback_ = false;
  // Dequantizes FP8 weights to FP16/BF16 scratch buffers before invoking the A16 MoE runner.
  bool use_fp8_dequant_fallback_ = false;
  // WFP4AFP8 (W4A8) requires SM100+ (Blackwell) block-scaled tensor ops. On older GPUs we
  // dequantize MXFP4 weights to FP16/BF16 and run the dense A16 MoE runner.
  bool use_wfp4afp8_dequant_fallback_ = false;
  std::string quant_type_;  // "int", "fp4", "fp8", or "wfp4afp8"

  std::unique_ptr<onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunnerInterface> m_moe_runner;

  // Pre-packed buffers
  // Note: For QMoE, we need both Scales (for dequant) and Bias (derived from ZP/Scale) during inference.
  // PrePack logic:
  // - Copies scales to GPU buffer (if in CPU) or just keeps them. For simplicity, we allocate and copy.
  // - Computes Bias from ZP and Scale using PrePack kernel.
  // - For ``quant_type == "int"``, also prepacks the per-expert int4/int8
  //   weight tensors into the CUTLASS fpA_intB layout, mirroring
  //   ``MatMulNBits.PrePack_B``. Without this, callers would have to
  //   pre-prepack the weights offline using ``pack_weights_for_cuda_mixed_gemm``,
  //   which is asymmetric with how ``MatMulNBits`` is consumed and forces
  //   a CUDA-enabled ORT build for any offline quantization tooling.
  IAllocatorUniquePtr<void> packed_fc1_weights_;
  IAllocatorUniquePtr<void> packed_fc2_weights_;
  IAllocatorUniquePtr<void> packed_fc1_scales_;
  IAllocatorUniquePtr<void> packed_fc1_bias_;
  IAllocatorUniquePtr<void> packed_fc2_scales_;
  IAllocatorUniquePtr<void> packed_fc2_bias_;

  // FP4 pre-packed buffers
  IAllocatorUniquePtr<void> packed_fp4_fc1_weights_;
  IAllocatorUniquePtr<void> packed_fp4_fc2_weights_;
  IAllocatorUniquePtr<void> packed_fp4_fc1_block_scales_;
  IAllocatorUniquePtr<void> packed_fp4_fc2_block_scales_;

  // Per-expert global weight scales used by FP4 and FP8 modes.
  IAllocatorUniquePtr<void> packed_fc1_global_scale_;
  IAllocatorUniquePtr<void> packed_fc2_global_scale_;

  // Per-tensor or per-expert FP8 activation global scales used by W4A8 (WFP4AFP8) Variant A.
  // Inputs 17/18 in the QMoE schema. Optional; absent for the MXFP8 block-scaled variant.
  IAllocatorUniquePtr<void> packed_fc1_act_scale_;
  IAllocatorUniquePtr<void> packed_fc2_act_scale_;

  mutable onnxruntime::llm::kernels::cutlass_kernels::MoeGemmProfiler mGemmProfiler;
  struct RouteTuningKey {
    int sm = 0;
    int dtype = 0;
    int expert_weight_bits = 0;
    int64_t block_size = 0;
    bool has_zero_points = false;
    int row_bucket = 0;
    int expanded_row_bucket = 0;
    int64_t hidden_size = 0;
    int64_t inter_size = 0;
    int top_k = 0;
    int num_experts = 0;
    int activation_type = 0;
    int swiglu_fusion = 0;
    int ep_size = 1;

    bool operator==(const RouteTuningKey& other) const {
      return sm == other.sm && dtype == other.dtype && expert_weight_bits == other.expert_weight_bits &&
             block_size == other.block_size && has_zero_points == other.has_zero_points &&
             row_bucket == other.row_bucket && expanded_row_bucket == other.expanded_row_bucket &&
             hidden_size == other.hidden_size && inter_size == other.inter_size && top_k == other.top_k &&
             num_experts == other.num_experts && activation_type == other.activation_type &&
             swiglu_fusion == other.swiglu_fusion && ep_size == other.ep_size;
    }
  };

  struct RouteTuningKeyHash {
    size_t operator()(const RouteTuningKey& key) const {
      size_t hash = 1469598103934665603ULL;
      auto combine = [&hash](auto value) {
        hash ^= static_cast<size_t>(value);
        hash *= 1099511628211ULL;
      };
      combine(key.sm);
      combine(key.dtype);
      combine(key.expert_weight_bits);
      combine(key.block_size);
      combine(key.has_zero_points);
      combine(key.row_bucket);
      combine(key.expanded_row_bucket);
      combine(key.hidden_size);
      combine(key.inter_size);
      combine(key.top_k);
      combine(key.num_experts);
      combine(key.activation_type);
      combine(key.swiglu_fusion);
      combine(key.ep_size);
      return hash;
    }
  };

  struct RouteTuningResult {
    onnxruntime::llm::kernels::cutlass_kernels::MoeRoutePolicy policy;
    float auto_time_ms = 0.0f;
    float gemv_where_supported_time_ms = 0.0f;
    float grouped_gemm_time_ms = 0.0f;
    float fc1_gemv_fc2_gemm_time_ms = 0.0f;
    float fc1_gemm_fc2_gemv_time_ms = 0.0f;
  };

  mutable std::unordered_map<RouteTuningKey, RouteTuningResult, RouteTuningKeyHash> qmoe_route_tuning_cache_;
  mutable std::mutex mGemmProfilerMutex;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
