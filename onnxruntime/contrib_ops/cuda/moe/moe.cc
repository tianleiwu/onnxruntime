// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "core/common/safeint.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_type_conversion.h"
#include "contrib_ops/cuda/moe/moe.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                    \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                    \
      MoE, kMSDomain, 1, T, kCudaExecutionProvider, \
      (*KernelDefBuilder::Create()).MayInplace(0, 0).TypeConstraint("T", DataTypeImpl::GetTensorType<T>()), MoE<T>);

REGISTER_KERNEL_TYPED(float)
REGISTER_KERNEL_TYPED(MLFloat16)
REGISTER_KERNEL_TYPED(BFloat16)

template <typename T>
MoE<T>::MoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
}

template <typename T>
Status MoE<T>::ComputeInternal(OpKernelContext* context) const {
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(3);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(4);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(5);
  const Tensor* fc3_experts_weights_optional = context->Input<Tensor>(6);
  const Tensor* fc3_experts_bias_optional = context->Input<Tensor>(7);

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, nullptr, nullptr,
      fc2_experts_weights, fc2_experts_bias_optional, nullptr, nullptr,
      fc3_experts_weights_optional, fc3_experts_bias_optional, nullptr, nullptr,
      1,  //  no quantization so pack size is 1
      activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu,
      0));  // no block-wise quantization for regular MoE

  using CudaT = typename OrtToCudaType<T>::type;
  auto stream_obj = context->GetComputeStream();
  cudaStream_t stream = static_cast<cudaStream_t>(stream_obj->GetHandle());

  auto& device_prop = GetDeviceProp();
  int sm = device_prop.major * 10 + device_prop.minor;

  // SM90 TMA WS kernels only support f16/bf16, not float32.
  // Force SM80 path for float32 to use legacy kernels.
  if constexpr (std::is_same_v<T, float>) {
    if (sm >= 90) {
      sm = 80;
    }
  }

  onnxruntime::llm::kernels::cutlass_kernels::CutlassMoeFCRunner<CudaT, CudaT> moe_runner(sm,
                                                                                          activation_type_,
                                                                                          fc3_experts_weights_optional != nullptr,
                                                                                          normalize_routing_weights_,
                                                                                          use_sparse_mixer_);

  constexpr bool use_lora = false;
  constexpr bool use_deepseek_block_scale = false;
  constexpr bool min_latency_mode = false;
  constexpr bool use_awq = false;
  onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};

  size_t ws_size = moe_runner.getWorkspaceSize(
      static_cast<size_t>(moe_params.num_rows), static_cast<size_t>(moe_params.hidden_size),
      static_cast<size_t>(moe_params.inter_size), static_cast<size_t>(moe_params.num_experts), static_cast<size_t>(k_),
      activation_type_, parallelism_config, use_lora, use_deepseek_block_scale, min_latency_mode, use_awq);

  // Scratch buffer for workspace + expert_scales + expert_indices + permutation_map
  size_t scales_bytes = moe_params.num_rows * k_ * sizeof(float);
  size_t indices_bytes = moe_params.num_rows * k_ * sizeof(int);
  size_t permutation_bytes = moe_params.num_rows * k_ * sizeof(int);
  size_t total_scratch_bytes = ws_size + scales_bytes + indices_bytes + permutation_bytes;

  auto work_space = GetScratchBuffer<void>(total_scratch_bytes, stream_obj);
  char* workspace_ptr = reinterpret_cast<char*>(work_space.get());
  float* expert_scales = reinterpret_cast<float*>(workspace_ptr + ws_size);
  int* expert_indices = reinterpret_cast<int*>(workspace_ptr + ws_size + scales_bytes);
  int* unpermuted_row_to_permuted_row = reinterpret_cast<int*>(workspace_ptr + ws_size + scales_bytes + indices_bytes);

  // Perform Softmax + TopK
  bool is_fp16 = input->IsDataType<MLFloat16>();
  if (is_fp16) {
    LaunchSoftmaxTopK(
        reinterpret_cast<const half*>(router_probs->DataRaw()),
        expert_scales,
        expert_indices,
        static_cast<int>(moe_params.num_rows),
        static_cast<int>(moe_params.num_experts),
        static_cast<int>(k_),
        normalize_routing_weights_,
        stream);
  } else {
    LaunchSoftmaxTopK(
        reinterpret_cast<const float*>(router_probs->DataRaw()),
        expert_scales,
        expert_indices,
        static_cast<int>(moe_params.num_rows),
        static_cast<int>(moe_params.num_experts),
        static_cast<int>(k_),
        normalize_routing_weights_,
        stream);
  }

  Tensor* output = context->Output(0, input->Shape());

  onnxruntime::llm::kernels::cutlass_kernels::QuantParams quant_params{};  // Default constructor
  onnxruntime::llm::kernels::LoraParams lora_params{};
  onnxruntime::llm::kernels::cutlass_kernels::MoeMinLatencyParams min_latency_params;

  // =============================================================================
  // WEIGHT LAYOUT TRANSPOSITION (Short-term fix)
  // =============================================================================
  // ORT input layout:  FC1=[E, hidden_size, inter_size], FC2=[E, inter_size, hidden_size]
  // Kernel expects:    FC1=[E, inter_size, hidden_size], FC2=[E, hidden_size, inter_size]
  // So FC1 needs transpose from [K, N] to [N, K] per expert, and FC2 is already correct shape
  // but stored transposed. Must transpose both.
  //
  // TODO(long-term): Consider updating ONNX op schema to match kernel layout, or
  // have the kernel accept ORT layout directly to avoid runtime transposes.
  // =============================================================================

  // Calculate buffer sizes
  size_t fc1_block_size = static_cast<size_t>(moe_params.inter_size) * static_cast<size_t>(moe_params.hidden_size);
  size_t fc2_block_size = static_cast<size_t>(moe_params.hidden_size) * static_cast<size_t>(moe_params.inter_size);
  int H = static_cast<int>(moe_params.hidden_size);
  int I = static_cast<int>(moe_params.inter_size);
  int E = static_cast<int>(moe_params.num_experts);

  // Allocate buffers
  size_t fc1_total_size = (fc3_experts_weights_optional != nullptr)
                              ? E * fc1_block_size * 2 * sizeof(CudaT)  // fused
                              : E * fc1_block_size * sizeof(CudaT);
  size_t fc2_total_size = E * fc2_block_size * sizeof(CudaT);

  auto fc1_processed_buffer = GetScratchBuffer<void>(fc1_total_size, stream_obj);
  CudaT* fc1_processed_ptr = reinterpret_cast<CudaT*>(fc1_processed_buffer.get());

  // FC1 Handling (validating dims against H and I)
  const auto& fc1_dims = fc1_experts_weights->Shape().GetDims();
  const CudaT* fc1_input_ptr = reinterpret_cast<const CudaT*>(fc1_experts_weights->DataRaw());

  // Detect fused SwiGLU weights: swiglu_fusion_ != 0 indicates FC1 contains pre-fused gate+value weights
  // When fused, FC1 has shape [E, 2*I, H] instead of [E, I, H] and FC3 is not provided
  bool is_fused_swiglu = (swiglu_fusion_ != 0) && (fc3_experts_weights_optional == nullptr);

  // For fused SwiGLU weights, each expert block size is 2*I*H, not I*H
  size_t fc1_per_expert_size = is_fused_swiglu ? (2 * fc1_block_size) : fc1_block_size;

  // Recalculate total size for fused case
  if (is_fused_swiglu) {
    fc1_total_size = E * fc1_per_expert_size * sizeof(CudaT);
  }

  // If input matches [E, H, I], it's KxN layout but kernel wants NxK [E, I, H]. Needs transpose.
  bool fc1_needs_transpose = (fc1_dims[1] == H && fc1_dims[2] == I);

  // Debug: print transpose decision for float types
  if constexpr (std::is_same_v<T, float>) {
    printf("DEBUG [moe.cc float]: H=%d, I=%d, E=%d\\n", H, I, E);
    printf("DEBUG [moe.cc float]: fc1_dims=[%lld, %lld, %lld]\\n",
           (long long)fc1_dims[0], (long long)fc1_dims[1], (long long)fc1_dims[2]);
    printf("DEBUG [moe.cc float]: fc1_needs_transpose=%d (expected: fc1_dims[1]==H && fc1_dims[2]==I)\\n",
           fc1_needs_transpose);
    fflush(stdout);
  }

  if (fc3_experts_weights_optional != nullptr) {  // SwiGLU
    const CudaT* fc3_input_ptr = reinterpret_cast<const CudaT*>(fc3_experts_weights_optional->DataRaw());

    for (int e = 0; e < E; ++e) {
      CudaT* dest_fc1 = fc1_processed_ptr + 2 * e * fc1_block_size;
      CudaT* dest_fc3 = fc1_processed_ptr + (2 * e + 1) * fc1_block_size;

      if (fc1_needs_transpose) {
        // Transpose [H, I] -> [I, H]
        LaunchTranspose2D<CudaT>(fc1_input_ptr + e * fc1_block_size, dest_fc1, H, I, stream);
        LaunchTranspose2D<CudaT>(fc3_input_ptr + e * fc1_block_size, dest_fc3, H, I, stream);
      } else {
        // Copy [I, H] directly
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dest_fc1, fc1_input_ptr + e * fc1_block_size, fc1_block_size * sizeof(CudaT), cudaMemcpyDeviceToDevice, stream));
        CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(dest_fc3, fc3_input_ptr + e * fc1_block_size, fc1_block_size * sizeof(CudaT), cudaMemcpyDeviceToDevice, stream));
      }
    }
  } else {  // Non-SwiGLU FC1 (may still be fused SwiGLU weights if is_fused_swiglu is true)
    if (fc1_needs_transpose) {
      for (int e = 0; e < E; ++e) {
        // Use fc1_per_expert_size for correct offset in fused weight case
        LaunchTranspose2D<CudaT>(fc1_input_ptr + e * fc1_per_expert_size, fc1_processed_ptr + e * fc1_per_expert_size, H, is_fused_swiglu ? 2 * I : I, stream);
      }
    } else {
      CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(fc1_processed_ptr, fc1_input_ptr, fc1_total_size, cudaMemcpyDeviceToDevice, stream));
    }
  }

  // FC2 Handling
  const auto& fc2_dims = fc2_experts_weights->Shape().GetDims();
  const CudaT* fc2_input_ptr = reinterpret_cast<const CudaT*>(fc2_experts_weights->DataRaw());
  CudaT* fc2_processed_ptr = nullptr;

  // Kernel expects FC2 as [hidden, inter] (N, K).
  // If input is [inter, hidden] (dims matches I, H), it is KxN layout, needs transpose to NxK.
  bool fc2_needs_transpose = (fc2_dims[1] == I && fc2_dims[2] == H);

  // Debug: print FC2 transpose decision for float types
  if constexpr (std::is_same_v<T, float>) {
    printf("DEBUG [moe.cc float]: fc2_dims=[%lld, %lld, %lld]\\n",
           (long long)fc2_dims[0], (long long)fc2_dims[1], (long long)fc2_dims[2]);
    printf("DEBUG [moe.cc float]: fc2_needs_transpose=%d (expected: fc2_dims[1]==I && fc2_dims[2]==H)\\n",
           fc2_needs_transpose);
    printf("DEBUG [moe.cc float]: moe_params: num_rows=%lld, hidden_size=%lld, inter_size=%lld, num_experts=%lld\\n",
           (long long)moe_params.num_rows, (long long)moe_params.hidden_size,
           (long long)moe_params.inter_size, (long long)moe_params.num_experts);
    fflush(stdout);
  }

  if (fc2_needs_transpose) {
    auto fc2_buffer = GetScratchBuffer<void>(fc2_total_size, stream_obj);
    fc2_processed_ptr = reinterpret_cast<CudaT*>(fc2_buffer.get());
    for (int e = 0; e < E; ++e) {
      // Transpose [I, H] -> [H, I]
      LaunchTranspose2D<CudaT>(fc2_input_ptr + e * fc2_block_size, fc2_processed_ptr + e * fc2_block_size, I, H, stream);
    }
  } else {
    // Layout matches kernel expectation [H, I]. Use directly.
    fc2_processed_ptr = const_cast<CudaT*>(fc2_input_ptr);
  }

  moe_runner.runMoe(
      reinterpret_cast<const CudaT*>(input->template Data<T>()),
      nullptr,         // input_sf
      expert_indices,  // token_selected_experts
      expert_scales,   // token_final_scales
      fc1_processed_ptr,
      fc1_experts_bias_optional == nullptr ? nullptr : reinterpret_cast<const CudaT*>(fc1_experts_bias_optional->template Data<T>()),
      activation_type_,
      fc2_processed_ptr,
      fc2_experts_bias_optional == nullptr ? nullptr : reinterpret_cast<const CudaT*>(fc2_experts_bias_optional->template Data<T>()),
      quant_params,
      static_cast<int>(moe_params.num_rows), static_cast<int>(moe_params.hidden_size),
      static_cast<int>(moe_params.inter_size), static_cast<int>(moe_params.num_experts),
      static_cast<int>(k_),
      workspace_ptr,
      reinterpret_cast<void*>(output->template MutableData<T>()),
      unpermuted_row_to_permuted_row,
      parallelism_config,
      false,  // enable_alltoall
      use_lora,
      lora_params,
      use_deepseek_block_scale,
      min_latency_mode,
      min_latency_params,
      {activation_alpha_, activation_beta_, swiglu_fusion_, swiglu_limit_},
      stream);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
