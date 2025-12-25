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

// REGISTER_KERNEL_TYPED(float)
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
  const int sm = device_prop.major * 10 + device_prop.minor;

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

  const CudaT* fc1_weights_ptr = reinterpret_cast<const CudaT*>(fc1_experts_weights->DataRaw());

  // Fuse FC1 and FC3 if needed (SwiGLU)
  IAllocatorUniquePtr<void> fused_fc1_buffer;
  if (fc3_experts_weights_optional != nullptr) {
    size_t fused_size = moe_params.num_experts * moe_params.hidden_size * moe_params.inter_size * 2 * sizeof(CudaT);
    fused_fc1_buffer = GetScratchBuffer<void>(fused_size, stream_obj);
    CudaT* fused_ptr = reinterpret_cast<CudaT*>(fused_fc1_buffer.get());
    const CudaT* fc3_ptr = reinterpret_cast<const CudaT*>(fc3_experts_weights_optional->DataRaw());

    size_t width_bytes = moe_params.inter_size * sizeof(CudaT);
    size_t height = moe_params.num_experts * moe_params.hidden_size;
    size_t src_pitch = width_bytes;
    size_t dst_pitch = 2 * width_bytes;

    // Copy FC1
    CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(fused_ptr, dst_pitch, fc1_weights_ptr, src_pitch, width_bytes, height, cudaMemcpyDeviceToDevice, stream));

    // Copy FC3
    CUDA_RETURN_IF_ERROR(cudaMemcpy2DAsync(reinterpret_cast<uint8_t*>(fused_ptr) + width_bytes, dst_pitch, fc3_ptr, src_pitch, width_bytes, height, cudaMemcpyDeviceToDevice, stream));

    fc1_weights_ptr = fused_ptr;
  }

  moe_runner.runMoe(
      reinterpret_cast<const CudaT*>(input->template Data<T>()),
      nullptr,         // input_sf
      expert_indices,  // token_selected_experts
      expert_scales,   // token_final_scales
      fc1_weights_ptr,
      fc1_experts_bias_optional == nullptr ? nullptr : reinterpret_cast<const CudaT*>(fc1_experts_bias_optional->template Data<T>()),
      activation_type_,
      reinterpret_cast<const CudaT*>(fc2_experts_weights->DataRaw()),
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
      stream);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
