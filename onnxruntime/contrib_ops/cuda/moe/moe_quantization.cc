// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#if 0  // disable QMoE for now
#ifdef __GNUC__
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-parameter"
#pragma GCC diagnostic ignored "-Wunused-local-typedefs"
#endif

#include "contrib_ops/cuda/moe/moe_quantization.h"
#include <type_traits>
#include "cutlass/numeric_types.h"
#include "core/common/safeint.h"
#include "contrib_ops/cuda/moe/qmoe_kernels.h"

using namespace onnxruntime::cuda;
using namespace ::onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {
namespace cuda {

#define REGISTER_KERNEL_TYPED(T)                                        \
  ONNX_OPERATOR_TYPED_KERNEL_EX(                                        \
      QMoE,                                                             \
      kMSDomain,                                                        \
      1,                                                                \
      T,                                                                \
      kCudaExecutionProvider,                                           \
      (*KernelDefBuilder::Create())                                     \
          .MayInplace(0, 0)                                             \
          .TypeConstraint("T", DataTypeImpl::GetTensorType<T>())        \
          .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>()) \
          .TypeConstraint("T2", DataTypeImpl::GetTensorType<T>()),      \
      QMoE);

REGISTER_KERNEL_TYPED(MLFloat16)

#define QUICK_BUILD 1
#if QUICK_BUILD == 0
REGISTER_KERNEL_TYPED(BFloat16)
#endif

QMoE::QMoE(const OpKernelInfo& op_kernel_info) : CudaKernel(op_kernel_info), MoEBase(op_kernel_info, GetDeviceProp()) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
#if QUICK_BUILD
  ORT_ENFORCE(expert_weight_bits_ == 8,
              "expert_weight_bits must be 8, but got ", expert_weight_bits_);
#else
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);
#endif

  this->block_size_ = op_kernel_info.GetAttrOrDefault<int64_t>("block_size", -1);

  using namespace onnxruntime::llm::kernels::cutlass_kernels;

  constexpr int kInputIndexFc3Weight = 8;
  has_fc3_ = op_kernel_info.GetInputCount() > kInputIndexFc3Weight && op_kernel_info.node().InputDefs()[kInputIndexFc3Weight]->Exists();

  int32_t input_type = op_kernel_info.node().InputDefs()[0]->TypeAsProto()->tensor_type().elem_type();

  bool is_fp16 = input_type == ONNX_NAMESPACE::TensorProto_DataType::TensorProto_DataType_FLOAT16;

#if QUICK_BUILD
  if (is_fp16) {
    if (expert_weight_bits_ == 8) {
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, uint8_t, half>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    }
  }
#else
  if (is_fp16) {
    if (expert_weight_bits_ == 4) {
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, cutlass::uint4b_t, half>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    } else {  // expert_weight_bits_ == 8
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<half, uint8_t, half>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    }
  } else {  // BFloat16
    if (expert_weight_bits_ == 4) {
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, cutlass::uint4b_t, __nv_bfloat16>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    } else {  // expert_weight_bits_ == 8
      m_moe_runner = std::make_unique<CutlassMoeFCRunner<__nv_bfloat16, uint8_t, __nv_bfloat16>>(
          sm_, activation_type_, has_fc3_, normalize_routing_weights_, use_sparse_mixer_);
    }
  }
#endif
}

Status QMoE::ComputeInternal(OpKernelContext* context) const {
  printf("DEBUG: QMoE ComputeInternal running on CUDA\n");
  const Tensor* input = context->Input<Tensor>(0);
  const Tensor* router_probs = context->Input<Tensor>(1);
  const Tensor* fc1_experts_weights = context->Input<Tensor>(2);
  const Tensor* fc1_scales = context->Input<Tensor>(3);
  const Tensor* fc1_experts_bias_optional = context->Input<Tensor>(4);
  const Tensor* fc2_experts_weights = context->Input<Tensor>(5);
  const Tensor* fc2_scales = context->Input<Tensor>(6);
  const Tensor* fc2_experts_bias_optional = context->Input<Tensor>(7);
  const Tensor* fc3_experts_weights_optional = context->Input<Tensor>(8);
  const Tensor* fc3_scales_optional = context->Input<Tensor>(9);
  const Tensor* fc3_experts_bias_optional = context->Input<Tensor>(10);

  const Tensor* fc1_zeros = context->Input<Tensor>(11);
  const Tensor* fc2_zeros = context->Input<Tensor>(12);
  const Tensor* fc3_zeros = context->Input<Tensor>(13);
  (void)fc3_zeros;  // Cast to void to avoid unused warning

  int64_t pack_size = expert_weight_bits_ == 4 ? 2 : 1;
  bool is_fused_swiglu = activation_type_ == onnxruntime::llm::kernels::cutlass_kernels::ActivationType::Swiglu;
  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(onnxruntime::contrib::moe_helper::CheckInputs(
      moe_params, input, router_probs, fc1_experts_weights,
      fc1_experts_bias_optional, fc1_scales, fc1_zeros,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales, fc2_zeros,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional, fc3_zeros,
      pack_size, is_fused_swiglu, block_size_));

  constexpr bool use_lora = false;
  constexpr bool use_deepseek_fp8_block_scale = false;
  constexpr bool min_latency_mode = false;
  bool use_awq = (fc1_zeros != nullptr);
  onnxruntime::llm::kernels::cutlass_kernels::MOEParallelismConfig parallelism_config{};

  size_t workspace_size = m_moe_runner->getWorkspaceSize(
      moe_params.num_rows, moe_params.hidden_size, moe_params.inter_size, moe_params.num_experts, k_,
      activation_type_, parallelism_config, use_lora, use_deepseek_fp8_block_scale, min_latency_mode, use_awq);

  // Scratch buffer for workspace + expert_scales + expert_indices
  // expert_scales: num_rows * k * sizeof(float)
  // expert_indices: num_rows * k * sizeof(int)
  size_t scales_bytes = moe_params.num_rows * k_ * sizeof(float);
  size_t indices_bytes = moe_params.num_rows * k_ * sizeof(int);
  size_t total_scratch_bytes = workspace_size + scales_bytes + indices_bytes;

  auto work_space = GetScratchBuffer<void>(total_scratch_bytes, context->GetComputeStream());
  char* workspace_ptr = reinterpret_cast<char*>(work_space.get());
  float* expert_scales = reinterpret_cast<float*>(workspace_ptr + workspace_size);
  int* expert_indices = reinterpret_cast<int*>(workspace_ptr + workspace_size + scales_bytes);

  cudaStream_t stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());

  // Perform Softmax + TopK
  // Input router_probs is (num_rows, num_experts)
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
    // Fallback for float
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

  // TODO: Add support for fc1_zeros and fc2_zeros (handled below)

  onnxruntime::llm::kernels::cutlass_kernels::QuantParams quant_params;
  if (block_size_ > 0) {
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::GroupWise(
        block_size_,
        fc1_scales->DataRaw(),
        fc2_scales->DataRaw(),
        nullptr,
        nullptr,
        fc1_zeros ? fc1_zeros->DataRaw() : nullptr,
        fc2_zeros ? fc2_zeros->DataRaw() : nullptr);
  } else {
    // Per-column quantization
    quant_params = onnxruntime::llm::kernels::cutlass_kernels::QuantParams::Int(
        fc1_scales->DataRaw(),
        fc2_scales->DataRaw());
  }

  Tensor* output = context->Output(0, input->Shape());

  onnxruntime::llm::kernels::LoraParams lora_params;
  onnxruntime::llm::kernels::cutlass_kernels::MoeMinLatencyParams min_latency_params;

  m_moe_runner->runMoe(
      input->DataRaw(),
      nullptr,
      expert_indices,
      expert_scales,
      fc1_experts_weights->DataRaw(),
      fc1_experts_bias_optional ? fc1_experts_bias_optional->DataRaw() : nullptr,
      activation_type_,
      fc2_experts_weights->DataRaw(),
      fc2_experts_bias_optional ? fc2_experts_bias_optional->DataRaw() : nullptr,
      quant_params,
      moe_params.num_rows,
      moe_params.hidden_size,
      moe_params.inter_size,
      moe_params.num_experts,
      k_,
      workspace_ptr,
      output->MutableDataRaw(),
      nullptr,
      parallelism_config,
      false,  // enable_alltoall
      use_lora,
      lora_params,
      use_deepseek_fp8_block_scale,
      min_latency_mode,
      min_latency_params,
      {activation_alpha_, activation_beta_, swiglu_fusion_, swiglu_limit_},
      stream);

  return Status::OK();
}

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
#endif
