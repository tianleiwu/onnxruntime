// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cpu/quantization/moe_quantization_cpu.h"
#include "core/framework/allocator.h"
#include "core/framework/buffer_deleter.h"
#include "core/mlas/inc/mlas.h"
#include "core/mlas/inc/mlas_q4.h"
#include "core/mlas/inc/mlas_qnbit.h"
#include "core/platform/threadpool.h"
#include "contrib_ops/cpu/moe/moe_utils.h"
#include <algorithm>
#include <vector>

using namespace onnxruntime::common;
using namespace ONNX_NAMESPACE;

namespace onnxruntime {
namespace contrib {

#define REGISTER_KERNEL()                                                                           \
  ONNX_OPERATOR_KERNEL_EX(QMoE, kMSDomain, 1, kCpuExecutionProvider,                                \
                          (*KernelDefBuilder::Create())                                             \
                              .MayInplace(0, 0)                                                     \
                              .TypeConstraint("T", BuildKernelDefConstraints<MLFloat16, float>())   \
                              .TypeConstraint("T1", BuildKernelDefConstraints<uint8_t>())           \
                              .TypeConstraint("T2", BuildKernelDefConstraints<MLFloat16, float>()), \
                          QMoE);

REGISTER_KERNEL();

QMoE::QMoE(const OpKernelInfo& op_kernel_info)
    : OpKernel(op_kernel_info),
      MoEBaseCPU(op_kernel_info),
      prepacked_fc1_weights_data_(nullptr),
      prepacked_fc2_weights_data_(nullptr),
      weights_allocator_(nullptr),
      is_prepacked_(false) {
  ORT_ENFORCE(op_kernel_info.GetAttr<int64_t>("expert_weight_bits", &expert_weight_bits_).IsOK());
  ORT_ENFORCE(expert_weight_bits_ == 8 || expert_weight_bits_ == 4,
              "expert_weight_bits must be 4 or 8, but got ", expert_weight_bits_);
}

Status QMoE::Compute(OpKernelContext* context) const {
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

  MoEParameters moe_params;
  ORT_RETURN_IF_ERROR(::onnxruntime::contrib::moe_helper::CheckInputs<Tensor>(
      moe_params, input, router_probs,
      fc1_experts_weights, fc1_experts_bias_optional, fc1_scales,
      fc2_experts_weights, fc2_experts_bias_optional, fc2_scales,
      fc3_experts_weights_optional, fc3_experts_bias_optional, fc3_scales_optional,
      expert_weight_bits_ == 4 ? 2 : 1,
      activation_type_ == ActivationType::SwiGLU));

  // Dispatch based on input data type
  if (input->IsDataType<MLFloat16>()) {
    if (expert_weight_bits_ == 4) {
      return QuantizedMoEImpl<true, MLFloat16>(context, moe_params, input, router_probs,
                                               fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                               fc2_experts_bias_optional, fc3_experts_weights_optional,
                                               fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
    } else {
      return QuantizedMoEImpl<false, MLFloat16>(context, moe_params, input, router_probs,
                                                fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                                fc2_experts_bias_optional, fc3_experts_weights_optional,
                                                fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
    }
  } else if (input->IsDataType<float>()) {
    if (expert_weight_bits_ == 4) {
      return QuantizedMoEImpl<true, float>(context, moe_params, input, router_probs,
                                           fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                           fc2_experts_bias_optional, fc3_experts_weights_optional,
                                           fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
    } else {
      return QuantizedMoEImpl<false, float>(context, moe_params, input, router_probs,
                                            fc1_experts_weights, fc1_experts_bias_optional, fc2_experts_weights,
                                            fc2_experts_bias_optional, fc3_experts_weights_optional,
                                            fc3_experts_bias_optional, fc1_scales, fc2_scales, fc3_scales_optional);
    }
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QMoE only supports float and MLFloat16 data types, but got ",
                           DataTypeImpl::ToString(input->DataType()));
  }
}

template <bool UseUInt4x2, typename T>
Status QMoE::QuantizedMoEImpl(OpKernelContext* context,
                              MoEParameters& moe_params,
                              const Tensor* input,
                              const Tensor* router_probs,
                              const Tensor* fc1_experts_weights,
                              const Tensor* fc1_experts_bias_optional,
                              const Tensor* fc2_experts_weights,
                              const Tensor* fc2_experts_bias_optional,
                              const Tensor* fc3_experts_weights_optional,
                              const Tensor* fc3_experts_bias_optional,
                              const Tensor* fc1_scales,
                              const Tensor* fc2_scales,
                              const Tensor* fc3_scales_optional) const {
  // SwiGLU validation - FC3 not supported
  bool is_swiglu = (activation_type_ == ActivationType::SwiGLU);
  if (is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "SwiGLU activation is not supported with fc3.");
  }
  if (!is_swiglu && fc3_experts_weights_optional != nullptr) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "FC3 gating is not yet implemented on CPU.");
  }

  // Check if we need to repack weights
  if (!is_prepacked_ ||
      cached_num_experts_ != moe_params.num_experts ||
      cached_hidden_size_ != moe_params.hidden_size ||
      cached_inter_size_ != moe_params.inter_size ||
      cached_is_swiglu_ != is_swiglu) {
    // Need to prepack weights
    Status status = const_cast<QMoE*>(this)->PrepackAndDequantizeWeights<UseUInt4x2>(
        context, moe_params, fc1_experts_weights, fc2_experts_weights,
        fc1_scales, fc2_scales, is_swiglu);
    ORT_RETURN_IF_ERROR(status);
  }

  auto* thread_pool = context->GetOperatorThreadPool();
  const bool is_deterministic = context->GetUseDeterministicCompute();

  const T* input_data = input->Data<T>();
  const T* router_probs_data = router_probs->Data<T>();
  const T* fc1_bias_data = fc1_experts_bias_optional ? fc1_experts_bias_optional->Data<T>() : nullptr;
  const T* fc2_bias_data = fc2_experts_bias_optional ? fc2_experts_bias_optional->Data<T>() : nullptr;

  Tensor* output = context->Output(0, input->Shape());
  T* output_data = output->MutableData<T>();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const int64_t total_output_size = input->Shape().Size();

  // Initialize output with optimized pattern based on data type
  if constexpr (std::is_same_v<T, MLFloat16>) {
    std::fill_n(output_data, static_cast<size_t>(total_output_size), MLFloat16(0.0f));
  } else {
    std::memset(output_data, 0, static_cast<size_t>(total_output_size) * sizeof(T));
  }

  IAllocatorUniquePtr<float> output_float_buffer;
  float* output_float_ptr = nullptr;
  if constexpr (std::is_same_v<T, MLFloat16>) {
    output_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(total_output_size));
    output_float_ptr = output_float_buffer.get();
  } else {
    output_float_ptr = reinterpret_cast<float*>(output_data);
  }
  std::fill_n(output_float_ptr, static_cast<size_t>(total_output_size), 0.0f);

  IAllocatorUniquePtr<float> input_float_buffer;
  float* input_float_ptr = nullptr;
  IAllocatorUniquePtr<float> router_probs_float_buffer;
  float* router_probs_float_ptr = nullptr;
  IAllocatorUniquePtr<float> fc1_bias_float_buffer;
  float* fc1_bias_float_ptr = nullptr;
  IAllocatorUniquePtr<float> fc2_bias_float_buffer;
  float* fc2_bias_float_ptr = nullptr;

  const int64_t fc1_bias_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
  const int64_t fc2_bias_size = moe_params.hidden_size;

  if constexpr (std::is_same_v<T, MLFloat16>) {
    input_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size));
    input_float_ptr = input_float_buffer.get();
    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(input_data), input_float_ptr, static_cast<size_t>(moe_params.num_rows * moe_params.hidden_size), thread_pool);

    router_probs_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(moe_params.num_rows * moe_params.num_experts));
    router_probs_float_ptr = router_probs_float_buffer.get();
    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(router_probs_data), router_probs_float_ptr, static_cast<size_t>(moe_params.num_rows * moe_params.num_experts), thread_pool);

    if (fc1_bias_data) {
      fc1_bias_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(moe_params.num_experts * fc1_bias_size));
      fc1_bias_float_ptr = fc1_bias_float_buffer.get();
      MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc1_bias_data), fc1_bias_float_ptr, static_cast<size_t>(moe_params.num_experts * fc1_bias_size), thread_pool);
    }
    if (fc2_bias_data) {
      fc2_bias_float_buffer = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(moe_params.num_experts * fc2_bias_size));
      fc2_bias_float_ptr = fc2_bias_float_buffer.get();
      MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc2_bias_data), fc2_bias_float_ptr, static_cast<size_t>(moe_params.num_experts * fc2_bias_size), thread_pool);
    }
  } else {
    input_float_ptr = const_cast<float*>(reinterpret_cast<const float*>(input_data));
    router_probs_float_ptr = const_cast<float*>(reinterpret_cast<const float*>(router_probs_data));
    if (fc1_bias_data) {
      fc1_bias_float_ptr = const_cast<float*>(reinterpret_cast<const float*>(fc1_bias_data));
    }
    if (fc2_bias_data) {
      fc2_bias_float_ptr = const_cast<float*>(reinterpret_cast<const float*>(fc2_bias_data));
    }
  }

  const int64_t fc1_output_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;
  const float* dequant_fc1_weights = prepacked_fc1_weights_data_;
  const float* dequant_fc2_weights = prepacked_fc2_weights_data_;

  // ====================================================================================================
  // UNIFIED BATCHED GEMM IMPLEMENTATION
  // This approach is thread-safe and efficient for all batch sizes, fixing the race condition.
  // ====================================================================================================

  // Phase 1: Create routing maps to group tokens by expert.
  std::vector<std::vector<int64_t>> expert_to_token_map(moe_params.num_experts);
  std::vector<std::vector<float>> expert_to_weight_map(moe_params.num_experts);
  std::vector<std::vector<std::pair<int64_t, int64_t>>> token_to_expert_info(moe_params.num_rows);

  for (int64_t i = 0; i < moe_params.num_rows; ++i) {
    const float* router_probs_row = router_probs_float_ptr + i * moe_params.num_experts;
    for (int64_t j = 0; j < moe_params.num_experts; ++j) {
      if (router_probs_row[j] > 1e-4f) {
        int64_t index_in_expert_batch = expert_to_token_map[j].size();
        expert_to_token_map[j].push_back(i);
        expert_to_weight_map[j].push_back(router_probs_row[j]);
        token_to_expert_info[i].emplace_back(j, index_in_expert_batch);
      }
    }
  }

  // This holds the output of each expert's batched computation.
  std::vector<IAllocatorUniquePtr<float>> expert_results(moe_params.num_experts);

  // Phase 2: Parallel computation over experts. Each expert processes its batch of tokens.
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts), 0.0,
      [&](std::ptrdiff_t expert_start, std::ptrdiff_t expert_end) {
        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const auto& tokens_for_expert = expert_to_token_map[expert_idx];
          const int64_t M = tokens_for_expert.size();
          if (M == 0) {
            continue;
          }

          // Gather inputs for the batch
          auto batched_input = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(M * moe_params.hidden_size));
          for (int64_t i = 0; i < M; ++i) {
            memcpy(batched_input.get() + i * moe_params.hidden_size,
                   input_float_ptr + tokens_for_expert[i] * moe_params.hidden_size,
                   static_cast<size_t>(moe_params.hidden_size) * sizeof(float));
          }

          // FC1
          auto fc1_out = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(M * fc1_output_size));
          const float* fc1_expert_weights = dequant_fc1_weights + expert_idx * moe_params.hidden_size * fc1_output_size;
          MlasGemm(CblasNoTrans, CblasTrans,
                   static_cast<size_t>(M), static_cast<size_t>(fc1_output_size), static_cast<size_t>(moe_params.hidden_size),
                   1.0f,
                   batched_input.get(), static_cast<size_t>(moe_params.hidden_size),
                   fc1_expert_weights, static_cast<size_t>(fc1_output_size),
                   0.0f,
                   fc1_out.get(), static_cast<size_t>(fc1_output_size),
                   nullptr);

          // Bias and Activation
          const float* fc1_expert_bias = fc1_bias_float_ptr ? fc1_bias_float_ptr + expert_idx * fc1_output_size : nullptr;
          const int64_t inter_size = is_swiglu ? moe_params.inter_size : fc1_output_size;

          for (int64_t i = 0; i < M; ++i) {
            float* fc1_out_row = fc1_out.get() + i * fc1_output_size;
            if (fc1_expert_bias) {
              for (int64_t j = 0; j < fc1_output_size; ++j) {
                fc1_out_row[j] += fc1_expert_bias[j];
              }
            }
            if (is_swiglu) {
              contrib::ApplySwiGLUActivation(fc1_out_row, moe_params.inter_size, true);
            } else {
              for (int64_t j = 0; j < inter_size; ++j) {
                fc1_out_row[j] = ApplyActivation(fc1_out_row[j], activation_type_);
              }
            }
          }

          // FC2
          const int64_t fc2_input_size = is_swiglu ? moe_params.inter_size : fc1_output_size;
          auto fc2_out = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(M * moe_params.hidden_size));
          const float* fc2_expert_weights = dequant_fc2_weights + expert_idx * moe_params.inter_size * moe_params.hidden_size;
          MlasGemm(CblasNoTrans, CblasTrans,
                   static_cast<size_t>(M), static_cast<size_t>(moe_params.hidden_size), static_cast<size_t>(fc2_input_size),
                   1.0f,
                   fc1_out.get(), static_cast<size_t>(fc2_input_size),
                   fc2_expert_weights, static_cast<size_t>(moe_params.hidden_size),
                   0.0f,
                   fc2_out.get(), static_cast<size_t>(moe_params.hidden_size),
                   nullptr);

          // FC2 Bias
          const float* fc2_expert_bias = fc2_bias_float_ptr ? fc2_bias_float_ptr + expert_idx * moe_params.hidden_size : nullptr;
          if (fc2_expert_bias) {
            for (int64_t i = 0; i < M; ++i) {
              float* fc2_out_row = fc2_out.get() + i * moe_params.hidden_size;
              for (int64_t j = 0; j < moe_params.hidden_size; ++j) {
                fc2_out_row[j] += fc2_expert_bias[j];
              }
            }
          }

          // Store result for this expert
          expert_results[expert_idx] = std::move(fc2_out);
        }
      });

  // Phase 3: Accumulate results in parallel over tokens (scatter-add). This is thread-safe.
  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_rows), 0.0,
      [&](std::ptrdiff_t token_start, std::ptrdiff_t token_end) {
        for (std::ptrdiff_t i = token_start; i < token_end; ++i) {
          float* token_output = output_float_ptr + i * moe_params.hidden_size;
          for (const auto& info : token_to_expert_info[i]) {
            int64_t expert_idx = info.first;
            int64_t pos_in_expert_batch = info.second;

            if (expert_results[expert_idx] == nullptr) continue;

            const float* expert_result_row = expert_results[expert_idx].get() + pos_in_expert_batch * moe_params.hidden_size;
            float routing_weight = expert_to_weight_map[expert_idx][pos_in_expert_batch];

            // Accumulate weighted result
            for (int64_t j = 0; j < moe_params.hidden_size; ++j) {
              token_output[j] += expert_result_row[j] * routing_weight;
            }
          }
        }
      });

  // Convert results back to the appropriate output type
  if constexpr (std::is_same_v<T, MLFloat16>) {
    MlasConvertFloatToHalfBuffer(output_float_ptr,
                                 reinterpret_cast<MLAS_FP16*>(output_data),
                                 static_cast<size_t>(total_output_size));
  }

  if (!is_swiglu) {
    ORT_UNUSED_PARAMETER(fc3_experts_bias_optional);
    ORT_UNUSED_PARAMETER(fc3_scales_optional);
  }

  return Status::OK();
}

template <bool UseUInt4x2>
Status QMoE::PrepackAndDequantizeWeights(OpKernelContext* context,
                                         MoEParameters& moe_params,
                                         const Tensor* fc1_experts_weights,
                                         const Tensor* fc2_experts_weights,
                                         const Tensor* fc1_scales,
                                         const Tensor* fc2_scales,
                                         bool is_swiglu) {
  auto* thread_pool = context->GetOperatorThreadPool();

  const uint8_t* fc1_weights_data = fc1_experts_weights->Data<uint8_t>();
  const uint8_t* fc2_weights_data = fc2_experts_weights->Data<uint8_t>();
  const void* fc1_scales_data_typed = fc1_scales->DataRaw();
  const void* fc2_scales_data_typed = fc2_scales->DataRaw();
  bool is_fp32_scales = fc1_scales->IsDataType<float>();

  AllocatorPtr allocator;
  ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&allocator));

  const int64_t fc1_scales_size = moe_params.num_experts * (is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size);
  const int64_t fc2_scales_size = moe_params.num_experts * moe_params.hidden_size;

  auto fc1_scales_float = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(fc1_scales_size));
  auto fc2_scales_float = IAllocator::MakeUniquePtr<float>(allocator, static_cast<size_t>(fc2_scales_size));

  if (is_fp32_scales) {
    std::memcpy(fc1_scales_float.get(), fc1_scales_data_typed, static_cast<size_t>(fc1_scales_size) * sizeof(float));
    std::memcpy(fc2_scales_float.get(), fc2_scales_data_typed, static_cast<size_t>(fc2_scales_size) * sizeof(float));
  } else {
    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc1_scales_data_typed),
                                           fc1_scales_float.get(),
                                           static_cast<size_t>(fc1_scales_size),
                                           thread_pool);
    MlasConvertHalfToFloatBufferInParallel(reinterpret_cast<const MLAS_FP16*>(fc2_scales_data_typed),
                                           fc2_scales_float.get(),
                                           static_cast<size_t>(fc2_scales_size),
                                           thread_pool);
  }

  const float* fc1_scales_data = fc1_scales_float.get();
  const float* fc2_scales_data = fc2_scales_float.get();

  const bool is_4bit = UseUInt4x2;
  const int64_t act_multiplier = is_swiglu ? 2 : 1;
  const int64_t fc1_output_size = is_swiglu ? 2 * moe_params.inter_size : moe_params.inter_size;

  const int64_t fc1_weight_stride = is_4bit ? (moe_params.hidden_size * fc1_output_size / 2) : (moe_params.hidden_size * moe_params.inter_size * act_multiplier);
  const int64_t fc2_weight_stride = is_4bit ? (moe_params.inter_size * moe_params.hidden_size / 2) : (moe_params.inter_size * moe_params.hidden_size);

  if (weights_allocator_ == nullptr) {
    AllocatorPtr temp_allocator;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&temp_allocator));
    weights_allocator_ = temp_allocator;
  }

  const size_t fc1_weights_size = static_cast<size_t>(moe_params.num_experts * moe_params.hidden_size * fc1_output_size);
  const size_t fc2_weights_size = static_cast<size_t>(moe_params.num_experts * moe_params.inter_size * moe_params.hidden_size);

  prepacked_fc1_weights_ = IAllocator::MakeUniquePtr<float>(weights_allocator_, fc1_weights_size);
  prepacked_fc2_weights_ = IAllocator::MakeUniquePtr<float>(weights_allocator_, fc2_weights_size);

  prepacked_fc1_weights_data_ = prepacked_fc1_weights_.get();
  prepacked_fc2_weights_data_ = prepacked_fc2_weights_.get();

  auto DequantizeWeight = [&](const uint8_t* weights, size_t linear_idx,
                              const float* scales, int64_t scale_idx) -> float {
    if (is_4bit) {
      size_t packed_idx = linear_idx >> 1;
      uint8_t packed_value = weights[packed_idx];
      uint8_t quantized_weight = (linear_idx & 1) ? (packed_value >> 4) : (packed_value & 0x0F);
      int8_t signed_weight = static_cast<int8_t>(quantized_weight - 8);
      return static_cast<float>(signed_weight) * scales[scale_idx];
    } else {
      int8_t signed_weight = static_cast<int8_t>(weights[linear_idx] - 128);
      return static_cast<float>(signed_weight) * scales[scale_idx];
    }
  };

  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts), 0.0,
      [&](ptrdiff_t expert_start, ptrdiff_t expert_end) {
        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const uint8_t* fc1_expert_weights = fc1_weights_data + expert_idx * fc1_weight_stride;
          const float* fc1_expert_scales = fc1_scales_data + expert_idx * fc1_output_size;
          float* dequant_fc1_expert = prepacked_fc1_weights_data_ + expert_idx * moe_params.hidden_size * fc1_output_size;

          const int64_t output_cols = fc1_output_size;
          for (int64_t in_col = 0; in_col < moe_params.hidden_size; ++in_col) {
            for (int64_t out_col = 0; out_col < output_cols; ++out_col) {
              size_t linear_idx = static_cast<size_t>(in_col * output_cols + out_col);
              size_t output_idx = static_cast<size_t>(out_col * moe_params.hidden_size + in_col);
              dequant_fc1_expert[output_idx] = DequantizeWeight(fc1_expert_weights, linear_idx, fc1_expert_scales, out_col);
            }
          }
        }
      });

  concurrency::ThreadPool::TryParallelFor(
      thread_pool, static_cast<std::ptrdiff_t>(moe_params.num_experts), 0.0,
      [&](ptrdiff_t expert_start, ptrdiff_t expert_end) {
        for (std::ptrdiff_t expert_idx = expert_start; expert_idx < expert_end; ++expert_idx) {
          const uint8_t* fc2_expert_weights = fc2_weights_data + expert_idx * fc2_weight_stride;
          const float* fc2_expert_scales = fc2_scales_data + expert_idx * moe_params.hidden_size;
          float* dequant_fc2_expert = prepacked_fc2_weights_data_ + expert_idx * moe_params.inter_size * moe_params.hidden_size;

          for (int64_t in_col = 0; in_col < moe_params.inter_size; ++in_col) {
            for (int64_t out_col = 0; out_col < moe_params.hidden_size; ++out_col) {
              size_t linear_idx = static_cast<size_t>(in_col * moe_params.hidden_size + out_col);
              size_t output_idx = static_cast<size_t>(out_col * moe_params.inter_size + in_col);
              dequant_fc2_expert[output_idx] = DequantizeWeight(fc2_expert_weights, linear_idx, fc2_expert_scales, out_col);
            }
          }
        }
      });

  cached_num_experts_ = moe_params.num_experts;
  cached_hidden_size_ = moe_params.hidden_size;
  cached_inter_size_ = moe_params.inter_size;
  cached_is_swiglu_ = is_swiglu;
  is_prepacked_ = true;

  return Status::OK();
}

// Explicit template instantiations
template Status QMoE::QuantizedMoEImpl<true, MLFloat16>(OpKernelContext* context, MoEParameters& moe_params, const Tensor* input, const Tensor* router_probs, const Tensor* fc1_experts_weights, const Tensor* fc1_experts_bias_optional, const Tensor* fc2_experts_weights, const Tensor* fc2_experts_bias_optional, const Tensor* fc3_experts_weights_optional, const Tensor* fc3_experts_bias_optional, const Tensor* fc1_scales, const Tensor* fc2_scales, const Tensor* fc3_scales_optional) const;
template Status QMoE::QuantizedMoEImpl<false, MLFloat16>(OpKernelContext* context, MoEParameters& moe_params, const Tensor* input, const Tensor* router_probs, const Tensor* fc1_experts_weights, const Tensor* fc1_experts_bias_optional, const Tensor* fc2_experts_weights, const Tensor* fc2_experts_bias_optional, const Tensor* fc3_experts_weights_optional, const Tensor* fc3_experts_bias_optional, const Tensor* fc1_scales, const Tensor* fc2_scales, const Tensor* fc3_scales_optional) const;
template Status QMoE::QuantizedMoEImpl<true, float>(OpKernelContext* context, MoEParameters& moe_params, const Tensor* input, const Tensor* router_probs, const Tensor* fc1_experts_weights, const Tensor* fc1_experts_bias_optional, const Tensor* fc2_experts_weights, const Tensor* fc2_experts_bias_optional, const Tensor* fc3_experts_weights_optional, const Tensor* fc3_experts_bias_optional, const Tensor* fc1_scales, const Tensor* fc2_scales, const Tensor* fc3_scales_optional) const;
template Status QMoE::QuantizedMoEImpl<false, float>(OpKernelContext* context, MoEParameters& moe_params, const Tensor* input, const Tensor* router_probs, const Tensor* fc1_experts_weights, const Tensor* fc1_experts_bias_optional, const Tensor* fc2_experts_weights, const Tensor* fc2_experts_bias_optional, const Tensor* fc3_experts_weights_optional, const Tensor* fc3_experts_bias_optional, const Tensor* fc1_scales, const Tensor* fc2_scales, const Tensor* fc3_scales_optional) const;

}  // namespace contrib
}  // namespace onnxruntime
