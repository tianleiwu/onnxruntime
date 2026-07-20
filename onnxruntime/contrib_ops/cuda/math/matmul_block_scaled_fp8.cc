// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

#if !defined(DISABLE_FLOAT8_TYPES)
#define MATMUL_BLOCK_SCALED_FP8_ACTIVATION_CONSTRAINTS BuildKernelDefConstraints<Float8E4M3FN, MLFloat16>()

ONNX_OPERATOR_KERNEL_EX(
    MatMulBlockScaledFp8,
    kMSDomain,
    1,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .TypeConstraint("TA", MATMUL_BLOCK_SCALED_FP8_ACTIVATION_CONSTRAINTS)
        .TypeConstraint("TB", BuildKernelDefConstraints<Float8E4M3FN>())
        .TypeConstraint("TS", BuildKernelDefConstraints<float, MLFloat16>())
        .TypeConstraint("TY", BuildKernelDefConstraints<BFloat16, MLFloat16>()),
    MatMulBlockScaledFp8);
#endif

MatMulBlockScaledFp8::MatMulBlockScaledFp8(const OpKernelInfo& info)
    : CudaKernel(info), block_size_(info.GetAttrOrDefault<int64_t>("block_size", 128)) {
  ORT_ENFORCE(block_size_ > 0, "block_size must be positive.");
  sm_ = GetDeviceProp().major * 10 + GetDeviceProp().minor;
}

Status MatMulBlockScaledFp8::ComputeInternal(OpKernelContext* context) const {
#if defined(DISABLE_FLOAT8_TYPES)
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockScaledFp8 requires float8 support.");
#else
  const Tensor* input_a = context->Input<Tensor>(0);
  const Tensor* input_b = context->Input<Tensor>(1);
  const Tensor* scale_a = context->Input<Tensor>(2);
  const Tensor* scale_b = context->Input<Tensor>(3);

  ORT_ENFORCE(input_a->Shape().NumDimensions() >= 1, "A must have rank at least 1.");
  ORT_ENFORCE(input_b->Shape().NumDimensions() == 2, "B must have rank 2.");
  ORT_ENFORCE(scale_a->Shape().NumDimensions() == 2, "scaleA must have rank 2.");
  ORT_ENFORCE(scale_b->Shape().NumDimensions() == 2, "scaleB must have rank 2.");

  const auto& a_shape = input_a->Shape();
  const auto& b_shape = input_b->Shape();
  const int64_t a_rank = a_shape.NumDimensions();
  ORT_ENFORCE(a_shape[a_rank - 1] == b_shape[1], "A and B have incompatible K dimensions.");

  const int64_t m = a_shape.SizeToDimension(a_rank - 1);
  const int64_t k = a_shape[a_rank - 1];
  const int64_t n = b_shape[0];
  const int64_t k_blocks = (k + block_size_ - 1) / block_size_;
  ORT_ENFORCE(scale_a->Shape() == TensorShape({m, k_blocks}),
              "scaleA must have shape [M, ceil(K / block_size)].");
  ORT_ENFORCE(scale_b->Shape() == TensorShape({n, k_blocks}),
              "scaleB must have shape [N, ceil(K / block_size)].");

  TensorShapeVector output_shape = a_shape.AsShapeVector();
  output_shape.back() = n;
  Tensor* output = context->Output(0, TensorShape(output_shape));
  const bool fp16_io = input_a->IsDataType<MLFloat16>();
  const bool fp16_scales = scale_a->IsDataType<MLFloat16>();
  ORT_ENFORCE(fp16_io == output->IsDataType<MLFloat16>(),
              "MatMulBlockScaledFp8 supports FP8 A with BF16 Y or FP16 A with FP16 Y.");
  ORT_ENFORCE(fp16_scales == scale_b->IsDataType<MLFloat16>(),
              "scaleA and scaleB must have the same element type.");

  const int m_i = gsl::narrow_cast<int>(m);
  const int n_i = gsl::narrow_cast<int>(n);
  const int k_i = gsl::narrow_cast<int>(k);

#if defined(ORT_ENABLE_BLOCKQUANT_SM90) || defined(ORT_ENABLE_BLOCKQUANT_SM100)
  // Fast path: CUTLASS tensor-core blockwise-scaled FP8 GEMM on Hopper / Blackwell.
  // Restricted to the FP8 A -> BF16 Y case with block_size 128 and K/N aligned to 16 (fp8 128-bit access).
  if (!fp16_io && output->IsDataType<BFloat16>() && block_size_ == 128 &&
      (k % 16 == 0) && (n % 16 == 0)) {
    const float* sfa = nullptr;
    const float* sfb = nullptr;
    IAllocatorUniquePtr<float> sfa_fp32;
    IAllocatorUniquePtr<float> sfb_fp32;
    if (fp16_scales) {
      const size_t a_scale_count = static_cast<size_t>(m) * static_cast<size_t>(k_blocks);
      const size_t b_scale_count = static_cast<size_t>(k_blocks) * static_cast<size_t>(n);
      sfa_fp32 = GetScratchBuffer<float>(a_scale_count, context->GetComputeStream());
      sfb_fp32 = GetScratchBuffer<float>(b_scale_count, context->GetComputeStream());
      LaunchConvertHalfToFloat(scale_a->DataRaw(), sfa_fp32.get(),
                               static_cast<int64_t>(a_scale_count), Stream(context));
      LaunchConvertHalfToFloat(scale_b->DataRaw(), sfb_fp32.get(),
                               static_cast<int64_t>(b_scale_count), Stream(context));
      sfa = sfa_fp32.get();
      sfb = sfb_fp32.get();
    } else {
      sfa = scale_a->Data<float>();
      sfb = scale_b->Data<float>();
    }

    bool handled = false;
    Status fast_status = Status::OK();
#if defined(ORT_ENABLE_BLOCKQUANT_SM90)
    if (sm_ == 90) {
      const size_t ws = GetBlockQuantizedFp8GemmSm90WorkspaceSize(m_i, n_i, k_i);
      auto ws_buf = GetScratchBuffer<uint8_t>(ws, context->GetComputeStream());
      fast_status = LaunchBlockQuantizedFp8GemmSm90(
          input_a->DataRaw(), input_b->DataRaw(), sfa, sfb, output->MutableDataRaw(),
          m_i, n_i, k_i, gsl::narrow_cast<int>(block_size_), ws_buf.get(), ws, Stream(context));
      handled = true;
    }
#endif
#if defined(ORT_ENABLE_BLOCKQUANT_SM100)
    if (!handled && sm_ >= 100) {
      const size_t ws = GetBlockQuantizedFp8GemmSm100WorkspaceSize(m_i, n_i, k_i);
      auto ws_buf = GetScratchBuffer<uint8_t>(ws, context->GetComputeStream());
      fast_status = LaunchBlockQuantizedFp8GemmSm100(
          input_a->DataRaw(), input_b->DataRaw(), sfa, sfb, output->MutableDataRaw(),
          m_i, n_i, k_i, gsl::narrow_cast<int>(block_size_), ws_buf.get(), ws, Stream(context));
      handled = true;
    }
#endif
    if (handled) {
      return fast_status;
    }
  }
#endif

  return LaunchMatMulBlockScaledFp8(input_a->DataRaw(), input_b->DataRaw(),
                                    scale_a->DataRaw(), scale_b->DataRaw(),
                                    output->MutableDataRaw(), m_i,
                                    n_i, k_i,
                                    gsl::narrow_cast<int>(block_size_), fp16_io, fp16_scales, Stream(context));
#endif
}

}  // namespace onnxruntime::contrib::cuda