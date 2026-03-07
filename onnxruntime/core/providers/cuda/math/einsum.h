// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/platform/threadpool.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/cuda_execution_provider.h"
#ifdef BUILD_CUDA_EP_AS_PLUGIN
#include "core/providers/cpu/math/einsum_utils/einsum_typed_compute_processor.h"
#include "core/providers/cuda/cuda_kernel.h"
#else
#include "core/providers/cpu/math/einsum.h"
#endif
#include "einsum_utils/einsum_auxiliary_ops.h"

namespace onnxruntime {
namespace cuda {

#ifdef BUILD_CUDA_EP_AS_PLUGIN

class Einsum final : public CudaKernel {
 public:
  explicit Einsum(const OpKernelInfo& info) : CudaKernel(info) {
    ORT_ENFORCE(info.GetAttr<std::string>("equation", &equation_).IsOK(),
                "Missing 'equation' attribute");
    einsum_equation_preprocessor_ = std::make_unique<EinsumEquationPreprocessor>(equation_);
    cuda_ep_ = static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider());
  }

  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  Status DeviceCompute(OpKernelContext* context, const std::vector<const Tensor*>& inputs,
                       AllocatorPtr allocator, concurrency::ThreadPool* tp) const;

  std::string equation_;
  std::unique_ptr<EinsumEquationPreprocessor> einsum_equation_preprocessor_;
  const CUDAExecutionProvider* cuda_ep_;
};

#else

class Einsum final : public onnxruntime::Einsum {
 public:
  Einsum(const OpKernelInfo& info) : onnxruntime::Einsum(info) {
    // We need to cast away the const as PerThreadCublasHandle() is currently a non-const method
    // TODO: Clean up the CUDAExecutionProvider interface to avoid this
    cuda_ep_ = static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider());
  }

  Status Compute(OpKernelContext* context) const override;

 private:
  Status DeviceCompute(OpKernelContext* context, const std::vector<const Tensor*>& inputs,
                       AllocatorPtr allocator, concurrency::ThreadPool* tp) const override;

  // Members of Einsum CUDA kernel
  using onnxruntime::Einsum::einsum_equation_preprocessor_;
  using onnxruntime::Einsum::equation_;

  // We need to access to the CUDA EP instance to get the cublas/cudnn handles
  const CUDAExecutionProvider* cuda_ep_;
};

#endif  // BUILD_CUDA_EP_AS_PLUGIN

}  // namespace cuda
}  // namespace onnxruntime
