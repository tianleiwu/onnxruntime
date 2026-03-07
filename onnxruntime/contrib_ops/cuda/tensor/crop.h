// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/providers/cuda/cuda_kernel.h"

#ifdef BUILD_CUDA_EP_AS_PLUGIN

// Plugin build: inline CropBase that works with the adapter's OpKernelInfo
// (the CPU contrib_ops/cpu/crop.h CropBase uses framework OpKernelInfo which
// is a different type from ep::adapter::OpKernelInfo in the plugin build).
namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

class CropBase {
 protected:
  CropBase(const OpKernelInfo& info)
      : border_(info.GetAttrsOrDefault<int64_t>("border")),
        scale_(info.GetAttrsOrDefault<int64_t>("scale")) {}

  Status ValidateInput(const Tensor* X) const {
    if (border_.size() != 4)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Attribute border needs 4 elements, got ", border_.size());
    const auto& dims = X->Shape().GetDims();
    if (dims.size() != 4)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "Input expected 4 dimensions [N,C,H,W], got ", dims.size());
    return Status::OK();
  }

  const std::vector<int64_t> border_;
  const std::vector<int64_t> scale_;
};

template <typename T>
class Crop final : public CropBase, public CudaKernel {
 public:
  Crop(const OpKernelInfo& info) : CropBase(info), CudaKernel(info) {}
  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#else  // !BUILD_CUDA_EP_AS_PLUGIN

#include "contrib_ops/cpu/crop.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

using namespace onnxruntime::cuda;

template <typename T>
class Crop final : public contrib::CropBase, public CudaKernel {
 public:
  Crop(const OpKernelInfo& info) : contrib::CropBase(info), CudaKernel(info) {
  }

  Status ComputeInternal(OpKernelContext* context) const override;
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#endif  // BUILD_CUDA_EP_AS_PLUGIN
