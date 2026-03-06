// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/data_transfer.h"
#include "core/framework/tensor.h"
#include "core/framework/stream_handles.h"

#include <cuda_runtime_api.h>

namespace onnxruntime {
namespace cuda_plugin {

class CudaPluginDataTransfer final : public IDataTransfer {
 public:
  bool CanCopy(const OrtDevice& src_device, const OrtDevice& dst_device) const override {
    const bool src_is_cpu = src_device.Type() == OrtDevice::CPU;
    const bool dst_is_cpu = dst_device.Type() == OrtDevice::CPU;
    const bool src_is_gpu = src_device.Type() == OrtDevice::GPU;
    const bool dst_is_gpu = dst_device.Type() == OrtDevice::GPU;

    return (src_is_cpu && dst_is_gpu) ||
           (src_is_gpu && dst_is_cpu) ||
           (src_is_gpu && dst_is_gpu);
  }

  common::Status CopyTensor(const Tensor& src, Tensor& dst) const override {
    return CopyTensorImpl(src, dst, nullptr);
  }

  common::Status CopyTensorAsync(const Tensor& src, Tensor& dst, Stream& stream) const override {
    return CopyTensorImpl(src, dst, static_cast<cudaStream_t>(stream.GetHandle()));
  }

 private:
  common::Status CopyTensorImpl(const Tensor& src, Tensor& dst, cudaStream_t stream) const {
    if (src.Shape().Size() != dst.Shape().Size()) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "Tensor size mismatch: source tensor size is ", src.Shape().Size(),
                             ", destination tensor size is ", dst.Shape().Size());
    }

    const size_t bytes = src.SizeInBytes();
    if (bytes == 0) {
      return Status::OK();
    }

    cudaMemcpyKind copy_kind;
    const bool src_is_cpu = src.Location().device.Type() == OrtDevice::CPU;
    const bool dst_is_cpu = dst.Location().device.Type() == OrtDevice::CPU;
    const bool src_is_gpu = src.Location().device.Type() == OrtDevice::GPU;
    const bool dst_is_gpu = dst.Location().device.Type() == OrtDevice::GPU;

    if (src_is_cpu && dst_is_gpu) {
      copy_kind = cudaMemcpyHostToDevice;
    } else if (src_is_gpu && dst_is_cpu) {
      copy_kind = cudaMemcpyDeviceToHost;
    } else if (src_is_gpu && dst_is_gpu) {
      copy_kind = cudaMemcpyDeviceToDevice;
    } else {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported copy direction");
    }

    const auto error = stream != nullptr
                           ? cudaMemcpyAsync(dst.MutableDataRaw(), src.DataRaw(), bytes, copy_kind, stream)
                           : cudaMemcpy(dst.MutableDataRaw(), src.DataRaw(), bytes, copy_kind);

    if (error != cudaSuccess) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL,
                             "CUDA memcpy failed: ", cudaGetErrorName(error), ": ", cudaGetErrorString(error));
    }

    return Status::OK();
  }
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
