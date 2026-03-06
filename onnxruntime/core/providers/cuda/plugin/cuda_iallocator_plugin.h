// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/framework/allocator.h"

#include <cuda_runtime_api.h>

namespace onnxruntime {
namespace cuda_plugin {

class CudaDeviceIAllocator : public IAllocator {
 public:
  explicit CudaDeviceIAllocator(OrtDevice::DeviceId device_id)
      : IAllocator(OrtMemoryInfo(CUDA, OrtAllocatorType::OrtDeviceAllocator,
                                 OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                           OrtDevice::VendorIds::NVIDIA, device_id),
                                 OrtMemTypeDefault)),
        device_id_(device_id) {
  }

  void* Alloc(size_t size) override {
    if (size == 0) {
      return nullptr;
    }

    SetDevice();

    void* pointer = nullptr;
    const auto error = cudaMalloc(&pointer, size);
    if (error != cudaSuccess) {
      ORT_THROW("cudaMalloc failed: ", cudaGetErrorName(error), ": ", cudaGetErrorString(error));
    }
    return pointer;
  }

  void Free(void* p) override {
    if (p == nullptr) {
      return;
    }

    SetDevice();
    static_cast<void>(cudaFree(p));
  }

 private:
  void SetDevice() const {
    static_cast<void>(cudaSetDevice(device_id_));
  }

  OrtDevice::DeviceId device_id_;
};

class CudaPinnedIAllocator : public IAllocator {
 public:
  explicit CudaPinnedIAllocator(OrtDevice::DeviceId device_id)
      : IAllocator(OrtMemoryInfo(CUDA_PINNED, OrtAllocatorType::OrtDeviceAllocator,
                                 OrtDevice(OrtDevice::GPU, OrtDevice::MemType::HOST_ACCESSIBLE,
                                           OrtDevice::VendorIds::NVIDIA, device_id),
                                 OrtMemTypeCPUOutput)) {
  }

  void* Alloc(size_t size) override {
    if (size == 0) {
      return nullptr;
    }

    void* pointer = nullptr;
    const auto error = cudaHostAlloc(&pointer, size, cudaHostAllocDefault);
    if (error != cudaSuccess) {
      ORT_THROW("cudaHostAlloc failed: ", cudaGetErrorName(error), ": ", cudaGetErrorString(error));
    }
    return pointer;
  }

  void Free(void* p) override {
    if (p == nullptr) {
      return;
    }

    static_cast<void>(cudaFreeHost(p));
  }
};

}  // namespace cuda_plugin
}  // namespace onnxruntime
