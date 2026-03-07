// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_ep_provider.h"

#include "cuda_idata_transfer_plugin.h"
#include "cuda_stream_plugin.h"

#include <cstring>

namespace onnxruntime {

std::atomic<CUDAExecutionProvider*> CUDAExecutionProvider::active_provider_{nullptr};

CUDAExecutionProvider::CUDAExecutionProvider(int device_id,
                                             bool use_tf32,
                                             int cudnn_conv_algo,
                                             bool cudnn_conv1d_pad_to_nc1d,
                                             bool cudnn_conv_use_max_workspace,
                                             bool skip_layer_norm_strict_mode,
                                             bool prefer_nhwc)
    : IExecutionProvider{kCudaExecutionProvider,
                         OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                   OrtDevice::VendorIds::NVIDIA, device_id)},
      device_id_{device_id},
      use_tf32_{use_tf32},
      cudnn_conv_algo_{cudnn_conv_algo},
      cudnn_conv1d_pad_to_nc1d_{cudnn_conv1d_pad_to_nc1d},
      cudnn_conv_use_max_workspace_{cudnn_conv_use_max_workspace},
      skip_layer_norm_strict_mode_{skip_layer_norm_strict_mode},
      prefer_nhwc_{prefer_nhwc} {
  if (cudaGetDeviceProperties(&device_prop_, device_id_) != cudaSuccess) {
    std::memset(&device_prop_, 0, sizeof(device_prop_));
    device_prop_.major = -1;
  }

  active_provider_.store(this, std::memory_order_release);
}

CUDAExecutionProvider::~CUDAExecutionProvider() {
  CUDAExecutionProvider* expected = this;
  active_provider_.compare_exchange_strong(expected, nullptr, std::memory_order_acq_rel);
}

void CUDAExecutionProvider::RegisterStream(cuda_plugin::CudaSyncStream& stream) {
  std::lock_guard<std::mutex> lock(stream_mutex_);
  streams_[stream.GetCudaStream()] = &stream;
  if (compute_stream_ == nullptr) {
    compute_stream_ = &stream;
  }
}

void CUDAExecutionProvider::UnregisterStream(cudaStream_t stream) {
  std::lock_guard<std::mutex> lock(stream_mutex_);
  auto it = streams_.find(stream);
  if (it == streams_.end()) {
    return;
  }

  if (compute_stream_ == it->second) {
    compute_stream_ = nullptr;
  }

  streams_.erase(it);
  if (compute_stream_ == nullptr && !streams_.empty()) {
    compute_stream_ = streams_.begin()->second;
  }
}

cuda_plugin::CudaSyncStream* CUDAExecutionProvider::GetSyncStream(cudaStream_t stream) const {
  std::lock_guard<std::mutex> lock(stream_mutex_);
  auto it = streams_.find(stream);
  return it != streams_.end() ? it->second : nullptr;
}

cuda_plugin::CudaSyncStream* CUDAExecutionProvider::GetComputeSyncStream() const {
  std::lock_guard<std::mutex> lock(stream_mutex_);
  return compute_stream_;
}

cublasHandle_t CUDAExecutionProvider::GetCublasHandle(cudaStream_t stream) const {
  auto* sync_stream = GetSyncStream(stream);
  return sync_stream ? sync_stream->GetCublasHandle() : nullptr;
}

cudnnHandle_t CUDAExecutionProvider::GetCudnnHandle(cudaStream_t stream) const {
  auto* sync_stream = GetSyncStream(stream);
  return sync_stream ? sync_stream->GetCudnnHandle() : nullptr;
}

cublasHandle_t CUDAExecutionProvider::PerThreadDefaultCublasHandle() const {
  auto* stream = GetComputeSyncStream();
  return stream ? stream->GetCublasHandle() : nullptr;
}

cudnnHandle_t CUDAExecutionProvider::PerThreadDefaultCudnnHandle() const {
  auto* stream = GetComputeSyncStream();
  return stream ? stream->GetCudnnHandle() : nullptr;
}

cublasLtHandle_t CUDAExecutionProvider::PerThreadCublasLtHandle() const {
  auto* stream = GetComputeSyncStream();
  return stream ? stream->GetCublasLtHandle() : nullptr;
}

cudaStream_t CUDAExecutionProvider::ComputeStream() const {
  auto* stream = GetComputeSyncStream();
  return stream ? stream->GetCudaStream() : nullptr;
}

std::unique_ptr<IDataTransfer> CUDAExecutionProvider::GetDataTransfer() const {
  return std::make_unique<cuda_plugin::CudaPluginDataTransfer>();
}

const AttentionKernelOptions* CUDAExecutionProvider::GetAttentionKernelOptions() const {
  static AttentionKernelOptions options;
  return &options;
}

ITuningContext* CUDAExecutionProvider::GetTuningContext() const {
  return nullptr;
}

const CUDAExecutionProvider* CUDAExecutionProvider::GetActiveProvider() noexcept {
  return active_provider_.load(std::memory_order_acquire);
}

}  // namespace onnxruntime
