// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "cuda_plugin_utils.h"

#include "core/framework/data_transfer.h"
#include "core/framework/execution_provider.h"
#include "core/framework/tuning_context.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"

#include <atomic>
#include <memory>
#include <mutex>
#include <unordered_map>

namespace onnxruntime {

namespace cuda_plugin {
class CudaSyncStream;
}

class CUDAExecutionProvider : public IExecutionProvider {
 public:
  CUDAExecutionProvider(int device_id,
                        bool use_tf32,
                        int cudnn_conv_algo,
                        bool cudnn_conv1d_pad_to_nc1d,
                        bool cudnn_conv_use_max_workspace,
                        bool skip_layer_norm_strict_mode,
                        bool prefer_nhwc);
  ~CUDAExecutionProvider() override;

  int GetDeviceId() const override { return device_id_; }
  const cudaDeviceProp& GetDeviceProp() const { return device_prop_; }
  bool UseTF32() const { return use_tf32_; }
  int GetCudnnConvAlgo() const { return cudnn_conv_algo_; }
  bool GetCudnnConv1dPadToNc1d() const { return cudnn_conv1d_pad_to_nc1d_; }
  bool GetCudnnConvUseMaxWorkspace() const { return cudnn_conv_use_max_workspace_; }
  bool IsSkipLayerNormInStrictMode() const { return skip_layer_norm_strict_mode_; }
  bool IsNHWCPreferred() const { return prefer_nhwc_; }
  bool DoCopyOnDefaultStream() const { return true; }
  bool IsFuseConvBias() const { return false; }

  void RegisterStream(cuda_plugin::CudaSyncStream& stream);
  void UnregisterStream(cudaStream_t stream);
  cuda_plugin::CudaSyncStream* GetSyncStream(cudaStream_t stream) const;
  cuda_plugin::CudaSyncStream* GetComputeSyncStream() const;
  cublasHandle_t GetCublasHandle(cudaStream_t stream) const;
  cudnnHandle_t GetCudnnHandle(cudaStream_t stream) const;
  cublasHandle_t PerThreadDefaultCublasHandle() const;
  cudnnHandle_t PerThreadDefaultCudnnHandle() const;
  cublasLtHandle_t PerThreadCublasLtHandle() const;
  cudaStream_t ComputeStream() const;

  std::unique_ptr<IDataTransfer> GetDataTransfer() const override;
  const AttentionKernelOptions* GetAttentionKernelOptions() const;
  ITuningContext* GetTuningContext() const;

  static const CUDAExecutionProvider* GetActiveProvider() noexcept;

  template <typename T>
  const T* GetConstOnes(size_t count, cudaStream_t stream) const {
    static std::unique_ptr<cuda::IConstantBuffer<T>> buf;
    static std::once_flag flag;
    std::call_once(flag, []() { buf = cuda::CreateConstantOnes<T>(); });
    return buf->GetBuffer(stream, count);
  }

 private:
  int device_id_;
  cudaDeviceProp device_prop_{};
  bool use_tf32_;
  int cudnn_conv_algo_;
  bool cudnn_conv1d_pad_to_nc1d_;
  bool cudnn_conv_use_max_workspace_;
  bool skip_layer_norm_strict_mode_;
  bool prefer_nhwc_;

  mutable std::mutex stream_mutex_;
  std::unordered_map<cudaStream_t, cuda_plugin::CudaSyncStream*> streams_;
  cuda_plugin::CudaSyncStream* compute_stream_ = nullptr;

  static std::atomic<CUDAExecutionProvider*> active_provider_;
};

using CudaEpProvider = CUDAExecutionProvider;

}  // namespace onnxruntime
