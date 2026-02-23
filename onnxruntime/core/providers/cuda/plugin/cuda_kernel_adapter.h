// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include "core/common/status.h"
#include "core/framework/allocator.h"
#include "core/framework/tensor_shape.h"
#include "core/providers/cuda/plugin/cuda_stream_plugin.h"
#include "core/session/onnxruntime_cxx_api.h"

#include <atomic>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace onnxruntime {
namespace cuda {

namespace detail {

struct CudaKernelAdapterRuntimeConfig {
  std::atomic<bool> use_tf32{true};
  std::atomic<int> device_id{0};
};

inline CudaKernelAdapterRuntimeConfig& GetCudaKernelAdapterRuntimeConfig() {
  static CudaKernelAdapterRuntimeConfig config;
  return config;
}

inline size_t BytesForCount(size_t count_or_bytes, size_t element_size) {
  if (element_size == 0) {
    return count_or_bytes;
  }

  if (count_or_bytes > (std::numeric_limits<size_t>::max() / element_size)) {
    return 0;
  }

  return count_or_bytes * element_size;
}

template <typename T>
inline T OneValue() {
  return static_cast<T>(1);
}

template <>
inline half OneValue<half>() {
  return __float2half(1.0f);
}

template <typename T>
struct ConstOnesState {
  std::mutex mutex;
  std::vector<T*> buffers;
  T* largest_buffer = nullptr;
  size_t largest_count = 0;

  ~ConstOnesState() {
    for (T* p : buffers) {
      if (p != nullptr) {
        cudaFree(p);
      }
    }
  }
};

template <typename T>
inline ConstOnesState<T>& GetConstOnesState() {
  static ConstOnesState<T> state;
  return state;
}

}  // namespace detail

inline void SetCudaKernelAdapterRuntimeConfig(bool use_tf32, int device_id) {
  auto& config = detail::GetCudaKernelAdapterRuntimeConfig();
  config.use_tf32.store(use_tf32, std::memory_order_relaxed);
  config.device_id.store(device_id, std::memory_order_relaxed);
}

class Tensor {
 public:
  explicit Tensor(Ort::ConstValue value) : const_value_(std::move(value)), is_mutable_(false) {
    auto info = const_value_.GetTensorTypeAndShapeInfo();
    shape_ = TensorShape(info.GetShape());
    element_type_ = info.GetElementType();
  }

  explicit Tensor(Ort::UnownedValue value) : unowned_value_(std::move(value)), is_mutable_(true) {
    auto info = unowned_value_.GetTensorTypeAndShapeInfo();
    shape_ = TensorShape(info.GetShape());
    element_type_ = info.GetElementType();
  }

  const TensorShape& Shape() const { return shape_; }
  int32_t GetElementType() const { return static_cast<int32_t>(element_type_); }

  template <typename T>
  const T* Data() const {
    return is_mutable_ ? unowned_value_.GetTensorData<T>() : const_value_.GetTensorData<T>();
  }

  template <typename T>
  T* MutableData() {
    if (!is_mutable_) {
      throw std::runtime_error("Attempted MutableData() on a read-only tensor");
    }
    return unowned_value_.GetTensorMutableData<T>();
  }

  const void* DataRaw() const {
    return is_mutable_ ? unowned_value_.GetTensorRawData() : const_value_.GetTensorRawData();
  }

 private:
  Ort::ConstValue const_value_{nullptr};
  Ort::UnownedValue unowned_value_{nullptr};
  bool is_mutable_ = false;
  TensorShape shape_;
  ONNXTensorElementDataType element_type_ = ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
};

class OpKernelInfo {
 public:
  explicit OpKernelInfo(const OrtKernelInfo* info) : info_(info) {}

  template <typename T>
  Status GetAttr(const std::string& name, T* value) const {
    if (value == nullptr) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetAttr output pointer must be non-null");
    }
    try {
      *value = info_.GetAttribute<T>(name.c_str());
      return Status::OK();
    } catch (const Ort::Exception& ex) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "GetAttr failed for '", name, "': ", ex.what());
    }
  }

  template <typename T>
  T GetAttrOrDefault(const std::string& name, T default_value) const {
    T value{};
    return GetAttr(name, &value).IsOK() ? value : default_value;
  }

  struct NodeInfo {
    int since_version = 0;
    int SinceVersion() const { return since_version; }
  };

  NodeInfo node() const {
    return NodeInfo{info_.GetOperatorSinceVersion()};
  }

 private:
  Ort::ConstKernelInfo info_;
};

class OpKernelContext {
 public:
  explicit OpKernelContext(OrtKernelContext* context) : context_(context) {}

  template <typename T>
  const T* Input(int index) {
    static_assert(std::is_same_v<T, Tensor>, "Plugin adapter currently supports Input<Tensor>() only");

    Ort::ConstValue value = context_.GetInput(static_cast<size_t>(index));
    if (!value) {
      return nullptr;
    }

    inputs_.push_back(std::make_unique<Tensor>(std::move(value)));
    return static_cast<const T*>(inputs_.back().get());
  }

  Tensor* Output(int index, const TensorShape& shape) {
    const auto& dims = shape.GetDims();
    Ort::UnownedValue value = context_.GetOutput(static_cast<size_t>(index), dims.data(), dims.size());
    outputs_.push_back(std::make_unique<Tensor>(std::move(value)));
    return outputs_.back().get();
  }

  int InputCount() const { return static_cast<int>(context_.GetInputCount()); }
  int OutputCount() const { return static_cast<int>(context_.GetOutputCount()); }

  void* GetComputeStream() const { return context_.GetGPUComputeStream(); }

 private:
  Ort::KernelContext context_;
  std::vector<std::unique_ptr<Tensor>> inputs_;
  std::vector<std::unique_ptr<Tensor>> outputs_;
};

class CudaKernel {
 public:
  explicit CudaKernel(const OpKernelInfo& info) : info_(info) {
    const auto& runtime_config = detail::GetCudaKernelAdapterRuntimeConfig();
    use_tf32_ = runtime_config.use_tf32.load(std::memory_order_relaxed);
    device_id_ = runtime_config.device_id.load(std::memory_order_relaxed);

    int current_device = device_id_;
    if (cudaGetDevice(&current_device) == cudaSuccess) {
      device_id_ = current_device;
    }

    if (cudaGetDeviceProperties(&device_prop_, device_id_) != cudaSuccess) {
      std::memset(&device_prop_, 0, sizeof(device_prop_));
      device_prop_.major = -1;
      device_prop_.minor = -1;
    }
  }
  virtual ~CudaKernel() = default;

  Status Compute(OpKernelContext* context) const {
    Status s = ComputeInternal(context);
    if (s.IsOK()) {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "CUDA error ", cudaGetErrorName(err), ":", cudaGetErrorString(err));
      }
    }

    return s;
  }

  virtual Status ComputeInternal(OpKernelContext* context) const = 0;

  cudaStream_t Stream(OpKernelContext* context) const {
    return context ? static_cast<cudaStream_t>(context->GetComputeStream()) : nullptr;
  }

  cudnnHandle_t GetCudnnHandle(OpKernelContext* context) const {
    cudaStream_t stream = Stream(context);
    auto* sync_stream = cuda_plugin::CudaSyncStream::FromCudaStream(stream);
    return sync_stream ? sync_stream->GetCudnnHandle() : nullptr;
  }

  cublasHandle_t GetCublasHandle(OpKernelContext* context) const {
    cudaStream_t stream = Stream(context);
    auto* sync_stream = cuda_plugin::CudaSyncStream::FromCudaStream(stream);
    return sync_stream ? sync_stream->GetCublasHandle() : nullptr;
  }

  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t count_or_bytes, void* stream) const {
    if (count_or_bytes == 0) {
      return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    }

    constexpr size_t kElementSize = std::is_void_v<T> ? 0 : sizeof(T);
    const size_t bytes = detail::BytesForCount(count_or_bytes, kElementSize);
    if (bytes == 0) {
      return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    }

    void* p = nullptr;
    cudaError_t alloc_err = cudaMalloc(&p, bytes);
    if (alloc_err != cudaSuccess) {
      return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    }

    const auto cuda_stream = static_cast<cudaStream_t>(stream);
    return IAllocatorUniquePtr<T>(
        static_cast<T*>(p),
        [cuda_stream](T* ptr) {
          if (ptr == nullptr) {
            return;
          }
#if CUDART_VERSION >= 11020
          if (cuda_stream != nullptr) {
            cudaFreeAsync(ptr, cuda_stream);
            return;
          }
#endif
          cudaFree(ptr);
        });
  }

  inline void AddDeferredReleaseCPUPtr(void* p, void* stream) const {
    if (p == nullptr) {
      return;
    }

    auto* sync_stream =
        cuda_plugin::CudaSyncStream::FromCudaStream(static_cast<cudaStream_t>(stream));
    if (sync_stream != nullptr) {
      sync_stream->EnqueueDeferredCPUBuffer(p);
      return;
    }

    // Fallback: if no tracked stream exists, release pinned host memory immediately.
    cudaFreeHost(p);
  }

  template <typename T>
  inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t count_or_bytes) const {
    if (count_or_bytes == 0) {
      return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    }

    constexpr size_t kElementSize = std::is_void_v<T> ? 0 : sizeof(T);
    const size_t bytes = detail::BytesForCount(count_or_bytes, kElementSize);
    if (bytes == 0) {
      return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    }

    void* p = nullptr;
    if (cudaHostAlloc(&p, bytes, cudaHostAllocDefault) != cudaSuccess) {
      return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    }

    return IAllocatorUniquePtr<T>(
        static_cast<T*>(p),
        [](T* ptr) {
          if (ptr != nullptr) {
            cudaFreeHost(ptr);
          }
        });
  }

  const cudaDeviceProp& GetDeviceProp() const { return device_prop_; }
  bool UseTF32() const { return use_tf32_; }
  bool IsArchAvailable(int arch) const { return device_prop_.major >= arch; }

  const OpKernelInfo& Info() const { return info_; }

 protected:
  template <typename T>
  inline const T* GetConstOnes(size_t count, cudaStream_t stream) const {
    if (count == 0) {
      return nullptr;
    }

    auto& state = detail::GetConstOnesState<T>();
    std::lock_guard<std::mutex> lock(state.mutex);

    if (count > state.largest_count) {
      T* device_ptr = nullptr;
      const size_t bytes = detail::BytesForCount(count, sizeof(T));
      if (bytes == 0) {
        return nullptr;
      }

      if (cudaMalloc(&device_ptr, bytes) != cudaSuccess || device_ptr == nullptr) {
        return nullptr;
      }

      std::vector<T> host_ones(count, detail::OneValue<T>());
      if (stream != nullptr) {
        if (cudaMemcpyAsync(device_ptr, host_ones.data(), bytes, cudaMemcpyHostToDevice, stream) != cudaSuccess) {
          cudaFree(device_ptr);
          return nullptr;
        }
      } else {
        if (cudaMemcpy(device_ptr, host_ones.data(), bytes, cudaMemcpyHostToDevice) != cudaSuccess) {
          cudaFree(device_ptr);
          return nullptr;
        }
      }

      state.buffers.push_back(device_ptr);
      state.largest_buffer = device_ptr;
      state.largest_count = count;
    }

    return state.largest_buffer;
  }

 private:
  OpKernelInfo info_;
  cudaDeviceProp device_prop_{};
  bool use_tf32_ = true;
  int device_id_ = 0;
};

}  // namespace cuda

#undef ONNX_OPERATOR_KERNEL_EX
#define ONNX_OPERATOR_KERNEL_EX(...)

#undef ONNX_OPERATOR_VERSIONED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(...)

#undef ONNX_OPERATOR_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TYPED_KERNEL_EX(...)

#undef ONNX_OPERATOR_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TWO_TYPED_KERNEL_EX(...)

#undef ONNX_OPERATOR_THREE_TYPED_KERNEL_EX
#define ONNX_OPERATOR_THREE_TYPED_KERNEL_EX(...)

#undef ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(...)

#undef ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX
// These macro guards are necessary because when building as a plugin, we may include
// framework headers that also define these macros. We use guards or undefs to ensure
// the plugin's simplified versions are used without causing redefinition errors.
#ifndef ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TWO_TYPED_KERNEL_EX(...)
#endif

}  // namespace onnxruntime
