// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// cuda_kernel_adapter.h — Compatibility shim for migrating CUDA kernels to the
// plugin EP architecture.
//
// This header provides:
//   - CudaKernel base class (scratch buffers, CUDA handles, etc.)
//   - Error-return macros (CUDA_RETURN_IF_ERROR, etc.)
//   - Type mapping helpers (ToCudaType)
//   - Math/compute shims (HalfGemmOptions, CublasMathModeSetter)
//   - Self-registering BuildKernelCreateInfo<> macros via PluginKernelCollector
//   - CUDAExecutionProvider shim class
//   - CPU provider shims for the plugin build

#pragma once

#include "core/common/status.h"
#include "core/common/narrow.h"
#include "core/common/float16.h"
#include "core/common/float8.h"
#include "core/framework/float4.h"
#include "core/framework/allocator.h"
#include "core/framework/data_transfer.h"
#include "core/framework/tuning_context.h"
#include "core/framework/tensor_shape.h"
#include "core/util/math.h"
#include <gsl/gsl>

#include <cublas_v2.h>
#include <cudnn.h>
#include "core/providers/cuda/plugin/cuda_ep_provider.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include "core/providers/cuda/shared_inc/fast_divmod.h"
#include "contrib_ops/cuda/bert/attention_kernel_options.h"

#ifdef __CUDACC__
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#endif

// ===================================================================
// Macros will be defined later to override core definitions.
// ===================================================================

#include "core/providers/cuda/plugin/cuda_stream_plugin.h"
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/session/onnxruntime_cxx_api.h"

namespace onnxruntime {
// Lightweight Stream shim for plugin build: wraps a raw cudaStream_t as a
// framework-compatible Stream* that can be passed to _impl.cu functions which
// call stream->GetHandle().  Stack-allocated; does NOT own the stream.
// Only available in .cc translation units (not .cu) since Stream is incomplete in NVCC context.
#ifndef __CUDACC__
struct PluginStreamShim : public onnxruntime::Stream {
  explicit PluginStreamShim(void* cuda_stream_handle)
      : onnxruntime::Stream(cuda_stream_handle,
                            OrtDevice(OrtDevice::GPU, OrtDevice::MemType::DEFAULT,
                                      OrtDevice::VendorIds::NVIDIA, 0)) {}
};
#endif
}  // namespace onnxruntime

// ===================================================================
// Section 1: Include path selection
// ===================================================================

#include "core/graph/constants.h"
#include "ep/adapters.h"
#include "core/framework/op_kernel.h"
#include "core/providers/common.h"

namespace onnxruntime {
namespace cuda {

#ifndef CUDA_STREAM_FROM_CTX
// Helper for kernels that need a cudaStream_t from OpKernelContext in plugin build.
#define CUDA_STREAM_FROM_CTX(ctx) static_cast<cudaStream_t>((GetComputeStream(ctx) != nullptr) ? GetComputeStream(ctx)->GetHandle() : nullptr)
#endif

// Forward declare the template for kernel registration macros to specialize
// inside the onnxruntime::cuda namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

// Tensor creation helper to replace deprecated Tensor::Create
inline std::unique_ptr<::onnxruntime::Tensor> TensorCreate(MLDataType type, const TensorShape& shape, AllocatorPtr allocator) {
  return std::make_unique<::onnxruntime::Tensor>(type, shape, std::move(allocator));
}

using ::onnxruntime::HandleNegativeAxis;

}  // namespace cuda

#ifndef DISABLE_CONTRIB_OPS
namespace contrib {
namespace cuda {

// Forward declare the template for kernel registration macros to specialize
// inside the onnxruntime::contrib::cuda namespace.
template <typename T>
KernelCreateInfo BuildKernelCreateInfo();

inline std::unique_ptr<::onnxruntime::Tensor> TensorCreate(MLDataType type, const TensorShape& shape, AllocatorPtr allocator) {
  return std::make_unique<::onnxruntime::Tensor>(type, shape, std::move(allocator));
}

using ::onnxruntime::HandleNegativeAxis;

}  // namespace cuda
}  // namespace contrib
#endif
}  // namespace onnxruntime

// ===================================================================
// Section 2: Error-return macros (redefined for all plugin paths)
// ===================================================================

// Redefine error macros for ported code to use our adapter-specific Status translation
#undef CUDA_RETURN_IF_ERROR
#define CUDA_RETURN_IF_ERROR(expr)                                                                                                                             \
  {                                                                                                                                                            \
    cudaError_t _err = (expr);                                                                                                                                 \
    if (_err != cudaSuccess) {                                                                                                                                 \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("CUDA error: ") + cudaGetErrorString(_err)); \
    }                                                                                                                                                          \
  }

#undef CUBLAS_RETURN_IF_ERROR
#define CUBLAS_RETURN_IF_ERROR(expr)                                                                                                                                              \
  {                                                                                                                                                                               \
    cublasStatus_t _status = (expr);                                                                                                                                              \
    if (_status != CUBLAS_STATUS_SUCCESS) {                                                                                                                                       \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("cuBLAS error: ") + std::to_string(static_cast<int>(_status))); \
    }                                                                                                                                                                             \
  }

#undef CUDNN_RETURN_IF_ERROR
#define CUDNN_RETURN_IF_ERROR(expr)                                                                                                                                 \
  {                                                                                                                                                                 \
    cudnnStatus_t _status = (expr);                                                                                                                                 \
    if (_status != CUDNN_STATUS_SUCCESS) {                                                                                                                          \
      return onnxruntime::common::Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, std::string("cuDNN error: ") + cudnnGetErrorString(_status)); \
    }                                                                                                                                                               \
  }

// ===================================================================
// Section 3: Self-registering kernel collector using BuildKernelCreateInfo<>
//
// Each ONNX_OPERATOR_*_KERNEL_EX macro expansion both:
//   1. Creates a BuildKernelCreateInfo<CLASS>() template specialization
//      (identical to the framework's macro output, but using adapter types)
//   2. Auto-registers the BuildKernelCreateInfoFn pointer into a global
//      PluginKernelCollector singleton
//
// At registration time, the factory iterates the collector and calls
// adapter::KernelRegistry::Register(build_fn()) for each compiled kernel.
// Only ops whose .cc files are actually compiled get registered.
// ===================================================================

#include <vector>

namespace onnxruntime {
namespace cuda {

/// Singleton collector for BuildKernelCreateInfoFn pointers.
/// Each compiled kernel .cc file's macro expansion auto-registers here.
class PluginKernelCollector {
 public:
  static PluginKernelCollector& Instance() {
    static PluginKernelCollector instance;
    return instance;
  }

  void Add(BuildKernelCreateInfoFn fn) { entries_.push_back(fn); }
  const std::vector<BuildKernelCreateInfoFn>& Entries() const { return entries_; }

 private:
  std::vector<BuildKernelCreateInfoFn> entries_;
};

}  // namespace cuda
}  // namespace onnxruntime

// --- Macro overrides: produce BuildKernelCreateInfo<> AND auto-register ---
//
// These macros mirror the framework's ONNX_OPERATOR_*_KERNEL_EX definitions
// from core/framework/op_kernel.h, but additionally register each
// BuildKernelCreateInfoFn into PluginKernelCollector at static init time.

#define ORT_ADAPTER_CONCAT_IMPL(x, y) x##y
#define ORT_ADAPTER_CONCAT(x, y) ORT_ADAPTER_CONCAT_IMPL(x, y)

#undef ONNX_OPERATOR_KERNEL_EX
#define ONNX_OPERATOR_KERNEL_EX(name, domain, ver, provider, builder, ...)                         \
  class ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name);                              \
  template <>                                                                                      \
  KernelCreateInfo                                                                                 \
  BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>() {          \
    return KernelCreateInfo(                                                                       \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(provider).Build(),     \
        static_cast<KernelCreatePtrFn>(                                                            \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status { \
              out = std::make_unique<__VA_ARGS__>(info);                                           \
              return Status::OK();                                                                 \
            }));                                                                                   \
  }                                                                                                \
  static const bool ORT_ADAPTER_CONCAT(_autoreg_##name##_, __COUNTER__) =                          \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                 \
           &BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(provider, domain, ver, name)>),  \
       true);

#undef ONNX_OPERATOR_VERSIONED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_KERNEL_EX(name, domain, startver, endver, provider, builder, ...)                \
  class ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name);                     \
  template <>                                                                                                    \
  KernelCreateInfo                                                                                               \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(provider, domain, startver, endver, name)>() { \
    return KernelCreateInfo(                                                                                     \
        builder.SetName(#name).SetDomain(domain).SinceVersion(startver, endver).Provider(provider).Build(),      \
        static_cast<KernelCreatePtrFn>(                                                                          \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {               \
              out = std::make_unique<__VA_ARGS__>(info);                                                         \
              return Status::OK();                                                                               \
            }));                                                                                                 \
  }                                                                                                              \
  static const bool ORT_ADAPTER_CONCAT(_autoreg_##name##_, __COUNTER__) =                                        \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                               \
           &BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_KERNEL_CLASS_NAME(                                     \
               provider, domain, startver, endver, name)>),                                                      \
       true);

#undef ONNX_OPERATOR_TYPED_KERNEL_EX
#define ONNX_OPERATOR_TYPED_KERNEL_EX(name, domain, ver, type, provider, builder, ...)                        \
  class ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name);                             \
  template <>                                                                                                 \
  KernelCreateInfo                                                                                            \
  BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)>() {         \
    return KernelCreateInfo(                                                                                  \
        builder.SetName(#name).SetDomain(domain).SinceVersion(ver).Provider(provider).Build(),                \
        static_cast<KernelCreatePtrFn>(                                                                       \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {            \
              out = std::make_unique<__VA_ARGS__>(info);                                                      \
              return Status::OK();                                                                            \
            }));                                                                                              \
  }                                                                                                           \
  static const bool ORT_ADAPTER_CONCAT(_autoreg_##name##_##type##_, __COUNTER__) =                            \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                            \
           &BuildKernelCreateInfo<ONNX_OPERATOR_TYPED_KERNEL_CLASS_NAME(provider, domain, ver, type, name)>), \
       true);

#undef ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX
#define ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_EX(name, domain, startver, endver, type, provider, builder, ...) \
  class ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(provider, domain, startver, endver, type, name);      \
  template <>                                                                                                 \
  KernelCreateInfo                                                                                            \
  BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(                                      \
      provider, domain, startver, endver, type, name)>() {                                                    \
    return KernelCreateInfo(                                                                                  \
        builder.SetName(#name).SetDomain(domain).SinceVersion(startver, endver).Provider(provider).Build(),   \
        static_cast<KernelCreatePtrFn>(                                                                       \
            [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {            \
              out = std::make_unique<__VA_ARGS__>(info);                                                      \
              return Status::OK();                                                                            \
            }));                                                                                              \
  }                                                                                                           \
  static const bool ORT_ADAPTER_CONCAT(_autoreg_##name##_##type##_, __COUNTER__) =                            \
      (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(                                            \
           &BuildKernelCreateInfo<ONNX_OPERATOR_VERSIONED_TYPED_KERNEL_CLASS_NAME(                            \
               provider, domain, startver, endver, type, name)>),                                             \
       true);

// ===================================================================
// Section 4: Logging shim (adapter path only)
// Replaces LOGS_DEFAULT with a no-op stream to avoid pulling in the
// full ORT logging framework inside the plugin shared library.
// ===================================================================

// Explicit function instantiation — called once per unique class in each .cc file
#define ONNX_OPERATOR_TYPED_KERNEL_COMPUTE_INSTANTIATION(cls) template Status cls::ComputeInternal(OpKernelContext* context) const;

#undef CREATE_MESSAGE
#undef LOGS
#undef LOGS_DEFAULT
#undef ORT_LOG_MESSAGE

namespace onnxruntime {
namespace cuda {
struct PluginNoOpLogStream {
  template <typename T>
  PluginNoOpLogStream& operator<<(const T&) { return *this; }
};
}  // namespace cuda
}  // namespace onnxruntime

#ifndef LOGS_DEFAULT
#define LOGS_DEFAULT(severity) ::onnxruntime::cuda::PluginNoOpLogStream()
#endif

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

// ===================================================================
// Section 5: Shared helpers for migrated kernels
// ===================================================================

namespace detail {
template <typename T>
struct SizeOf {
  static constexpr size_t value = sizeof(T);
};
template <>
struct SizeOf<void> {
  static constexpr size_t value = 0;
};
inline size_t BytesForCount(size_t count_or_bytes, size_t element_size) {
  if (element_size == 0) return count_or_bytes;
  if (count_or_bytes > (std::numeric_limits<size_t>::max() / element_size)) return 0;
  return count_or_bytes * element_size;
}
}  // namespace detail
}  // namespace cuda

namespace cuda {

// Global aliases and shims
using Status = onnxruntime::common::Status;
using MLFloat16 = onnxruntime::MLFloat16;
using BFloat16 = onnxruntime::BFloat16;
using Float8E4M3FN = onnxruntime::Float8E4M3FN;
using Float8E4M3FNUZ = onnxruntime::Float8E4M3FNUZ;
using Float8E5M2 = onnxruntime::Float8E5M2;
using Float8E5M2FNUZ = onnxruntime::Float8E5M2FNUZ;

// Type mapping for CUDA
template <typename T>
struct ToCudaType {
  typedef T MappedType;
  static MappedType FromFloat(float f) { return static_cast<MappedType>(f); }
};

template <>
struct ToCudaType<MLFloat16> {
  typedef half MappedType;
  static MappedType FromFloat(float f) {
    uint16_t h = onnxruntime::math::floatToHalf(f);
    return *reinterpret_cast<MappedType*>(&h);
  }
};

#ifdef __CUDACC__
template <>
struct ToCudaType<BFloat16> {
  typedef nv_bfloat16 MappedType;
  static MappedType FromFloat(float f) {
    return nv_bfloat16(f);
  }
};

// Forward declare templates from common.cuh to allow specialization
// Match signatures from common.cuh exactly (no default parameters)
template <typename T, bool detect_positive, bool detect_negative>
struct _IsInf;
template <typename T>
struct _IsNan;

namespace bf16_isinf_nan {
template <typename T>
struct IsInfTyped;
template <>
struct IsInfTyped<nv_bfloat16> {
  static __device__ __inline__ bool IsInf(nv_bfloat16 a) {
    uint16_t val = *reinterpret_cast<const uint16_t*>(&a);
    return (val & 0x7F80) == 0x7F80 && (val & 0x007F) == 0x0000;
  }
  static __device__ __inline__ bool IsInfPos(nv_bfloat16 a) {
    return *reinterpret_cast<const uint16_t*>(&a) == 0x7F80;
  }
  static __device__ __inline__ bool IsInfNeg(nv_bfloat16 a) {
    return *reinterpret_cast<const uint16_t*>(&a) == 0xFF80;
  }
};
}  // namespace bf16_isinf_nan

// Specialize for nv_bfloat16 to avoid ambiguity with isnan/isinf overloads
template <>
struct _IsNan<nv_bfloat16> {
  __device__ __inline__ bool operator()(nv_bfloat16 a) const {
    uint16_t val = *reinterpret_cast<const uint16_t*>(&a);
    return (val & 0x7F80) == 0x7F80 && (val & 0x007F) != 0x0000;
  }
};

template <bool detect_positive, bool detect_negative>
struct _IsInf<nv_bfloat16, detect_positive, detect_negative> {
  __device__ __inline__ bool operator()(nv_bfloat16 a) const {
    if constexpr (detect_positive && detect_negative) {
      return bf16_isinf_nan::IsInfTyped<nv_bfloat16>::IsInf(a);
    } else if constexpr (detect_positive) {
      return bf16_isinf_nan::IsInfTyped<nv_bfloat16>::IsInfPos(a);
    } else if constexpr (detect_negative) {
      return bf16_isinf_nan::IsInfTyped<nv_bfloat16>::IsInfNeg(a);
    } else {
      return false;
    }
  }
};
#endif

// ===================================================================
// Section 6b: CPU provider shims for the plugin build
// Inline implementations of CPU utility functions that CUDA kernels
// reference (e.g., OneHot validation, GatherElements shape checks).
// These are normally provided by onnxruntime_providers but the plugin
// does not link against it.
//
// We temporarily close namespace cuda so these shims live directly in
// namespace onnxruntime, where unqualified lookup from onnxruntime::cuda
// will find them, and where onnxruntime::GatherElements resolves correctly.
// ===================================================================

}  // namespace cuda

// Shim for ValidateInputs from core/providers/cpu/tensor/onehot.h
inline Status ValidateInputs(const Tensor* depth, const Tensor* values) {
  if (!depth->Shape().IsScalar()) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Invalid argument for depth; it's not a scalar.");
  }
  if (!(values->Shape().NumDimensions() == 1 && values->Shape().Size() == 2)) {
    return Status(common::ONNXRUNTIME, common::INVALID_ARGUMENT,
                  "Invalid argument for values; either it's rank is more than 1"
                  " or it has more than 2 elements");
  }
  return Status::OK();
}

// Shim for PrepareOutputShape from core/providers/cpu/tensor/onehot.h
inline Status PrepareOutputShape(const Tensor* indices, const int64_t depth_val, const int64_t axis,
                                 int64_t& prefix_dim_size, int64_t& suffix_dim_size,
                                 TensorShapeVector& output_shape) {
  const auto& indices_shape = indices->Shape();
  const auto indices_dims = indices_shape.GetDims();
  const auto indices_num_dims = indices_shape.NumDimensions();
  output_shape = indices_shape.AsShapeVector();
  const auto output_rank = static_cast<int64_t>(indices_num_dims) + 1;
  auto true_axis = HandleNegativeAxis(axis, output_rank);
  output_shape.insert(output_shape.begin() + true_axis, depth_val);
  prefix_dim_size = 1;
  for (int64_t i = 0; i < true_axis; ++i) {
    prefix_dim_size *= indices_dims[narrow<size_t>(i)];
  }
  suffix_dim_size = indices_shape.Size() / prefix_dim_size;
  return Status::OK();
}

// Shim for GatherElements::ValidateInputShapes from
// core/providers/cpu/tensor/gather_elements.h
class GatherElements {
 public:
  static Status ValidateInputShapes(const TensorShape& input_data_shape,
                                    const TensorShape& indices_shape,
                                    int64_t axis) {
    int64_t input_data_rank = static_cast<int64_t>(input_data_shape.NumDimensions());
    int64_t indices_rank = static_cast<int64_t>(indices_shape.NumDimensions());
    if (input_data_rank < 1)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GatherElements op: Cannot operate on scalar input");
    if (input_data_rank != indices_rank)
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                             "GatherElements op: Rank of input 'data' needs to be equal to rank of input 'indices'");
    for (int64_t i = 0; i < indices_rank; ++i) {
      if (i != axis) {
        if (indices_shape[narrow<size_t>(i)] < 0 ||
            indices_shape[narrow<size_t>(i)] > input_data_shape[narrow<size_t>(i)])
          return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                                 "GatherElements op: 'indices' shape should have values within bounds of 'data' shape. "
                                 "Invalid value in indices shape is: ",
                                 indices_shape[narrow<size_t>(i)]);
      }
    }
    return Status::OK();
  }
};

namespace cuda {  // re-open onnxruntime::cuda

// ===================================================================
// Section 7: CudaKernel base class
// Base class for all migrated CUDA kernels. Provides scratch-buffer
// management, CUDA handle access (cuBLAS, cuDNN), device property
// queries, and the CudaAsyncBuffer helper for host→device transfers.
// ===================================================================

// Additional adapter logic for CudaKernel

class CudaKernel : public OpKernel {
 public:
  explicit CudaKernel(const OpKernelInfo& info)
      : OpKernel(info),
        info_(info),
        provider_(const_cast<CUDAExecutionProvider*>(static_cast<const CUDAExecutionProvider*>(info.GetExecutionProvider()))) {}
  virtual ~CudaKernel() = default;
  Status Compute(OpKernelContext* ctx) const {
    Status s = ComputeInternal(ctx);
    if (s.IsOK()) {
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "CUDA error: " + std::string(cudaGetErrorString(err)));
    }
    return s;
  }
  virtual Status ComputeInternal(OpKernelContext* ctx) const = 0;

  inline cudaStream_t DefaultCudaStream() const { return provider_->ComputeStream(); }
  inline cublasHandle_t DefaultCublasHandle() const { return provider_->PerThreadDefaultCublasHandle(); }
  inline cublasLtHandle_t CublasLtHandle() const { return provider_->PerThreadCublasLtHandle(); }
  inline cudnnHandle_t DefaultCudnnHandle() const { return provider_->PerThreadDefaultCudnnHandle(); }

  inline Status CopyTensor(const onnxruntime::Tensor& src, onnxruntime::Tensor& dst, onnxruntime::Stream& stream) const {
    if (src.Shape().Size() == 0) return Status::OK();
    if (cudaMemcpyAsync(dst.MutableDataRaw(), src.DataRaw(), src.SizeInBytes(), cudaMemcpyDeviceToDevice, (cudaStream_t)stream.GetHandle()) != cudaSuccess) {
      return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "Memcpy fail");
    }
    return Status::OK();
  }

  cudaStream_t Stream(OpKernelContext* ctx) const {
    if (!ctx) return nullptr;
    return static_cast<cudaStream_t>(ctx->GetGPUComputeStream());
  }

  // Compatibility helper used by kernels that need the underlying ORT stream object.
  inline onnxruntime::Stream* GetComputeStream(OpKernelContext* ctx) const {
    return ctx ? ctx->GetComputeStream() : nullptr;
  }

  static cudnnHandle_t GetCudnnHandle(cudaStream_t s) {
    auto* provider = CUDAExecutionProvider::GetActiveProvider();
    return provider ? provider->GetCudnnHandle(s) : nullptr;
  }
  static inline cudnnHandle_t GetCudnnHandle(onnxruntime::Stream* stream) {
    return stream ? GetCudnnHandle(static_cast<cudaStream_t>(stream->GetHandle())) : nullptr;
  }
  cudnnHandle_t GetCudnnHandle(OpKernelContext* ctx) const { return GetCudnnHandle(Stream(ctx)); }

  static cublasHandle_t GetCublasHandle(cudaStream_t s) {
    auto* provider = CUDAExecutionProvider::GetActiveProvider();
    return provider ? provider->GetCublasHandle(s) : nullptr;
  }
  static inline cublasHandle_t GetCublasHandle(onnxruntime::Stream* stream) {
    return stream ? GetCublasHandle(static_cast<cudaStream_t>(stream->GetHandle())) : nullptr;
  }
  cublasHandle_t GetCublasHandle(OpKernelContext* ctx) const { return GetCublasHandle(Stream(ctx)); }

  const cudaDeviceProp& GetDeviceProp() const { return provider_->GetDeviceProp(); }
  bool UseTF32() const { return provider_->UseTF32(); }
  bool IsArchAvailable(int arch) const { return provider_->GetDeviceProp().major >= arch; }
  const OpKernelInfo& Info() const { return info_; }
  const onnxruntime::AttentionKernelOptions* GetAttentionKernelOptions() const {
    return provider_->GetAttentionKernelOptions();
  }
  ITuningContext* GetTuningContext() const {
    return provider_->GetTuningContext();
  }

  // GetConstOnes: returns a device buffer of constant ones.
  // Delegates to IConstantBuffer from cuda_utils.h (compiled in cuda_utils.cu).
  template <typename T>
  const T* GetConstOnes(size_t count, cudaStream_t stream) const {
    return provider_->template GetConstOnes<T>(count, stream);
  }

  template <typename T>
  using IAllocatorUniquePtr = std::unique_ptr<T, std::function<void(T*)>>;
  template <typename T>
  inline IAllocatorUniquePtr<T> GetScratchBuffer(size_t cnt, onnxruntime::Stream* stream) const {
    if (cnt == 0) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    size_t sz = detail::BytesForCount(cnt, detail::SizeOf<T>::value);
    void* p = nullptr;
    auto* stream_handle = stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
    if (cudaMalloc(&p, sz) != cudaSuccess) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    return IAllocatorUniquePtr<T>(static_cast<T*>(p), [stream_handle](T* ptr) {
      if (ptr) {
        if (stream_handle) {
          cudaFreeAsync(ptr, stream_handle);
        } else {
          cudaFree(ptr);
        }
      }
    });
  }
  template <typename T>
  inline IAllocatorUniquePtr<T> GetTransientScratchBuffer(size_t cnt) const {
    return GetScratchBuffer<T>(cnt, static_cast<onnxruntime::Stream*>(nullptr));
  }
  inline void AddDeferredReleaseCPUPtr(void* p, onnxruntime::Stream* stream) const {
    if (!p) return;
    auto* sync = provider_->GetSyncStream(stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr);
    if (sync) {
      sync->EnqueueDeferredCPUBuffer(p);
      return;
    }
    cudaFreeHost(p);
  }
  template <typename T>
  inline IAllocatorUniquePtr<T> AllocateBufferOnCPUPinned(size_t cnt) const {
    if (cnt == 0) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    size_t sz = detail::BytesForCount(cnt, detail::SizeOf<T>::value);
    void* p = nullptr;
    if (cudaHostAlloc(&p, sz, cudaHostAllocDefault) != cudaSuccess) return IAllocatorUniquePtr<T>(nullptr, [](T*) {});
    return IAllocatorUniquePtr<T>(static_cast<T*>(p), [](T* ptr) { if (ptr) cudaFreeHost(ptr); });
  }

  template <typename T>
  class CudaAsyncBuffer {
   public:
    CudaAsyncBuffer(const CudaKernel* ok) : gpu_(nullptr, [](T*) {}), count_(0), op_kernel_(ok) {}
    CudaAsyncBuffer(const CudaKernel* ok, size_t n) : CudaAsyncBuffer(ok) { AllocCpuPtr(n); }
    CudaAsyncBuffer(const CudaKernel* ok, const T& v, size_t n) : CudaAsyncBuffer(ok, n) {
      T* p = CpuPtr();
      for (size_t i = 0; i != n; ++i) *p++ = v;
    }
    CudaAsyncBuffer(const CudaKernel* ok, gsl::span<T const> vec) : CudaAsyncBuffer(ok, vec.size()) { memcpy(CpuPtr(), vec.data(), vec.size() * sizeof(T)); }
    void AllocCpuPtr(size_t n) {
      cpu_ = op_kernel_->AllocateBufferOnCPUPinned<T>(n);
      if (!cpu_) throw std::runtime_error("alloc fail");
      count_ = n;
    }
    Status CopyToGpu(onnxruntime::Stream* stream) {
      if (cpu_) {
        auto* stream_handle = stream ? static_cast<cudaStream_t>(stream->GetHandle()) : nullptr;
        gpu_ = op_kernel_->GetScratchBuffer<T>(count_, stream);
        if (cudaMemcpyAsync(gpu_.get(), cpu_.get(), count_ * sizeof(T), cudaMemcpyHostToDevice, stream_handle) != cudaSuccess) return Status(onnxruntime::common::ONNXRUNTIME, onnxruntime::common::FAIL, "Memcpy fail");
        op_kernel_->AddDeferredReleaseCPUPtr(cpu_.release(), stream);
      }
      return Status::OK();
    }
    T* CpuPtr() const { return cpu_.get(); }
    gsl::span<T> CpuSpan() const { return gsl::span<T>(CpuPtr(), count_); }
    T* GpuPtr() const { return gpu_.get(); }
    size_t count() const { return count_; }

   protected:
    IAllocatorUniquePtr<T> gpu_;
    std::unique_ptr<T, std::function<void(T*)>> cpu_{nullptr, [](T*) {}};
    size_t count_;
    const CudaKernel* op_kernel_;
  };

 private:
  const OpKernelInfo& info_;
  CUDAExecutionProvider* provider_;
};

// ===================================================================
// Section 8: Compute helper shims (HalfGemmOptions, CublasMathModeSetter)
// ===================================================================

// Shims for HalfGemmOptions and CublasMathModeSetter required by fpgeneric.h
class HalfGemmOptions {
 public:
  static const HalfGemmOptions* GetInstance() {
    static HalfGemmOptions instance;
    return &instance;
  }
  cublasMath_t GetMathMode() const { return CUBLAS_DEFAULT_MATH; }
  bool IsCompute16F() const { return false; }
#if defined(CUBLAS_COMPUTE_32F)
  cublasComputeType_t GetComputeType() const { return CUBLAS_COMPUTE_32F; }
#else
  cudaDataType_t GetComputeType() const { return CUDA_R_32F; }
#endif
};

class CublasMathModeSetter {
 public:
  CublasMathModeSetter(const cudaDeviceProp& prop, cublasHandle_t handle, cublasMath_t mode) : handle_(handle) {
    enable_ = (mode == CUBLAS_TF32_TENSOR_OP_MATH ? prop.major >= 8 : true);
    if (enable_) {
      cublasGetMathMode(handle, &mode_);
      enable_ = (mode_ != mode);
      if (enable_) {
        cublasSetMathMode(handle, mode);
      }
    }
  }

  ~CublasMathModeSetter() {
    if (enable_) {
      cublasSetMathMode(handle_, mode_);
    }
  }

 private:
  cublasHandle_t handle_;
  cublasMath_t mode_ = CUBLAS_DEFAULT_MATH;
  bool enable_;
};

#include "core/providers/cuda/cuda_common_type_helpers.h"

}  // namespace cuda

// Global aliases for convenience
using MLFloat16 = onnxruntime::MLFloat16;
using BFloat16 = onnxruntime::BFloat16;
using Float8E4M3FN = onnxruntime::Float8E4M3FN;
using Float8E4M3FNUZ = onnxruntime::Float8E4M3FNUZ;
using Float8E5M2 = onnxruntime::Float8E5M2;
using Float8E5M2FNUZ = onnxruntime::Float8E5M2FNUZ;

}  // namespace onnxruntime
