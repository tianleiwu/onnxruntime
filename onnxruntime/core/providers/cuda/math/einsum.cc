// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "einsum.h"

namespace onnxruntime {

#ifndef BUILD_CUDA_EP_AS_PLUGIN

// This function must exist due to the C++ base class constructor needing this to be defined for the vtable, but it is never called.
Status Einsum::DeviceCompute(OpKernelContext* /*context*/, const std::vector<const Tensor*>& /*inputs*/,
                             AllocatorPtr /*allocator*/, concurrency::ThreadPool* /*tp*/) const {
  assert(false);
  return Status::OK();
}

#endif  // BUILD_CUDA_EP_AS_PLUGIN

namespace cuda {

#ifdef BUILD_CUDA_EP_AS_PLUGIN

class ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, Einsum);

// Keep the adapter macros provider-agnostic. Einsum is registered manually for
// the plugin build because this kernel needs plugin-specific execution wiring
// and must be claimed under the plugin EP name without changing global macro
// behavior for every adapted CUDA kernel.
template <>
KernelCreateInfo BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, Einsum)>() {
  return KernelCreateInfo(
      (*KernelDefBuilder::Create())
          .TypeConstraint("T", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(),
                                                       DataTypeImpl::GetTensorType<double>(),
                                                       DataTypeImpl::GetTensorType<MLFloat16>()})
          .SetName("Einsum")
          .SetDomain(kOnnxDomain)
          .SinceVersion(12)
          .Provider("CudaPluginExecutionProvider")
          .Build(),
      static_cast<KernelCreatePtrFn>(
          [](FuncManager&, const OpKernelInfo& info, std::unique_ptr<OpKernel>& out) -> Status {
            out = std::make_unique<Einsum>(info);
            return Status::OK();
          }));
}

static const bool kEinsumPluginKernelRegistered =
    (::onnxruntime::cuda::PluginKernelCollector::Instance().Add(
         &BuildKernelCreateInfo<ONNX_OPERATOR_KERNEL_CLASS_NAME(kCudaExecutionProvider, kOnnxDomain, 12, Einsum)>),
     true);

#else

ONNX_OPERATOR_KERNEL_EX(
    Einsum,
    kOnnxDomain,
    12,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create()).TypeConstraint("T", std::vector<MLDataType>{DataTypeImpl::GetTensorType<float>(), DataTypeImpl::GetTensorType<double>(), DataTypeImpl::GetTensorType<MLFloat16>()}),
    Einsum);

#endif

#ifndef BUILD_CUDA_EP_AS_PLUGIN

Status Einsum::Compute(OpKernelContext* context) const {
  return onnxruntime::Einsum::Compute(context);
}

#else

Status Einsum::ComputeInternal(OpKernelContext* context) const {
  int num_inputs = context->InputCount();
  if (num_inputs == 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "Einsum op: There must be atleast one input");
  }

  std::vector<const Tensor*> inputs;
  inputs.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    inputs.push_back(context->Input<Tensor>(i));
  }

  AllocatorPtr allocator;
  auto status = context->GetTempSpaceAllocator(&allocator);
  if (!status.IsOK()) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, RUNTIME_EXCEPTION,
                           "There was a problem acquiring temporary memory allocator in Einsum op");
  }

  concurrency::ThreadPool* tp = nullptr;
  return DeviceCompute(context, inputs, allocator, tp);
}

#endif  // BUILD_CUDA_EP_AS_PLUGIN

Status Einsum::DeviceCompute(OpKernelContext* context, const std::vector<const Tensor*>& inputs,
                             AllocatorPtr allocator, concurrency::ThreadPool* tp) const {
  auto* stream = context->GetComputeStream();
  ORT_RETURN_IF(!stream, "stream is null");
  cublasHandle_t cublas_handle = CudaKernel::GetCublasHandle(stream);
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  EinsumOp::EinsumCudaAssets einsum_cuda_assets(cublas_handle, cuda_ep_, stream, allocator);
#else
  EinsumOp::EinsumCudaAssets einsum_cuda_assets(cublas_handle, cuda_ep_, stream, Info().GetAllocator(OrtMemType::OrtMemTypeDefault));
#endif

// EinsumComputePreprocessor section -
#ifdef BUILD_CUDA_EP_AS_PLUGIN
  EinsumComputePreprocessor einsum_compute_preprocessor(*einsum_equation_preprocessor_, inputs, allocator,
                                                        &einsum_cuda_assets);
  einsum_compute_preprocessor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Diagonal,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose);
  ORT_RETURN_IF_ERROR(einsum_compute_preprocessor.Run());
#else
  auto einsum_compute_preprocessor = EinsumComputePreprocessor::Create(*einsum_equation_preprocessor_, inputs, allocator,
                                                                       &einsum_cuda_assets);
  einsum_compute_preprocessor->SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Diagonal,
                                                EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose);
  ORT_RETURN_IF_ERROR(einsum_compute_preprocessor->Run());
#endif

  // EinsumComputeProcessor section -
  if (inputs[0]->IsDataType<float>()) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
    EinsumTypedComputeProcessor<float> einsum_compute_processor(context, allocator, tp,
                                                                nullptr,  // mlas_backend_config
                                                                einsum_compute_preprocessor,
                                                                &einsum_cuda_assets);
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<float>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<float>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ZeroBuffer);
    return einsum_compute_processor.Run();
#else
    auto einsum_compute_processor = EinsumTypedComputeProcessor<float>::Create(context, allocator, tp,
                                                                               nullptr,  // mlas_backend_config
                                                                               *einsum_compute_preprocessor,
                                                                               &einsum_cuda_assets);
    einsum_compute_processor->SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<float>,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<float>,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::ZeroBuffer);
    return einsum_compute_processor->Run();
#endif
  } else if (inputs[0]->IsDataType<double>()) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
    EinsumTypedComputeProcessor<double> einsum_compute_processor(context, allocator, tp,
                                                                 nullptr,  // mlas_backend_config
                                                                 einsum_compute_preprocessor,
                                                                 &einsum_cuda_assets);
    // Set device specific methods (CPU methods) to be used during processing
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<double>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<double>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ZeroBuffer);
    return einsum_compute_processor.Run();
#else
    auto einsum_compute_processor = EinsumTypedComputeProcessor<double>::Create(context, allocator, tp,
                                                                                nullptr,  // mlas_backend_config
                                                                                *einsum_compute_preprocessor,
                                                                                &einsum_cuda_assets);
    einsum_compute_processor->SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<double>,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<double>,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::ZeroBuffer);
    return einsum_compute_processor->Run();
#endif
  } else if (inputs[0]->IsDataType<MLFloat16>()) {
#ifdef BUILD_CUDA_EP_AS_PLUGIN
    EinsumTypedComputeProcessor<MLFloat16> einsum_compute_processor(context, allocator, tp,
                                                                    nullptr,  // mlas_backend_config
                                                                    einsum_compute_preprocessor,
                                                                    &einsum_cuda_assets);
    einsum_compute_processor.SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<MLFloat16>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<MLFloat16>,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy,
                                              EinsumOp::DeviceHelpers::CudaDeviceHelpers::ZeroBuffer);
    return einsum_compute_processor.Run();
#else
    auto einsum_compute_processor = EinsumTypedComputeProcessor<MLFloat16>::Create(context, allocator, tp,
                                                                                   nullptr,  // mlas_backend_config
                                                                                   *einsum_compute_preprocessor,
                                                                                   &einsum_cuda_assets);
    einsum_compute_processor->SetDeviceHelpers(EinsumOp::DeviceHelpers::CudaDeviceHelpers::Transpose,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::MatMul<MLFloat16>,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::ReduceSum<MLFloat16>,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::DataCopy,
                                               EinsumOp::DeviceHelpers::CudaDeviceHelpers::ZeroBuffer);
    return einsum_compute_processor->Run();
#endif
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "Einsum op: An implementation for the input type ",
                         inputs[0]->DataType(), " is not supported yet");
}

}  // namespace cuda

}  // namespace onnxruntime
