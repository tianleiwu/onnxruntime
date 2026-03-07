// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

// This module hosts the following abstraction -

// 1) EinsumTypedComputeProcessor - The core logic of the Einsum operator. Invoked from Einsum Compute().

#pragma once

#include "einsum_auxiliary_ops.h"
#include "einsum_compute_preprocessor.h"

namespace onnxruntime {

#ifdef BUILD_CUDA_EP_AS_PLUGIN
using EinsumOpKernelContext = cuda::OpKernelContext;
#else
using EinsumOpKernelContext = OpKernelContext;
#endif

// This method does the heavy-lifting compute portion of Einsum Compute()
template <typename T>
class EinsumTypedComputeProcessor {
 public:
  explicit EinsumTypedComputeProcessor(EinsumOpKernelContext* context, AllocatorPtr allocator,
                                       concurrency::ThreadPool* tp,
                                       const void* mlas_backend_config,
                                       EinsumComputePreprocessor& einsum_compute_preprocessor,
                                       void* einsum_cuda_assets)
      : context_(context),
        allocator_(allocator),
        tp_(tp),
        mlas_backend_config_(mlas_backend_config),
        einsum_compute_preprocessor_(einsum_compute_preprocessor),
        einsum_ep_assets_(einsum_cuda_assets) {}

  // Pass-in device specific functions
  // (Pass-in CPU implementation or CUDA implementation function depending on the kernel using this class)
  void SetDeviceHelpers(const EinsumOp::DeviceHelpers::Transpose& device_transpose_func,
                        const EinsumOp::DeviceHelpers::MatMul<T>& device_matmul_func,
                        const EinsumOp::DeviceHelpers::ReduceSum<T>& device_reduce_sum_func,
                        const EinsumOp::DeviceHelpers::DataCopy& device_data_copy_func,
                        const EinsumOp::DeviceHelpers::ZeroBuffer& device_zero_buffer_func);

  Status Run();

 private:
  // Private methods -

  // Processes Einsum operands in a pair-wise fashion
  // Employs Transpose, ReduceSum, and MatMul under the hood
  // to achieve MatMul(a, b) and reduces (by summing) along specified axes
  std::unique_ptr<Tensor> PairwiseOperandProcess(const Tensor& left,
                                                 const TensorShape& left_shape_override,
                                                 const Tensor& right,
                                                 const TensorShape& right_shape_override,
                                                 const gsl::span<const int64_t>& reduce_dims,
                                                 bool is_final_pair);

  // Here we take a "candidate output"(candidate output is a tensor that is a permutation and / or a reshape away from the final output),
  // and after a few operations to get it to the required output structure, copy it to the op's output
  // The candidate output might contain dims that may not be part of the op's output (i.e.) the dims will have to be unsqueezed
  void FinalizeOutput(const Tensor& candidate_output,
                      const gsl::span<const int64_t>& ordered_subscript_indices_in_candidate);

  // Private members -
  EinsumOpKernelContext* context_;
  AllocatorPtr allocator_;
  concurrency::ThreadPool* tp_;      // CPU only: Thread pool for parallelism
  const void* mlas_backend_config_;  // CPU only: MLAS backend kernel selector config
  EinsumComputePreprocessor& einsum_compute_preprocessor_;

  EinsumOp::DeviceHelpers::Transpose device_transpose_func_;
  EinsumOp::DeviceHelpers::MatMul<T> device_matmul_func_;
  EinsumOp::DeviceHelpers::ReduceSum<T> device_reduce_sum_func_;
  EinsumOp::DeviceHelpers::DataCopy device_data_copy_func_;
  EinsumOp::DeviceHelpers::ZeroBuffer device_zero_buffer_func_;

  // Holds EP-specific assets required for (auxiliary) ops that need to be executed on non-CPU EPs
  void* einsum_ep_assets_;
};

namespace EinsumOp {

// Thin wrapper over the Transpose op to be called from Einsum that does some checks and invokes the device specific helper
inline std::unique_ptr<Tensor> Transpose(const Tensor& input, const TensorShape& input_shape_override,
                                         const gsl::span<const size_t>& permutation, AllocatorPtr allocator,
                                         void* einsum_cuda_assets, const DeviceHelpers::Transpose& device_transpose_func) {
  auto input_rank = input_shape_override.NumDimensions();
  ORT_ENFORCE(input_rank == permutation.size(), "Length of permutation must match the rank of the input to be permutated");

  TensorShapeVector output_dims;
  output_dims.reserve(input_rank);

  for (const auto& dim : permutation) {
    output_dims.push_back(input_shape_override[dim]);
  }

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  std::unique_ptr<Tensor> output = std::make_unique<Tensor>(input.DataType(), output_dims, allocator);

  TensorShape overridden_shape(input_shape_override);

  auto status = device_transpose_func(permutation, input, *output, &overridden_shape, einsum_cuda_assets);

  if (!status.IsOK()) {
    ORT_THROW(common::ONNXRUNTIME, common::FAIL, "Einsum op: Transpose failed: ", status.ErrorMessage());
  }
  return output;
}

// Thin wrapper over the MatMul op to be called from Einsum that does some checks and invokes the device specific helper
// Not using the MatMulHelper for checks and to compute output dims as it adds a lot of checking overhead involving transposes of the inputs
// In our case, we have a more simplistic version which doesn't need to have those checks
template <typename T>
inline std::unique_ptr<Tensor> MatMul(const Tensor& input_1, const gsl::span<const int64_t>& input_shape_1_override,
                                      const Tensor& input_2, const gsl::span<const int64_t>& input_shape_2_override,
                                      AllocatorPtr allocator, concurrency::ThreadPool* tp,
                                      const void* mlas_backend_config,
                                      void* einsum_cuda_assets,
                                      const DeviceHelpers::MatMul<T>& device_matmul_func) {
  // Sanity checks before the actual MatMul
  ORT_ENFORCE(input_1.DataType() == input_2.DataType(), "Data types of the inputs must match for MatMul");
  ORT_ENFORCE(input_shape_1_override.size() == 3 && input_shape_2_override.size() == 3, "Only 1 batch dimension is allowed for MatMul");
  ORT_ENFORCE(input_shape_1_override[0] == input_shape_2_override[0], "Batch dimension should match for MatMul;");
  ORT_ENFORCE(input_shape_1_override[2] == input_shape_2_override[1], "Incompatible matrix dimensions for matMul");

  size_t batches = static_cast<size_t>(input_shape_1_override[0]);
  size_t M = static_cast<size_t>(input_shape_1_override[1]);
  size_t K = static_cast<size_t>(input_shape_1_override[2]);
  size_t N = static_cast<size_t>(input_shape_2_override[2]);

  size_t left_offset = M * K;
  size_t right_offset = K * N;
  size_t output_offset = M * N;

  TensorShapeVector output_dims;
  output_dims.reserve(3);
  output_dims.push_back(static_cast<int64_t>(batches));
  output_dims.push_back(static_cast<int64_t>(M));
  output_dims.push_back(static_cast<int64_t>(N));

  // Pass in allocator as that will be used as an allocator deleter by the framework
  // and it will de-allocate the memory for this intermediate tensor when it goes out of scope
  std::unique_ptr<Tensor> output = std::make_unique<Tensor>(input_1.DataType(), output_dims, allocator);

  const T* input_1_data = input_1.Data<T>();
  const T* input_2_data = input_2.Data<T>();
  T* output_data = output->MutableData<T>();

  auto status = device_matmul_func(input_1_data, input_2_data, output_data,
                                   left_offset, right_offset, output_offset, batches, M, K, N, tp, mlas_backend_config, einsum_cuda_assets);

  if (!status.IsOK()) {
    ORT_THROW(common::ONNXRUNTIME, common::FAIL, "Einsum op: Exception during MatMul operation: ",
              status.ErrorMessage());
  }

  return output;
}

// Thin wrapper over the ReduceSum op
template <typename T>
inline std::unique_ptr<Tensor> ReduceSum(const Tensor& input, const TensorShape& input_shape_override,
                                         gsl::span<const int64_t> reduce_axes, AllocatorPtr allocator,
                                         concurrency::ThreadPool* tp, void* einsum_cuda_assets,
                                         const DeviceHelpers::ReduceSum<T>& device_reduce_sum_func) {
  return device_reduce_sum_func(input, reduce_axes, true, allocator, &input_shape_override, tp, einsum_cuda_assets);
}

}  // namespace EinsumOp
}  // namespace onnxruntime
