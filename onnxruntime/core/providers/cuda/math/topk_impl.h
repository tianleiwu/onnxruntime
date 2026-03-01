// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once
#include <stdint.h>
#include "core/providers/cuda/shared_inc/cuda_utils.h"
#include "core/providers/cuda/cuda_kernel.h"
#include "core/common/common.h"

namespace onnxruntime {
namespace cuda {

template <typename T>
#ifndef BUILD_CUDA_EP_AS_PLUGIN
Status TopKImpl(const CudaKernel* kernel, bool use_deterministic_compute, onnxruntime::Stream* ort_stream,
#else
Status TopKImpl(const CudaKernel* kernel, bool use_deterministic_compute, cudaStream_t stream,
#endif
                const T* input_x, T* output_v, int64_t* output_i, const TArray<int64_t>& elem_nums,
                size_t size, int32_t axis, int64_t K, int64_t largest, int64_t sorted, int64_t N, int64_t dimension);

}  // namespace cuda
}  // namespace onnxruntime
