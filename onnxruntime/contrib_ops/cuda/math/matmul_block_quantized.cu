// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_quantized.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime::contrib::cuda {

#if !defined(DISABLE_FLOAT8_TYPES) && CUDA_VERSION >= 11080
template <typename InputType, typename OutputType, typename ScaleType>
__global__ void MatMulBlockQuantizedKernel(const InputType* input_a,
                                           const __nv_fp8_e4m3* input_b,
                                           const ScaleType* scale_a,
                                           const ScaleType* scale_b,
                                           OutputType* output,
                                           int m,
                                           int n,
                                           int k,
                                           int block_size) {
  const int column = blockIdx.x * blockDim.x + threadIdx.x;
  const int row = blockIdx.y * blockDim.y + threadIdx.y;
  if (row >= m || column >= n) {
    return;
  }

  float sum = 0.0f;
  for (int k_index = 0; k_index < k; ++k_index) {
    const int scale_index = k_index / block_size;
    const float a_value = static_cast<float>(input_a[row * k + k_index]) *
                          static_cast<float>(scale_a[row * ((k + block_size - 1) / block_size) + scale_index]);
    const float b_value = static_cast<float>(input_b[k_index * n + column]) *
                          static_cast<float>(scale_b[scale_index * n + column]);
    sum += a_value * b_value;
  }

  output[row * n + column] = OutputType(sum);
}
#endif

namespace {
__global__ void ConvertHalfToFloatKernel(const half* src, float* dst, int64_t count) {
  const int64_t index = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (index < count) {
    dst[index] = __half2float(src[index]);
  }
}
}  // namespace

void LaunchConvertHalfToFloat(const void* src_fp16, float* dst, int64_t count, cudaStream_t stream) {
  if (count <= 0) {
    return;
  }
  constexpr int threads = 256;
  const int blocks = static_cast<int>((count + threads - 1) / threads);
  ConvertHalfToFloatKernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const half*>(src_fp16), dst, count);
}

Status LaunchMatMulBlockQuantized(const void* input_a,
                                  const void* input_b,
                                  const void* scale_a,
                                  const void* scale_b,
                                  void* output,
                                  int m,
                                  int n,
                                  int k,
                                  int block_size,
                                  bool fp16_io,
                                  bool fp16_scales,
                                  cudaStream_t stream) {
#if !defined(DISABLE_FLOAT8_TYPES) && CUDA_VERSION >= 11080
  constexpr dim3 threads{16, 16};
  const dim3 blocks{static_cast<unsigned int>((n + threads.x - 1) / threads.x),
                    static_cast<unsigned int>((m + threads.y - 1) / threads.y)};
  const auto* b = reinterpret_cast<const __nv_fp8_e4m3*>(input_b);
  if (fp16_io && fp16_scales) {
    MatMulBlockQuantizedKernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const half*>(input_a), b,
                                                               reinterpret_cast<const half*>(scale_a), reinterpret_cast<const half*>(scale_b),
                                                               reinterpret_cast<half*>(output), m, n, k, block_size);
  } else if (fp16_io) {
    MatMulBlockQuantizedKernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const half*>(input_a), b,
                                                               reinterpret_cast<const float*>(scale_a), reinterpret_cast<const float*>(scale_b),
                                                               reinterpret_cast<half*>(output), m, n, k, block_size);
  } else if (fp16_scales) {
    MatMulBlockQuantizedKernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_fp8_e4m3*>(input_a), b,
                                                               reinterpret_cast<const half*>(scale_a), reinterpret_cast<const half*>(scale_b),
                                                               reinterpret_cast<__nv_bfloat16*>(output), m, n, k, block_size);
  } else {
    MatMulBlockQuantizedKernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_fp8_e4m3*>(input_a), b,
                                                               reinterpret_cast<const float*>(scale_a), reinterpret_cast<const float*>(scale_b),
                                                               reinterpret_cast<__nv_bfloat16*>(output), m, n, k, block_size);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(input_a);
  ORT_UNUSED_PARAMETER(input_b);
  ORT_UNUSED_PARAMETER(scale_a);
  ORT_UNUSED_PARAMETER(scale_b);
  ORT_UNUSED_PARAMETER(output);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(fp16_io);
  ORT_UNUSED_PARAMETER(fp16_scales);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockQuantized requires CUDA 11.8 or later.");
#endif
}

}  // namespace onnxruntime::contrib::cuda