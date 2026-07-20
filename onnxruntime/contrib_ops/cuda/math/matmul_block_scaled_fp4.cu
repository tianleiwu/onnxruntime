// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp4.h"

#include <cuda.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
#include <cuda_fp4.h>
#include <cuda_fp8.h>
#endif

#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime::contrib::cuda {

#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080

namespace {

template <typename T>
__device__ __forceinline__ T FromFloat(float v);

template <>
__device__ __forceinline__ half FromFloat<half>(float v) {
  return __float2half(v);
}

template <>
__device__ __forceinline__ nv_bfloat16 FromFloat<nv_bfloat16>(float v) {
  return __float2bfloat16(v);
}

template <typename T>
__device__ __forceinline__ float ToFloat(T v);

template <>
__device__ __forceinline__ float ToFloat<half>(half v) {
  return __half2float(v);
}

template <>
__device__ __forceinline__ float ToFloat<nv_bfloat16>(nv_bfloat16 v) {
  return __bfloat162float(v);
}

template <typename T>
__global__ void DequantizeNvFp4Kernel(T* __restrict__ out,
                                      const uint8_t* __restrict__ b_packed,
                                      const uint8_t* __restrict__ weight_scale,
                                      const float* __restrict__ weight_scale_2,
                                      int n,
                                      int k,
                                      int k_blocks,
                                      int block_size) {
  const int half_k = k >> 1;
  const long long total = static_cast<long long>(n) * half_k;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }

  const int row = static_cast<int>(idx / half_k);
  const int pair = static_cast<int>(idx - static_cast<long long>(row) * half_k);
  const int k0 = pair << 1;

  const uint8_t packed = b_packed[idx];
  const __half2_raw hr = __nv_cvt_fp4x2_to_halfraw2(static_cast<__nv_fp4x2_storage_t>(packed), __NV_E2M1);
  const __half2 h2 = __half2(hr);
  const float2 v = __half22float2(h2);

  const float g = *weight_scale_2;
  const int blk0 = k0 / block_size;
  const int blk1 = (k0 + 1) / block_size;
  const float s0 = __half2float(__nv_cvt_fp8_to_halfraw(
                       static_cast<__nv_fp8_storage_t>(weight_scale[row * k_blocks + blk0]), __NV_E4M3)) *
                   g;
  const float s1 = __half2float(__nv_cvt_fp8_to_halfraw(
                       static_cast<__nv_fp8_storage_t>(weight_scale[row * k_blocks + blk1]), __NV_E4M3)) *
                   g;

  const long long out_base = static_cast<long long>(row) * k + k0;
  out[out_base] = FromFloat<T>(v.x * s0);
  out[out_base + 1] = FromFloat<T>(v.y * s1);
}

template <typename T>
__global__ void AddBiasKernel(T* __restrict__ y, const T* __restrict__ bias, int m, int n) {
  const long long total = static_cast<long long>(m) * n;
  const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (idx >= total) {
    return;
  }
  const int col = static_cast<int>(idx % n);
  y[idx] = FromFloat<T>(ToFloat<T>(y[idx]) + ToFloat<T>(bias[col]));
}

}  // namespace

#endif  // CUDA_VERSION >= 12080

Status LaunchDequantizeNvFp4(void* b_dequant,
                             const void* b_packed,
                             const void* weight_scale,
                             const float* weight_scale_2,
                             int n,
                             int k,
                             int block_size,
                             bool is_bf16,
                             cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  const int half_k = k >> 1;
  const long long total = static_cast<long long>(n) * half_k;
  if (total == 0) {
    return Status::OK();
  }
  const int k_blocks = (k + block_size - 1) / block_size;
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  const uint8_t* bp = reinterpret_cast<const uint8_t*>(b_packed);
  const uint8_t* ws = reinterpret_cast<const uint8_t*>(weight_scale);

  if (is_bf16) {
    DequantizeNvFp4Kernel<nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(b_dequant), bp, ws, weight_scale_2, n, k, k_blocks, block_size);
  } else {
    DequantizeNvFp4Kernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(b_dequant), bp, ws, weight_scale_2, n, k, k_blocks, block_size);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(b_dequant);
  ORT_UNUSED_PARAMETER(b_packed);
  ORT_UNUSED_PARAMETER(weight_scale);
  ORT_UNUSED_PARAMETER(weight_scale_2);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(k);
  ORT_UNUSED_PARAMETER(block_size);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockScaledFp4 requires CUDA 12.8 or newer for NVFP4 support.");
#endif
}

Status LaunchAddBiasNvFp4(void* y,
                          const void* bias,
                          int m,
                          int n,
                          bool is_bf16,
                          cudaStream_t stream) {
#if defined(CUDA_VERSION) && CUDA_VERSION >= 12080
  const long long total = static_cast<long long>(m) * n;
  if (total == 0) {
    return Status::OK();
  }
  constexpr int kThreads = 256;
  const int blocks = static_cast<int>((total + kThreads - 1) / kThreads);
  if (is_bf16) {
    AddBiasKernel<nv_bfloat16><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<nv_bfloat16*>(y), reinterpret_cast<const nv_bfloat16*>(bias), m, n);
  } else {
    AddBiasKernel<half><<<blocks, kThreads, 0, stream>>>(
        reinterpret_cast<half*>(y), reinterpret_cast<const half*>(bias), m, n);
  }
  return CUDA_CALL(cudaGetLastError());
#else
  ORT_UNUSED_PARAMETER(y);
  ORT_UNUSED_PARAMETER(bias);
  ORT_UNUSED_PARAMETER(m);
  ORT_UNUSED_PARAMETER(n);
  ORT_UNUSED_PARAMETER(is_bf16);
  ORT_UNUSED_PARAMETER(stream);
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockScaledFp4 requires CUDA 12.8 or newer for NVFP4 support.");
#endif
}

}  // namespace onnxruntime::contrib::cuda
