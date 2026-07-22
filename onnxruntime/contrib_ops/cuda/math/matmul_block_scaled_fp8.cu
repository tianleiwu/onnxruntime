// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/math/matmul_block_scaled_fp8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp8.h>

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime::contrib::cuda {

#if !defined(DISABLE_FLOAT8_TYPES) && CUDA_VERSION >= 11080
template <typename InputType, typename OutputType, typename ScaleType>
__global__ void MatMulBlockScaledFp8Kernel(const InputType* input_a,
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

  const int k_blocks = (k + block_size - 1) / block_size;
  float sum = 0.0f;
  for (int k_index = 0; k_index < k; ++k_index) {
    const int scale_index = k_index / block_size;
    const float a_value = static_cast<float>(input_a[row * k + k_index]) *
                          static_cast<float>(scale_a[row * k_blocks + scale_index]);
    const float b_value = static_cast<float>(input_b[column * k + k_index]) *
                          static_cast<float>(scale_b[column * k_blocks + scale_index]);
    sum += a_value * b_value;
  }

  output[row * n + column] = OutputType(sum);
}

// -----------------------------------------------------------------------------
// Fused GEMV fast path for the decode phase (small M).
//
// Each warp computes exactly one output element Y[row, col]. The 32 lanes of the
// warp cooperatively reduce over K using 16-wide vectorized loads, so the packed
// FP8 weight B is streamed exactly once with fully coalesced 512-byte warp
// transactions. Block scales are applied once per K-block (not per element): a
// lane's 16-element chunk is guaranteed to lie inside a single K-block whenever
// block_size is a multiple of 16, so scale_a/scale_b are folded in per chunk.
//
// This avoids both the materialized dequant buffer and the tensor-core GEMM,
// which is heavily underutilized at M == 1. It runs on any architecture
// (SM80/SM90/SM100/SM120) because it only relies on warp-shuffle intrinsics.
template <typename AType>
__device__ __forceinline__ void LoadFp8Gemv16A(const AType* ptr, float (&out)[16]);

template <>
__device__ __forceinline__ void LoadFp8Gemv16A<__nv_fp8_e4m3>(const __nv_fp8_e4m3* ptr, float (&out)[16]) {
  const uint4 raw = *reinterpret_cast<const uint4*>(ptr);
  const __nv_fp8_e4m3* v = reinterpret_cast<const __nv_fp8_e4m3*>(&raw);
#pragma unroll
  for (int i = 0; i < 16; ++i) {
    out[i] = static_cast<float>(v[i]);
  }
}

template <>
__device__ __forceinline__ void LoadFp8Gemv16A<half>(const half* ptr, float (&out)[16]) {
  const uint4* p = reinterpret_cast<const uint4*>(ptr);
  const uint4 lo = p[0];
  const uint4 hi = p[1];
  const half* vlo = reinterpret_cast<const half*>(&lo);
  const half* vhi = reinterpret_cast<const half*>(&hi);
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[i] = __half2float(vlo[i]);
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    out[8 + i] = __half2float(vhi[i]);
  }
}

template <int RowsPerWarp, typename AType, typename OutputType, typename ScaleType>
__global__ void MatMulBlockScaledFp8GemvKernel(const AType* __restrict__ input_a,
                                               const __nv_fp8_e4m3* __restrict__ input_b,
                                               const ScaleType* __restrict__ scale_a,
                                               const ScaleType* __restrict__ scale_b,
                                               OutputType* __restrict__ output,
                                               int m,
                                               int n,
                                               int k,
                                               int block_size,
                                               int k_blocks) {
  const int lane = threadIdx.x;                           // 0..31
  const int col = blockIdx.x * blockDim.y + threadIdx.y;  // n
  const int row_base = blockIdx.y * RowsPerWarp;          // m
  if (row_base >= m || col >= n) {
    return;
  }

  const __nv_fp8_e4m3* b_row = input_b + static_cast<size_t>(col) * k;
  const ScaleType* sb_row = scale_b + static_cast<size_t>(col) * k_blocks;

  constexpr int kElemsPerLane = 16;
  const int stride = 32 * kElemsPerLane;  // 512 elements per warp iteration

  float acc[RowsPerWarp] = {};
  for (int base = 0; base < k; base += stride) {
    const int koff = base + lane * kElemsPerLane;
    if (koff < k) {
      const uint4 b_raw = *reinterpret_cast<const uint4*>(b_row + koff);
      const __nv_fp8_e4m3* bp = reinterpret_cast<const __nv_fp8_e4m3*>(&b_raw);
      const int kb = koff / block_size;
      const float b_scale = static_cast<float>(sb_row[kb]);
#pragma unroll
      for (int row_offset = 0; row_offset < RowsPerWarp; ++row_offset) {
        const int row = row_base + row_offset;
        if (row < m) {
          const AType* a_row = input_a + static_cast<size_t>(row) * k;
          const ScaleType* sa_row = scale_a + static_cast<size_t>(row) * k_blocks;
          float a_vals[16];
          LoadFp8Gemv16A<AType>(a_row + koff, a_vals);

          float partial = 0.0f;
#pragma unroll
          for (int i = 0; i < kElemsPerLane; ++i) {
            partial += a_vals[i] * static_cast<float>(bp[i]);
          }
          acc[row_offset] += partial * static_cast<float>(sa_row[kb]) * b_scale;
        }
      }
    }
  }

#pragma unroll
  for (int row_offset = 0; row_offset < RowsPerWarp; ++row_offset) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      acc[row_offset] += __shfl_down_sync(0xffffffffu, acc[row_offset], offset);
    }
  }
  if (lane == 0) {
#pragma unroll
    for (int row_offset = 0; row_offset < RowsPerWarp; ++row_offset) {
      const int row = row_base + row_offset;
      if (row < m) {
        output[static_cast<size_t>(row) * n + col] = OutputType(acc[row_offset]);
      }
    }
  }
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

Status LaunchMatMulBlockScaledFp8(const void* input_a,
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
    MatMulBlockScaledFp8Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const half*>(input_a), b,
                                                               reinterpret_cast<const half*>(scale_a), reinterpret_cast<const half*>(scale_b),
                                                               reinterpret_cast<half*>(output), m, n, k, block_size);
  } else if (fp16_io) {
    MatMulBlockScaledFp8Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const half*>(input_a), b,
                                                               reinterpret_cast<const float*>(scale_a), reinterpret_cast<const float*>(scale_b),
                                                               reinterpret_cast<half*>(output), m, n, k, block_size);
  } else if (fp16_scales) {
    MatMulBlockScaledFp8Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_fp8_e4m3*>(input_a), b,
                                                               reinterpret_cast<const half*>(scale_a), reinterpret_cast<const half*>(scale_b),
                                                               reinterpret_cast<__nv_bfloat16*>(output), m, n, k, block_size);
  } else {
    MatMulBlockScaledFp8Kernel<<<blocks, threads, 0, stream>>>(reinterpret_cast<const __nv_fp8_e4m3*>(input_a), b,
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
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockScaledFp8 requires CUDA 11.8 or later.");
#endif
}

Status LaunchMatMulBlockScaledFp8Gemv(const void* input_a,
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
  const int k_blocks = (k + block_size - 1) / block_size;
  constexpr int kWarpsPerBlock = 8;
  const dim3 threads{32, kWarpsPerBlock};
  const auto* b = reinterpret_cast<const __nv_fp8_e4m3*>(input_b);
  const auto launch = [&]<int RowsPerWarp>() {
    const dim3 blocks{static_cast<unsigned int>((n + kWarpsPerBlock - 1) / kWarpsPerBlock),
                      static_cast<unsigned int>((m + RowsPerWarp - 1) / RowsPerWarp)};
    if (fp16_io && fp16_scales) {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(input_a), b,
          reinterpret_cast<const half*>(scale_a), reinterpret_cast<const half*>(scale_b),
          reinterpret_cast<half*>(output), m, n, k, block_size, k_blocks);
    } else if (fp16_io) {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const half*>(input_a), b,
          reinterpret_cast<const float*>(scale_a), reinterpret_cast<const float*>(scale_b),
          reinterpret_cast<half*>(output), m, n, k, block_size, k_blocks);
    } else if (fp16_scales) {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_fp8_e4m3*>(input_a), b,
          reinterpret_cast<const half*>(scale_a), reinterpret_cast<const half*>(scale_b),
          reinterpret_cast<__nv_bfloat16*>(output), m, n, k, block_size, k_blocks);
    } else {
      MatMulBlockScaledFp8GemvKernel<RowsPerWarp><<<blocks, threads, 0, stream>>>(
          reinterpret_cast<const __nv_fp8_e4m3*>(input_a), b,
          reinterpret_cast<const float*>(scale_a), reinterpret_cast<const float*>(scale_b),
          reinterpret_cast<__nv_bfloat16*>(output), m, n, k, block_size, k_blocks);
    }
  };

  if (m == 1) {
    launch.template operator()<1>();
  } else if (m <= 2) {
    launch.template operator()<2>();
  } else {
    launch.template operator()<4>();
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
  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "MatMulBlockScaledFp8 requires CUDA 11.8 or later.");
#endif
}

}  // namespace onnxruntime::contrib::cuda