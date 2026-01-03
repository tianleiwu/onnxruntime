// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/attention_impl.h"
#include "contrib_ops/cuda/bert/attention_kv_cache.h"
#include "core/providers/cuda/cu_inc/common.cuh"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void ConcatTensorToTensor(const int tensor_add_sequence_length,
                                     const T* tensor_in,
                                     const T* tensor_add,
                                     T* tensor_out) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int chunk_id = blockIdx.z;

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  // K: number of identical tensors
  // tensor_in:    K x BxNxPxH
  // tensor_add:   K x BxNxLxH
  // tensor_out:   K x BxNxTxH, where T = P + L
  const int tensor_in_sequence_length = all_sequence_length - tensor_add_sequence_length;

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  int out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
  if (s < tensor_in_sequence_length) {
    const int past_SH = tensor_in_sequence_length * H;
    const int past_NSH = num_heads * past_SH;
    const int in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
    tensor_out[out_offset] = tensor_in[in_offset];
  } else if (s < all_sequence_length) {
    const int SH = tensor_add_sequence_length * H;
    const int NSH = num_heads * SH;
    const int in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
    tensor_out[out_offset] = tensor_add[in_offset];
  }
}

template <typename T>
__global__ void ConcatTensorToTensorLarge(const int tensor_add_sequence_length,
                                          const int H,
                                          const T* tensor_in,
                                          const T* tensor_add,
                                          T* tensor_out) {
  // Use when (H*)*num_heads > 1024
  int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int chunk_id = blockIdx.z;

  const int all_sequence_length = gridDim.x;
  const int batch_size = gridDim.y;
  const int num_heads = blockDim.y;
  const int stride = blockDim.x;

  // K: number of identical tensor
  // tensor_in:    K x BxNxPxH
  // tensor_add:   K x BxNxLxH
  // tensor_out:   K x BxNxTxH
  const int tensor_in_sequence_length = all_sequence_length - tensor_add_sequence_length;

  const int present_SH = all_sequence_length * H;
  const int present_NSH = num_heads * present_SH;
  while (h < H) {
    int out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
    if (s < tensor_in_sequence_length) {
      const int past_SH = tensor_in_sequence_length * H;
      const int past_NSH = num_heads * past_SH;
      const int in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
      tensor_out[out_offset] = tensor_in[in_offset];
    } else if (s < all_sequence_length) {
      const int SH = tensor_add_sequence_length * H;
      const int NSH = num_heads * SH;
      const int in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
      tensor_out[out_offset] = tensor_add[in_offset];
    }

    h += stride;
  }
}

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const float* tensor_in,
                                  const float* tensor_add,
                                  float* tensor_out) {
  const dim3 grid(all_sequence_length, batch_size, matrix_num);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                               reinterpret_cast<const float2*>(tensor_in),
                                                               reinterpret_cast<const float2*>(tensor_add),
                                                               reinterpret_cast<float2*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                                    H,
                                                                    reinterpret_cast<const float2*>(tensor_in),
                                                                    reinterpret_cast<const float2*>(tensor_add),
                                                                    reinterpret_cast<float2*>(tensor_out));
    }
  } else {
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatTensorToTensor<float><<<grid, block, 0, stream>>>(sequence_length, tensor_in, tensor_add, tensor_out);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float><<<grid, block, 0, stream>>>(sequence_length,
                                                                   head_size,
                                                                   tensor_in,
                                                                   tensor_add,
                                                                   tensor_out);
    }
  }
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const half* tensor_in,
                                  const half* tensor_add,
                                  half* tensor_out) {
  const dim3 grid(all_sequence_length, batch_size, matrix_num);
  if (0 == (head_size % 4)) {
    const int H = head_size / 4;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                               reinterpret_cast<const float2*>(tensor_in),
                                                               reinterpret_cast<const float2*>(tensor_add),
                                                               reinterpret_cast<float2*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float2><<<grid, block, 0, stream>>>(sequence_length,
                                                                    H,
                                                                    reinterpret_cast<const float2*>(tensor_in),
                                                                    reinterpret_cast<const float2*>(tensor_add),
                                                                    reinterpret_cast<float2*>(tensor_out));
    }
  } else if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<half2><<<grid, block, 0, stream>>>(sequence_length,
                                                              reinterpret_cast<const half2*>(tensor_in),
                                                              reinterpret_cast<const half2*>(tensor_add),
                                                              reinterpret_cast<half2*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<half2><<<grid, block, 0, stream>>>(sequence_length,
                                                                   H,
                                                                   reinterpret_cast<const half2*>(tensor_in),
                                                                   reinterpret_cast<const half2*>(tensor_add),
                                                                   reinterpret_cast<half2*>(tensor_out));
    }
  } else {  // this should be an "odd" case. probably not worth catching it in the half2 kernel.
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatTensorToTensor<half><<<grid, block, 0, stream>>>(sequence_length, tensor_in, tensor_add, tensor_out);
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<half><<<grid, block, 0, stream>>>(sequence_length,
                                                                  head_size,
                                                                  tensor_in,
                                                                  tensor_add,
                                                                  tensor_out);
    }
  }
  return CUDA_CALL(cudaGetLastError());
}

Status LaunchConcatTensorToTensor(cudaStream_t stream,
                                  const int all_sequence_length,
                                  const int sequence_length,
                                  const int batch_size,
                                  const int head_size,
                                  const int num_heads,
                                  const int max_threads_per_block,
                                  const int matrix_num,
                                  const BFloat16* tensor_in,
                                  const BFloat16* tensor_add,
                                  BFloat16* tensor_out) {
  assert(num_heads <= max_threads_per_block);
  const dim3 grid(all_sequence_length, batch_size, matrix_num);
  if (0 == (head_size & 1)) {
    const int H = head_size / 2;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<__nv_bfloat162><<<grid, block, 0, stream>>>(
          sequence_length,
          reinterpret_cast<const __nv_bfloat162*>(tensor_in),
          reinterpret_cast<const __nv_bfloat162*>(tensor_add),
          reinterpret_cast<__nv_bfloat162*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<__nv_bfloat162><<<grid, block, 0, stream>>>(
          sequence_length,
          H,
          reinterpret_cast<const __nv_bfloat162*>(tensor_in),
          reinterpret_cast<const __nv_bfloat162*>(tensor_add),
          reinterpret_cast<__nv_bfloat162*>(tensor_out));
    }
  } else {
    if (head_size * num_heads <= max_threads_per_block) {
      const dim3 block(head_size, num_heads, 1);
      ConcatTensorToTensor<__nv_bfloat16><<<grid, block, 0, stream>>>(
          sequence_length,
          reinterpret_cast<const __nv_bfloat16*>(tensor_in),
          reinterpret_cast<const __nv_bfloat16*>(tensor_add),
          reinterpret_cast<__nv_bfloat16*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<__nv_bfloat16><<<grid, block, 0, stream>>>(
          sequence_length,
          head_size,
          reinterpret_cast<const __nv_bfloat16*>(tensor_in),
          reinterpret_cast<const __nv_bfloat16*>(tensor_add),
          reinterpret_cast<__nv_bfloat16*>(tensor_out));
    }
  }

  return CUDA_CALL(cudaGetLastError());
}

// ----------------------------------------------------------------------------------
// Below kernels are for past and present sharing buffer
// ----------------------------------------------------------------------------------

template <typename T>
__global__ void AddBiasTransAppendKvToPresentSmall(
    const T* qkv, const T* biases, T* present,
    const int head_size, const int past_sequence_length, const int max_sequence_length) {
  // Input:  BxSxMxNxH  (Format 1)
  // Output: (2, B, N, [P..P+S) of MaxS, H),
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int N = blockDim.y;
  const int S = gridDim.x;
  const int B = gridDim.y;

  constexpr int M = 3;           // Matrix count in qkv
  const int m = blockIdx.z + 1;  // k = 1, v = 2

  const int NH = N * head_size;
  const int NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int MsH = max_sequence_length * head_size;
  const int NMsH = N * MsH;
  const int BNMsH = B * NMsH;
  present += ((past_sequence_length + s) * head_size + n * MsH + b * NMsH + (m - 1) * BNMsH);

  for (int h = threadIdx.x; h < head_size; h += blockDim.x) {
    T bias = (biases ? biases[h] : (T)0.0f);
    present[h] = qkv[h] + bias;
  }
}

template <typename T>
__global__ void AddBiasTransAppendKvToPresent(
    const T* qkv, const T* biases, T* present,
    const int head_size, const int past_sequence_length, const int max_sequence_length) {
  // Input:  BxSxMxNxH  (Format 1)
  // Output: (2, B, N, [P..P+S) of MaxS, H),
  // B is batch_size, S is sequence_length, M is number of matrices, N is num_heads, H is head_size
  const int n = blockIdx.x;
  const int s = blockIdx.y;
  const int b = (blockIdx.z >> 1);
  const int N = gridDim.x;
  const int S = gridDim.y;
  const int B = (gridDim.z >> 1);

  constexpr int M = 3;                   // Matrix count in qkv
  const int m = (blockIdx.z & 0x1) + 1;  // k = 1, v = 2

  const int NH = N * head_size;
  const int NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int MsH = max_sequence_length * head_size;
  const int NMsH = N * MsH;
  const int BNMsH = B * NMsH;
  present += ((past_sequence_length + s) * head_size + n * MsH + b * NMsH + (m - 1) * BNMsH);

  for (int h = threadIdx.x; h < head_size; h += blockDim.x) {
    T bias = (biases ? biases[h] : (T)0.0f);
    present[h] = qkv[h] + bias;
  }
}

// qkv buffer is merged tensor of shape (B,S,3,N,H), k v is the second/third of the 3.
// bias is of shape (3, NxH) or nullptr
// append to present of (2, B, N, (P..T) of M, H),
template <typename T>
Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                           const int max_sequence_length,
                                           const int past_sequence_length,
                                           const int sequence_length,
                                           const int batch_size,
                                           const int head_size,
                                           const int num_heads,
                                           const int max_threads_per_block,
                                           const T* biases,
                                           const T* qkv_buffer,
                                           T* present) {
  assert(head_size <= (1 << 30));

  int64_t nh = (int64_t)head_size * num_heads;
  if (nh <= max_threads_per_block) {
    const dim3 grid(sequence_length, batch_size, 2);  // 2 for k and v
    const dim3 block(max_threads_per_block / num_heads, num_heads, 1);

    AddBiasTransAppendKvToPresentSmall<T><<<grid, block, 0, stream>>>(
        qkv_buffer, biases, present, head_size, past_sequence_length, max_sequence_length);
  } else {
    const dim3 grid(num_heads, sequence_length, batch_size * 2);  // 2 for k and v
    const dim3 block(std::min(head_size, max_threads_per_block), 1, 1);
    AddBiasTransAppendKvToPresent<T><<<grid, block, 0, stream>>>(
        qkv_buffer, biases, present, head_size, past_sequence_length, max_sequence_length);
  }

  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                                    const int max_sequence_length,
                                                    const int total_sequence_length,
                                                    const int sequence_length,
                                                    const int batch_size,
                                                    const int head_size,
                                                    const int num_heads,
                                                    const int max_threads_per_block,
                                                    const float* bias,
                                                    const float* qkv_buffer,
                                                    float* present);

template Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                                    const int max_sequence_length,
                                                    const int total_sequence_length,
                                                    const int sequence_length,
                                                    const int batch_size,
                                                    const int head_size,
                                                    const int num_heads,
                                                    const int max_threads_per_block,
                                                    const half* bias,
                                                    const half* qkv_buffer,
                                                    half* present);

template Status LaunchAddBiasTransAppendKvToPresent(cudaStream_t stream,
                                                    const int max_sequence_length,
                                                    const int total_sequence_length,
                                                    const int sequence_length,
                                                    const int batch_size,
                                                    const int head_size,
                                                    const int num_heads,
                                                    const int max_threads_per_block,
                                                    const BFloat16* bias,
                                                    const BFloat16* qkv_buffer,
                                                    BFloat16* present);

// Kernel to append new and past kv in either BSNH or BNSH format
// Adapted from ConcatTensorToTensor kernel in attention_kv_cache.cu file
// Dispatcher for RoPE application based on VectorT and ElementT
template <typename VectorT, typename ElementT>
struct RotaryDispatcher {
  __device__ static void apply(VectorT& /*val*/, const VectorT* /*cos_cache*/, const VectorT* /*sin_cache*/, const int /*rotary_dim*/, const int /*h_idx*/, const int /*pos_id*/, const bool /*interleaved*/, const VectorT* /*new_kv_base*/, const int /*in_offset*/) {
    // Default implementation: skip
  }
};

template <>
struct RotaryDispatcher<float2, float> {
  __device__ static void apply(float2& val, const float2* cos_cache, const float2* sin_cache, const int rotary_dim, const int h_idx, const int pos_id, const bool interleaved, const float2* new_kv_base, const int in_offset) {
    if (2 * h_idx >= rotary_dim) return;

    const float* cos_ptr = reinterpret_cast<const float*>(cos_cache);
    const float* sin_ptr = reinterpret_cast<const float*>(sin_cache);
    const float* kv_ptr = reinterpret_cast<const float*>(new_kv_base);
    int scalar_in_offset = in_offset * 2;
    int scalar_h = h_idx * 2;
    int half_rot = rotary_dim / 2;

    float c, s;
    float x = val.x;
    float y = val.y;

    if (interleaved) {
      int cs_idx = pos_id * half_rot + h_idx;
      c = cos_ptr[cs_idx];
      s = sin_ptr[cs_idx];
      val.x = x * c - y * s;
      val.y = x * s + y * c;
    } else {
      if (scalar_h < half_rot) {
        int cs_idx = pos_id * half_rot + scalar_h;
        c = cos_ptr[cs_idx];
        s = sin_ptr[cs_idx];
        float pair_x = kv_ptr[scalar_in_offset + scalar_h + half_rot];
        val.x = x * c - pair_x * s;
      } else {
        int cs_idx = pos_id * half_rot + (scalar_h - half_rot);
        c = cos_ptr[cs_idx];
        s = sin_ptr[cs_idx];
        float pair_x = kv_ptr[scalar_in_offset + scalar_h - half_rot];
        val.x = x * c + pair_x * s;
      }

      int scalar_hy = scalar_h + 1;
      if (scalar_hy < half_rot) {
        int cs_idx = pos_id * half_rot + scalar_hy;
        c = cos_ptr[cs_idx];
        s = sin_ptr[cs_idx];
        float pair_y = kv_ptr[scalar_in_offset + scalar_hy + half_rot];
        val.y = y * c - pair_y * s;
      } else {
        int cs_idx = pos_id * half_rot + (scalar_hy - half_rot);
        c = cos_ptr[cs_idx];
        s = sin_ptr[cs_idx];
        float pair_y = kv_ptr[scalar_in_offset + scalar_hy - half_rot];
        val.y = y * c + pair_y * s;
      }
    }
  }
};

template <>
struct RotaryDispatcher<float4, float> {
  __device__ static void apply(float4& val, const float4* cos_cache, const float4* sin_cache, const int rotary_dim, const int h_idx, const int pos_id, const bool interleaved, const float4* new_kv_base, const int in_offset) {
    float2 p1 = make_float2(val.x, val.y);
    float2 p2 = make_float2(val.z, val.w);
    const float2* c = reinterpret_cast<const float2*>(cos_cache);
    const float2* s = reinterpret_cast<const float2*>(sin_cache);
    const float2* b = reinterpret_cast<const float2*>(new_kv_base);
    RotaryDispatcher<float2, float>::apply(p1, c, s, rotary_dim, h_idx * 2, pos_id, interleaved, b, in_offset * 2);
    RotaryDispatcher<float2, float>::apply(p2, c, s, rotary_dim, h_idx * 2 + 1, pos_id, interleaved, b, in_offset * 2);
    val.x = p1.x;
    val.y = p1.y;
    val.z = p2.x;
    val.w = p2.y;
  }
};

template <>
struct RotaryDispatcher<float2, half> {
  __device__ static void apply(float2& val, const float2* cos_cache, const float2* sin_cache, const int rotary_dim, const int h_idx, const int pos_id, const bool interleaved, const float2* new_kv_base, const int in_offset) {
    if (2 * h_idx * 2 >= rotary_dim) return;
    half2* v_ptr = reinterpret_cast<half2*>(&val);
    half2 v0 = v_ptr[0];
    half2 v1 = v_ptr[1];
    const half2* cos_ptr = reinterpret_cast<const half2*>(cos_cache);
    const half2* sin_ptr = reinterpret_cast<const half2*>(sin_cache);
    int half_rot = rotary_dim / 2;

    if (interleaved) {
      int f0 = 2 * h_idx;
      int cs0 = pos_id * half_rot + f0;
      half2 c_pair = cos_ptr[cs0 / 2];
      half2 s_pair = sin_ptr[cs0 / 2];

      half c0 = c_pair.x;
      half s0 = s_pair.x;
      float c0f = __half2float(c0);
      float s0f = __half2float(s0);
      float e0f = __half2float(v0.x);
      float e1f = __half2float(v0.y);
      v0.x = __float2half(e0f * c0f - e1f * s0f);
      v0.y = __float2half(e0f * s0f + e1f * c0f);

      half c1 = c_pair.y;
      half s1 = s_pair.y;
      float c1f = __half2float(c1);
      float s1f = __half2float(s1);
      float e2f = __half2float(v1.x);
      float e3f = __half2float(v1.y);
      v1.x = __float2half(e2f * c1f - e3f * s1f);
      v1.y = __float2half(e2f * s1f + e3f * c1f);
    }
    v_ptr[0] = v0;
    v_ptr[1] = v1;
  }
};

template <>
struct RotaryDispatcher<float2, BFloat16> {
  __device__ static void apply(float2& val, const float2* cos_cache, const float2* sin_cache, const int rotary_dim, const int h_idx, const int pos_id, const bool interleaved, const float2* new_kv_base, const int in_offset) {
    if (2 * h_idx * 2 >= rotary_dim) return;
    using namespace onnxruntime::cuda;
    __nv_bfloat162* v_ptr = reinterpret_cast<__nv_bfloat162*>(&val);
    __nv_bfloat162 v0 = v_ptr[0];
    __nv_bfloat162 v1 = v_ptr[1];
    const __nv_bfloat162* cos_ptr = reinterpret_cast<const __nv_bfloat162*>(cos_cache);
    const __nv_bfloat162* sin_ptr = reinterpret_cast<const __nv_bfloat162*>(sin_cache);
    int half_rot = rotary_dim / 2;

    if (interleaved) {
      int f0 = 2 * h_idx;
      int cs0 = pos_id * half_rot + f0;
      __nv_bfloat162 c_pair = cos_ptr[cs0 / 2];
      __nv_bfloat162 s_pair = sin_ptr[cs0 / 2];
      __nv_bfloat16 c0 = c_pair.x;
      __nv_bfloat16 s0 = s_pair.x;
      float c0f = __bfloat162float(c0);
      float s0f = __bfloat162float(s0);
      float e0f = __bfloat162float(v0.x);
      float e1f = __bfloat162float(v0.y);
      v0.x = __float2bfloat16(e0f * c0f - e1f * s0f);
      v0.y = __float2bfloat16(e0f * s0f + e1f * c0f);

      __nv_bfloat16 c1 = c_pair.y;
      __nv_bfloat16 s1 = s_pair.y;
      float c1f = __bfloat162float(c1);
      float s1f = __bfloat162float(s1);
      float e2f = __bfloat162float(v1.x);
      float e3f = __bfloat162float(v1.y);
      v1.x = __float2bfloat16(e2f * c1f - e3f * s1f);
      v1.y = __float2bfloat16(e2f * s1f + e3f * c1f);
    }
    v_ptr[0] = v0;
    v_ptr[1] = v1;
  }
};

template <typename T, typename ElementT>
__global__ void ConcatNewToPastKV(const int new_seqlen,
                                  const int past_buffer_seqlen,
                                  const T* past_kv,
                                  const T* new_kv,
                                  T* present_kv,
                                  const int* seqlens_k,
                                  const bool past_only,
                                  // const int* seqlens_q,
                                  const bool is_bsnh,
                                  const T* cos_cache,
                                  const T* sin_cache,
                                  const int rotary_dim,
                                  const int64_t* position_ids,
                                  const bool interleaved) {  // refers to past; otherwise bnsh
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int present_buffer_seqlen = gridDim.x;
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  const int present_batch_stride = present_buffer_seqlen * num_heads * H;
  const int row_stride = is_bsnh ? num_heads * H : H;
  const int present_head_stride = is_bsnh ? H : present_buffer_seqlen * H;

  // past_kv:     BPNH or BNPH
  // new_kv:      BLNH
  // present_kv:  BTNH or BNTH, where T = P + L

  // prompt, token, and interactive decoding cases
  const int past_seqlen = seqlens_k == nullptr ? 0 : seqlens_k[b] + 1 - new_seqlen;

  int out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;
  if (s < past_seqlen) {
    const int past_batch_stride = past_buffer_seqlen * num_heads * H;
    const int past_head_stride = is_bsnh ? H : past_buffer_seqlen * H;
    const int in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
    present_kv[out_offset] = past_kv[in_offset];
  } else if (!past_only && s < past_seqlen + new_seqlen) {
    // Note: new KV always BSNH
    const int new_batch_stride = new_seqlen * num_heads * H;
    const int new_row_stride = num_heads * H;
    const int new_head_stride = H;
    const int in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;

    // Apply Rotation if needed
    T val = new_kv[in_offset];
    if (cos_cache != nullptr && rotary_dim > 0) {
      int pos_id = 0;
      if (position_ids) {
        int new_s_idx = s - past_seqlen;
        if (new_s_idx >= 0 && new_s_idx < new_seqlen) {
          pos_id = static_cast<int>(position_ids[b * new_seqlen + new_s_idx]);
        } else {
          pos_id = s;
        }
      } else {
        pos_id = s;
      }

      RotaryDispatcher<T, ElementT>::apply(val, cos_cache, sin_cache, rotary_dim, h, pos_id, interleaved, new_kv, in_offset - h);
    }

    present_kv[out_offset] = val;
  }
}

// Use when (H*)*num_heads > 1024
template <typename T, typename ElementT>
__global__ void ConcatNewToPastKVLarge(const int new_seqlen,
                                       const int past_buffer_seqlen,
                                       const int H,
                                       const int num_heads,
                                       const T* past_kv,
                                       const T* new_kv,
                                       T* present_kv,
                                       const int* seqlens_k,
                                       const bool past_only,
                                       const bool is_bsnh,
                                       const T* cos_cache,
                                       const T* sin_cache,
                                       const int rotary_dim,
                                       const int64_t* position_ids,
                                       const bool interleaved) {
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int present_buffer_seqlen = gridDim.y;

    const int present_batch_stride = present_buffer_seqlen * num_heads * H;
    const int row_stride = is_bsnh ? num_heads * H : H;
    const int present_head_stride = is_bsnh ? H : present_buffer_seqlen * H;

    // past_kv:     BPNH or BNPH
    // new_kv:      BLNH
    // present_kv:  BTNH or BNTH, where T = P + L

    // prompt, token, and interactive decoding cases
    const int past_seqlen = seqlens_k == nullptr ? 0 : seqlens_k[b] + 1 - new_seqlen;

    int out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;
    if (s < past_seqlen) {
      const int past_batch_stride = past_buffer_seqlen * num_heads * H;
      const int past_head_stride = is_bsnh ? H : past_buffer_seqlen * H;
      const int in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
      present_kv[out_offset] = past_kv[in_offset];
    } else if (!past_only && s < past_seqlen + new_seqlen) {
      const int new_batch_stride = new_seqlen * num_heads * H;
      const int new_row_stride = num_heads * H;
      const int new_head_stride = H;
      const int in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;

      T val = new_kv[in_offset];
      if (cos_cache != nullptr && rotary_dim > 0) {
        // Same logic
        int pos_id = s;
        int new_s_idx = s - past_seqlen;
        if (position_ids && new_s_idx >= 0 && new_s_idx < new_seqlen) {
          pos_id = static_cast<int>(position_ids[b * new_seqlen + new_s_idx]);
        }

        RotaryDispatcher<T, ElementT>::apply(val, cos_cache, sin_cache, rotary_dim, h, pos_id, interleaved, new_kv, in_offset - h);
        // Add barrier to prevent race condition if new_kv and present_kv alias (in-place).
        // Thread i needs to read neighbor x[i+offset] from Global Memory before Thread j writes to x[i+offset].
        // RotaryDispatcher reads from Global Memory.
        // We must ensure all reads are done before any writes in the block.
        __syncthreads();
      }
      present_kv[out_offset] = val;
    }
  }
}

// Concat new to kv buffer in place
template <typename T>
Status LaunchConcatNewToPastKV(const int batch_size,
                               const int kv_num_heads,
                               const int head_size,
                               const int kv_sequence_length,
                               const int past_sequence_length,
                               const int present_sequence_length,
                               const bool is_bsnh,
                               const int* seqlens_k,
                               const T* past_key,
                               const T* past_value,
                               const T* new_key,
                               const T* new_value,
                               T* present_key,
                               T* present_value,
                               cudaStream_t stream,
                               const int max_threads_per_block,
                               const bool past_only,
                               const T* cos_cache,
                               const T* sin_cache,
                               const int rotary_dim,
                               const int64_t* position_ids,
                               const bool interleaved) {
  // We use float2 vectorization (8 bytes).
  // num_elements_per_thread = 8 / sizeof(T).
  // H = head_size / num_elements_per_thread.
  int num_elements_per_thread = 8 / sizeof(T);
  if (num_elements_per_thread == 0) num_elements_per_thread = 1;  // safety
  const int H = head_size / num_elements_per_thread;
  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(present_sequence_length, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);

    // Apply RoPE ONLY to KEYS, not values
    ConcatNewToPastKV<float2, T><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                             past_sequence_length,
                                                             reinterpret_cast<const float2*>(past_key),
                                                             reinterpret_cast<const float2*>(new_key),
                                                             reinterpret_cast<float2*>(present_key),
                                                             seqlens_k,
                                                             past_only,
                                                             is_bsnh,
                                                             reinterpret_cast<const float2*>(cos_cache),
                                                             reinterpret_cast<const float2*>(sin_cache),
                                                             rotary_dim, position_ids, interleaved);

    ConcatNewToPastKV<float2, T><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                             past_sequence_length,
                                                             reinterpret_cast<const float2*>(past_value),
                                                             reinterpret_cast<const float2*>(new_value),
                                                             reinterpret_cast<float2*>(present_value),
                                                             seqlens_k,
                                                             past_only,
                                                             is_bsnh,
                                                             nullptr, nullptr, 0, nullptr, false);  // No RoPE for Values
  } else {
    int steps = (H * kv_num_heads + 255) / 256;
    const dim3 grid(steps, present_sequence_length, batch_size);
    const dim3 block(256, 1, 1);
    ConcatNewToPastKVLarge<float2, T><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                                  past_sequence_length,
                                                                  H,
                                                                  kv_num_heads,
                                                                  reinterpret_cast<const float2*>(past_key),
                                                                  reinterpret_cast<const float2*>(new_key),
                                                                  reinterpret_cast<float2*>(present_key),
                                                                  seqlens_k,
                                                                  past_only,
                                                                  is_bsnh,
                                                                  reinterpret_cast<const float2*>(cos_cache),
                                                                  reinterpret_cast<const float2*>(sin_cache),
                                                                  rotary_dim, position_ids, interleaved);

    ConcatNewToPastKVLarge<float2, T><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                                  past_sequence_length,
                                                                  H,
                                                                  kv_num_heads,
                                                                  reinterpret_cast<const float2*>(past_value),
                                                                  reinterpret_cast<const float2*>(new_value),
                                                                  reinterpret_cast<float2*>(present_value),
                                                                  seqlens_k,
                                                                  past_only,
                                                                  is_bsnh,
                                                                  nullptr, nullptr, 0, nullptr, false);
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConcatNewToPastKV<half>(const int batch_size,
                                              const int kv_num_heads,
                                              const int head_size,
                                              const int kv_sequence_length,
                                              const int past_sequence_length,
                                              const int present_sequence_length,
                                              const bool is_bsnh,
                                              const int* seqlens_k,
                                              const half* past_key,
                                              const half* past_value,
                                              const half* new_key,
                                              const half* new_value,
                                              half* present_key,
                                              half* present_value,
                                              cudaStream_t stream,
                                              const int max_threads_per_block,
                                              const bool past_only,
                                              const half* cos_cache,
                                              const half* sin_cache,
                                              const int rotary_dim,
                                              const int64_t* position_ids,
                                              const bool interleaved);

template Status LaunchConcatNewToPastKV<BFloat16>(const int batch_size,
                                                  const int kv_num_heads,
                                                  const int head_size,
                                                  const int kv_sequence_length,
                                                  const int past_sequence_length,
                                                  const int present_sequence_length,
                                                  const bool is_bsnh,
                                                  const int* seqlens_k,
                                                  const BFloat16* past_key,
                                                  const BFloat16* past_value,
                                                  const BFloat16* new_key,
                                                  const BFloat16* new_value,
                                                  BFloat16* present_key,
                                                  BFloat16* present_value,
                                                  cudaStream_t stream,
                                                  const int max_threads_per_block,
                                                  const bool past_only,
                                                  const BFloat16* cos_cache,
                                                  const BFloat16* sin_cache,
                                                  const int rotary_dim,
                                                  const int64_t* position_ids,
                                                  const bool interleaved);

template Status LaunchConcatNewToPastKV<float>(const int batch_size,
                                               const int kv_num_heads,
                                               const int head_size,
                                               const int kv_sequence_length,
                                               const int past_sequence_length,
                                               const int present_sequence_length,
                                               const bool is_bsnh,
                                               const int* seqlens_k,
                                               const float* past_key,
                                               const float* past_value,
                                               const float* new_key,
                                               const float* new_value,
                                               float* present_key,
                                               float* present_value,
                                               cudaStream_t stream,
                                               const int max_threads_per_block,
                                               const bool past_only,
                                               const float* cos_cache,
                                               const float* sin_cache,
                                               const int rotary_dim,
                                               const int64_t* position_ids,
                                               const bool interleaved);

// Kernel to append new kv to kv buffer in place
template <typename T>
__global__ void ConcatKVInPlace(const int max_seqlen,
                                T* kv_buff,
                                const T* new_kv,
                                const int* seqlens_k,
                                const int* total_seqlens_k,
                                const bool is_past_kv_bnsh_format,
                                const bool is_new_kv_bnsh_format) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int new_seqlen = gridDim.x;
  const int kv_num_heads = blockDim.y;
  const int H = blockDim.x;

  const int past_seq_len = (total_seqlens_k != nullptr)
                               ? (total_seqlens_k[b] - new_seqlen)
                               : (seqlens_k == nullptr ? 0 : (seqlens_k[b] + 1 - new_seqlen));

  int out_offset = is_past_kv_bnsh_format
                       ? INDEX_4D(kv_num_heads, max_seqlen, H, b, n, s + past_seq_len, h)
                       : INDEX_4D(max_seqlen, kv_num_heads, H, b, s + past_seq_len, n, h);

  int in_offset = is_new_kv_bnsh_format
                      ? INDEX_4D(kv_num_heads, new_seqlen, H, b, n, s, h)
                      : INDEX_4D(new_seqlen, kv_num_heads, H, b, s, n, h);

  kv_buff[out_offset] = new_kv[in_offset];
}

template <typename T>
__global__ void ConcatKVInPlaceLarge(const int max_seqlen,
                                     const int H,
                                     const int kv_num_heads,
                                     T* kv_buff,
                                     const T* new_kv,
                                     const int* seqlens_k,
                                     const int* total_seqlens_k,
                                     const bool is_past_kv_bnsh_format,
                                     const bool is_new_kv_bnsh_format) {  // refers to kv buff; otherwise bnsh
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * kv_num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int new_seqlen = gridDim.y;
    const int past_seq_len = (total_seqlens_k != nullptr)
                                 ? (total_seqlens_k[b] - new_seqlen)
                                 : (seqlens_k == nullptr ? 0 : (seqlens_k[b] + 1 - new_seqlen));

    int out_offset = is_past_kv_bnsh_format
                         ? INDEX_4D(kv_num_heads, max_seqlen, H, b, n, s + past_seq_len, h)
                         : INDEX_4D(max_seqlen, kv_num_heads, H, b, s + past_seq_len, n, h);

    int in_offset = is_new_kv_bnsh_format
                        ? INDEX_4D(kv_num_heads, new_seqlen, H, b, n, s, h)
                        : INDEX_4D(new_seqlen, kv_num_heads, H, b, s, n, h);

    kv_buff[out_offset] = new_kv[in_offset];
  }
}

// Concat new to kv buffer in place
template <typename T>
Status LaunchConcatKVInPlace(int batch_size,
                             int kv_num_heads,
                             int head_size,
                             int max_sequence_length,
                             const int* seqlens_k,
                             const int* total_seqlens_k,
                             int new_seq_len,
                             const T* new_key,
                             const T* new_value,
                             T* present_key,
                             T* present_value,
                             const bool is_past_kv_bnsh_format,
                             const bool is_new_kv_bnsh_format,
                             cudaStream_t stream,
                             const int max_threads_per_block) {
  // static_assert(sizeof(T) == 2);
  assert(head_size % 4 == 0);

  const int H = head_size / 4;
  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(new_seq_len, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                        reinterpret_cast<float2*>(present_key),
                                                        reinterpret_cast<const float2*>(new_key),
                                                        seqlens_k,
                                                        total_seqlens_k,
                                                        is_past_kv_bnsh_format,
                                                        is_new_kv_bnsh_format);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                        reinterpret_cast<float2*>(present_value),
                                                        reinterpret_cast<const float2*>(new_value),
                                                        seqlens_k,
                                                        total_seqlens_k,
                                                        is_past_kv_bnsh_format,
                                                        is_new_kv_bnsh_format);
  } else {
    int steps = int(ceil(float(H * kv_num_heads) / 256.0));
    const dim3 grid(steps, new_seq_len, batch_size);
    const dim3 block(256, 1, 1);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(present_key),
                                                             reinterpret_cast<const float2*>(new_key),
                                                             seqlens_k,
                                                             total_seqlens_k,
                                                             is_past_kv_bnsh_format,
                                                             is_new_kv_bnsh_format);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(present_value),
                                                             reinterpret_cast<const float2*>(new_value),
                                                             seqlens_k,
                                                             total_seqlens_k,
                                                             is_past_kv_bnsh_format,
                                                             is_new_kv_bnsh_format);
  }
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConcatKVInPlace<half>(int batch_size,
                                            int kv_num_heads,
                                            int head_size,
                                            int max_sequence_length,
                                            const int* seqlens_k,
                                            const int* total_seqlens_k,
                                            int new_seq_len,
                                            const half* new_key,
                                            const half* new_value,
                                            half* present_key,
                                            half* present_value,
                                            bool is_past_kv_bnsh_format,
                                            bool is_new_kv_bnsh_format,
                                            cudaStream_t stream,
                                            const int max_threads_per_block);

template Status LaunchConcatKVInPlace<BFloat16>(int batch_size,
                                                int kv_num_heads,
                                                int head_size,
                                                int max_sequence_length,
                                                const int* seqlens_k,
                                                const int* total_seqlens_k,
                                                int new_seq_len,
                                                const BFloat16* new_key,
                                                const BFloat16* new_value,
                                                BFloat16* present_key,
                                                BFloat16* present_value,
                                                bool is_past_kv_bnsh_format,
                                                bool is_new_kv_bnsh_format,
                                                cudaStream_t stream,
                                                const int max_threads_per_block);

template Status LaunchConcatKVInPlace<float>(int batch_size,
                                             int kv_num_heads,
                                             int head_size,
                                             int max_sequence_length,
                                             const int* seqlens_k,
                                             const int* total_seqlens_k,
                                             int new_seq_len,
                                             const float* new_key,
                                             const float* new_value,
                                             float* present_key,
                                             float* present_value,
                                             bool is_past_kv_bnsh_format,
                                             bool is_new_kv_bnsh_format,
                                             cudaStream_t stream,
                                             const int max_threads_per_block);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
