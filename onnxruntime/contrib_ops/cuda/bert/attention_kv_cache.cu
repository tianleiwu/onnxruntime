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

  const int64_t present_SH = int64_t(all_sequence_length) * H;
  const int64_t present_NSH = num_heads * present_SH;
  int64_t out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
  if (s < tensor_in_sequence_length) {
    const int64_t past_SH = int64_t(tensor_in_sequence_length) * H;
    const int64_t past_NSH = num_heads * past_SH;
    const int64_t in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
    tensor_out[out_offset] = tensor_in[in_offset];
  } else if (s < all_sequence_length) {
    const int64_t SH = int64_t(tensor_add_sequence_length) * H;
    const int64_t NSH = num_heads * SH;
    const int64_t in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
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

  const int64_t present_SH = int64_t(all_sequence_length) * H;
  const int64_t present_NSH = num_heads * present_SH;
  while (h < H) {
    int64_t out_offset = b * present_NSH + n * present_SH + s * H + h + chunk_id * (present_NSH * batch_size);
    if (s < tensor_in_sequence_length) {
      const int64_t past_SH = int64_t(tensor_in_sequence_length) * H;
      const int64_t past_NSH = num_heads * past_SH;
      const int64_t in_offset = b * past_NSH + n * past_SH + s * H + h + chunk_id * (past_NSH * batch_size);
      tensor_out[out_offset] = tensor_in[in_offset];
    } else if (s < all_sequence_length) {
      const int64_t SH = int64_t(tensor_add_sequence_length) * H;
      const int64_t NSH = num_heads * SH;
      const int64_t in_offset = b * NSH + n * SH + (s - tensor_in_sequence_length) * H + h + chunk_id * (NSH * batch_size);
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
#ifndef NDEBUG
  CUDA_CALL(cudaStreamSynchronize(stream));
#endif
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
#ifndef NDEBUG
  CUDA_CALL(cudaStreamSynchronize(stream));
#endif
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
  if (0 == (head_size % 8)) {
    const int H = head_size / 8;
    if (H * num_heads <= max_threads_per_block) {
      const dim3 block(H, num_heads, 1);
      ConcatTensorToTensor<float4><<<grid, block, 0, stream>>>(
          sequence_length,
          reinterpret_cast<const float4*>(tensor_in),
          reinterpret_cast<const float4*>(tensor_add),
          reinterpret_cast<float4*>(tensor_out));
    } else {
      const dim3 block(max_threads_per_block / num_heads, num_heads, 1);
      ConcatTensorToTensorLarge<float4><<<grid, block, 0, stream>>>(
          sequence_length,
          H,
          reinterpret_cast<const float4*>(tensor_in),
          reinterpret_cast<const float4*>(tensor_add),
          reinterpret_cast<float4*>(tensor_out));
    }
  } else if (0 == (head_size & 1)) {
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

#ifndef NDEBUG
  CUDA_CALL(cudaStreamSynchronize(stream));
#endif
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

  constexpr int M = static_cast<int>(QKV::COUNT);  // Matrix count in qkv
  const int m = blockIdx.z + 1;                    // k = 1, v = 2

  const int64_t NH = N * head_size;
  const int64_t NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int64_t MsH = int64_t(max_sequence_length) * head_size;
  const int64_t NMsH = N * MsH;
  const int64_t BNMsH = B * NMsH;
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

  constexpr int M = static_cast<int>(QKV::COUNT);  // Matrix count in qkv
  const int m = (blockIdx.z & 0x1) + 1;            // k = 1, v = 2

  const int64_t NH = N * head_size;
  const int64_t NHS = NH * S;

  qkv += (n * head_size + (s * M + m) * NH + b * M * NHS);
  if (biases) {
    biases += (m * NH + n * head_size);
  }

  const int64_t MsH = int64_t(max_sequence_length) * head_size;
  const int64_t NMsH = N * MsH;
  const int64_t BNMsH = B * NMsH;
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
// Helper to apply RoPE rotation
template <typename T, typename ElementT>
__device__ __forceinline__ void ApplyRotaryEmbedding(T& val, const T* cos_cache, const T* sin_cache,
                                                     const int rotary_dim, const int h_idx, const int pos_id,
                                                     const bool interleaved, const T* new_kv_base,
                                                     const int64_t in_offset) {
  // Check if we are within rotary dimension
  // For vector types, we need to check if ANY element is within range, or handle partial rotation?
  // Our caller ensures h_idx corresponds to vector start.
  // Generally we process 'vectors' of elements.

  // NOTE: This helper assumes T fits within the processing granularity (float2/float4).
  // The logic below is adapted from the original dispatcher specializations.

  // This is a placeholder for the specialized logic which is quite different per type
  // We will keep specialization but clean up the body.
}

template <typename VectorT, typename ElementT>
struct RotaryDispatcher {
  __device__ static void apply(VectorT& val, const VectorT* cos_cache, const VectorT* sin_cache,
                               const int rotary_dim, const int h_idx, const int pos_id,
                               const bool interleaved, const VectorT* new_kv_base, const int64_t in_offset);
};

// Specialization for float2 (float)
template <>
struct RotaryDispatcher<float2, float> {
  __device__ static void apply(float2& val, const float2* cos_cache, const float2* sin_cache,
                               const int rotary_dim, const int h_idx, const int pos_id,
                               const bool interleaved, const float2* new_kv_base, const int64_t in_offset) {
    if (2 * h_idx >= rotary_dim) return;

    const float* cos_ptr = reinterpret_cast<const float*>(cos_cache);
    const float* sin_ptr = reinterpret_cast<const float*>(sin_cache);
    const float* kv_ptr = reinterpret_cast<const float*>(new_kv_base);

    // Use int64_t for byte offsets if needed, but here we index float array
    int64_t scalar_in_offset = in_offset * 2;
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
      // Half-Split Logic
      // Process x (idx = scalar_h)
      {
        int idx = scalar_h;
        if (idx < rotary_dim) {  // Should be true given h_idx check
          int pair_idx = (idx < half_rot) ? (idx + half_rot) : (idx - half_rot);
          float sign = (idx < half_rot) ? -1.0f : 1.0f;
          int cos_idx = idx % half_rot;
          int cs_idx = pos_id * half_rot + cos_idx;

          c = cos_ptr[cs_idx];
          s = sin_ptr[cs_idx];
          // Potential gather from new_kv if we are doing fused append+rotate from a source
          // The source is 'new_kv_base'.
          float pair_val = kv_ptr[scalar_in_offset + pair_idx];
          val.x = x * c + sign * pair_val * s;
        }
      }

      // Process y (idx = scalar_h + 1)
      {
        int idx = scalar_h + 1;
        if (idx < rotary_dim) {
          int pair_idx = (idx < half_rot) ? (idx + half_rot) : (idx - half_rot);
          float sign = (idx < half_rot) ? -1.0f : 1.0f;
          int cos_idx = idx % half_rot;
          int cs_idx = pos_id * half_rot + cos_idx;

          c = cos_ptr[cs_idx];
          s = sin_ptr[cs_idx];
          float pair_val = kv_ptr[scalar_in_offset + pair_idx];
          val.y = y * c + sign * pair_val * s;
        }
      }
    }
  }
};

// Specialization for float4 (float)
template <>
struct RotaryDispatcher<float4, float> {
  __device__ static void apply(float4& val, const float4* cos_cache, const float4* sin_cache,
                               const int rotary_dim, const int h_idx, const int pos_id,
                               const bool interleaved, const float4* new_kv_base, const int64_t in_offset) {
    float2 p1 = make_float2(val.x, val.y);
    float2 p2 = make_float2(val.z, val.w);
    const float2* c = reinterpret_cast<const float2*>(cos_cache);
    const float2* s = reinterpret_cast<const float2*>(sin_cache);
    const float2* b = reinterpret_cast<const float2*>(new_kv_base);

    // Update offsets for float2 components
    RotaryDispatcher<float2, float>::apply(p1, c, s, rotary_dim, h_idx * 2, pos_id, interleaved, b, in_offset * 2);
    RotaryDispatcher<float2, float>::apply(p2, c, s, rotary_dim, h_idx * 2 + 1, pos_id, interleaved, b, in_offset * 2);

    val.x = p1.x;
    val.y = p1.y;
    val.z = p2.x;
    val.w = p2.y;
  }
};

// Specialization for float2 (half)
template <>
struct RotaryDispatcher<float2, half> {
  __device__ static void apply(float2& val, const float2* cos_cache, const float2* sin_cache,
                               const int rotary_dim, const int h_idx, const int pos_id,
                               const bool interleaved, const float2* new_kv_base, const int64_t in_offset) {
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

      const half2 c_pair = cos_ptr[cs0 / 2];
      const half2 s_pair = sin_ptr[cs0 / 2];

      const float2 c_f = __half22float2(c_pair);
      const float2 s_f = __half22float2(s_pair);

      // Rotate v0 (pair 0)
      const float2 e0 = __half22float2(v0);
      v0 = __float22half2_rn(make_float2(e0.x * c_f.x - e0.y * s_f.x, e0.x * s_f.x + e0.y * c_f.x));

      // Rotate v1 (pair 1)
      const float2 e1 = __half22float2(v1);
      v1 = __float22half2_rn(make_float2(e1.x * c_f.y - e1.y * s_f.y, e1.x * s_f.y + e1.y * c_f.y));
    } else {
      // Half-Split Logic
      // Elements i and i + H/2 are paired.
      // We have 4 elements: 4*h_idx, +1, +2, +3.
      // We need to fetch pairs from new_kv_base.

      const half* kv_ptr = reinterpret_cast<const half*>(new_kv_base);
      int base_idx = 4 * h_idx;
      int64_t scalar_in_offset = in_offset * 4;  // 4 halfs per float2

      auto rotate_element = [&](int idx, half& val) {
        if (idx >= rotary_dim) return;  // Should be covered
        int pair_idx = (idx < half_rot) ? (idx + half_rot) : (idx - half_rot);
        float sign = (idx < half_rot) ? -1.0f : 1.0f;
        int cos_idx = idx % half_rot;
        int cs_idx = pos_id * half_rot + cos_idx;

        half c_val = reinterpret_cast<const half*>(cos_ptr)[cs_idx];
        half s_val = reinterpret_cast<const half*>(sin_ptr)[cs_idx];  // Original used cos_ptr? No sin_ptr

        float val_f = __half2float(val);
        float pair_f = __half2float(kv_ptr[scalar_in_offset + pair_idx]);
        float cf = __half2float(c_val);
        float sf = __half2float(s_val);

        val = __float2half(val_f * cf + sign * pair_f * sf);
      };

      rotate_element(base_idx, v0.x);
      rotate_element(base_idx + 1, v0.y);
      rotate_element(base_idx + 2, v1.x);
      rotate_element(base_idx + 3, v1.y);
    }
    v_ptr[0] = v0;
    v_ptr[1] = v1;
  }
};

// Specialization for float2 (BFloat16)
template <>
struct RotaryDispatcher<float2, BFloat16> {
  __device__ static void apply(float2& val, const float2* cos_cache, const float2* sin_cache,
                               const int rotary_dim, const int h_idx, const int pos_id,
                               const bool interleaved, const float2* new_kv_base, const int64_t in_offset) {
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

      // Process v0 (pair 1)
      // v0.x, v0.y
      float c0f = __bfloat162float(c_pair.x);
      float s0f = __bfloat162float(s_pair.x);
      float e0x = __bfloat162float(v0.x);
      float e0y = __bfloat162float(v0.y);
      v0.x = __float2bfloat16(e0x * c0f - e0y * s0f);
      v0.y = __float2bfloat16(e0x * s0f + e0y * c0f);

      // Process v1 (pair 2)
      float c1f = __bfloat162float(c_pair.y);
      float s1f = __bfloat162float(s_pair.y);
      float e1x = __bfloat162float(v1.x);
      float e1y = __bfloat162float(v1.y);
      v1.x = __float2bfloat16(e1x * c1f - e1y * s1f);
      v1.y = __float2bfloat16(e1x * s1f + e1y * c1f);

    } else {
      // Half-Split Logic
      const __nv_bfloat16* kv_ptr = reinterpret_cast<const __nv_bfloat16*>(new_kv_base);
      int base_idx = 4 * h_idx;
      int64_t scalar_in_offset = in_offset * 4;

      auto rotate_element_bf16 = [&](int idx, __nv_bfloat16& val) {
        if (idx >= rotary_dim) return;
        int pair_idx = (idx < half_rot) ? (idx + half_rot) : (idx - half_rot);
        float sign = (idx < half_rot) ? -1.0f : 1.0f;
        int cos_idx = idx % half_rot;
        int cs_idx = pos_id * half_rot + cos_idx;

        __nv_bfloat16 c_val = reinterpret_cast<const __nv_bfloat16*>(cos_ptr)[cs_idx];
        __nv_bfloat16 s_val = reinterpret_cast<const __nv_bfloat16*>(sin_ptr)[cs_idx];

        float val_f = __bfloat162float(val);
        float pair_f = __bfloat162float(kv_ptr[scalar_in_offset + pair_idx]);
        float cf = __bfloat162float(c_val);
        float sf = __bfloat162float(s_val);

        val = __float2bfloat16(val_f * cf + sign * pair_f * sf);
      };

      rotate_element_bf16(base_idx, v0.x);
      rotate_element_bf16(base_idx + 1, v0.y);
      rotate_element_bf16(base_idx + 2, v1.x);
      rotate_element_bf16(base_idx + 3, v1.y);
    }
    v_ptr[0] = v0;
    v_ptr[1] = v1;
  }
};

// Fused versions that handle both K and V using blockIdx.z
// blockIdx.z == 0 -> Process K (with RoPE if enabled)
// blockIdx.z == 1 -> Process V (no RoPE)
// Grid.z should be 2.

template <typename T, typename ElementT>
__global__ void ConcatNewToPastKVFused(const int new_seqlen,
                                       const int past_buffer_seqlen,
                                       const T* past_key,
                                       const T* past_value,
                                       const T* new_key,
                                       const T* new_value,
                                       T* present_key,
                                       T* present_value,
                                       const int* past_seq_lens,
                                       const int* total_seq_lens,
                                       const bool past_only,
                                       const bool is_bsnh,
                                       const T* cos_cache,
                                       const T* sin_cache,
                                       const int rotary_dim,
                                       const int64_t* position_ids,
                                       const bool interleaved) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;
  const int kind = blockIdx.z;  // 0 for K, 1 for V

  const int present_buffer_seqlen = gridDim.x;  // gridDim.x is present_sequence_length
  const int num_heads = blockDim.y;
  const int H = blockDim.x;

  const int64_t present_batch_stride = int64_t(present_buffer_seqlen) * num_heads * H;
  const int64_t row_stride = is_bsnh ? num_heads * H : H;
  const int64_t present_head_stride = is_bsnh ? H : int64_t(present_buffer_seqlen) * H;

  // Determine pointers based on kind
  const T* past_ptr = (kind == 0) ? past_key : past_value;
  const T* new_ptr = (kind == 0) ? new_key : new_value;
  T* present_ptr = (kind == 0) ? present_key : present_value;

  const int past_seqlen = past_seq_lens[b];

  int64_t out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;

  if (s < past_seqlen) {
    const int64_t past_batch_stride = int64_t(past_buffer_seqlen) * num_heads * H;
    const int64_t past_head_stride = is_bsnh ? H : int64_t(past_buffer_seqlen) * H;
    const int64_t in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
    present_ptr[out_offset] = past_ptr[in_offset];
  } else if (!past_only && s < past_seqlen + new_seqlen) {
    const int64_t new_batch_stride = int64_t(new_seqlen) * num_heads * H;
    const int64_t new_row_stride = num_heads * H;
    const int64_t new_head_stride = H;
    const int64_t in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;

    T val = new_ptr[in_offset];

    // Apply RoPE only for K (kind == 0)
    if (kind == 0 && cos_cache != nullptr && rotary_dim > 0) {
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

      // Check bounds for pos_id to be safe?
      // RoPE cache size usually matches max_seq_len.

      RotaryDispatcher<T, ElementT>::apply(val, cos_cache, sin_cache, rotary_dim, h, pos_id, interleaved, new_key, in_offset - h);
    }
    present_ptr[out_offset] = val;
  }
}

template <typename T, typename ElementT>
__global__ void ConcatNewToPastKVFusedLarge(const int new_seqlen,
                                            const int past_buffer_seqlen,
                                            const int H,
                                            const int num_heads,
                                            const T* past_key,
                                            const T* past_value,
                                            const T* new_key,
                                            const T* new_value,
                                            T* present_key,
                                            T* present_value,
                                            const int* past_seq_lens,
                                            const int* total_seq_lens,
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
    const int b = blockIdx.z / 2;     // Integer div
    const int kind = blockIdx.z % 2;  // 0 for K, 1 for V

    const int present_buffer_seqlen = gridDim.y;
    // gridDim.z is batch_size * 2

    const int64_t present_batch_stride = int64_t(present_buffer_seqlen) * num_heads * H;
    const int64_t row_stride = is_bsnh ? num_heads * H : H;
    const int64_t present_head_stride = is_bsnh ? H : int64_t(present_buffer_seqlen) * H;

    const T* past_ptr = (kind == 0) ? past_key : past_value;
    const T* new_ptr = (kind == 0) ? new_key : new_value;
    T* present_ptr = (kind == 0) ? present_key : present_value;

    const int past_seqlen = past_seq_lens[b];

    const int64_t out_offset = b * present_batch_stride + s * row_stride + n * present_head_stride + h;

    if (s < past_seqlen) {
      const int64_t past_batch_stride = int64_t(past_buffer_seqlen) * num_heads * H;
      const int64_t past_head_stride = is_bsnh ? H : int64_t(past_buffer_seqlen) * H;
      const int64_t in_offset = b * past_batch_stride + s * row_stride + n * past_head_stride + h;
      present_ptr[out_offset] = past_ptr[in_offset];
    } else if (!past_only && s < past_seqlen + new_seqlen) {
      const int64_t new_batch_stride = int64_t(new_seqlen) * num_heads * H;
      const int64_t new_row_stride = num_heads * H;
      const int64_t new_head_stride = H;
      const int64_t in_offset = b * new_batch_stride + (s - past_seqlen) * new_row_stride + n * new_head_stride + h;

      T val = new_ptr[in_offset];

      if (kind == 0 && cos_cache != nullptr && rotary_dim > 0) {
        int pos_id = s;
        int new_s_idx = s - past_seqlen;
        if (position_ids && new_s_idx >= 0 && new_s_idx < new_seqlen) {
          pos_id = static_cast<int>(position_ids[b * new_seqlen + new_s_idx]);
        }

        RotaryDispatcher<T, ElementT>::apply(val, cos_cache, sin_cache, rotary_dim, h, pos_id, interleaved, new_key, in_offset - h);
      }
      present_ptr[out_offset] = val;
    }
  }
}

template <typename T>
Status LaunchConcatNewToPastKV(const int batch_size,
                               const int kv_num_heads,
                               const int head_size,
                               const int kv_sequence_length,
                               const int past_sequence_length,
                               const int present_sequence_length,
                               const bool is_bsnh,
                               const int* past_seq_lens,
                               const int* total_seq_lens,
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
  constexpr int num_elements_per_thread = std::max(1, 8 / int(sizeof(T)));
  const int H = head_size / num_elements_per_thread;

  if (H * kv_num_heads <= max_threads_per_block) {
    // Grid Z dim is 2: 0 for K, 1 for V
    const dim3 grid(present_sequence_length, batch_size, 2);
    const dim3 block(H, kv_num_heads, 1);

    ConcatNewToPastKVFused<float2, T><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                                  past_sequence_length,
                                                                  reinterpret_cast<const float2*>(past_key),
                                                                  reinterpret_cast<const float2*>(past_value),
                                                                  reinterpret_cast<const float2*>(new_key),
                                                                  reinterpret_cast<const float2*>(new_value),
                                                                  reinterpret_cast<float2*>(present_key),
                                                                  reinterpret_cast<float2*>(present_value),
                                                                  past_seq_lens,
                                                                  total_seq_lens,
                                                                  past_only,
                                                                  is_bsnh,
                                                                  reinterpret_cast<const float2*>(cos_cache),
                                                                  reinterpret_cast<const float2*>(sin_cache),
                                                                  rotary_dim, position_ids, interleaved);
  } else {
    // Large kernel version
    int steps = (H * kv_num_heads + 255) / 256;
    // Grid Z dim is batch_size * 2
    // We encode b and kind in blockIdx.z in the kernel
    const dim3 grid(steps, present_sequence_length, batch_size * 2);
    const dim3 block(256, 1, 1);

    ConcatNewToPastKVFusedLarge<float2, T><<<grid, block, 0, stream>>>(kv_sequence_length,
                                                                       past_sequence_length,
                                                                       H,
                                                                       kv_num_heads,
                                                                       reinterpret_cast<const float2*>(past_key),
                                                                       reinterpret_cast<const float2*>(past_value),
                                                                       reinterpret_cast<const float2*>(new_key),
                                                                       reinterpret_cast<const float2*>(new_value),
                                                                       reinterpret_cast<float2*>(present_key),
                                                                       reinterpret_cast<float2*>(present_value),
                                                                       past_seq_lens,
                                                                       total_seq_lens,
                                                                       past_only,
                                                                       is_bsnh,
                                                                       reinterpret_cast<const float2*>(cos_cache),
                                                                       reinterpret_cast<const float2*>(sin_cache),
                                                                       rotary_dim, position_ids, interleaved);
  }
#ifndef NDEBUG
  CUDA_CALL(cudaStreamSynchronize(stream));
#endif
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConcatNewToPastKV<half>(const int batch_size,
                                              const int kv_num_heads,
                                              const int head_size,
                                              const int kv_sequence_length,
                                              const int past_sequence_length,
                                              const int present_sequence_length,
                                              const bool is_bsnh,
                                              const int* past_seq_lens,
                                              const int* total_seq_lens,
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
                                                  const int* past_seq_lens,
                                                  const int* total_seq_lens,
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
                                               const int* past_seq_lens,
                                               const int* total_seq_lens,
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
                                const int* past_seq_lens,
                                const int* total_seq_lens,
                                const bool is_past_kv_bnsh_format,
                                const bool is_new_kv_bnsh_format) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int new_seqlen = gridDim.x;
  const int kv_num_heads = blockDim.y;
  const int H = blockDim.x;

  const int past_seq_len = (past_seq_lens != nullptr) ? past_seq_lens[b] : (total_seq_lens[b] - new_seqlen);

  int64_t out_offset = is_past_kv_bnsh_format
                           ? INDEX_4D(int64_t(kv_num_heads), int64_t(max_seqlen), int64_t(H), int64_t(b), int64_t(n), int64_t(s + past_seq_len), int64_t(h))
                           : INDEX_4D(int64_t(max_seqlen), int64_t(kv_num_heads), int64_t(H), int64_t(b), int64_t(s + past_seq_len), int64_t(n), int64_t(h));

  int64_t in_offset = is_new_kv_bnsh_format
                          ? INDEX_4D(int64_t(kv_num_heads), int64_t(new_seqlen), int64_t(H), int64_t(b), int64_t(n), int64_t(s), int64_t(h))
                          : INDEX_4D(int64_t(new_seqlen), int64_t(kv_num_heads), int64_t(H), int64_t(b), int64_t(s), int64_t(n), int64_t(h));

  if (s + past_seq_len < total_seq_lens[b]) {
    kv_buff[out_offset] = new_kv[in_offset];
  }
}

template <typename T>
__global__ void ConcatKVInPlaceLarge(const int max_seqlen,
                                     const int H,
                                     const int kv_num_heads,
                                     T* kv_buff,
                                     const T* new_kv,
                                     const int* past_seq_lens,
                                     const int* total_seq_lens,
                                     const bool is_past_kv_bnsh_format,
                                     const bool is_new_kv_bnsh_format) {  // refers to kv buff; otherwise bnsh
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * kv_num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;
    const int new_seqlen = gridDim.y;
    const int past_seq_len = (past_seq_lens != nullptr) ? past_seq_lens[b] : (total_seq_lens[b] - new_seqlen);

    int64_t out_offset = is_past_kv_bnsh_format
                             ? INDEX_4D(int64_t(kv_num_heads), int64_t(max_seqlen), int64_t(H), int64_t(b), int64_t(n), int64_t(s + past_seq_len), int64_t(h))
                             : INDEX_4D(int64_t(max_seqlen), int64_t(kv_num_heads), int64_t(H), int64_t(b), int64_t(s + past_seq_len), int64_t(n), int64_t(h));

    int64_t in_offset = is_new_kv_bnsh_format
                            ? INDEX_4D(int64_t(kv_num_heads), int64_t(new_seqlen), int64_t(H), int64_t(b), int64_t(n), int64_t(s), int64_t(h))
                            : INDEX_4D(int64_t(new_seqlen), int64_t(kv_num_heads), int64_t(H), int64_t(b), int64_t(s), int64_t(n), int64_t(h));

    if (s + past_seq_len < total_seq_lens[b]) {
      kv_buff[out_offset] = new_kv[in_offset];
    }
  }
}

template <typename T>
Status LaunchConcatKVInPlace(int batch_size,
                             int kv_num_heads,
                             int head_size,
                             int max_sequence_length,
                             const int* past_seq_lens,
                             const int* total_seq_lens,
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
                                                        past_seq_lens,
                                                        total_seq_lens,
                                                        is_past_kv_bnsh_format,
                                                        is_new_kv_bnsh_format);
    ConcatKVInPlace<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                        reinterpret_cast<float2*>(present_value),
                                                        reinterpret_cast<const float2*>(new_value),
                                                        past_seq_lens,
                                                        total_seq_lens,
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
                                                             past_seq_lens,
                                                             total_seq_lens,
                                                             is_past_kv_bnsh_format,
                                                             is_new_kv_bnsh_format);
    ConcatKVInPlaceLarge<float2><<<grid, block, 0, stream>>>(max_sequence_length,
                                                             H,
                                                             kv_num_heads,
                                                             reinterpret_cast<float2*>(present_value),
                                                             reinterpret_cast<const float2*>(new_value),
                                                             past_seq_lens,
                                                             total_seq_lens,
                                                             is_past_kv_bnsh_format,
                                                             is_new_kv_bnsh_format);
  }
#ifndef NDEBUG
  CUDA_CALL(cudaStreamSynchronize(stream));
#endif
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConcatKVInPlace<half>(int batch_size,
                                            int kv_num_heads,
                                            int head_size,
                                            int max_sequence_length,
                                            const int* past_seq_lens,
                                            const int* total_seq_lens,
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
                                                const int* past_seq_lens,
                                                const int* total_seq_lens,
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
                                             const int* past_seq_lens,
                                             const int* total_seq_lens,
                                             int new_seq_len,
                                             const float* new_key,
                                             const float* new_value,
                                             float* present_key,
                                             float* present_value,
                                             bool is_past_kv_bnsh_format,
                                             bool is_new_kv_bnsh_format,
                                             cudaStream_t stream,
                                             const int max_threads_per_block);

// ============================================================================
// TRULY FUSED K+V KERNEL: Single kernel for both K and V
// This eliminates the separate ConcatKVInPlace call for V, saving one kernel launch.
// RoPE should be applied BEFORE calling this kernel.
// ============================================================================

// Fused kernel: Append K and V in a single kernel
// Each thread handles one element of K and one element of V
template <typename T>
__global__ void ConcatKVInPlaceFused(const int max_seqlen,
                                     const int new_seqlen,
                                     T* k_buff,
                                     T* v_buff,
                                     const T* new_k,
                                     const T* new_v,
                                     const int* past_seq_lens,
                                     const int* total_seq_lens,
                                     const bool is_past_kv_bnsh_format,
                                     const bool is_new_kv_bnsh_format) {
  const int h = threadIdx.x;
  const int n = threadIdx.y;
  const int s = blockIdx.x;
  const int b = blockIdx.y;

  const int kv_num_heads = blockDim.y;
  const int H = blockDim.x;

  const int past_seq_len = (past_seq_lens != nullptr) ? past_seq_lens[b] : (total_seq_lens[b] - new_seqlen);

  // Early exit to prevent out-of-bounds access and redundant writes
  if (s + past_seq_len >= total_seq_lens[b]) {
    return;
  }

  // Use int64_t for offsets to prevent overflow
  int64_t out_offset = is_past_kv_bnsh_format
                           ? INDEX_4D(int64_t(kv_num_heads), int64_t(max_seqlen), int64_t(H), int64_t(b), int64_t(n), int64_t(s + past_seq_len), int64_t(h))
                           : INDEX_4D(int64_t(max_seqlen), int64_t(kv_num_heads), int64_t(H), int64_t(b), int64_t(s + past_seq_len), int64_t(n), int64_t(h));

  int64_t in_offset = is_new_kv_bnsh_format
                          ? INDEX_4D(int64_t(kv_num_heads), int64_t(new_seqlen), int64_t(H), int64_t(b), int64_t(n), int64_t(s), int64_t(h))
                          : INDEX_4D(int64_t(new_seqlen), int64_t(kv_num_heads), int64_t(H), int64_t(b), int64_t(s), int64_t(n), int64_t(h));

  // Simple copy for K and V
  k_buff[out_offset] = new_k[in_offset];
  v_buff[out_offset] = new_v[in_offset];
}

// Large version for when H * kv_num_heads > max_threads_per_block
template <typename T>
__global__ void ConcatKVInPlaceFusedLarge(const int max_seqlen,
                                          const int new_seqlen,
                                          const int H,
                                          const int kv_num_heads,
                                          T* k_buff,
                                          T* v_buff,
                                          const T* new_k,
                                          const T* new_v,
                                          const int* past_seq_lens,
                                          const int* total_seq_lens,
                                          const bool is_past_kv_bnsh_format,
                                          const bool is_new_kv_bnsh_format) {
  int i = threadIdx.x + (blockDim.x * blockIdx.x);
  if (i < H * kv_num_heads) {
    const int h = i % H;
    const int n = i / H;
    const int s = blockIdx.y;
    const int b = blockIdx.z;

    const int past_seq_len = (past_seq_lens != nullptr) ? past_seq_lens[b] : (total_seq_lens[b] - new_seqlen);

    if (s + past_seq_len >= total_seq_lens[b]) {
      return;
    }

    int64_t out_offset = is_past_kv_bnsh_format
                             ? INDEX_4D(int64_t(kv_num_heads), int64_t(max_seqlen), int64_t(H), int64_t(b), int64_t(n), int64_t(s + past_seq_len), int64_t(h))
                             : INDEX_4D(int64_t(max_seqlen), int64_t(kv_num_heads), int64_t(H), int64_t(b), int64_t(s + past_seq_len), int64_t(n), int64_t(h));

    int64_t in_offset = is_new_kv_bnsh_format
                            ? INDEX_4D(int64_t(kv_num_heads), int64_t(new_seqlen), int64_t(H), int64_t(b), int64_t(n), int64_t(s), int64_t(h))
                            : INDEX_4D(int64_t(new_seqlen), int64_t(kv_num_heads), int64_t(H), int64_t(b), int64_t(s), int64_t(n), int64_t(h));

    k_buff[out_offset] = new_k[in_offset];
    v_buff[out_offset] = new_v[in_offset];
  }
}

// Launcher for fused K+V append
template <typename T>
Status LaunchConcatKVInPlaceFused(int batch_size,
                                  int kv_num_heads,
                                  int head_size,
                                  int max_sequence_length,
                                  const int* past_seq_lens,
                                  const int* total_seq_lens,
                                  int new_seq_len,
                                  const T* new_key,
                                  const T* new_value,
                                  T* present_key,
                                  T* present_value,
                                  bool is_past_kv_bnsh_format,
                                  bool is_new_kv_bnsh_format,
                                  cudaStream_t stream,
                                  const int max_threads_per_block) {
  // Determine vectorization factor (float2 is 8 bytes)
  constexpr int vector_bytes = sizeof(float2);
  constexpr int element_bytes = sizeof(T);
  constexpr int elements_per_vector = vector_bytes / element_bytes;

  if (head_size % elements_per_vector != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size must be divisible by ", elements_per_vector, " for vectorized kernel.");
  }

  const int H = head_size / elements_per_vector;

  if (H * kv_num_heads <= max_threads_per_block) {
    const dim3 grid(new_seq_len, batch_size, 1);
    const dim3 block(H, kv_num_heads, 1);

    // Single kernel for both K and V
    ConcatKVInPlaceFused<float2><<<grid, block, 0, stream>>>(
        max_sequence_length,
        new_seq_len,
        reinterpret_cast<float2*>(present_key),
        reinterpret_cast<float2*>(present_value),
        reinterpret_cast<const float2*>(new_key),
        reinterpret_cast<const float2*>(new_value),
        past_seq_lens,
        total_seq_lens,
        is_past_kv_bnsh_format,
        is_new_kv_bnsh_format);
  } else {
    int steps = int(ceil(float(H * kv_num_heads) / 256.0));
    const dim3 grid(steps, new_seq_len, batch_size);
    const dim3 block(256, 1, 1);

    ConcatKVInPlaceFusedLarge<float2><<<grid, block, 0, stream>>>(
        max_sequence_length,
        new_seq_len,
        H,
        kv_num_heads,
        reinterpret_cast<float2*>(present_key),
        reinterpret_cast<float2*>(present_value),
        reinterpret_cast<const float2*>(new_key),
        reinterpret_cast<const float2*>(new_value),
        past_seq_lens,
        total_seq_lens,
        is_past_kv_bnsh_format,
        is_new_kv_bnsh_format);
  }
#ifndef NDEBUG
  CUDA_CALL(cudaStreamSynchronize(stream));
#endif
  return CUDA_CALL(cudaGetLastError());
}

template Status LaunchConcatKVInPlaceFused<half>(int batch_size,
                                                 int kv_num_heads,
                                                 int head_size,
                                                 int max_sequence_length,
                                                 const int* past_seq_lens,
                                                 const int* total_seq_lens,
                                                 int new_seq_len,
                                                 const half* new_key,
                                                 const half* new_value,
                                                 half* present_key,
                                                 half* present_value,
                                                 bool is_past_kv_bnsh_format,
                                                 bool is_new_kv_bnsh_format,
                                                 cudaStream_t stream,
                                                 const int max_threads_per_block);

template Status LaunchConcatKVInPlaceFused<BFloat16>(int batch_size,
                                                     int kv_num_heads,
                                                     int head_size,
                                                     int max_sequence_length,
                                                     const int* past_seq_lens,
                                                     const int* total_seq_lens,
                                                     int new_seq_len,
                                                     const BFloat16* new_key,
                                                     const BFloat16* new_value,
                                                     BFloat16* present_key,
                                                     BFloat16* present_value,
                                                     bool is_past_kv_bnsh_format,
                                                     bool is_new_kv_bnsh_format,
                                                     cudaStream_t stream,
                                                     const int max_threads_per_block);

template Status LaunchConcatKVInPlaceFused<float>(int batch_size,
                                                  int kv_num_heads,
                                                  int head_size,
                                                  int max_sequence_length,
                                                  const int* past_seq_lens,
                                                  const int* total_seq_lens,
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
