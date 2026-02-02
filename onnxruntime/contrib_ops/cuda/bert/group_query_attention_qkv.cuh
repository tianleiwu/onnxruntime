// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cuda_fp16.h>

#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "contrib_ops/cpu/bert/attention_common.h"
#include "contrib_ops/cuda/bert/rotary_common.cuh"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Fused kernel: Unpack QKV + Apply RoPE to Q and K + Append K/V directly to cache + Quantize if needed
//
// This kernel performs the following:
// 1. Unpacks Q, K, V from input tensor(s).
// 2. Applies Rotary Positional Embedding (RoPE) to Q and K.
// 3. Appends K and V to the KV cache, performing on-the-fly quantization (Int8/Int4) if configured.
// 4. Writes the rotated Q back to global memory (for subsequent Flash Attention).
template <typename T, int BIT_WIDTH = 16, int MAX_HEAD_SIZE = 256>
__global__ void UnpackRoPEAppend(
    const T* packed_qkv,
    const T* query,
    const T* key,
    const T* value,
    T* unpacked_q,
    void* k_cache,
    void* v_cache,
    const float* k_scale,
    const float* v_scale,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int d,           // packed QKV hidden stride = (num_heads + 2*kv_num_heads) * head_size
    const int max_seqlen,  // KV cache max sequence length
    const int* past_seq_lens,
    const T* cos_cache,
    const T* sin_cache,
    const int rotary_dim,
    const int64_t* position_ids,
    const bool interleaved,
    const bool is_cache_bnsh,
    const bool per_channel) {
  using LoadT = float4;
  constexpr int elements_per_thread = sizeof(LoadT) / sizeof(T);

  const int s = blockIdx.x;
  const int head_idx = blockIdx.y;
  const int b = blockIdx.z;
  const int tid = threadIdx.x;
  const int h = tid * elements_per_thread;

  // Guard work with 'valid' instead of early return to ensure all threads reach __syncthreads()
  const bool valid = (h < head_size);

  const int q_hidden = num_heads * head_size;
  const int k_hidden = kv_num_heads * head_size;
  const int sequence_length = gridDim.x;

  __shared__ T shared_head[MAX_HEAD_SIZE];

  // Determine Head Type and Offset within hidden dimension
  enum HeadType { QUERY,
                  KEY,
                  VALUE };
  HeadType head_type;
  int n;  // Index within its specific type
  int offset_in_hidden;

  if (head_idx < num_heads) {
    head_type = QUERY;
    n = head_idx;
    offset_in_hidden = n * head_size;
  } else if (head_idx < num_heads + kv_num_heads) {
    head_type = KEY;
    n = head_idx - num_heads;
    offset_in_hidden = q_hidden + n * head_size;
  } else {
    head_type = VALUE;
    n = head_idx - (num_heads + kv_num_heads);
    offset_in_hidden = q_hidden + k_hidden + n * head_size;
  }

  // 1. Load data into Registers
  alignas(16) T vals[elements_per_thread];
  if (valid) {
    if (packed_qkv != nullptr) {
      const int64_t packed_idx = static_cast<int64_t>(b) * sequence_length * d +
                                 static_cast<int64_t>(s) * d +
                                 static_cast<int64_t>(offset_in_hidden) + h;
      *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(packed_qkv)[packed_idx / elements_per_thread];
    } else {
      if (head_type == QUERY) {
        const int64_t q_idx = static_cast<int64_t>(b) * sequence_length * q_hidden +
                              static_cast<int64_t>(s) * q_hidden +
                              static_cast<int64_t>(n) * head_size + h;
        *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(query)[q_idx / elements_per_thread];
      } else if (head_type == KEY) {
        const int64_t k_idx = static_cast<int64_t>(b) * sequence_length * k_hidden +
                              static_cast<int64_t>(s) * k_hidden +
                              static_cast<int64_t>(n) * head_size + h;
        *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(key)[k_idx / elements_per_thread];
      } else {
        const int64_t v_idx = static_cast<int64_t>(b) * sequence_length * k_hidden +
                              static_cast<int64_t>(s) * k_hidden +
                              static_cast<int64_t>(n) * head_size + h;
        *reinterpret_cast<LoadT*>(vals) = reinterpret_cast<const LoadT*>(value)[v_idx / elements_per_thread];
      }
    }
  }

  // 2. Process RoPE
  // Optimization: Only use shared memory for non-interleaved mode
  const bool is_qk = (head_type == QUERY || head_type == KEY);
  if (valid && rotary_dim > 0 && is_qk && !interleaved) {
    T* shared_ptr = &shared_head[h];
    *reinterpret_cast<LoadT*>(shared_ptr) = *reinterpret_cast<LoadT*>(vals);
  }

  // CRITICAL: Barrier must be outside the 'if(valid)' and 'if(is_qk)' blocks
  // to ensure every thread in the block participates.
  __syncthreads();

  if (valid && rotary_dim > 0 && is_qk) {
    const int past_seq_len = past_seq_lens[b];
    const int64_t pos_base = static_cast<int64_t>(b) * sequence_length;
    int pos_id = (position_ids != nullptr) ? static_cast<int>(position_ids[pos_base + s]) : (past_seq_len + s);
    const int h_idx = h / elements_per_thread;

    onnxruntime::contrib::cuda::RotaryDispatcher<LoadT, T>::apply(
        *reinterpret_cast<LoadT*>(vals),
        reinterpret_cast<const LoadT*>(cos_cache),
        reinterpret_cast<const LoadT*>(sin_cache),
        rotary_dim, h_idx, pos_id, interleaved,
        reinterpret_cast<const LoadT*>(shared_head),
        0);
  }

  // 3. Store results back to Global Memory
  if (valid) {
    if (head_type == QUERY) {
      if (unpacked_q != nullptr) {
        const int64_t q_out_idx = static_cast<int64_t>(b) * sequence_length * q_hidden +
                                  static_cast<int64_t>(s) * q_hidden +
                                  static_cast<int64_t>(n) * head_size + h;
        reinterpret_cast<LoadT*>(unpacked_q)[q_out_idx / elements_per_thread] = *reinterpret_cast<LoadT*>(vals);
      }
    } else {
      const int cache_s = past_seq_lens[b] + s;
      if (cache_s < max_seqlen) {
        void* cache_ptr = (head_type == KEY) ? k_cache : v_cache;
        if (cache_ptr != nullptr) {
          int64_t cache_idx;
          if (is_cache_bnsh) {
            // BNSH layout: [Batch, NumHeads, SeqLen, HeadSize]
            cache_idx = static_cast<int64_t>(b) * kv_num_heads * max_seqlen * head_size +
                        static_cast<int64_t>(n) * max_seqlen * head_size +
                        static_cast<int64_t>(cache_s) * head_size +
                        h;
          } else {
            // BSNH layout: [Batch, SeqLen, NumHeads, HeadSize]
            cache_idx = static_cast<int64_t>(b) * max_seqlen * kv_num_heads * head_size +
                        static_cast<int64_t>(cache_s) * kv_num_heads * head_size +
                        static_cast<int64_t>(n) * head_size +
                        h;
          }

          if constexpr (BIT_WIDTH == 16 || BIT_WIDTH == 32) {
            reinterpret_cast<LoadT*>(cache_ptr)[cache_idx / elements_per_thread] = *reinterpret_cast<LoadT*>(vals);
          } else if constexpr (BIT_WIDTH == 8) {
            const float* scale_buffer = (head_type == KEY) ? k_scale : v_scale;
            uint64_t packed = 0;
            for (int i = 0; i < elements_per_thread; ++i) {
              float scale_val = per_channel ? scale_buffer[n * head_size + h + i] : scale_buffer[0];
              float inv_s = (scale_val == 0.0f) ? 0.0f : 1.0f / scale_val;
              int8_t q = static_cast<int8_t>(max(-128.0f, min(127.0f, rintf(static_cast<float>(vals[i]) * inv_s))));
              packed |= (static_cast<uint64_t>(static_cast<uint8_t>(q)) << (i * 8));
            }
            reinterpret_cast<uint64_t*>(cache_ptr)[cache_idx / elements_per_thread] = packed;
          } else if constexpr (BIT_WIDTH == 4) {
            constexpr float kInt4Min = -8.0f;
            constexpr float kInt4Max = 7.0f;
            const float* scale_buffer = (head_type == KEY) ? k_scale : v_scale;
            // Pack 8 4-bit values into one 32-bit integer (each thread handles 8 elements, i.e., 4 float4 loads? No, loop is i < 4)
            // Loop runs 4 times. 'vals' has elements_per_thread = 8 (for half).
            // Actually, let's verify assumptions.
            // If T=half, elements_per_thread=8. Loop i=0..3 handles 4 pairs = 8 elements. Correct.
            // Packing: 2 4-bit values -> 1 uint8. 4 uint8s -> 1 uint32.
            uint32_t packed = 0;
            for (int i = 0; i < 4; ++i) {
              float s0 = per_channel ? scale_buffer[n * head_size + h + i * 2] : scale_buffer[0];
              float s1 = per_channel ? scale_buffer[n * head_size + h + i * 2 + 1] : scale_buffer[0];
              int8_t q0 = static_cast<int8_t>(max(kInt4Min, min(kInt4Max, rintf(static_cast<float>(vals[i * 2]) * (s0 == 0 ? 0 : 1.0f / s0)))));
              int8_t q1 = static_cast<int8_t>(max(kInt4Min, min(kInt4Max, rintf(static_cast<float>(vals[i * 2 + 1]) * (s1 == 0 ? 0 : 1.0f / s1)))));
              uint8_t p = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
              packed |= (static_cast<uint32_t>(p) << (i * 8));
            }
            reinterpret_cast<uint32_t*>(cache_ptr)[cache_idx / elements_per_thread] = packed;
          }
        }
      }
    }
  }
}

template <typename T, int BIT_WIDTH>
Status DispatchUnpackRoPEAppendHeadSize(
    const dim3& grid, const dim3& block, cudaStream_t stream,
    const T* packed_qkv, const T* query, const T* key, const T* value,
    T* unpacked_q, void* k_cache, void* v_cache,
    const float* k_scale, const float* v_scale,
    const int num_heads, const int kv_num_heads, const int head_size, const int d,
    const int max_seqlen, const int* past_seq_lens,
    const T* cos_cache, const T* sin_cache, const int rotary_dim,
    const int64_t* position_ids, const bool interleaved, const bool is_cache_bnsh, const bool per_channel) {
  if (head_size <= 64) {
    UnpackRoPEAppend<T, BIT_WIDTH, 64><<<grid, block, 0, stream>>>(
        packed_qkv, query, key, value, unpacked_q, k_cache, v_cache, k_scale, v_scale,
        num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else if (head_size <= 128) {
    UnpackRoPEAppend<T, BIT_WIDTH, 128><<<grid, block, 0, stream>>>(
        packed_qkv, query, key, value, unpacked_q, k_cache, v_cache, k_scale, v_scale,
        num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else if (head_size <= 256) {
    UnpackRoPEAppend<T, BIT_WIDTH, 256><<<grid, block, 0, stream>>>(
        packed_qkv, query, key, value, unpacked_q, k_cache, v_cache, k_scale, v_scale,
        num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size (", head_size, ") exceeds maximum supported MAX_HEAD_SIZE (256).");
  }
  return CUDA_CALL(cudaGetLastError());
}

template <typename T>
Status LaunchUnpackRoPEAppend(
    const T* packed_qkv, const T* query, const T* key, const T* value,
    T* unpacked_q, void* k_cache, void* v_cache,
    const float* k_scale, const float* v_scale,
    const int num_heads, const int kv_num_heads, const int head_size,
    const int sequence_length, const int batch_size, const int max_seqlen,
    const int* past_seq_lens, const T* cos_cache, const T* sin_cache,
    const int rotary_dim, const int64_t* position_ids, const bool interleaved,
    const bool is_cache_bnsh, const KVQuantizationType k_quant_type,
    const int bit_width, cudaStream_t stream, const int max_threads_per_block) {
  constexpr int elements_per_vector = sizeof(float4) / sizeof(T);

  if (head_size % elements_per_vector != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size must be divisible by vector size (16 bytes).");
  }

  if (rotary_dim > head_size) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "rotary_dim (", rotary_dim, ") cannot exceed head_size (", head_size, ").");
  }

  if (!interleaved && rotary_dim % 2 != 0) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Non-interleaved RoPE requires even rotary_dim.");
  }

  const int total_heads = num_heads + 2 * kv_num_heads;
  const int d = total_heads * head_size;

  const int threads_per_block = (head_size + elements_per_vector - 1) / elements_per_vector;
  if (threads_per_block > max_threads_per_block) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Head size too large for current block configuration.");
  }

  if (total_heads > 65535) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Total heads (", total_heads, ") exceeds CUDA grid limit (65535).");
  }
  if (batch_size > 65535) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "batch_size (", batch_size, ") exceeds CUDA grid limit (65535).");
  }

  const dim3 grid(sequence_length, total_heads, batch_size);
  const dim3 block(threads_per_block);

  bool per_channel = (k_quant_type == KVQuantizationType::PER_CHANNEL);

  if (bit_width == 16 || bit_width == 32) {
    return DispatchUnpackRoPEAppendHeadSize<T, 16>(
        grid, block, stream, packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
        k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else if (bit_width == 8) {
    return DispatchUnpackRoPEAppendHeadSize<T, 8>(
        grid, block, stream, packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
        k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  } else if (bit_width == 4) {
    return DispatchUnpackRoPEAppendHeadSize<T, 4>(
        grid, block, stream, packed_qkv, query, key, value, unpacked_q, k_cache, v_cache,
        k_scale, v_scale, num_heads, kv_num_heads, head_size, d, max_seqlen, past_seq_lens,
        cos_cache, sin_cache, rotary_dim, position_ids, interleaved, is_cache_bnsh, per_channel);
  }

  return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "Unsupported bit_width (", bit_width, ") for GQA quantization.");
}

// Explicit template instantiations
template Status LaunchUnpackRoPEAppend<half>(
    const half*, const half*, const half*, const half*, half*, void*, void*, const float*, const float*,
    int, int, int, int, int, int, const int*, const half*, const half*, int, const int64_t*, bool, bool,
    KVQuantizationType, int, cudaStream_t, int);

template Status LaunchUnpackRoPEAppend<BFloat16>(
    const BFloat16*, const BFloat16*, const BFloat16*, const BFloat16*, BFloat16*, void*, void*, const float*, const float*,
    int, int, int, int, int, int, const int*, const BFloat16*, const BFloat16*, int, const int64_t*, bool, bool,
    KVQuantizationType, int, cudaStream_t, int);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
