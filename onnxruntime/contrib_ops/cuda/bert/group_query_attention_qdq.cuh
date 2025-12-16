// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

// #include <cstdio> // Added for printf
#include <cuda_fp16.h>

#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Dequantization Kernel for KV cache.
template <typename T, typename T_QUANT, typename T_SCALE>
__global__ void DequantizeKernel(T* dequantized_data,
                                 const T_QUANT* quantized_data,
                                 const T_SCALE* scale, const int* seqlens,
                                 int batch_size, int num_heads,
                                 int cache_sequence_length, int sequence_length,
                                 int head_size, bool is_past, int bit_width,
                                 KVQuantizationType quant_type) {
  int S = cache_sequence_length;
  int total_elements = batch_size * num_heads * S * head_size;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements;
       i += blockDim.x * gridDim.x) {
    int h = i % head_size;
    int s = (i / head_size) % S;
    int n = (i / (head_size * S)) % num_heads;
    int b = i / (num_heads * head_size * S);

    // Correctly identify padding in the past_kv cache.
    // In the decoding case, `seqlens` contains `past_len + new_len - 1`.
    // We need the actual past_len to mask the padding correctly.
    if (is_past && seqlens != nullptr) {
      // For a given batch entry `b`, the actual length of the past sequence is `seqlens[b] + 1 - sequence_length`.
      // If `s` (the current sequence index) is beyond this length, it's padding and should be zeroed.
      int past_len_b = seqlens[b] + 1 - sequence_length;
      if (s >= past_len_b) {
        dequantized_data[i] = static_cast<T>(0.0f);
        continue;
      }
    }

    float scale_val = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR) {
      scale_val = static_cast<float>(scale[0]);
    } else {  // PER_CHANNEL
      int scale_idx = n * head_size + h;
      scale_val = static_cast<float>(scale[scale_idx]);
    }

    float quantized_float;
    if (bit_width == 8) {
      quantized_float = static_cast<float>(
          reinterpret_cast<const int8_t*>(quantized_data)[i]);
    } else {  // 4
      const uint8_t packed_val =
          reinterpret_cast<const uint8_t*>(quantized_data)[i / 2];
      quantized_float = (i % 2 == 0)
                            ? static_cast<float>((packed_val & 0x0F) - 8)
                            : static_cast<float>((packed_val >> 4) - 8);
    }

    dequantized_data[i] = static_cast<T>(quantized_float * scale_val);
  }
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchDequantizeKV(cudaStream_t stream, T* dequantized_data,
                          const T_QUANT* quantized_data, const T_SCALE* scale,
                          const int* seqlens, int batch_size, int num_heads,
                          int cache_sequence_length, int sequence_length,
                          int head_size, bool is_past, int bit_width,
                          KVQuantizationType quant_type) {
  int S = cache_sequence_length;
  if (S == 0) return Status::OK();

  int total_elements = batch_size * num_heads * S * head_size;
  const int threads_per_block = 256;
  const int blocks =
      (total_elements + threads_per_block - 1) / threads_per_block;

  DequantizeKernel<T, T_QUANT, T_SCALE><<<blocks, threads_per_block, 0, stream>>>(
      dequantized_data, quantized_data, scale, seqlens, batch_size, num_heads,
      cache_sequence_length, sequence_length, head_size, is_past, bit_width,
      quant_type);

  return CUDA_CALL(cudaGetLastError());
}

// Quantization Kernel for KV cache.
template <typename T, typename T_QUANT, typename T_SCALE>
__global__ void QuantizeKernel(T_QUANT* quantized_data,
                               const T* dequantized_data, const T_SCALE* scale,
                               const int* seqlens, int total_elements,
                               int cache_sequence_length, int num_heads, int head_size,
                               int bit_width, KVQuantizationType quant_type) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements;
       i += blockDim.x * gridDim.x) {
    int h = i % head_size;
    int s = (i / head_size) % cache_sequence_length;
    int n = (i / (head_size * cache_sequence_length)) % num_heads;
    int b = i / (num_heads * head_size * cache_sequence_length);

    // Zero out padding in the present_kv cache.
    // `seqlens` (seqlens_k) provides the total valid sequence length for each batch item.
    // If the current sequence index `s` is in the padded region, write zero.
    int total_valid_len_b = seqlens[b] + 1;
    if (s >= total_valid_len_b) {
      if (bit_width == 8) {
        reinterpret_cast<int8_t*>(quantized_data)[i] = 0;
      } else {  // 4
        // To avoid race conditions, only the thread for the even index writes the packed byte.
        if (i % 2 == 0) {
          uint8_t zero_nibble = (0 + 8) & 0x0F;
          uint8_t high_nibble;
          // Check if the next element is also in a padded region.
          if (s >= total_valid_len_b - (i % 2 == 0 ? 1 : 0)) {
            high_nibble = (0 + 8) & 0x0F;
          } else {
            // This path is complex; the safest approach is ensuring padded dequantized_data is zero,
            // but for robustness, we handle it here. Let's assume the adjacent value needs to be calculated.
            // (A simpler implementation would be to ensure `dequantized_data` is zeroed out before this kernel)
            // For now, let's just write zero for both nibbles if the first is padding.
            high_nibble = (0 + 8) & 0x0F;
          }
          reinterpret_cast<uint8_t*>(quantized_data)[i / 2] = zero_nibble | (high_nibble << 4);
        }
      }
      continue;
    }

    float scale_val = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR) {
      scale_val = static_cast<float>(scale[0]);
    } else {  // PER_CHANNEL
      int scale_idx = n * head_size + h;
      scale_val = static_cast<float>(scale[scale_idx]);
    }

    float inv_scale = (scale_val == 0.0f) ? 0.0f : 1.0f / scale_val;
    float val_float = static_cast<float>(dequantized_data[i]) * inv_scale;

    if (bit_width == 8) {
      int32_t val_int32 = static_cast<int32_t>(rintf(val_float));
      reinterpret_cast<int8_t*>(quantized_data)[i] =
          static_cast<int8_t>(max(-128, min(127, val_int32)));
    } else {  // 4
      int32_t val_int32 = static_cast<int32_t>(rintf(val_float));
      int8_t val_int8 = static_cast<int8_t>(max(-8, min(7, val_int32)));

      if (i % 2 == 0) {
        int8_t next_val_int8 = 0;
        if (i + 1 < total_elements) {
          int s_next = ((i + 1) / head_size) % cache_sequence_length;
          // Check if the next element is in a padded region as well.
          if (s_next >= total_valid_len_b) {
            next_val_int8 = 0;
          } else {
            float scale_val_next = 1.0f;
            if (quant_type == KVQuantizationType::PER_TENSOR) {
              scale_val_next = static_cast<float>(scale[0]);
            } else {  // PER_CHANNEL
              int h_next = (i + 1) % head_size;
              int n_next = ((i + 1) / (head_size * cache_sequence_length)) % num_heads;
              int scale_idx_next = n_next * head_size + h_next;
              scale_val_next = static_cast<float>(scale[scale_idx_next]);
            }

            float inv_scale_next =
                (scale_val_next == 0.0f) ? 0.0f : 1.0f / scale_val_next;
            float next_val_float =
                static_cast<float>(dequantized_data[i + 1]) * inv_scale_next;

            int32_t next_val_int32 = static_cast<int32_t>(rintf(next_val_float));
            next_val_int8 = static_cast<int8_t>(max(-8, min(7, next_val_int32)));
          }
        }

        uint8_t low_nibble = (val_int8 + 8) & 0x0F;
        uint8_t high_nibble = (next_val_int8 + 8) & 0x0F;
        reinterpret_cast<uint8_t*>(quantized_data)[i / 2] =
            low_nibble | (high_nibble << 4);
      }
    }
  }
}

// Append kernel for dynamic offset quantization
template <typename T, typename T_QUANT, typename T_SCALE>
__global__ void QuantizeAppendKernel(T_QUANT* cache_data,
                                     const T* new_data,
                                     const T_SCALE* scale,
                                     const int* past_seqlens,
                                     int max_seq_len,
                                     int num_heads,
                                     int head_size,
                                     int bit_width,
                                     KVQuantizationType quant_type) {
  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_elements = gridDim.x * blockDim.x;
  if (idx >= total_elements) return;

  int h_packed = idx % elements_per_head_packed;
  int tmp = idx / elements_per_head_packed;
  int n = tmp % num_heads;
  int b = tmp / num_heads;

  if (b >= gridDim.x) return;

  int past_len = past_seqlens[b];

  // [DEBUG PRINT]
  // if (b == 0 && n == 0 && h_packed == 0) {
  //     printf("[QuantAppend] B=%d N=%d PastLen=%d HeadSize=%d BitWidth=%d Type=%d\n",
  //            b, num_heads, past_len, head_size, bit_width, (int)quant_type);
  // }

  int64_t cache_offset = (int64_t)b * num_heads * max_seq_len * elements_per_head_packed +
                         (int64_t)n * max_seq_len * elements_per_head_packed +
                         (int64_t)past_len * elements_per_head_packed +
                         h_packed;

  if (bit_width == 8) {
    int h = h_packed;
    int64_t src_idx = (int64_t)b * num_heads * head_size + n * head_size + h;
    float val = static_cast<float>(new_data[src_idx]);

    float s = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR)
      s = (float)scale[0];
    else if (quant_type == KVQuantizationType::PER_CHANNEL)
      s = (float)scale[n * head_size + h];

    float inv_s = (s == 0.0f) ? 0.0f : 1.0f / s;
    int8_t q_val = static_cast<int8_t>(max(-128.0f, min(127.0f, rintf(val * inv_s))));
    reinterpret_cast<int8_t*>(cache_data)[cache_offset] = q_val;

    // [DEBUG PRINT DATA]
    // if (b == 0 && n == 0 && h < 4) {
    //     printf("[QuantAppend-8bit] H=%d Val=%f Scale=%f InvS=%f QVal=%d Offset=%lld\n",
    //            h, val, s, inv_s, (int)q_val, cache_offset);
    // }

  } else {  // Int4
    int h0 = h_packed * 2;
    int h1 = h0 + 1;

    int64_t src_idx0 = (int64_t)b * num_heads * head_size + n * head_size + h0;
    float val0 = static_cast<float>(new_data[src_idx0]);

    float val1 = 0.0f;
    if (h1 < head_size) {
      int64_t src_idx1 = (int64_t)b * num_heads * head_size + n * head_size + h1;
      val1 = static_cast<float>(new_data[src_idx1]);
    }

    float s0 = 1.0f;
    float s1 = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR) {
      s0 = (float)scale[0];
      s1 = (float)scale[0];
    } else {
      s0 = (float)scale[n * head_size + h0];
      if (h1 < head_size) s1 = (float)scale[n * head_size + h1];
    }

    int8_t q0 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(val0 * (s0 == 0 ? 0 : 1.0f / s0)))));
    int8_t q1 = static_cast<int8_t>(max(-8.0f, min(7.0f, rintf(val1 * (s1 == 0 ? 0 : 1.0f / s1)))));

    uint8_t packed = ((q0 + 8) & 0x0F) | (((q1 + 8) & 0x0F) << 4);
    reinterpret_cast<uint8_t*>(cache_data)[cache_offset] = packed;

    // [DEBUG PRINT DATA]
    // if (b == 0 && n == 0 && h0 < 4) {
    //     printf("[QuantAppend-4bit] H0=%d Val0=%f Scale0=%f Q0=%d | H1=%d Val1=%f Scale1=%f Q1=%d | Packed=%x\n",
    //            h0, val0, s0, (int)q0, h1, val1, s1, (int)q1, (int)packed);
    // }
  }
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchQuantizeKV(cudaStream_t stream, T_QUANT* quantized_data,
                        const T* dequantized_data, const T_SCALE* scale,
                        const int* seqlens, int batch_size, int num_heads,
                        int cache_sequence_length, int head_size, int bit_width,
                        KVQuantizationType quant_type) {
  if (cache_sequence_length == 0) return Status::OK();

  int total_elements = batch_size * num_heads * cache_sequence_length * head_size;
  const int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;

  QuantizeKernel<T, T_QUANT, T_SCALE><<<blocks, threads_per_block, 0, stream>>>(
      quantized_data, dequantized_data, scale, seqlens, total_elements,
      cache_sequence_length, num_heads, head_size, bit_width, quant_type);

  return CUDA_CALL(cudaGetLastError());
}

template <typename T, typename T_QUANT, typename T_SCALE>
Status LaunchQuantizeAppendKV(cudaStream_t stream, T_QUANT* cache_data,
                              const T* new_data, const T_SCALE* scale,
                              const int* past_seqlens, int batch_size, int num_heads,
                              int max_seq_len, int head_size, int bit_width,
                              KVQuantizationType quant_type) {
  int elements_per_head_packed = (bit_width == 4) ? (head_size + 1) / 2 : head_size;
  int total_threads = batch_size * num_heads * elements_per_head_packed;
  const int threads_per_block = 256;
  int blocks = (total_threads + threads_per_block - 1) / threads_per_block;

  QuantizeAppendKernel<T, T_QUANT, T_SCALE><<<blocks, threads_per_block, 0, stream>>>(
      cache_data, new_data, scale, past_seqlens, max_seq_len, num_heads, head_size, bit_width, quant_type);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations for launchers
template Status LaunchDequantizeKV<half, int8_t, half>(cudaStream_t, half*,
                                                       const int8_t*, const half*,
                                                       const int*, int, int, int, int,
                                                       int, bool, int,
                                                       KVQuantizationType);
template Status LaunchDequantizeKV<half, uint8_t, half>(cudaStream_t, half*,
                                                        const uint8_t*, const half*,
                                                        const int*, int, int, int,
                                                        int, int, bool, int,
                                                        KVQuantizationType);
template Status LaunchDequantizeKV<BFloat16, int8_t, BFloat16>(
    cudaStream_t, BFloat16*, const int8_t*, const BFloat16*, const int*, int, int,
    int, int, int, bool, int, KVQuantizationType);
template Status LaunchDequantizeKV<BFloat16, uint8_t, BFloat16>(
    cudaStream_t, BFloat16*, const uint8_t*, const BFloat16*, const int*, int, int,
    int, int, int, bool, int, KVQuantizationType);

template Status LaunchQuantizeKV<half, int8_t, half>(
    cudaStream_t, int8_t*, const half*, const half*, const int*, int, int, int, int,
    int, KVQuantizationType);
template Status LaunchQuantizeKV<half, uint8_t, half>(
    cudaStream_t, uint8_t*, const half*, const half*, const int*, int, int, int, int,
    int, KVQuantizationType);
template Status LaunchQuantizeKV<BFloat16, int8_t, BFloat16>(
    cudaStream_t, int8_t*, const BFloat16*, const BFloat16*, const int*, int, int, int,
    int, int, KVQuantizationType);
template Status LaunchQuantizeKV<BFloat16, uint8_t, BFloat16>(
    cudaStream_t, uint8_t*, const BFloat16*, const BFloat16*, const int*, int, int, int,
    int, int, KVQuantizationType);

template Status LaunchQuantizeAppendKV<half, int8_t, half>(
    cudaStream_t, int8_t*, const half*, const half*, const int*, int, int, int, int,
    int, KVQuantizationType);
template Status LaunchQuantizeAppendKV<half, uint8_t, half>(
    cudaStream_t, uint8_t*, const half*, const half*, const int*, int, int, int, int,
    int, KVQuantizationType);
template Status LaunchQuantizeAppendKV<BFloat16, int8_t, BFloat16>(
    cudaStream_t, int8_t*, const BFloat16*, const BFloat16*, const int*, int, int, int,
    int, int, KVQuantizationType);
template Status LaunchQuantizeAppendKV<BFloat16, uint8_t, BFloat16>(
    cudaStream_t, uint8_t*, const BFloat16*, const BFloat16*, const int*, int, int, int,
    int, int, KVQuantizationType);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
