// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.
#pragma once

#include <cuda_fp16.h>

#include "contrib_ops/cuda/bert/group_query_attention_impl.h"
#include "core/providers/cuda/cuda_common.h"
#include "core/providers/cuda/shared_inc/cuda_call.h"

using namespace onnxruntime::cuda;

namespace onnxruntime {
namespace contrib {
namespace cuda {

// New Dequantization/Quantization Kernels
template <typename T, typename T_QUANT>
__global__ void DequantizeKernel(T* dequantized_data,
                                 const T_QUANT* quantized_data,
                                 const float* scale, const int* seqlens,
                                 int batch_size, int num_heads,
                                 int past_sequence_length, int sequence_length,
                                 int head_size, bool is_past, int bit_width,
                                 KVQuantizationType quant_type) {
  int S = is_past ? past_sequence_length : sequence_length;
  int total_elements = batch_size * num_heads * S * head_size;

  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements;
       i += blockDim.x * gridDim.x) {
    int h = i % head_size;
    int s = (i / head_size) % S;
    int n = (i / head_size / S) % num_heads;
    int b = (i / head_size / S / num_heads);

    float scale_val = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR) {
      scale_val = scale[0];
    } else {  // PER_CHANNEL
      int scale_idx = n * head_size + h;
      scale_val = scale[scale_idx];
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

template <typename T, typename T_QUANT>
Status LaunchDequantizeKV(cudaStream_t stream, T* dequantized_data,
                          const T_QUANT* quantized_data, const float* scale,
                          const int* seqlens, int batch_size, int num_heads,
                          int past_sequence_length, int sequence_length,
                          int head_size, bool is_past, int bit_width,
                          KVQuantizationType quant_type) {
  int S = is_past ? past_sequence_length : sequence_length;
  if (S == 0) return Status::OK();

  int total_elements = batch_size * num_heads * S * head_size;
  const int threads_per_block = 256;
  const int blocks =
      (total_elements + threads_per_block - 1) / threads_per_block;

  DequantizeKernel<T, T_QUANT><<<blocks, threads_per_block, 0, stream>>>(
      dequantized_data, quantized_data, scale, seqlens, batch_size, num_heads,
      past_sequence_length, sequence_length, head_size, is_past, bit_width,
      quant_type);

  return CUDA_CALL(cudaGetLastError());
}

template <typename T, typename T_QUANT>
__global__ void QuantizeKernel(T_QUANT* quantized_data,
                               const T* dequantized_data, const float* scale,
                               int total_elements, int sequence_length,
                               int num_heads, int head_size, int bit_width,
                               KVQuantizationType quant_type) {
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < total_elements;
       i += blockDim.x * gridDim.x) {
    int h = i % head_size;
    int s = (i / head_size) % sequence_length;
    int n = (i / (head_size * sequence_length)) % num_heads;

    float scale_val = 1.0f;
    if (quant_type == KVQuantizationType::PER_TENSOR) {
      scale_val = scale[0];
    } else {  // PER_CHANNEL
      int scale_idx = n * head_size + h;
      scale_val = scale[scale_idx];
    }

    float inv_scale = (scale_val == 0.0f) ? 0.0f : 1.0f / scale_val;
    float val_float = static_cast<float>(dequantized_data[i]) * inv_scale;

    if (bit_width == 8) {
      int32_t val_int32 = static_cast<int32_t>(roundf(val_float));
      reinterpret_cast<int8_t*>(quantized_data)[i] =
          static_cast<int8_t>(max(-128, min(127, val_int32)));
    } else {  // 4
      int32_t val_int32 = static_cast<int32_t>(roundf(val_float));
      int8_t val_int8 = static_cast<int8_t>(max(-8, min(7, val_int32)));

      // This part has a race condition if multiple threads write to the same
      // byte. A simple approach is to have only one thread per byte write the
      // packed result.
      if (i % 2 == 0) {
        int8_t next_val_int8 =
            0;  // Default for padding if it's the last element
        if (i + 1 < total_elements) {
          // Calculate scale for the next element
          float scale_val_next = 1.0f;
          if (quant_type == KVQuantizationType::PER_TENSOR) {
            scale_val_next = scale[0];
          } else {  // PER_CHANNEL
            int h_next = (i + 1) % head_size;
            int s_next = ((i + 1) / head_size) % sequence_length;
            int n_next = ((i + 1) / (head_size * sequence_length)) % num_heads;
            int scale_idx_next = n_next * head_size + h_next;
            scale_val_next = scale[scale_idx_next];
          }

          float inv_scale_next =
              (scale_val_next == 0.0f) ? 0.0f : 1.0f / scale_val_next;
          float next_val_float =
              static_cast<float>(dequantized_data[i + 1]) * inv_scale_next;

          int32_t next_val_int32 = static_cast<int32_t>(roundf(next_val_float));
          next_val_int8 = static_cast<int8_t>(max(-8, min(7, next_val_int32)));
        }

        uint8_t low_nibble = (val_int8 + 8) & 0x0F;
        uint8_t high_nibble = (next_val_int8 + 8) & 0x0F;
        reinterpret_cast<uint8_t*>(quantized_data)[i / 2] =
            low_nibble | (high_nibble << 4);
      }
    }
  }
}

template <typename T, typename T_QUANT>
Status LaunchQuantizeKV(cudaStream_t stream, T_QUANT* quantized_data,
                        T* dequantized_data, const float* scale, int batch_size,
                        int num_heads, int sequence_length, int head_size,
                        int bit_width, KVQuantizationType quant_type) {
  if (sequence_length == 0) return Status::OK();

  int total_elements = batch_size * num_heads * sequence_length * head_size;
  const int threads_per_block = 256;
  int blocks = (total_elements + threads_per_block - 1) / threads_per_block;
  if (bit_width == 4) {
    // Each thread handles two elements to avoid race conditions when packing
    // int4s
    blocks =
        ((total_elements + 1) / 2 + threads_per_block - 1) / threads_per_block;
  }

  QuantizeKernel<T, T_QUANT><<<blocks, threads_per_block, 0, stream>>>(
      quantized_data, dequantized_data, scale, total_elements, sequence_length,
      num_heads, head_size, bit_width, quant_type);

  return CUDA_CALL(cudaGetLastError());
}

// Explicit instantiations for launchers
template Status LaunchDequantizeKV<half, int8_t>(cudaStream_t, half*,
                                                 const int8_t*, const float*,
                                                 const int*, int, int, int, int,
                                                 int, bool, int,
                                                 KVQuantizationType);
template Status LaunchDequantizeKV<half, uint8_t>(cudaStream_t, half*,
                                                  const uint8_t*, const float*,
                                                  const int*, int, int, int,
                                                  int, int, bool, int,
                                                  KVQuantizationType);
template Status LaunchDequantizeKV<BFloat16, int8_t>(
    cudaStream_t, BFloat16*, const int8_t*, const float*, const int*, int, int,
    int, int, int, bool, int, KVQuantizationType);
template Status LaunchDequantizeKV<BFloat16, uint8_t>(
    cudaStream_t, BFloat16*, const uint8_t*, const float*, const int*, int, int,
    int, int, int, bool, int, KVQuantizationType);

template Status LaunchQuantizeKV<half, int8_t>(cudaStream_t, int8_t*, half*,
                                               const float*, int, int, int, int,
                                               int, KVQuantizationType);
template Status LaunchQuantizeKV<half, uint8_t>(cudaStream_t, uint8_t*, half*,
                                                const float*, int, int, int,
                                                int, int, KVQuantizationType);
template Status LaunchQuantizeKV<BFloat16, int8_t>(cudaStream_t, int8_t*,
                                                   BFloat16*, const float*, int,
                                                   int, int, int, int,
                                                   KVQuantizationType);
template Status LaunchQuantizeKV<BFloat16, uint8_t>(cudaStream_t, uint8_t*,
                                                    BFloat16*, const float*,
                                                    int, int, int, int, int,
                                                    KVQuantizationType);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

#undef OFFSET_BNSH
#undef OFFSET_BSNH
