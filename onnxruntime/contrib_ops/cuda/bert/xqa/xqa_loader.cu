// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xqa_loader.h"
#include <cassert>

namespace onnxruntime {
namespace contrib {
namespace cuda {

// Forward declarations of instantiated kernels from H128 and H64 namespaces
namespace H128 {
template <typename T>
Status LaunchXQAKernelImpl(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size);

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

size_t GetXQAInt8ScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

size_t GetXQAInt8ScratchSizeBF16(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

size_t GetXQABf16ScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);
}  // namespace H128

namespace H64 {
template <typename T>
Status LaunchXQAKernelImpl(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size);

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

size_t GetXQAInt8ScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

size_t GetXQAInt8ScratchSizeBF16(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

size_t GetXQABf16ScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);
}  // namespace H64

namespace H256 {
template <typename T>
Status LaunchXQAKernelImpl(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size);

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

size_t GetXQAInt8ScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

size_t GetXQAInt8ScratchSizeBF16(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);

size_t GetXQABf16ScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);
}  // namespace H256

// Dispatcher Implementation
namespace H64 {
size_t GetXQAInt8ScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);
}
namespace H128 {
size_t GetXQAInt8ScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);
}
namespace H256 {
size_t GetXQAInt8ScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len);
}

template <typename T>
Status LaunchXQAKernel(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size) {
  if (device_prop.major < 8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA is only supported on Ampere (SM80) or newer GPUs.");
  }

  if (head_size == 256) {
    return H256::LaunchXQAKernelImpl<T>(
        device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size,
        max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, kv_quant_type, workspace, workspace_size);
  } else if (head_size == 128) {
    return H128::LaunchXQAKernelImpl<T>(
        device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size,
        max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, kv_quant_type, workspace, workspace_size);
  } else if (head_size == 64) {
    return H64::LaunchXQAKernelImpl<T>(
        device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size,
        max_seq_len, scale, is_bsnh, past_seq_lens, kv_cache_scale, kv_quant_type, workspace, workspace_size);
  } else {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA only supports head_size=64, 128, or 256. Input has ", head_size);
  }
}

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int head_size,
    int max_seq_len,
    int kv_quant_type,
    bool is_bf16) {
  if (device_prop.major < 8) {
    return 0;
  }

  // INT8 path
  if (kv_quant_type == 1) {
    if (is_bf16) {
      if (head_size == 128) {
        return H128::GetXQAInt8ScratchSizeBF16(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
      } else if (head_size == 64) {
        return H64::GetXQAInt8ScratchSizeBF16(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
      } else if (head_size == 256) {
        return H256::GetXQAInt8ScratchSizeBF16(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
      } else {
        return 0;  // Not supported
      }
    } else {
      if (head_size == 128) {
        return H128::GetXQAInt8ScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
      } else if (head_size == 64) {
        return H64::GetXQAInt8ScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
      } else if (head_size == 256) {
        return H256::GetXQAInt8ScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
      } else {
        return 0;  // Not supported
      }
    }
  }

  // FP16/BF16 path
  if (is_bf16) {
    if (head_size == 128) {
      return H128::GetXQABf16ScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
    } else if (head_size == 64) {
      return H64::GetXQABf16ScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
    } else if (head_size == 256) {
      return H256::GetXQABf16ScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
    }
  } else {
    if (head_size == 128) {
      return H128::GetXQAScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
    } else if (head_size == 64) {
      return H64::GetXQAScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
    } else if (head_size == 256) {
      return H256::GetXQAScratchSize(device_prop, batch_size, num_heads, kv_num_heads, max_seq_len);
    }
  }

  return 0;
}

// Instantiate template for half
template Status LaunchXQAKernel<half>(
    const cudaDeviceProp& device_prop,
    cudaStream_t stream,
    const void* query,
    const void* key_cache,
    const void* value_cache,
    void* output,
    const int batch_size,
    const int num_heads,
    const int kv_num_heads,
    const int head_size,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* past_seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
