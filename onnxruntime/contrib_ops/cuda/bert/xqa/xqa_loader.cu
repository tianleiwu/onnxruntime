// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "xqa_loader.h"
#include <cassert>

// Define global constants BEFORE including ANY header that uses them
#define HEAD_ELEMS 128
#define USE_PAGED_KV_CACHE 0
#define TOKENS_PER_PAGE 0
#define INPUT_FP16 1
#define ALLOW_MULTI_BLOCK_MODE 1

#pragma nv_diag_suppress 177
#pragma nv_diag_suppress 20012

// Include common headers once
#include "cuda_hint.cuh"
#include "mha.h"
// Include all helpers globally to ensure visibility
#include "ldgsts.cuh"
#include "mhaUtils.cuh"
#include "mha_components.cuh"
#include "mma.cuh"
#include "utils.cuh"
#include "hostUtils.h"

// Undefine HEAD_GRP_SIZE and M_TILESIZE to allow re-definition in impl gen
#undef HEAD_GRP_SIZE
#undef M_TILESIZE

namespace onnxruntime {
namespace contrib {
namespace cuda {

// ============================================================================
// FP16 KV Cache Instantiations (CACHE_ELEM_ENUM=0)
// Each group_size maps to appropriate M_TILESIZE:
//   - group_size 1-8:   M_TILESIZE=8
//   - group_size 16:    M_TILESIZE=16
//   - group_size 32:    M_TILESIZE=32
// ============================================================================

#define NAMESPACE_NAME grp1_fp16
#define GRP_SIZE 1
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp2_fp16
#define GRP_SIZE 2
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp4_fp16
#define GRP_SIZE 4
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp8_fp16
#define GRP_SIZE 8
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp16_fp16
#define GRP_SIZE 16
#define M_TILESIZE 16
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp32_fp16
#define GRP_SIZE 32
#define M_TILESIZE 32
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

// ============================================================================
// INT8 KV Cache Instantiations (CACHE_ELEM_ENUM=1)
// ============================================================================

#undef CACHE_ELEM_ENUM
#define CACHE_ELEM_ENUM 1

#define NAMESPACE_NAME grp8_int8
#define GRP_SIZE 8
#define M_TILESIZE 8
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp16_int8
#define GRP_SIZE 16
#define M_TILESIZE 16
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

#define NAMESPACE_NAME grp32_int8
#define GRP_SIZE 32
#define M_TILESIZE 32
#include "xqa_impl_gen.cuh"
#undef NAMESPACE_NAME
#undef GRP_SIZE
#undef M_TILESIZE

// Reset CACHE_ELEM_ENUM to default
#undef CACHE_ELEM_ENUM
#define CACHE_ELEM_ENUM 0

// ============================================================================
// Dispatcher
// ============================================================================

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
    const int actual_seq_len,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size) {
  if (head_size != 128) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA only supports head_size=128.");
  }
  if (!std::is_same<T, half>::value) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA only supports FP16 in generic path.");
  }

  int group_size = num_heads / kv_num_heads;

  // Dispatch based on kv_quant_type and group_size
  if (kv_quant_type == 1) {
    // INT8 KV Cache path
    switch (group_size) {
      case 8:
        return grp8_int8::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, workspace, workspace_size);
      case 16:
        return grp16_int8::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, workspace, workspace_size);
      case 32:
        return grp32_int8::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, workspace, workspace_size);
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA INT8 supports group_size 8, 16, 32. Input has ", group_size);
    }
  } else {
    // FP16 KV Cache path (default)
    switch (group_size) {
      case 1:
        return grp1_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, workspace, workspace_size);
      case 2:
        return grp2_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, workspace, workspace_size);
      case 4:
        return grp4_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, workspace, workspace_size);
      case 8:
        return grp8_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, workspace, workspace_size);
      case 16:
        return grp16_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, workspace, workspace_size);
      case 32:
        return grp32_fp16::Launch<T>(device_prop, stream, query, key_cache, value_cache, output, batch_size, num_heads, kv_num_heads, head_size, actual_seq_len, max_seq_len, scale, is_bsnh, seq_lens, kv_cache_scale, workspace, workspace_size);
      default:
        return ORT_MAKE_STATUS(ONNXRUNTIME, FAIL, "XQA supports group_size 1, 2, 4, 8, 16, 32. Input has ", group_size);
    }
  }
}

size_t GetXQAScratchSize(
    const cudaDeviceProp& device_prop,
    int batch_size,
    int num_heads,
    int kv_num_heads,
    int max_seq_len) {
  int group_size = num_heads / kv_num_heads;
  switch (group_size) {
    case 1:
      return grp1_fp16::GetScratchSize(device_prop, batch_size, kv_num_heads, max_seq_len);
    case 2:
      return grp2_fp16::GetScratchSize(device_prop, batch_size, kv_num_heads, max_seq_len);
    case 4:
      return grp4_fp16::GetScratchSize(device_prop, batch_size, kv_num_heads, max_seq_len);
    case 8:
      return grp8_fp16::GetScratchSize(device_prop, batch_size, kv_num_heads, max_seq_len);
    case 16:
      return grp16_fp16::GetScratchSize(device_prop, batch_size, kv_num_heads, max_seq_len);
    case 32:
      return grp32_fp16::GetScratchSize(device_prop, batch_size, kv_num_heads, max_seq_len);
    default:
      return 0;  // Not supported
  }
}

// Instantiate template for the dispatcher
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
    const int actual_seq_len,
    const int max_seq_len,
    const float scale,
    const bool is_bsnh,
    const int* seq_lens,
    const float* kv_cache_scale,
    const int kv_quant_type,
    void* workspace,
    size_t workspace_size);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
