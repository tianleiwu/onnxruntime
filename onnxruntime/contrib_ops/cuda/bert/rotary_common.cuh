// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include "core/providers/cuda/cuda_common.h"

namespace onnxruntime {
namespace contrib {
namespace cuda {

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
        half s_val = reinterpret_cast<const half*>(sin_ptr)[cs_idx];

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
      // v1.x, v1.y
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


// Specialization for float4 (half)
template <>
struct RotaryDispatcher<float4, half> {
  __device__ static void apply(float4& val, const float4* cos_cache, const float4* sin_cache,
                               const int rotary_dim, const int h_idx, const int pos_id,
                               const bool interleaved, const float4* new_kv_base, const int64_t in_offset) {
    float2 p1 = make_float2(val.x, val.y);
    float2 p2 = make_float2(val.z, val.w);
    const float2* c = reinterpret_cast<const float2*>(cos_cache);
    const float2* s = reinterpret_cast<const float2*>(sin_cache);
    const float2* b = reinterpret_cast<const float2*>(new_kv_base);

    RotaryDispatcher<float2, half>::apply(p1, c, s, rotary_dim, h_idx * 2, pos_id, interleaved, b, in_offset * 2);
    RotaryDispatcher<float2, half>::apply(p2, c, s, rotary_dim, h_idx * 2 + 1, pos_id, interleaved, b, in_offset * 2);

    val.x = p1.x;
    val.y = p1.y;
    val.z = p2.x;
    val.w = p2.y;
  }
};

// Specialization for float4 (BFloat16)
template <>
struct RotaryDispatcher<float4, BFloat16> {
  __device__ static void apply(float4& val, const float4* cos_cache, const float4* sin_cache,
                               const int rotary_dim, const int h_idx, const int pos_id,
                               const bool interleaved, const float4* new_kv_base, const int64_t in_offset) {
    float2 p1 = make_float2(val.x, val.y);
    float2 p2 = make_float2(val.z, val.w);
    const float2* c = reinterpret_cast<const float2*>(cos_cache);
    const float2* s = reinterpret_cast<const float2*>(sin_cache);
    const float2* b = reinterpret_cast<const float2*>(new_kv_base);

    RotaryDispatcher<float2, BFloat16>::apply(p1, c, s, rotary_dim, h_idx * 2, pos_id, interleaved, b, in_offset * 2);
    RotaryDispatcher<float2, BFloat16>::apply(p2, c, s, rotary_dim, h_idx * 2 + 1, pos_id, interleaved, b, in_offset * 2);

    val.x = p1.x;
    val.y = p1.y;
    val.z = p2.x;
    val.w = p2.y;
  }
};

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime
