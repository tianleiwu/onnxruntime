/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 ******************************************************************************/

#pragma once

#include <memory>
#include <utility>
#include <vector>
#include <algorithm>
#include <map>
#include <set>

#include "core/common/status.h"
#include "contrib_ops/cuda/bert/flash_attention/utils.h"

#define ALLOW_FLASH_FWD_QINT_DYNAMIC 0

#include "contrib_ops/cuda/bert/flash_attention/namespace_config.h"
#include "contrib_ops/cuda/bert/flash_attention/static_switch.h"
#include "contrib_ops/cuda/bert/flash_attention/flash.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_kernel.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_qint4_kernel.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_qint8_kernel.h"
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_qint8_decode_kernel.h"
#if ALLOW_FLASH_FWD_QINT_DYNAMIC
#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_qint8_dynamic_kernel.h"
#endif
#include "core/providers/cuda/shared_inc/cuda_call.h"
#include <stdio.h>

namespace FLASH_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#define ARCH_SUPPORTS_FLASH
#define KERNEL_PARAM_MODIFIER __grid_constant__
#else
#define KERNEL_PARAM_MODIFIER
#endif

// Define a macro for unsupported architecture handling to centralize the error message
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");

// Use a macro to clean up kernel definitions
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
  template <typename Kernel_traits, __VA_ARGS__>     \
  __global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_int4_dequant_kernel, bool Is_causal, bool Is_local, bool Has_alibi,
                            bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV,
                            int QUANT_TYPE) {
#if defined(ARCH_SUPPORTS_FLASH)
  static_assert(!(Is_causal && Is_local));  // Enforce constraints
  // Grid is dim3(num_m_block, params.b, params.h) => (m_block, batch, head)
  // So: blockIdx.x = m_block, blockIdx.y = batch (bidb), blockIdx.z = head (bidh)
  // With Split-K:
  // Grid is dim3(num_m_block, num_splits > 1 ? num_splits : params.b, num_splits > 1 ? params.b * params.h : params.h)
  // If Split: blockIdx.y = split_idx, blockIdx.z = batch * head + head_idx (collapsed)

  const int n_split_idx = Split ? blockIdx.y : 0;
  const int num_n_splits = Split ? gridDim.y : 1;
  const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
  const int bidh = Split ? blockIdx.z % params.h : blockIdx.z;

  flash::int4::compute_attn_1rowblock<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV, QUANT_TYPE>(
      params, bidb, bidh, blockIdx.x, n_split_idx, num_n_splits);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_int8_dequant_kernel, bool Is_causal, bool Is_local, bool Has_alibi,
                            bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV) {
#if defined(ARCH_SUPPORTS_FLASH)
  const int n_split_idx = Split ? blockIdx.y : 0;
  const int num_n_splits = Split ? gridDim.y : 1;
  const int bidb = Split ? blockIdx.z / params.h : blockIdx.y;
  const int bidh = Split ? blockIdx.z % params.h : blockIdx.z;

  flash::int8::compute_attn_1rowblock<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(
      params, bidb, bidh, blockIdx.x, n_split_idx, num_n_splits);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_int8_decode_kernel, bool Is_causal, bool Is_local, bool Has_alibi, bool Is_even_MN, bool Is_even_K, bool Is_softcap, bool Split, bool Append_KV) {
#if defined(ARCH_SUPPORTS_FLASH)
  assert(params.seqlen_q == 1 && params.h_h_k_ratio > 1);
  // Standard Flash Attention Grid Decoding
  // Grid: (m_block, splits/b, b*h/h)
  const int m_block = blockIdx.x;
  const int n_split_idx = Split ? blockIdx.y : 0;
  const int num_n_splits = Split ? gridDim.y : 1;
  const int num_kv_heads = params.h_k;
  const int bidb = Split ? blockIdx.z / num_kv_heads : blockIdx.y;
  const int kv_head_idx = Split ? blockIdx.z - bidb * num_kv_heads : blockIdx.z;
  flash::int8::compute_attn_1rowblock_gqa<Kernel_traits, Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV>(params, bidb, kv_head_idx, m_block, n_split_idx, num_n_splits);
#else
  FLASH_UNSUPPORTED_ARCH
#endif
}

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_splitkv_combine_kernel, int kBlockM, int Log_max_splits, bool Is_even_K) {
  static_assert(Log_max_splits >= 1);
  FLASH_NAMESPACE::combine_attn_seqk_parallel<Kernel_traits, kBlockM, Log_max_splits, Is_even_K>(params);
}

template <typename Kernel_traits, int QUANT_TYPE>
void run_flash_int4_dequant_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr size_t smem_size = sizeof(typename Kernel_traits::Element) *
                                   (Kernel_traits::kBlockM * Kernel_traits::kHeadDim +   // Q
                                    Kernel_traits::kBlockN * Kernel_traits::kHeadDim +   // K
                                    Kernel_traits::kBlockN * Kernel_traits::kHeadDim) +  // V
                               sizeof(typename Kernel_traits::ElementInt8) *
                                   (Kernel_traits::kBlockN * Kernel_traits::kHeadDim +                  // K_int8
                                    Kernel_traits::kBlockN * Kernel_traits::kHeadDim) +                 // V_int8
                               sizeof(typename Kernel_traits::Element) * 2 * Kernel_traits::kHeadDim +  // K_Scale + V_Scale
                               2048;                                                                    // Padding (1024 bytes) and extra buffer for safety (alignment etc.)

  const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);

  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                          params.seqlen_k % Kernel_traits::kBlockN == 0 &&
                          params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;

  QUANT_CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
      EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        SOFTCAP_SWITCH(params.softcap > 0.0, Is_softcap, [&] {
          BOOL_SWITCH(params.num_splits > 1, SplitConst, [&] {
            BOOL_SWITCH(params.knew_ptr != nullptr, Append_KV_Const, [&] {
              auto kernel = &flash_fwd_int4_dequant_kernel < Kernel_traits, Is_causal, false, false,
                   IsEvenMNConst && IsEvenKConst, IsEvenKConst, Is_softcap, SplitConst, Append_KV_Const, QUANT_TYPE > ;
              if (smem_size >= 48 * 1024) {
                cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
              }
              kernel<<<grid, Kernel_traits::kNThreads, static_cast<int>(smem_size), stream>>>(params);
            });
          });
        });
      });
    });
  });

  if (params.num_splits > 1) {
    constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
    dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);

    // Combine kernel requires kNThreads == 128.
    using CombineTraits = typename std::conditional<
        Kernel_traits::kNThreads == 128,
        Kernel_traits,
        flash::int8::Flash_dq_kernel_traits<Kernel_traits::kHeadDim, 64, 64, 4, typename Kernel_traits::Element>>::type;

    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
      int split_combine_num = params.num_splits;
      if (split_combine_num <= 2)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 1, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 4)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 2, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 8)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 3, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 16)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 4, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 32)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 5, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 64)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 6, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 128)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 7, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
    });
  }
}

template <typename Kernel_traits>
void run_flash_int8_decode_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  assert(params.seqlen_q == 1 && params.h_h_k_ratio > 1);

  // Specialized Decode Kernel Dispatch (Q=1)
  // Use the new optimized INT8 kernel (flash_int8_fwd_kernel.h)
  // This kernel uses standard grid (one block per Q-head), not GQA-aware blocking yet.

  using DecodeTraits = Kernel_traits;

  // Grid: (num_m_block, splits, b*h) or (num_m_block, splits, b*h_k) for GQA
  // For Q=1, num_m_block = 1.
  int num_heads_launch = (params.h_h_k_ratio > 1) ? params.h_k : params.h;
  dim3 grid_decode(1, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * num_heads_launch : num_heads_launch);

  // Calculate Smem Size (Physical size with padding)
  constexpr int kSmemSizeQ = DecodeTraits::kBlockM * DecodeTraits::kHeadDim * sizeof(typename DecodeTraits::Element);
  constexpr int kSmemSizeKInt8 = DecodeTraits::kBlockN * DecodeTraits::kSmemRowStrideInt8 * sizeof(typename DecodeTraits::ElementInt8);
  constexpr int kSmemSizeVInt8 = kSmemSizeKInt8;
  int smem_size_decode = kSmemSizeQ + kSmemSizeKInt8 + kSmemSizeVInt8;

  QUANT_CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
    BOOL_SWITCH(params.num_splits > 1, SplitConst, [&] {
      // Assume no alibi, standard local/softcap/etc for now within BOOL_SWITCH if needed.
      // For simplicity, hardcode some for decoding or match the existing BOOL_SWITCH structure.
      // run_flash_fwd_splitkv_kernel uses: Is_causal, Is_local, Has_alibi, Is_even_MN, Is_even_K, Is_softcap, Split, Append_KV

      bool Is_even_K = params.d == DecodeTraits::kHeadDim;
      bool Is_softcap = params.softcap > 0.0;
      // Is_local, Has_alibi, Append_KV unused. Is_even_MN = false.

      EVENK_SWITCH(Is_even_K, IsEvenKConst, [&] {
        SOFTCAP_SWITCH(Is_softcap, IsSoftcapConst, [&] {
          auto kernel = &flash_fwd_int8_decode_kernel<DecodeTraits, Is_causal, false, false, false, IsEvenKConst, IsSoftcapConst, SplitConst, false>;
          if (smem_size_decode >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size_decode);
          }
          kernel<<<grid_decode, DecodeTraits::kNThreads, smem_size_decode, stream>>>(params);
        });
      });
    });
  });

  // if (params.num_splits > 1) {
  //   constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
  //   dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);
  //   const bool is_even_K = params.d == DecodeTraits::kHeadDim;

  //   // Combine kernel requires kNThreads == 128. If DecodeTraits uses fewer threads (e.g. GQA optimization), switch traits.
  //   using CombineTraits = typename std::conditional<
  //       DecodeTraits::kNThreads == 128,
  //       DecodeTraits,
  //       flash::int8::Flash_dq_kernel_traits<DecodeTraits::kHeadDim, 64, 64, 4, typename DecodeTraits::Element>>::type;

  //   EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
  //     int split_combine_num = params.num_splits;
  //     if (split_combine_num <= 2)
  //       flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 1, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
  //     else if (split_combine_num <= 4)
  //       flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 2, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
  //     else if (split_combine_num <= 8)
  //       flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 3, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
  //     else if (split_combine_num <= 16)
  //       flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 4, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
  //     else if (split_combine_num <= 32)
  //       flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 5, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
  //     else if (split_combine_num <= 64)
  //       flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 6, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
  //     else if (split_combine_num <= 128)
  //       flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 7, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
  //   });
  // }
}

template <typename Kernel_traits>
void run_flash_int8_dequant_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  assert(params.knew_ptr == nullptr);
  // Smem layout: [sQ (FP16)] [sK (Int8, padded) x kNumStages] [sV (Int8, padded) x kNumStages]
  // Note: INT8 K/V buffers use kSmemRowStrideInt8 (144 for HeadDim=128) instead of kHeadDim (128)
  // to ensure 128-bit alignment for cp.async and avoid bank conflicts.
  // kNumStages=2 for double-buffering: allows overlapping memory loads with compute.
  constexpr size_t smem_size = sizeof(typename Kernel_traits::Element) *
                                   (Kernel_traits::kBlockM * Kernel_traits::kHeadDim) +  // Q (FP16, no padding)
                               sizeof(typename Kernel_traits::ElementInt8) *
                                   Kernel_traits::kNumStages *                                    // Double-buffering factor
                                   (Kernel_traits::kBlockN * Kernel_traits::kSmemRowStrideInt8 +  // K_int8 (padded stride)
                                    Kernel_traits::kBlockN * Kernel_traits::kSmemRowStrideInt8);  // V_int8 (padded stride)

  const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);

  // const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
  //                         params.seqlen_k % Kernel_traits::kBlockN == 0 &&
  //                         params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;
  const bool Is_softcap = params.softcap > 0.0;

  QUANT_CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
    BOOL_SWITCH(params.num_splits > 1, SplitConst, [&] {  // TODO Experiment: Disable Split to reduce compile time and see performance impact.
      EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        SOFTCAP_SWITCH(Is_softcap, IsSoftcapConst, [&] {
          // INT8 prefill usually doesn't need split-k (saturates GPU with batch/heads)
          // Softcap support in INT8 kernel usually not critical for now.
          constexpr bool Is_local = false;
          constexpr bool Is_even_MN = false;
          constexpr bool Append_KV_Const = false;
          constexpr bool Has_alibi = false;
          auto kernel = &flash_fwd_int8_dequant_kernel<Kernel_traits, Is_causal, Is_local, Has_alibi,
                                                       Is_even_MN, IsEvenKConst, IsSoftcapConst, SplitConst, Append_KV_Const>;
          if (smem_size >= 48 * 1024) {
            cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
          }
          kernel<<<grid, Kernel_traits::kNThreads, static_cast<int>(smem_size), stream>>>(params);
        });
      });
    });
  });

  if (params.num_splits > 1) {
    constexpr static int kBlockM = Kernel_traits::kHeadDim % 128 == 0 ? 4 : (Kernel_traits::kHeadDim % 64 == 0 ? 8 : 16);
    dim3 grid_combine((params.b * params.h * params.seqlen_q + kBlockM - 1) / kBlockM);

    // Combine kernel requires kNThreads == 128.
    using CombineTraits = typename std::conditional<
        Kernel_traits::kNThreads == 128,
        Kernel_traits,
        flash::int8::Flash_dq_kernel_traits<Kernel_traits::kHeadDim, 64, 64, 4, typename Kernel_traits::Element>>::type;

    EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
      int split_combine_num = params.num_splits;
      if (split_combine_num <= 2)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 1, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 4)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 2, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 8)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 3, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 16)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 4, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 32)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 5, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 64)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 6, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
      else if (split_combine_num <= 128)
        flash_fwd_splitkv_combine_kernel<CombineTraits, kBlockM, 7, IsEvenKConst><<<grid_combine, CombineTraits::kNThreads, 0, stream>>>(params);
    });
  }
}

#if ALLOW_FLASH_FWD_QINT_DYNAMIC

template <typename Kernel_traits>
void run_flash_int8_quant_fwd(Flash_fwd_params& params, cudaStream_t stream) {
  constexpr size_t smem_size = sizeof(typename Kernel_traits::Element) *
                                   (Kernel_traits::kBlockM * Kernel_traits::kHeadDim) +  // Q (FP16)
                               sizeof(typename Kernel_traits::ElementInt8) *
                                   (Kernel_traits::kBlockN * Kernel_traits::kSmemRowStrideInt8 +                             // K_int8
                                    Kernel_traits::kBlockN * Kernel_traits::kSmemRowStrideInt8) +                            // V_int8
                               sizeof(typename Kernel_traits::Element) * (Kernel_traits::kBlockM * Kernel_traits::kBlockN);  // P (FP16/BF16) reshuffle buffer

  const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
  dim3 grid(num_m_block, params.num_splits > 1 ? params.num_splits : params.b, params.num_splits > 1 ? params.b * params.h : params.h);

  const bool is_even_MN = params.cu_seqlens_q == nullptr && params.cu_seqlens_k == nullptr &&
                          params.seqlen_k % Kernel_traits::kBlockN == 0 &&
                          params.seqlen_q % Kernel_traits::kBlockM == 0;
  const bool is_even_K = params.d == Kernel_traits::kHeadDim;

  // Use BOOL_SWITCH for runtime parameters
  // Is_causal, Is_even_MN, Is_even_K
  QUANT_CAUSAL_SWITCH(params.is_causal, Is_causal, [&] {
    BOOL_SWITCH(is_even_MN, IsEvenMNConst, [&] {
      EVENK_SWITCH(is_even_K, IsEvenKConst, [&] {
        // No softcap or split kernels for now in quantization kernel
        auto kernel = &flash_fwd_int8_quant_kernel<Kernel_traits, Is_causal, false, false, IsEvenMNConst, IsEvenKConst, false, false, false, Flash_fwd_params>;
        if (smem_size >= 48 * 1024) {
          cudaFuncSetAttribute(kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem_size));
        }
        kernel<<<grid, Kernel_traits::kNThreads, static_cast<int>(smem_size), stream>>>(params);
      });
    });
  });
}

#endif
// Global Kernel Wrapper for Int8 Quant

template <typename T, int Headdim, int kQuantBits>
void run_mha_fwd_dequant_dispatch(Flash_fwd_params& params, cudaStream_t stream) {
  static_assert(Headdim == 128, "Dequant kernel currently only supports HeadDim=128");
  constexpr int kBlockM = 64;
  constexpr int kBlockN = 64;
  constexpr int kNWarps = 4;

  // if (params.kv_cache_bit_width == 8) {
  //   if (params.seqlen_q == 1) printf("DEBUG Dispatch: seqlen_q == 1 detected\n");
  //   printf("DEBUG Dispatch: seqlen_q=%d h_h_k_ratio=%d k_quant_type=%d\n", params.seqlen_q, params.h_h_k_ratio, params.k_quant_type);
  // }

  if constexpr (kQuantBits == 8) {
    if (params.kv_cache_bit_width == 8) {
#if ALLOW_FLASH_FWD_QINT_DYNAMIC
      if (params.query_dynamic_quant) {
        // Use new Int8 MMA kernel
        run_flash_int8_quant_fwd<flash::int8_mma::Flash_int8_quant_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps, T>>(params, stream);
        return;
      }
#endif

      if (params.k_quant_type != 0) {
        // TODO: review: whether we need params.h_h_k_ratio > 1 in the following check.
        if (params.seqlen_q == 1 && params.h_h_k_ratio > 1) {
          // Specialized GQA path (Optimization Round 2)
          // M=16 reduces compute waste for small query groups (e.g. 4 heads).
          // N=128 improves memory bandwidth usage.
          // Warps=1 works with N=128 and M=16.
          // Note: Ensure sufficient splits are used to saturate GPU.
          run_flash_int8_decode_fwd<flash::int8::Flash_dq_kernel_traits<Headdim, 16, 128, 1, T>>(params, stream);
        } else {
          run_flash_int8_dequant_fwd<flash::int8::Flash_dq_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps, T>>(params, stream);
        }
      }
    }
  }

  if constexpr (kEnableFlashAttention4Bit) {
    if (params.kv_cache_bit_width == 4) {
      // if (params.k_quant_type == 1) {
      //   run_flash_int4_dequant_fwd<flash::int4::Flash_dq_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps, T>, 1>(params, stream);
      // } else
      if (params.k_quant_type == 2) {
        run_flash_int4_dequant_fwd<flash::int4::Flash_dq_kernel_traits<Headdim, kBlockM, kBlockN, kNWarps, T>, 2>(params, stream);
      }
    }
  }
}

}  // namespace FLASH_NAMESPACE
