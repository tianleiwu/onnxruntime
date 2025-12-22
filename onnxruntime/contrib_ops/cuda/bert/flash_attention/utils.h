/******************************************************************************
 * Copyright (c) 2023, Tri Dao.
 ******************************************************************************/
#pragma once

#include <assert.h>
#include <stdint.h>
#include <stdlib.h>

#include <cuda_fp16.h>

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#include <cuda_bf16.h>
#endif

#include <cute/tensor.hpp>

#include <cutlass/array.h>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_conversion.h>
#include <cutlass/numeric_types.h>

////////////////////////////////////////////////////////////////////////////////////////////////////
namespace onnxruntime {
namespace flash {

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
__forceinline__ __device__ uint32_t relu2(const uint32_t x);

template <>
__forceinline__ __device__ uint32_t relu2<cutlass::half_t>(const uint32_t x) {
  uint32_t res;
  const uint32_t zero = 0u;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  asm volatile("max.f16x2 %0, %1, %2;\n"
               : "=r"(res)
               : "r"(x), "r"(zero));
#else
  asm volatile(
      "{\n"
      "\t .reg .f16x2 sela;\n"
      "\t set.gtu.u32.f16x2 sela, %1, %2;\n"
      "\t and.b32 %0, sela, %1;\n"
      "}\n"
      : "=r"(res)
      : "r"(x), "r"(zero));
#endif
  return res;
}

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
template <>
__forceinline__ __device__ uint32_t relu2<cutlass::bfloat16_t>(const uint32_t x) {
  uint32_t res;
  const uint32_t zero = 0u;
  asm volatile("max.bf16x2 %0, %1, %2;\n"
               : "=r"(res)
               : "r"(x), "r"(zero));
  return res;
}
#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800

template <typename T>
__forceinline__ __device__ uint32_t convert_relu2(const float2 x);

template <>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::half_t>(const float2 x) {
  uint32_t res;
  const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
  const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
  asm volatile("cvt.rn.relu.f16x2.f32 %0, %1, %2;\n"
               : "=r"(res)
               : "r"(b), "r"(a));
  return res;
}

template <>
__forceinline__ __device__ uint32_t convert_relu2<cutlass::bfloat16_t>(const float2 x) {
  uint32_t res;
  const uint32_t a = reinterpret_cast<const uint32_t&>(x.x);
  const uint32_t b = reinterpret_cast<const uint32_t&>(x.y);
  asm volatile("cvt.rn.relu.bf16x2.f32 %0, %1, %2;\n"
               : "=r"(res)
               : "r"(b), "r"(a));
  return res;
}

#endif

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct MaxOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) { return x > y ? x : y; }
};

template <>
struct MaxOp<float> {
  // This is slightly faster
  __device__ __forceinline__ float operator()(float const& x, float const& y) { return max(x, y); }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T>
struct SumOp {
  __device__ __forceinline__ T operator()(T const& x, T const& y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS>
struct Allreduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return Allreduce<OFFSET>::run(x, op);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Allreduce<2> {
  template <typename T, typename Operator>
  static __device__ __forceinline__ T run(T x, Operator& op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool A_in_regs = false, bool B_in_regs = false, typename Tensor0, typename Tensor1,
          typename Tensor2, typename Tensor3, typename Tensor4,
          typename TiledMma, typename TiledCopyA, typename TiledCopyB,
          typename ThrCopyA, typename ThrCopyB>
__forceinline__ __device__ void gemm(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsA,
                                     Tensor4 const& tCsB, TiledMma tiled_mma,
                                     TiledCopyA smem_tiled_copy_A, TiledCopyB smem_tiled_copy_B,
                                     ThrCopyA smem_thr_copy_A, ThrCopyB smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K
  Tensor tCrA_copy_view = smem_thr_copy_A.retile_D(tCrA);
  CUTE_STATIC_ASSERT_V(size<1>(tCsA) == size<1>(tCrA_copy_view));  // M
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));  // N
  if (!A_in_regs) {
    cute::copy(smem_tiled_copy_A, tCsA(_, _, _0{}), tCrA_copy_view(_, _, _0{}));
  }
  if (!B_in_regs) {
    cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
  }
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      if (!A_in_regs) {
        cute::copy(smem_tiled_copy_A, tCsA(_, _, i + 1), tCrA_copy_view(_, _, i + 1));
      }
      if (!B_in_regs) {
        cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
      }
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy>
__forceinline__ __device__ void gemm_rs(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsB,
                                        TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                        ThrCopy smem_thr_copy_B) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K
  Tensor tCrB_copy_view = smem_thr_copy_B.retile_D(tCrB);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_copy_view));  // N
  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_copy_view(_, _, _0{}));
#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_copy_view(_, _, i + 1));
    }
    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

// Original gemm_quant for legacy kernels
template <typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy, typename ScaleType>
__forceinline__ __device__ void gemm_quant(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsB,
                                           TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                           ThrCopy smem_thr_copy_B, ScaleType scale) {
  CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K

  // Create register tensor for quantized data
  using ElementQuant = typename TiledCopy::ValType;
  Tensor tCrB_quant = make_tensor<ElementQuant>(layout(tCrB));

  Tensor tCrB_quant_copy_view = smem_thr_copy_B.retile_D(tCrB_quant);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_quant_copy_view));  // N

  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_quant_copy_view(_, _, _0{}));

  using DQuantType = typename Tensor2::value_type;

#pragma unroll
  for (int i = 0; i < size<2>(tCrA); ++i) {
    if (i < size<2>(tCrA) - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, i + 1), tCrB_quant_copy_view(_, _, i + 1));
    }

    Tensor tCrB_quant_frag = tCrB_quant(_, _, i);
    Tensor tCrB_frag = tCrB(_, _, i);
#pragma unroll
    for (int j = 0; j < size(tCrB_frag); ++j) {
      tCrB_frag(j) = static_cast<DQuantType>(static_cast<float>(tCrB_quant_frag(j)) * scale);
    }

    cute::gemm(tiled_mma, tCrA(_, _, i), tCrB(_, _, i), acc);
  }
}

// gemm_quant_manual: A variant of gemm_quant designed for the dequantization kernel.
//
// Why this variant is needed:
// ---------------------------
// The standard `gemm_quant` uses `cute::gemm(tiled_mma, ...)` which enforces strict
// static assertions on tensor shapes:
//   - size<1>(tCrB) == size<2>(acc)  (MMA_N match)
//   - size<2>(tCrA) == size<2>(tCrB) (MMA_K match)
//
// In the dequantization kernel (flash_dq_fwd_kernel.h), we use a TiledMma with N=8
// (matching the MMA Atom's native N dimension) while using dummy FP16 tensors for
// partitioning Int8 data. This creates a mismatch where:
//   - tCrB (B operand) has 8 N-tiles (one per MMA atom)
//   - acc (accumulator) has a different tile structure
//
// This function bypasses cute::gemm's static assertions by:
// 1. Using a manual nested loop over M and N tiles
// 2. Directly instantiating the MMA Atom (TiledMma::Atom) instead of using TiledMma
// 3. Accessing accumulator slices with acc(_, m, n) which matches the Atom's output
//
// Optimizations included:
// - O3: Precompute scale as __half2 for vectorized multiply
// - O5: Process 2 logical K-tiles per quantized tile (Int8 has 2x elements packed)
// - Vectorized dequantization using __half2 intrinsics
//
template <bool Is_Int4, typename Tensor0, typename Tensor1, typename Tensor2, typename Tensor3,
          typename TiledMma, typename TiledCopy, typename ThrCopy, typename ScaleArg>
__forceinline__ __device__ void gemm_quant_manual(Tensor0& acc, Tensor1& tCrA, Tensor2& tCrB, Tensor3 const& tCsB,
                                                  TiledMma tiled_mma, TiledCopy smem_tiled_copy_B,
                                                  ThrCopy smem_thr_copy_B, ScaleArg scale) {
  // NOTE: Static assertions are intentionally disabled for this variant.
  // The tile shapes may not match cute::gemm's expectations due to the N=8 TiledMma
  // configuration used in the dequantization kernel.
  // CUTE_STATIC_ASSERT_V(size<1>(tCrA) == size<1>(acc));   // MMA_M
  // CUTE_STATIC_ASSERT_V(size<1>(tCrB) == size<2>(acc));   // MMA_N
  // CUTE_STATIC_ASSERT_V(size<2>(tCrA) == size<2>(tCrB));  // MMA_K

  // Create register tensor for quantized data
  using ElementQuant = typename TiledCopy::ValType;
  Tensor tCrB_quant = make_tensor<ElementQuant>(layout(tCrB));

  Tensor tCrB_quant_copy_view = smem_thr_copy_B.retile_D(tCrB_quant);
  CUTE_STATIC_ASSERT_V(size<1>(tCsB) == size<1>(tCrB_quant_copy_view));  // N

  cute::copy(smem_tiled_copy_B, tCsB(_, _, _0{}), tCrB_quant_copy_view(_, _, _0{}));

  // O3: Precompute scale as __half to avoid repeated float->half conversion
  using DQuantType = typename Tensor2::value_type;

  // Check if ScaleArg is a Tensor (Per-Channel) or Scalar (Per-Tensor)
  static constexpr bool Is_Scale_Tensor = cute::is_tensor<ScaleArg>::value;

  // For Scalar Scale (Per-Tensor)
  __half2 scale_h2_scalar;
  if constexpr (!Is_Scale_Tensor) {
    const __half scale_h = __float2half(static_cast<float>(scale));
    scale_h2_scalar = __half2half2(scale_h);
  }

  // O5: Process tiles
  constexpr int num_k_tiles = decltype(size<2>(tCrA))::value;

  // Helper to bypass cute::gemm static assertions on shape
  auto manual_gemm_helper = [&](auto const& tA_k, auto const& tB_k) {
    typename TiledMma::Atom atom_mma;
    CUTE_UNROLL
    for (int m = 0; m < size<1>(tA_k); ++m) {
      CUTE_UNROLL
      for (int n = 0; n < size<1>(tB_k); ++n) {
        cute::gemm(atom_mma, tA_k(_, m), tB_k(_, n), acc(_, m, n));
      }
    }
  };

  CUTE_UNROLL
  for (int qi = 0; qi < num_k_tiles; ++qi) {
    if (qi < num_k_tiles - 1) {
      cute::copy(smem_tiled_copy_B, tCsB(_, _, qi + 1), tCrB_quant_copy_view(_, _, qi + 1));
    }

    Tensor tCrB_quant_frag = tCrB_quant(_, _, qi);
    Tensor tCrB_frag = tCrB(_, _, qi);

    if constexpr (Is_Int4) {
      // Int4 Path (Duplicate Layout): Pairs of elements share a byte.
      // tCrB_quant_frag contains [b0, b0, b1, b1...]
      // Branchless unpacking + vectorized dequantization has perf regression so we use simple logic here
      CUTE_UNROLL
      for (int j = 0; j < size(tCrB_frag); j += 2) {
        // Read duplicated byte (take even index)
        auto packed_val = tCrB_quant_frag(j);
        uint8_t ub = static_cast<uint8_t>(packed_val);

        // Unpack 2's complement 4-bit values
        // v0: lower 4 bits (sign-extended)
        // v1: upper 4 bits (sign-extended)
        // Robust Int4 Unpacking (2s Complement)
        int8_t v0_raw = ub & 0xF;
        int8_t v0 = v0_raw >= 8 ? v0_raw - 16 : v0_raw;

        int8_t v1_raw = (ub >> 4) & 0xF;
        int8_t v1 = v1_raw >= 8 ? v1_raw - 16 : v1_raw;

        if constexpr (Is_Scale_Tensor) {
          tCrB_frag(j) = static_cast<DQuantType>(static_cast<float>(v0) * static_cast<float>(scale(j, 0, qi)));
          tCrB_frag(j + 1) = static_cast<DQuantType>(static_cast<float>(v1) * static_cast<float>(scale(j + 1, 0, qi)));
        } else {
          tCrB_frag(j) = static_cast<DQuantType>(static_cast<float>(v0) * static_cast<float>(scale));
          tCrB_frag(j + 1) = static_cast<DQuantType>(static_cast<float>(v1) * static_cast<float>(scale));
        }
      }
    } else {
      // Int8 Path: 1-to-1 mapping
      // Optimization: Vectorized dequantization using __half2 for 2x throughput
      if constexpr (std::is_same_v<DQuantType, cutlass::half_t>) {
        // Vectorized FP16 path for BOTH per-tensor and per-channel
        CUTE_UNROLL
        for (int j = 0; j < size(tCrB_frag); j += 2) {
          int8_t v0 = tCrB_quant_frag(j);
          int8_t v1 = tCrB_quant_frag(j + 1);

          __half2 vals_h2 = __halves2half2(__int2half_rn(v0), __int2half_rn(v1));
          __half2 result_h2;

          if constexpr (Is_Scale_Tensor) {
            // Per-channel: load 2 different scales
            __half2 scale_pair = __halves2half2(
                static_cast<__half>(scale(j, 0, qi)),
                static_cast<__half>(scale(j + 1, 0, qi)));
            result_h2 = __hmul2(vals_h2, scale_pair);
          } else {
            // Per-tensor: broadcast single scale
            __half scale_h = __float2half(static_cast<float>(scale));
            __half2 scale_h2 = __half2half2(scale_h);
            result_h2 = __hmul2(vals_h2, scale_h2);
          }

          tCrB_frag(j) = result_h2.x;
          tCrB_frag(j + 1) = result_h2.y;
        }
      } else {
        // Fallback for BF16 (scalar path)
        CUTE_UNROLL
        for (int j = 0; j < size(tCrB_frag); ++j) {
          auto val_q = tCrB_quant_frag(j);
          DQuantType s;
          if constexpr (Is_Scale_Tensor) {
            s = scale(j, 0, qi);
          } else {
            s = scale;
          }
          tCrB_frag(j) = static_cast<DQuantType>(static_cast<float>(val_q) * s);
        }
      }
    }

    manual_gemm_helper(tCrA(_, _, qi), tCrB_frag);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to (nrow=(2, MMA_M), ncol=(2, MMA_N))
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_rowcol(Layout acc_layout) {
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<_2>{});  // ((2, 2), MMA_M, MMA_N)
  return make_layout(make_layout(get<0, 1>(l), get<1>(l)), make_layout(get<0, 0>(l), get<2>(l)));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
// if using m16n8k16, or to (4, MMA_M, MMA_N) if using m16n8k8.
template <typename MMA_traits, typename Layout>
__forceinline__ __device__ auto convert_layout_acc_Aregs(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  constexpr int mma_shape_K = get<2>(typename MMA_traits::Shape_MNK{});
  static_assert(mma_shape_K == 8 || mma_shape_K == 16);
  if constexpr (mma_shape_K == 8) {
    return acc_layout;
  } else {
    auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
    return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

// Convert acc_layout from (MMA=4, MMA_M, MMA_N) to ((4, 2), MMA_M, MMA_N / 2)
template <typename Layout>
__forceinline__ __device__ auto convert_layout_acc_dropout(Layout acc_layout) {
  using X = Underscore;
  static_assert(decltype(size<0>(acc_layout))::value == 4);
  static_assert(decltype(rank(acc_layout))::value == 3);
  auto l = logical_divide(acc_layout, Shape<X, X, _2>{});  // (4, MMA_M, (2, MMA_N / 2)))
  return make_layout(make_layout(get<0>(l), get<2, 0>(l)), get<1>(l), get<2, 1>(l));
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  constexpr int numel = decltype(size(tensor))::value;
  cutlass::NumericArrayConverter<To_type, From_type, numel> convert_op;
  // HACK: this requires tensor to be "contiguous"
  auto frag = convert_op(*reinterpret_cast<const cutlass::Array<From_type, numel>*>(tensor.data()));
  return make_tensor(make_rmem_ptr<To_type>(&frag), tensor.layout());
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void relu_(Tensor<Engine, Layout>& tensor) {
  constexpr int numel = decltype(size(tensor))::value;
  static_assert(numel % 2 == 0);
  using value_t = typename Engine::value_type;
  // HACK: this requires tensor to be "contiguous"
  Tensor tensor_uint32 = recast<uint32_t>(tensor);
#pragma unroll
  for (int i = 0; i < size(tensor_uint32); ++i) {
    tensor_uint32(i) = relu2<value_t>(tensor_uint32(i));
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// On SM80 and above, we can fuse fp32 -> fp16/bf16 conversion and relu into 1 instruction
template <typename To_type, typename Engine, typename Layout>
__forceinline__ __device__ auto convert_type_relu(Tensor<Engine, Layout> const& tensor) {
  using From_type = typename Engine::value_type;
  static_assert(std::is_same_v<To_type, cutlass::half_t> || std::is_same_v<To_type, cutlass::bfloat16_t>);
  static_assert(std::is_same_v<float, From_type>);
  constexpr int numel = decltype(size(tensor))::value;
  static_assert(numel % 2 == 0);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // HACK: this requires tensor to be "contiguous"
  Tensor tensor_float2 = recast<float2>(tensor);
  Tensor out_uint32 = make_tensor<uint32_t>(tensor_float2.layout());
#pragma unroll
  for (int i = 0; i < size(out_uint32); ++i) {
    out_uint32(i) = convert_relu2<To_type>(tensor_float2(i));
  }
  Tensor out = make_tensor(make_rmem_ptr<To_type>(out_uint32.data()), tensor.layout());
#else
  Tensor out = flash::convert_type<To_type>(tensor);
  flash::relu_(out);
#endif
  return out;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// Blocks until all but N previous cp.async.commit_group operations have committed.
// This differs from cute::cp_async_wait in that when N = 0 we don't call cp.async.wait_all
// (which is equivalent to commit_group then wait_group 0).
// Instead we just call cp.async.wait_group 0, which is slightly faster.
// https://github.com/NVIDIA/cutlass/blob/master/include/cute/arch/copy_sm80.hpp#L113
template <int N>
CUTE_HOST_DEVICE void cp_async_wait() {
#if defined(CUTE_ARCH_CP_ASYNC_SM80_ENABLED)
  asm volatile("cp.async.wait_group %0;\n" ::"n"(N));
#endif
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_MN = true, bool Is_even_K = true, bool Clear_OOB_MN = false, bool Clear_OOB_K = true,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
__forceinline__ __device__ void copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const& S,
                                     Tensor<Engine1, Layout1>& D, Tensor<Engine2, Layout2> const& identity_MN,
                                     Tensor<Engine3, Layout3> const& predicate_K, const int max_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K
  // There's no case where !Clear_OOB_K && Clear_OOB_MN
  static_assert(!(Clear_OOB_MN && !Clear_OOB_K));
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || predicate_K(k)) {
          cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
        } else if (Clear_OOB_K) {
          cute::clear(D(_, m, k));
        }
      }
    } else if (Clear_OOB_MN) {
      cute::clear(D(_, m, _));
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <bool Is_even_K = true,
          typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2, typename Engine3, typename Layout3>
inline __device__ void copy_w_min_idx(Tensor<Engine0, Layout0> const& S,
                                      Tensor<Engine1, Layout1>& D, Tensor<Engine2, Layout2> const& identity_MN,
                                      Tensor<Engine3, Layout3> const& predicate_K,
                                      const int max_MN = 0, const int min_MN = 0) {
  CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
  CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
  CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));  // MMA
  CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));  // MMA_M
  CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));  // MMA_K
// if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, max_MN = %d, min_MN = %d\n", blockIdx.y, max_MN, min_MN); }
#pragma unroll
  for (int m = 0; m < size<1>(S); ++m) {
    // if (threadIdx.x == 0 && blockIdx.z == 0) { printf("blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
    if (get<0>(identity_MN(0, m, 0)) >= min_MN && get<0>(identity_MN(0, m, 0)) < max_MN) {
// if (threadIdx.x == 0 && blockIdx.z == 0) { printf("Inner loop, blockIdx.y = %d, m = %d\n", blockIdx.y, get<0>(identity_MN(0, m, 0))); }
#pragma unroll
      for (int k = 0; k < size<2>(S); ++k) {
        if (Is_even_K || predicate_K(k)) {
          cute::copy(S(_, m, k), D(_, m, k));
        }
      }
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename Engine, typename Layout>
__forceinline__ __device__ void apply_softcap(Tensor<Engine, Layout>& tensor, const float softcap) {
#pragma unroll
  for (int i = 0; i < size(tensor); ++i) {
    tensor(i) = cutlass::fast_tanh(tensor(i) * softcap);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace flash
}  // namespace onnxruntime
