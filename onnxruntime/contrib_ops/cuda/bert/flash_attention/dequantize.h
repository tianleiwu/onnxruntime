#pragma once

#include "contrib_ops/cuda/bert/flash_attention/utils.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

// Set to 1 to enable debug prints for this kernel
#define DEQUANT_DEBUG 0

namespace onnxruntime {
namespace flash {

using namespace cute;

template <
    bool Is_even_MN = true, bool Is_even_K = true, int QUANT_TYPE = 0, int BIT_WIDTH = 0,
    typename TiledCopy, typename SrcTensor, typename DstTensor,
    typename CoordTensor, typename PredTensor, typename ScaleType>
__noinline__ __device__ void copy_and_dequantize(
    TiledCopy const& tiled_copy,
    SrcTensor const& gmem_src,       // Thread-local view of quantized source tensor in gmem
    DstTensor& smem_dst,             // Thread-local view of dequantized destination tensor in smem
    CoordTensor const& identity_MN,  // Thread-local view of an identity tensor for global coordinates
    PredTensor const& predicate_K,   // Predicate for the K dimension
    const int max_MN,
    const ScaleType* scale,
    const int d,
    const int h_k_idx) {
#if DEQUANT_DEBUG
  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0) {
    printf("[DEQUANT_DEBUG] Enter copy_and_dequantize\n");
    printf("  gmem_ptr: %p, smem_ptr: %p, scale_ptr: %p\n", raw_pointer_cast(gmem_src.data()), raw_pointer_cast(smem_dst.data()), scale);
    printf("  d: %d, h_k_idx: %d, max_MN: %d, BIT_WIDTH: %d, QUANT_TYPE: %d\n", d, h_k_idx, max_MN, BIT_WIDTH, QUANT_TYPE);
  }
#endif

  using DQuantType = typename DstTensor::value_type;  // The dequantized type (e.g., half)
  using QType = typename SrcTensor::value_type;       // The quantized type (e.g., int8_t or uint8_t for int4)

#pragma unroll
  for (int m = 0; m < size<1>(gmem_src); ++m) {
    if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
#pragma unroll
      for (int k = 0; k < size<2>(gmem_src); ++k) {
        if (Is_even_K || predicate_K(k)) {
          auto gmem_slice = gmem_src(_, m, k);
          auto smem_slice = smem_dst(_, m, k);
          auto identity_slice = identity_MN(_, m, k);

          //
          // Step 1 & 2: Load raw quantized data from GMEM to registers.
          // The provided tiled_copy is for DQuantType. To load QType data, we create
          // a register tensor (tRrQ_raw) with a layout that matches the byte footprint
          // of the source but is typed to DQuantType for the copy operation.
          //
          // auto raw_layout = recast_layout<QType, DQuantType>(make_layout(shape(gmem_slice)));
          Tensor tRrQ_raw = make_tensor<QType>(shape(gmem_slice));
          copy(tiled_copy, gmem_slice, tRrQ_raw);

          copy(tiled_copy, gmem_slice, tRrQ_raw);

          //
          // Step 3: Dequantize the data now residing in registers.
          //
          Tensor tRsK = make_tensor<DQuantType>(shape(smem_slice));

          if constexpr (BIT_WIDTH == 8) {
            auto const* tRrQ_quant = reinterpret_cast<int8_t const*>(&tRrQ_raw(0));

#pragma unroll 1
            for (int i = 0; i < size(tRsK); ++i) {
              float val = static_cast<float>(tRrQ_quant[i]);
              float current_scale = 1.0f;
              int coord_k = get<1>(identity_slice(i));

              if constexpr (QUANT_TYPE == 1) {  // PER_TENSOR
                current_scale = static_cast<float>(scale[0]);
              } else if constexpr (QUANT_TYPE == 2) {  // PER_CHANNEL
                if (coord_k < d) {
                  int scale_idx = h_k_idx * d + coord_k;
#if DEQUANT_DEBUG
                  if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && m == 0 && k == 0 && i == 0) {
                    printf("[FlashDequant-8bit] Thread=%d CoordK=%d ScaleIdx=%d Raw=%f Scale=%f Result=%f\n",
                           threadIdx.x, coord_k, scale_idx, val, current_scale, val * current_scale);
                  }
#endif
                  current_scale = static_cast<float>(scale[scale_idx]);
                }
              }
              tRsK(i) = static_cast<DQuantType>(val * current_scale);
            }
          } else if constexpr (BIT_WIDTH == 4) {
            auto const* tRrQ_packed = reinterpret_cast<uint8_t const*>(&tRrQ_raw(0));

#pragma unroll 1
            for (int i = 0; i < size(tRsK); ++i) {
              uint8_t packed_val = tRrQ_packed[i / 2];
              int8_t unpacked_val = (i % 2 == 0)
                                        ? static_cast<int8_t>((packed_val & 0x0F) - 8)
                                        : static_cast<int8_t>((packed_val >> 4) - 8);

              float val = static_cast<float>(unpacked_val);
              float current_scale = 1.0f;
              int coord_k = get<1>(identity_slice(i));

              if constexpr (QUANT_TYPE == 1) {  // PER_TENSOR
                current_scale = static_cast<float>(scale[0]);
              } else if constexpr (QUANT_TYPE == 2) {  // PER_CHANNEL
                if (coord_k < d) {
                  int scale_idx = h_k_idx * d + coord_k;
#if DEQUANT_DEBUG
                  if (threadIdx.x == 0 && blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && m == 0 && k == 0 && i < 4) {
                    printf("  [DEBUG 4b] i=%d, coord_k=%d, scale_idx=%d, packed_val=0x%x\n", i, coord_k, scale_idx, (unsigned int)packed_val);
                  }
#endif
                  current_scale = static_cast<float>(scale[scale_idx]);
                }
              }
              tRsK(i) = static_cast<DQuantType>(val * current_scale);
            }
          }

          //
          // Step 4: Perform an efficient copy from registers to shared memory.
          //
          copy(tRsK, smem_slice);

        } else {
          clear(smem_dst(_, m, k));
        }
      }
    } else {
      clear(smem_dst(_, m, _));
    }
  }
}

}  // namespace flash
}  // namespace onnxruntime
