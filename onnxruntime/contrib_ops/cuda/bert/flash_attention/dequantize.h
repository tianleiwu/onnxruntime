#pragma once

#include "contrib_ops/cuda/bert/flash_attention/utils.h"
#include <cute/tensor.hpp>
#include <cutlass/cutlass.h>
#include <cutlass/numeric_types.h>

namespace onnxruntime {
namespace flash {

using namespace cute;

template <
    bool Is_even_MN = true, bool Is_even_K = true, int QUANT_TYPE = 0, int BIT_WIDTH = 0,
    typename TiledCopy, typename SrcTensor, typename DstTensor,
    typename CoordTensor, typename PredTensor, typename ScaleType>
__forceinline__ __device__ void copy_and_dequantize(
    TiledCopy const& tiled_copy,
    SrcTensor const& gmem_src,        // Thread-local view of quantized source tensor in gmem
    DstTensor& smem_dst,              // Thread-local view of dequantized destination tensor in smem
    CoordTensor const& identity_MN,   // Thread-local view of an identity tensor for global coordinates
    PredTensor const& predicate_K,    // Predicate for the K dimension
    const int max_MN,
    const ScaleType* scale,
    const int d,
    const int h_k_idx) {

    using DQuantType = typename DstTensor::value_type; // The dequantized type (e.g., half)
    using QType = typename SrcTensor::value_type;      // The quantized type (e.g., int8_t or uint8_t for int4)

    // This function mimics the structure of cute::copy but inserts a dequantization step.
    // The outer loops iterate over the "fragments" or "sub-tiles" that this thread is responsible for.
    #pragma unroll
    for (int m = 0; m < size<1>(gmem_src); ++m) {
        if (Is_even_MN || get<0>(identity_MN(0, m, 0)) < max_MN) {
            #pragma unroll
            for (int k = 0; k < size<2>(gmem_src); ++k) {
                if (Is_even_K || predicate_K(k)) {
                    // Sliced views for the current vector copy operation.
                    auto gmem_slice = gmem_src(_, m, k);
                    auto smem_slice = smem_dst(_, m, k);
                    auto identity_slice = identity_MN(_, m, k);

                    //
                    // Step 1: Create a register tensor to hold the raw bytes from gmem.
                    // The TiledCopy is configured for DQuantType, so we must load into a tensor of that type.
                    //
                    Tensor tRrQ_raw = make_tensor<DQuantType>(shape_div<sizeof(DQuantType)>(shape(gmem_slice)));

                    //
                    // Step 2: Perform an efficient, coalesced copy from global memory into registers.
                    // This loads the raw quantized bytes, interpreting them as the raw bits of DQuantType.
                    // The shape of tRrQ_raw is adjusted to ensure the byte counts match.
                    //
                    copy(tiled_copy, gmem_slice, tRrQ_raw);

                    //
                    // Step 3: Dequantize the data now residing in registers. This is fast.
                    //
                    Tensor tRsK = make_tensor<DQuantType>(shape(smem_slice));

                    if constexpr (BIT_WIDTH == 8) {
                        // Recast the raw data pointer in registers to the actual quantized type.
                        auto const* tRrQ_quant = reinterpret_cast<int8_t const*>(raw_pointer_cast(tRrQ_raw.data()));
                        #pragma unroll
                        for (int i = 0; i < size(tRsK); ++i) {
                            float val = static_cast<float>(tRrQ_quant[i]);
                            float current_scale = 1.0f;
                            int coord_k = get<1>(identity_slice(i)); // Get column coordinate for PER_CHANNEL scaling

                            if constexpr (QUANT_TYPE == 1) { // PER_TENSOR
                                current_scale = static_cast<float>(scale[0]);
                            } else if constexpr (QUANT_TYPE == 2) { // PER_CHANNEL
                                if (coord_k < d) { // Safety check for predicated-off elements
                                    current_scale = static_cast<float>(scale[h_k_idx * d + coord_k]);
                                }
                            }
                            tRsK(i) = static_cast<DQuantType>(val * current_scale);
                        }
                    } else if constexpr (BIT_WIDTH == 4) {
                        // Recast the register tensor pointer to access packed uint8_t data.
                        auto const* tRrQ_packed = reinterpret_cast<uint8_t const*>(raw_pointer_cast(tRrQ_raw.data()));

                        #pragma unroll
                        for (int i = 0; i < size(tRsK); ++i) {
                            // Each byte in tRrQ_packed contains two 4-bit values.
                            uint8_t packed_val = tRrQ_packed[i / 2];
                            // Unpack the correct 4-bit value based on whether i is even or odd.
                            int8_t unpacked_val = (i % 2 == 0)
                                                    ? static_cast<int8_t>((packed_val & 0x0F) - 8)
                                                    : static_cast<int8_t>((packed_val >> 4) - 8);

                            float val = static_cast<float>(unpacked_val);
                            float current_scale = 1.0f;
                            int coord_k = get<1>(identity_slice(i)); // Get column coordinate

                            if constexpr (QUANT_TYPE == 1) { // PER_TENSOR
                                current_scale = static_cast<float>(scale[0]);
                            } else if constexpr (QUANT_TYPE == 2) { // PER_CHANNEL
                                if (coord_k < d) { // Safety check for predicated-off elements
                                    current_scale = static_cast<float>(scale[h_k_idx * d + coord_k]);
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
                    // If predicate is false for this vector, clear the destination in smem.
                    clear(smem_dst(_, m, k));
                }
            }
        } else {
            // If the m-predicate is false, clear all corresponding k-tiles.
            clear(smem_dst(_, m, _));
        }
    }
}

}  // namespace flash
}  // namespace onnxruntime
