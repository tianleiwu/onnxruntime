// Copyright (c) 2023, Tri Dao.
// Copyright (c) 2024, Microsoft.
// Native INT8 dequant kernel instantiation for HeadDim=128 (BF16)

#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace onnxruntime {
namespace flash {

// Explicitly instantiate the dequant dispatcher for BF16, HeadDim=128
template void run_mha_fwd_dequant_dispatch<cutlass::bfloat16_t, 128>(Flash_fwd_params& params, cudaStream_t stream);

}  // namespace flash
}  // namespace onnxruntime
