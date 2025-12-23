// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "contrib_ops/cuda/bert/flash_attention/flash_fwd_launch_template.h"

namespace onnxruntime {
namespace flash {

// Explicitly instantiate the dequant dispatcher for BF16, HeadDim=128, QuantBits=4
template void run_mha_fwd_dequant_dispatch<cutlass::bfloat16_t, 128, 4>(Flash_fwd_params& params, cudaStream_t stream);

}  // namespace flash
}  // namespace onnxruntime
