/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuda_runtime.h>
#include <stdint.h>
#include <cstddef>
#include <vector>

namespace onnxruntime::llm {
namespace kernels {
namespace weight_only {

enum class QuantType {
  W8_A16,
  W4_A16,
  W4_AFP8
};

int get_arch_for_mixed_gemm_weight_preprocess(int arch);

// ``apply_bias_interleave`` controls the final ``add_bias_and_interleave`` step, which is
// specific to integer weights: it adds the int4/int8 zero-point bias and pair-interleaves the
// elements into the register order expected by ``FastInterleavedAndBiasedNumericArrayConverter``.
// MXFP4 (e2m1) weights are floating-point codes whose bit pattern would be corrupted by the
// integer bias add, and they are decoded with a linear (non-interleaved) converter, so callers
// preprocessing e2m1 weights must pass ``apply_bias_interleave=false`` to obtain only the
// row-permutation + subbyte-transpose + column-interleave (steps 1-3) layout.
void preprocess_weights_for_mixed_gemm_cuda(cudaStream_t stream,
                                            int arch,
                                            int8_t* preprocessed_quantized_weight,
                                            int8_t* row_major_quantized_weight,
                                            int32_t* d_permutation_map,
                                            std::vector<size_t> const& shape,
                                            QuantType quant_type,
                                            bool synchronize = true,
                                            bool apply_bias_interleave = true);

}  // namespace weight_only
}  // namespace kernels
}  // namespace onnxruntime::llm
