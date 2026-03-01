// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "cuda_plugin_kernels.h"
#include "cuda_stream_plugin.h"
#include "cuda_kernel_adapter.h"
#include "core/common/narrow.h"
#include "core/providers/cuda/activation/activations.h"
#include "core/providers/cuda/math/binary_elementwise_ops.h"
#include "core/providers/cuda/math/clip.h"
#include "core/providers/cuda/math/softmax.h"
#include "core/providers/cuda/math/unary_elementwise_ops.h"
#include "core/providers/cuda/reduction/reduction_ops.h"
#include "core/providers/cuda/tensor/concat.h"
#include "core/providers/cuda/tensor/cast_op.h"
#include "core/providers/cuda/tensor/gather.h"
#include "core/providers/cuda/tensor/split.h"
#include "core/providers/cuda/tensor/where.h"
#include "contrib_ops/cuda/bert/decoder_masked_multihead_attention.h"
#include "contrib_ops/cuda/bert/embed_layer_norm.h"
#include "contrib_ops/cuda/bert/fast_gelu.h"
#include "contrib_ops/cuda/bert/gemma_rotary_emb.h"
#include "contrib_ops/cuda/bert/group_query_attention.h"
#include "contrib_ops/cuda/bert/multihead_attention.h"
#include "contrib_ops/cuda/bert/rotary_embedding.h"
#include "contrib_ops/cuda/bert/skip_layer_norm.h"
#include "contrib_ops/cuda/bert/attention.h"
#include "contrib_ops/cuda/moe/moe.h"
#include "contrib_ops/cuda/quantization/gather_block_quantized.h"
#include "contrib_ops/cuda/quantization/matmul_nbits.h"
#include "contrib_ops/cuda/quantization/moe_quantization.h"

#include <cstring>
#include <map>
#include <set>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace onnxruntime {
namespace cuda_plugin {

OrtStatus* CreateCudaKernelRegistry(const OrtEpApi& ep_api,
                                    const char* ep_name,
                                    void* create_kernel_state,
                                    OrtKernelRegistry** out_registry) {
  return CreateCudaKernelRegistryFromOrtTables(ep_api, ep_name, create_kernel_state, out_registry);
}

}  // namespace cuda_plugin
}  // namespace onnxruntime
