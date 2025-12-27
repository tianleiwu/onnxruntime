
#include "contrib_ops/cuda/moe/qmoe_kernels.h"
#include "core/providers/cuda/cuda_common.h"
#include "contrib_ops/cuda/llm/moe_gemm/moe_kernels.h"
#include <cub/cub.cuh>
#include <limits>

namespace onnxruntime {
namespace contrib {
namespace cuda {

template <typename T>
__global__ void SoftmaxTopKKernel(const T* logits, float* topk_scales, int* topk_indices,
                                  int num_rows, int num_experts, int k, bool normalize_scales) {
  int row = blockIdx.x * blockDim.x + threadIdx.x;
  if (row >= num_rows) return;

  const T* row_logits = logits + row * num_experts;
  float* row_scales = topk_scales + row * k;
  int* row_indices = topk_indices + row * k;

  // 1. Find max for numerical stability
  float max_val = -1e20f;
  for (int i = 0; i < num_experts; ++i) {
    float val = static_cast<float>(row_logits[i]);
    if (val > max_val) max_val = val;
  }

  // 2. Compute exp sum
  float sum_exp = 0.0f;
  for (int i = 0; i < num_experts; ++i) {
    sum_exp += expf(static_cast<float>(row_logits[i]) - max_val);
  }

  // 3. Compute Softmax and find TopK
  // For small k, we can do a simple selection.
  // Note: This is efficient only for small k and small num_experts.

  // We can compute softmax values on the fly or store them.
  // Given we need topK, let's just compute all softmax values then pick top K.
  // (Optimization: use a heap or similar if K is small and N is large)

  for (int i = 0; i < k; ++i) {
    row_scales[i] = -1.0f;
    row_indices[i] = -1;
  }

  for (int i = 0; i < num_experts; ++i) {
    float prob = expf(static_cast<float>(row_logits[i]) - max_val) / sum_exp;

    // Insert into top-k logic
    // Simple insertion sort for very small k (e.g. k=2)
    for (int j = 0; j < k; ++j) {
      if (prob > row_scales[j]) {
        // Shift current values down
        for (int m = k - 1; m > j; --m) {
          row_scales[m] = row_scales[m - 1];
          row_indices[m] = row_indices[m - 1];
        }
        row_scales[j] = prob;
        row_indices[j] = i;
        break;
      }
    }
  }

  // 4. Normalize if requested
  if (normalize_scales) {
    float scale_sum = 0.0f;
    for (int i = 0; i < k; ++i) {
      scale_sum += row_scales[i];
    }
    if (scale_sum > 1e-6f) {
      for (int i = 0; i < k; ++i) {
        row_scales[i] /= scale_sum;
      }
    }
  }
}

void LaunchSoftmaxTopK(
    const float* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream) {
  int block = 256;
  int grid = (num_rows + block - 1) / block;
  SoftmaxTopKKernel<float><<<grid, block, 0, stream>>>(logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
}

void LaunchSoftmaxTopK(
    const half* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream) {
  int block = 256;
  int grid = (num_rows + block - 1) / block;
  SoftmaxTopKKernel<half><<<grid, block, 0, stream>>>(logits, topk_scales, topk_indices, num_rows, num_experts, k, normalize_scales);
}

// Transpose kernel: converts [rows, cols] to [cols, rows]
// For MoE: transposes weights from ORT layout [hidden_size, inter_size]
// to kernel layout [inter_size, hidden_size]
template <typename T>
__global__ void Transpose2DKernel(const T* input, T* output, int rows, int cols) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total = rows * cols;
  if (idx >= total) return;

  int r = idx / cols;  // row in input
  int c = idx % cols;  // col in input
  // output[c, r] = input[r, c]
  output[c * rows + r] = input[r * cols + c];
}

template <typename T>
void LaunchTranspose2D(
    const T* input,
    T* output,
    int rows,
    int cols,
    cudaStream_t stream) {
  int total = rows * cols;
  int block = 256;
  int grid = (total + block - 1) / block;
  Transpose2DKernel<T><<<grid, block, 0, stream>>>(input, output, rows, cols);
}

// Explicit template instantiations
template void LaunchTranspose2D<float>(const float*, float*, int, int, cudaStream_t);
template void LaunchTranspose2D<half>(const half*, half*, int, int, cudaStream_t);
template void LaunchTranspose2D<__nv_bfloat16>(const __nv_bfloat16*, __nv_bfloat16*, int, int, cudaStream_t);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

// Dummy definition for Lora_run to satisfy linker
namespace onnxruntime::llm::kernels {
int Lora_run(LoraImpl* /*impl*/, int64_t /*numTokens*/, int64_t /*numReqs*/, void const* /*input*/, int32_t const* /*loraRanks*/,
             void const* const* /*loraWeightsPtr*/, int /*weightIndex*/, void* const* /*outputs*/, void* /*workspace*/, cudaStream_t /*stream*/) {
  return 0;
}

}  // namespace onnxruntime::llm::kernels
