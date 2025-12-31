// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#pragma once

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace onnxruntime {
namespace contrib {
namespace cuda {

void LaunchSoftmaxTopK(
    const float* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream);

void LaunchSoftmaxTopK(
    const half* logits,
    float* topk_scales,
    int* topk_indices,
    int num_rows,
    int num_experts,
    int k,
    bool normalize_scales,
    cudaStream_t stream);

void LaunchSparseMixerTop2(
    const float* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream);

void LaunchSparseMixerTop2(
    const half* input,
    float* output,
    int* indices,
    int* source_rows,
    int num_rows,
    int num_experts,
    cudaStream_t stream);

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoEPrePackZP(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const float* scales,
    float* output,
    int num_elements,
    int N,
    cudaStream_t stream);

void LaunchQMoEPrePackPacked4BitZPKernel(
    const uint8_t* packed_zp,
    const half* scales,
    half* output,
    int num_elements,
    int N,
    cudaStream_t stream);

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const float* scales,
    float* output,
    int num_elements,
    float offset,
    cudaStream_t stream);

void LaunchQMoEPrePackOffsetBias(
    const uint8_t* zp,
    const half* scales,
    half* output,
    int num_elements,
    float offset,
    cudaStream_t stream);

void LaunchQMoEShiftWeights(
    const uint8_t* input,
    uint8_t* output,
    int num_elements,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const float* input,
    float* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const half* input,
    half* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

void LaunchQMoETranspose2D(
    const uint8_t* input,
    uint8_t* output,
    int batch_size,
    int rows,
    int cols,
    cudaStream_t stream);

}  // namespace cuda
}  // namespace contrib
}  // namespace onnxruntime

namespace onnxruntime::llm::kernels {
void LaunchBatchedTranspose(cudaStream_t stream, const void* input, void* output, int batch, int rows, int cols, int element_size);
}
