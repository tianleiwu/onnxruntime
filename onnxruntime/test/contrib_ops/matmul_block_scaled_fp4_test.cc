// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/conversion.h"

namespace onnxruntime::test {

#if defined(USE_CUDA)

// NVFP4 (E2M1) 4-bit magnitude nibble encodings (sign bit is 0x8):
//   +0.0 -> 0x0, +0.5 -> 0x1, +1.0 -> 0x2, +1.5 -> 0x3,
//   +2.0 -> 0x4, +3.0 -> 0x5, +4.0 -> 0x6, +6.0 -> 0x7
// A packed byte holds two values: low nibble is element 2j, high nibble is element 2j+1.
//
// E4M3 (float8e4m3fn) scale byte encodings:
//   1.0 -> 0x38, 2.0 -> 0x40, 0.5 -> 0x30

// A -> [M, K] all ones per row scaled by (m + 1); weights are constant per row, so the
// operator must reproduce Y[m, n] = W_val[n] * sum_k A[m, k].
TEST(MatMulBlockScaledFp4OpTest, WeightOnlyBasicFp16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockScaledFp4.";
  }

  constexpr int64_t m = 2;
  constexpr int64_t n = 2;
  constexpr int64_t k = 16;  // one block with block_size == 16

  // Weight row 0 = +1.0 (nibble 0x2 -> byte 0x22), row 1 = +2.0 (nibble 0x4 -> byte 0x44).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0x44;
  }
  // One E4M3 scale per row (single K block), both 1.0.
  std::vector<uint8_t> weight_scale = {0x38, 0x38};
  std::vector<float> weight_scale_2 = {1.0f};

  std::vector<float> a(m * k);
  for (int64_t row = 0; row < m; ++row) {
    for (int64_t col = 0; col < k; ++col) {
      a[row * k + col] = static_cast<float>(row + 1);
    }
  }
  // W[0, :] = 1.0, W[1, :] = 2.0; sum_k A[0, :] = 16, sum_k A[1, :] = 32.
  std::vector<float> expected = {16.0f, 32.0f, 32.0f, 64.0f};

  OpTester test("MatMulBlockScaledFp4", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("K", k);
  test.AddAttribute<int64_t>("N", n);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<MLFloat16>("A", {m, k}, FloatsToMLFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, 1}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOutput<MLFloat16>("Y", {m, n}, FloatsToMLFloat16s(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Exercises non-unit per-block E4M3 scales, a global weight_scale_2, negative weights, bias and
// a skipped optional input_scale, with BF16 activations/output.
TEST(MatMulBlockScaledFp4OpTest, WeightOnlyScalesBiasBf16) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device is required for MatMulBlockScaledFp4.";
  }

  constexpr int64_t m = 1;
  constexpr int64_t n = 2;
  constexpr int64_t k = 16;

  // Weight row 0 = +1.0 (0x22), row 1 = -1.0 (nibble 0xA -> byte 0xAA).
  std::vector<uint8_t> b(n * (k / 2));
  for (int64_t j = 0; j < k / 2; ++j) {
    b[0 * (k / 2) + j] = 0x22;
    b[1 * (k / 2) + j] = 0xAA;
  }
  // Row 0 scale = 2.0 (0x40), row 1 scale = 1.0 (0x38).
  std::vector<uint8_t> weight_scale = {0x40, 0x38};
  std::vector<float> weight_scale_2 = {3.0f};

  std::vector<float> a(m * k, 1.0f);
  // W[0, :] = 1.0 * 3.0 * 2.0 = 6.0; W[1, :] = -1.0 * 3.0 * 1.0 = -3.0; sum_k A = 16.
  // Y = {6*16, -3*16} + bias{1, 2} = {97, -46}.
  std::vector<float> bias = {1.0f, 2.0f};
  std::vector<float> expected = {97.0f, -46.0f};

  OpTester test("MatMulBlockScaledFp4", 1, onnxruntime::kMSDomain);
  test.AddAttribute<int64_t>("K", k);
  test.AddAttribute<int64_t>("N", n);
  test.AddAttribute<int64_t>("block_size", 16);
  test.AddInput<BFloat16>("A", {m, k}, FloatsToBFloat16s(a));
  test.AddInput<uint8_t>("B", {n, k / 2}, b);
  test.AddInput<uint8_t>("weight_scale", {n, 1}, weight_scale);
  test.AddInput<float>("weight_scale_2", {1}, weight_scale_2);
  test.AddOptionalInputEdge<float>();  // input_scale (skipped)
  test.AddInput<BFloat16>("bias", {n}, FloatsToBFloat16s(bias));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));
  test.SetOutputTolerance(0.5f);

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

#endif  // USE_CUDA

}  // namespace onnxruntime::test
