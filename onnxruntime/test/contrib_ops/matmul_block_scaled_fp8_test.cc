// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include "gtest/gtest.h"
#include "test/common/cuda_op_test_utils.h"
#include "test/common/tensor_op_test_utils.h"
#include "test/providers/provider_test_utils.h"
#include "test/unittest_util/conversion.h"

namespace onnxruntime::test {

#if defined(USE_CUDA) && !defined(DISABLE_FLOAT8_TYPES)
TEST(MatMulBlockScaledFp8OpTest, PerKBlockScales) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device does not support FP8 matrix data types.";
  }

  constexpr int64_t k = 128;
  OpTester test("MatMulBlockScaledFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", k);
  test.AddInput<Float8E4M3FN>("A", {2, 1, k}, std::vector<Float8E4M3FN>(2 * k, Float8E4M3FN(1.0f)));
  test.AddInput<Float8E4M3FN>("B", {2, k}, std::vector<Float8E4M3FN>(2 * k, Float8E4M3FN(1.0f)));
  test.AddInput<MLFloat16>("scaleA", {2, 1}, MakeMLFloat16({2.0f, 3.0f}));
  test.AddInput<MLFloat16>("scaleB", {2, 1}, MakeMLFloat16({4.0f, 5.0f}));
  test.AddOutput<BFloat16>("Y", {2, 1, 2}, MakeBFloat16({1024.0f, 1280.0f, 1536.0f, 1920.0f}));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(MatMulBlockScaledFp8OpTest, Fp16ActivationsAndOutput) {
  if (!HasCudaEnvironment(800)) {
    GTEST_SKIP() << "CUDA device does not support FP8 matrix data types.";
  }

  constexpr int64_t k = 128;
  OpTester test("MatMulBlockScaledFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", k);
  test.AddInput<MLFloat16>("A", {2, 1, k}, std::vector<MLFloat16>(2 * k, MLFloat16(1.0f)));
  test.AddInput<Float8E4M3FN>("B", {2, k}, std::vector<Float8E4M3FN>(2 * k, Float8E4M3FN(1.0f)));
  test.AddInput<float>("scaleA", {2, 1}, {2.0f, 3.0f});
  test.AddInput<float>("scaleB", {2, 1}, {4.0f, 5.0f});
  test.AddOutput<MLFloat16>("Y", {2, 1, 2}, MakeMLFloat16({1024.0f, 1280.0f, 1536.0f, 1920.0f}));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Exercises the CUTLASS tensor-core fast path (Hopper/Blackwell): FP8 A -> BF16 Y with
// block_size 128 and K/N aligned to 16. On architectures without the fast path this
// falls back to the reference kernel, so the expected output is identical either way.
// A and B are all ones with distinct per-row (scaleA) and per-column (scaleB) scales, so
// Y[m, n] = K * scaleA[m] * scaleB[n]; this also catches any A/B scale-layout transposition.
TEST(MatMulBlockScaledFp8OpTest, Fp8FastPathAlignedBF16) {
  if (!HasCudaEnvironment(900)) {
    GTEST_SKIP() << "CUDA device does not support the FP8 tensor-core fast path (requires SM90+).";
  }

  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t k = 128;  // single block with block_size == 128

  std::vector<float> scale_a(m);
  for (int64_t i = 0; i < m; ++i) {
    scale_a[i] = (i % 2 == 0) ? 1.0f : 0.5f;
  }
  std::vector<float> scale_b(n);
  for (int64_t j = 0; j < n; ++j) {
    scale_b[j] = (j < n / 2) ? 1.0f : 2.0f;
  }
  std::vector<float> expected(m * n);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      expected[i * n + j] = static_cast<float>(k) * scale_a[i] * scale_b[j];
    }
  }

  OpTester test("MatMulBlockScaledFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", k);
  test.AddInput<Float8E4M3FN>("A", {m, k}, std::vector<Float8E4M3FN>(m * k, Float8E4M3FN(1.0f)));
  test.AddInput<Float8E4M3FN>("B", {n, k}, std::vector<Float8E4M3FN>(k * n, Float8E4M3FN(1.0f)));
  test.AddInput<float>("scaleA", {m, 1}, scale_a);
  test.AddInput<float>("scaleB", {n, 1}, scale_b);
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Fast path with K = 256 == 2 * block_size: two K-blocks, each with distinct per-row/per-column
// scales, so Y[m, n] = block_size * (scaleA[m,0]*scaleB[0,n] + scaleA[m,1]*scaleB[1,n]). This
// verifies K-block iteration and the K-major scaleA / scaleB layouts across multiple blocks.
TEST(MatMulBlockScaledFp8OpTest, Fp8FastPathMultiBlockK256BF16) {
  if (!HasCudaEnvironment(900)) {
    GTEST_SKIP() << "CUDA device does not support the FP8 tensor-core fast path (requires SM90+).";
  }

  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t block_size = 128;
  constexpr int64_t k = 256;  // 2 * block_size
  constexpr int64_t k_blocks = k / block_size;

  std::vector<float> scale_a(m * k_blocks);  // row-major [M, k_blocks]
  for (int64_t i = 0; i < m; ++i) {
    scale_a[i * k_blocks + 0] = (i % 2 == 0) ? 1.0f : 0.5f;
    scale_a[i * k_blocks + 1] = (i % 2 == 0) ? 2.0f : 1.0f;
  }
  std::vector<float> scale_b(n * k_blocks);  // row-major [N, k_blocks]
  for (int64_t j = 0; j < n; ++j) {
    scale_b[j * k_blocks + 0] = (j < n / 2) ? 1.0f : 2.0f;
    scale_b[j * k_blocks + 1] = (j < n / 2) ? 0.5f : 1.0f;
  }
  std::vector<float> expected(m * n);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      float acc = 0.0f;
      for (int64_t b = 0; b < k_blocks; ++b) {
        acc += scale_a[i * k_blocks + b] * scale_b[j * k_blocks + b];
      }
      expected[i * n + j] = static_cast<float>(block_size) * acc;
    }
  }

  OpTester test("MatMulBlockScaledFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", block_size);
  test.AddInput<Float8E4M3FN>("A", {m, k}, std::vector<Float8E4M3FN>(m * k, Float8E4M3FN(1.0f)));
  test.AddInput<Float8E4M3FN>("B", {n, k}, std::vector<Float8E4M3FN>(k * n, Float8E4M3FN(1.0f)));
  test.AddInput<float>("scaleA", {m, k_blocks}, scale_a);
  test.AddInput<float>("scaleB", {n, k_blocks}, scale_b);
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Fast path with K = 16 (< block_size): a single partial K-block sharing one scale, so
// Y[m, n] = K * scaleA[m] * scaleB[n]. Exercises the K-residue / sub-block-size path.
TEST(MatMulBlockScaledFp8OpTest, Fp8FastPathPartialBlockK16BF16) {
  if (!HasCudaEnvironment(900)) {
    GTEST_SKIP() << "CUDA device does not support the FP8 tensor-core fast path (requires SM90+).";
  }

  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t block_size = 128;
  constexpr int64_t k = 16;  // single partial block (k < block_size)

  std::vector<float> scale_a(m);
  for (int64_t i = 0; i < m; ++i) {
    scale_a[i] = (i % 2 == 0) ? 1.0f : 0.5f;
  }
  std::vector<float> scale_b(n);
  for (int64_t j = 0; j < n; ++j) {
    scale_b[j] = (j < n / 2) ? 1.0f : 2.0f;
  }
  std::vector<float> expected(m * n);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      expected[i * n + j] = static_cast<float>(k) * scale_a[i] * scale_b[j];
    }
  }

  OpTester test("MatMulBlockScaledFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", block_size);
  test.AddInput<Float8E4M3FN>("A", {m, k}, std::vector<Float8E4M3FN>(m * k, Float8E4M3FN(1.0f)));
  test.AddInput<Float8E4M3FN>("B", {n, k}, std::vector<Float8E4M3FN>(k * n, Float8E4M3FN(1.0f)));
  test.AddInput<float>("scaleA", {m, 1}, scale_a);
  test.AddInput<float>("scaleB", {n, 1}, scale_b);
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

// Fast path with a weight that varies along N (constant along K) and unit scales, so
// Y[m, n] = K * B_value(n). All-ones-B tests cannot distinguish the [N, K] weight layout from a
// transposed [K, N] one (every element is 1.0), so this test pins down the weight layout: a kernel
// that read the weight as [K, N] would produce a constant across n instead of the per-n staircase.
TEST(MatMulBlockScaledFp8OpTest, Fp8FastPathWeightLayoutBF16) {
  if (!HasCudaEnvironment(900)) {
    GTEST_SKIP() << "CUDA device does not support the FP8 tensor-core fast path (requires SM90+).";
  }

  constexpr int64_t m = 16;
  constexpr int64_t n = 16;
  constexpr int64_t k = 128;  // single block with block_size == 128

  std::vector<Float8E4M3FN> b(n * k);  // row-major [N, K]; B[n, :] == 1.0 or 2.0 depending on n
  for (int64_t j = 0; j < n; ++j) {
    const float v = (j < n / 2) ? 1.0f : 2.0f;
    for (int64_t ki = 0; ki < k; ++ki) {
      b[j * k + ki] = Float8E4M3FN(v);
    }
  }
  std::vector<float> expected(m * n);
  for (int64_t i = 0; i < m; ++i) {
    for (int64_t j = 0; j < n; ++j) {
      expected[i * n + j] = static_cast<float>(k) * ((j < n / 2) ? 1.0f : 2.0f);
    }
  }

  OpTester test("MatMulBlockScaledFp8", 1, onnxruntime::kMSDomain);
  test.AddAttribute("block_size", k);
  test.AddInput<Float8E4M3FN>("A", {m, k}, std::vector<Float8E4M3FN>(m * k, Float8E4M3FN(1.0f)));
  test.AddInput<Float8E4M3FN>("B", {n, k}, b);
  test.AddInput<float>("scaleA", {m, 1}, std::vector<float>(m, 1.0f));
  test.AddInput<float>("scaleB", {n, 1}, std::vector<float>(n, 1.0f));
  test.AddOutput<BFloat16>("Y", {m, n}, FloatsToBFloat16s(expected));

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.push_back(DefaultCudaExecutionProvider());
  test.Run(OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}
#endif

}  // namespace onnxruntime::test