// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <vector>
#include <type_traits>
#include <memory>
#include <utility>
#include <cstdint>

#include "core/common/common.h"
#include "core/framework/execution_provider.h"
#include "gtest/gtest.h"
#include "test/providers/provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

// When uint8_t data type is used GatherBlockQuantize applies MatMulNBit's conventions for storing the data.
// That is when no zero points are specified a default zero point of 8 is used. This convertor hence
// compensates for that by adding 8 to the data values, so that the outputs match the results that
// we be seen with non uint8_t data types.
template <typename T1>
void PackDataForUint8TypeIfNecessary(std::vector<int>& data, std::vector<int64_t>& data_shape, int bits = 4) {
  if (!std::is_same_v<T1, uint8_t>) {
    return;
  }

  if (bits == 8) {
    return;  // No packing needed for 8 bits
  }

  int64_t total_elements = 1;
  for (const auto& dim : data_shape) {
    total_elements *= dim;
  }
  int64_t total_columns = data_shape.back();
  int64_t total_rows = total_elements / total_columns;

  // For uint8_t, we need to pack each pair of values (after adding 8) into a single uint8_t
  std::vector<int> packed_data;
  int64_t packed_columns = (total_columns + 1) / 2;
  packed_data.reserve(total_rows * packed_columns);
  for (int64_t row = 0; row < total_rows; ++row) {
    for (int64_t col = 0; col < total_columns; col += 2) {
      int low_nibble = (data[row * total_columns + col] + 8) & 0xF;
      int high_nibble = ((col + 1) < total_columns) ? ((data[row * total_columns + col + 1] + 8) & 0xF) : 0;
      int packed = (high_nibble << 4) | low_nibble;
      packed_data.push_back(packed);
    }
  }

  data = packed_data;
  data_shape[data_shape.size() - 1] = packed_columns;
}

// Combinations: types, gather_axis, quantize_axis, block_size, indices, scale shape vs data shape
template <typename T1, typename T2, typename Tind>
void RunGatherBlockQuantized(const std::vector<T1>& data,
                             const std::vector<int64_t>& data_shape,
                             const std::vector<Tind>& indices,
                             const std::vector<int64_t>& indices_shape,
                             const std::vector<T2>& scales,
                             const std::vector<int64_t>& scales_shape,
                             const std::vector<T1>& zero_points,
                             const std::vector<int64_t>& zero_points_shape,
                             const int64_t gather_axis,
                             const int64_t quantize_axis,
                             const int64_t block_size,
                             const int64_t bits,
                             const std::vector<T2>& output,
                             const std::vector<int64_t>& output_shape,
                             OpTester::ExpectResult expect_result = OpTester::ExpectResult::kExpectSuccess) {
  auto run_test = [&](bool indices_is_initializer) {
    OpTester test("GatherBlockQuantized", 1, kMSDomain);

    test.AddAttribute<int64_t>("gather_axis", gather_axis);
    test.AddAttribute<int64_t>("quantize_axis", quantize_axis);
    test.AddAttribute<int64_t>("block_size", block_size);
    test.AddAttribute<int64_t>("bits", bits);

    test.AddInput<T1>("data", data_shape, data);
    test.AddInput<Tind>("indices", indices_shape, indices, indices_is_initializer);
    test.AddInput<T2>("scales", scales_shape, scales);
    if (!zero_points.empty()) {
      test.AddInput<T1>("zero_points", zero_points_shape, zero_points);
    }

    test.AddOutput<T2>("output", output_shape, output);

    std::vector<std::unique_ptr<IExecutionProvider>> eps;
    eps.push_back(DefaultCpuExecutionProvider());
    test.Run(expect_result, "", {}, nullptr, &eps);
  };

  run_test(false);
  run_test(true);
}

template <typename T1, typename T2>
typename std::enable_if<
    (boost::mp11::mp_contains<TypeList<BFloat16, MLFloat16, float>, T1>::value && std::is_same<T2, float>::value) ||
        (std::is_integral<T1>::value && std::is_same<T2, int>::value),
    std::vector<T1>>::type
ToType(const std::vector<T2>& vec) {
  std::vector<T1> result;
  for (auto v : vec) {
    result.push_back(static_cast<T1>(v));
  }

  return result;
}

template <typename T>
typename std::enable_if<boost::mp11::mp_contains<TypeList<UInt4x2, Int4x2>, T>::value, std::vector<T>>::type
ToType(const std::vector<int>& vec) {
  std::vector<T> result;
  size_t i = 0;
  constexpr int offset = std::is_same<T, Int4x2>::value ? 0 : 8;
  for (i = 0; i + 1 < vec.size(); i += 2) {
    result.push_back(T(vec[i] + offset, vec[i + 1] + offset));
  }
  if (i < vec.size()) {
    result.push_back(T(vec[i] + offset, 0 + offset));
  }

  return result;
}

// The data and zero_points are not packed
template <typename T1, typename T2, typename Tind>
void RunUnpackedData(
    std::vector<int>& data,
    std::vector<int64_t>& data_shape,
    const std::vector<int>& indices,
    const std::vector<int64_t>& indices_shape,
    const std::vector<float>& scales,
    const std::vector<int64_t>& scales_shape,
    std::vector<int>& zero_points,
    const int64_t gather_axis,
    const int64_t quantize_axis,
    const int64_t block_size,
    const int64_t bits,
    const std::vector<float>& output,
    const std::vector<int64_t>& output_shape,
    bool expect_success) {
  PackDataForUint8TypeIfNecessary<T1>(data, data_shape, bits);

  int64_t total_elements = 1;
  for (const auto& dim : data_shape) {
    total_elements *= dim;
  }
  ORT_ENFORCE(data.size() == static_cast<size_t>(total_elements),
              "Data size does not match the shape provided. Expected: ", total_elements, ", got: ", data.size());

  auto expect_result = expect_success ? OpTester::ExpectResult::kExpectSuccess : OpTester::ExpectResult::kExpectFailure;
  if (zero_points.empty()) {
    // If no zero points are provided, we can skip packing them.
    RunGatherBlockQuantized(ToType<T1>(data),
                            data_shape,
                            ToType<Tind>(indices),
                            indices_shape,
                            ToType<T2>(scales),
                            scales_shape,
                            {},
                            {},
                            gather_axis,
                            quantize_axis,
                            block_size,
                            bits,
                            ToType<T2>(output),
                            output_shape,
                            expect_result);
    return;
  }

  std::vector<int64_t> zero_points_shape = scales_shape;
  PackDataForUint8TypeIfNecessary<T1>(zero_points, zero_points_shape, bits);

  total_elements = 1;
  for (const auto& dim : zero_points_shape) {
    total_elements *= dim;
  }
  ORT_ENFORCE(zero_points.size() == static_cast<size_t>(total_elements),
              "Zero points size does not match the shape provided. Expected: ", total_elements, ", got: ", zero_points.size());

  RunGatherBlockQuantized(ToType<T1>(data),
                          data_shape,
                          ToType<Tind>(indices),
                          indices_shape,
                          ToType<T2>(scales),
                          scales_shape,
                          ToType<T1>(zero_points),
                          zero_points_shape,
                          gather_axis,
                          quantize_axis,
                          block_size,
                          bits,
                          ToType<T2>(output),
                          output_shape,
                          expect_result);
}

template <typename T>
std::vector<T> ConvertZeroPoint(const std::vector<int>& vec, int bits) {
  if constexpr (std::is_same_v<T, uint8_t>) {
    std::vector<uint8_t> result;
    if (bits == 4) {
      for (size_t i = 0; i + 1 < vec.size(); i += 2) {
        result.push_back(static_cast<uint8_t>((vec[i + 1] + 8) * 16 + vec[i] + 8));
      }
    } else {
      ORT_ENFORCE(bits == 8, "Only 4 or 8 bits are supported for uint8_t zero points");
      for (int v : vec) {
        result.push_back(static_cast<uint8_t>(v + 128));
      }
    }
    return result;
  } else {
    return ToType<T>(vec);
  }
}

template <typename T>
std::vector<int64_t> GetPackedShape(const std::vector<int64_t>& unpacked_shape, int bits) {
  if constexpr (std::is_same<T, Int4x2>::value) {
    return unpacked_shape;
  }

  if constexpr (std::is_same<T, UInt4x2>::value) {
    return unpacked_shape;
  }

  if constexpr (std::is_same<T, uint8_t>::value) {
    if (bits == 8) {
      return unpacked_shape;
    } else {
      ORT_ENFORCE(bits == 4, "Only 4 or 8 bits are supported for uint8_t zero points");
      std::vector<int64_t> packed_shape = unpacked_shape;
      size_t last_dim_index = packed_shape.size() - 1;
      packed_shape[last_dim_index] = (packed_shape[last_dim_index] + 1) / 2;
      return packed_shape;
    }
  }

  // Unsupported type, return the same shape as scales
  return unpacked_shape;
}

template <typename T1, typename T2, typename Tind>
void Test_Fail_WithZeroPoints(int64_t gather_axis,
                              int64_t quantize_axis,
                              int64_t block_size,
                              int64_t bits = 4) {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               3.f, 4.f, 5.f, 6.f,
                               -6.f, -4.f, -2.f, 0.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, false);
}

TEST(GatherBlockQuantizedOpTest, UnsupportedTypes) {
  Test_Fail_WithZeroPoints<int8_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<int16_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<uint16_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<int32_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<uint32_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<int64_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<uint64_t, float, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<UInt4x2, float, int16_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<Int4x2, float, int16_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<UInt4x2, BFloat16, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<Int4x2, BFloat16, int32_t>(0, 2, 16);
  Test_Fail_WithZeroPoints<uint8_t, float, int16_t>(0, 2, 16);
}

template <typename T1, typename T2, typename Tind>
void Test_Fail_WithoutZeroPoints(int64_t gather_axis,
                                 int64_t quantize_axis,
                                 int64_t block_size,
                                 int64_t bits = 4) {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  PackDataForUint8TypeIfNecessary<T1>(data, data_shape);
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<int> zero_points = {};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               3.f, 4.f, 5.f, 6.f,
                               -6.f, -4.f, -2.f, 0.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, false);
}

TEST(GatherBlockQuantizedOpTest, UnsupportedUInt8DataType) {
  // Gather on axis other than 0 is not supported with uint8_t
  Test_Fail_WithoutZeroPoints<uint8_t, float, int32_t>(1, 2, 16);
  Test_Fail_WithoutZeroPoints<uint8_t, float, int16_t>(1, 2, 16);
}

TEST(GatherBlockQuantizedOpTest, InvalidBlockSize) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 8);
  Test_Fail_WithZeroPoints<Int4x2, float, int32_t>(0, 2, 17);
  Test_Fail_WithZeroPoints<uint8_t, float, int32_t>(0, 2, 17);
}

TEST(GatherBlockQuantizedOpTest, InvalidGatherAxis) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(3, 2, 16);
  Test_Fail_WithZeroPoints<Int4x2, float, int32_t>(-4, 2, 16);
  Test_Fail_WithZeroPoints<uint8_t, float, int32_t>(-4, 2, 16);
}

TEST(GatherBlockQuantizedOpTest, InvalidQuantizeAxis) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 3, 16);
  Test_Fail_WithZeroPoints<Int4x2, float, int32_t>(0, -4, 16);
  Test_Fail_WithZeroPoints<uint8_t, float, int32_t>(0, -4, 16);
}

TEST(GatherBlockQuantizedOpTest, NotSupportedBits) {
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 1);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 2);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 3);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 5);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 6);
  Test_Fail_WithZeroPoints<UInt4x2, float, int32_t>(0, 2, 16, 7);
}

template <typename T1, typename T2, typename Tind>
void Test_ShapeMismatch_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  PackDataForUint8TypeIfNecessary<T1>(data, data_shape);
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 2, 1};
  std::vector<int> zero_points = {-1, 1, 0, 0};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               3.f, 4.f, 5.f, 6.f,
                               -6.f, -4.f, -2.f, 0.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, false);
}

TEST(GatherBlockQuantizedOpTest, ShapeMismatch) {
  Test_ShapeMismatch_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_ShapeMismatch_WithZeroPoints<Int4x2, float, int32_t>();
  Test_ShapeMismatch_WithZeroPoints<uint8_t, float, int32_t>();
}

template <typename T1, typename T2, typename Tind>
void Test_InvalidIndices_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  PackDataForUint8TypeIfNecessary<T1>(data, data_shape);
  std::vector<int> indices = {2};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               3.f, 4.f, 5.f, 6.f,
                               -6.f, -4.f, -2.f, 0.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, false);
}

TEST(GatherBlockQuantizedOpTest, InvalidIndices) {
  Test_InvalidIndices_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_InvalidIndices_WithZeroPoints<Int4x2, float, int32_t>();
  Test_InvalidIndices_WithZeroPoints<uint8_t, float, int32_t>();
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis0_WithZeroPoints(int bits = 4) {
  std::vector<int> data = {-8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8, -7, -6, -5, -8,
                           -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4,
                           0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3, 0,
                           4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4,
                           4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 4,
                           -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4, -3, -2, -1, -4};
  std::vector<int64_t> data_shape = {2, 3, 17};
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f,
                               2.0f, 2.0f, 1.0f, 1.0f, 2.0f, 1.0f};
  std::vector<int64_t> scales_shape = {2, 3, 2};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1,
                                  1, -1, 1, 0, -1, 1};
  std::vector<float> output = {6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 6, 8, 10, 12, 10,
                               3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 3, 4, 5, 6, 4,
                               -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -6, -4, -2, 0, -5};
  std::vector<int64_t> output_shape = {1, 3, 17};

  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                -3, -1, block_size, bits, output, output_shape, true);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0WithZeroPoints) {
  Test_GatherAxis0_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_GatherAxis0_WithZeroPoints<Int4x2, float, int32_t>();
  Test_GatherAxis0_WithZeroPoints<UInt4x2, MLFloat16, int32_t>();
  Test_GatherAxis0_WithZeroPoints<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis0_WithZeroPoints<UInt4x2, float, int64_t>();
  Test_GatherAxis0_WithZeroPoints<Int4x2, float, int64_t>();
  Test_GatherAxis0_WithZeroPoints<UInt4x2, MLFloat16, int64_t>();
  Test_GatherAxis0_WithZeroPoints<Int4x2, MLFloat16, int64_t>();
  // Test_GatherAxis0_WithZeroPoints<uint8_t, float, int32_t>();
  // Test_GatherAxis0_WithZeroPoints<uint8_t, MLFloat16, int64_t>();
  // Test_GatherAxis0_WithZeroPoints<uint8_t, float, int32_t>(8);
  // Test_GatherAxis0_WithZeroPoints<uint8_t, MLFloat16, int64_t>(8);
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis0_NoZeroPoints(int bits = 4) {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  PackDataForUint8TypeIfNecessary<T1>(data, data_shape);
  std::vector<int> indices = {1};
  std::vector<int64_t> indices_shape = {1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 3, 1};
  std::vector<float> output = {8.f, 10.f, 12.f, 14.f,
                               4.f, 5.f, 6.f, 7.f,
                               -8.f, -6.f, -4.f, -2.f};
  std::vector<int64_t> output_shape = {1, 3, 4};

  std::vector<int> zero_points = {};
  constexpr int64_t gather_axis = 0;
  constexpr int64_t quantize_axis = 2;
  constexpr int64_t block_size = 16;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                -3, -1, block_size, bits, output, output_shape, true);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis0NoZeroPoints) {
  Test_GatherAxis0_NoZeroPoints<Int4x2, float, int32_t>();
  Test_GatherAxis0_NoZeroPoints<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis0_NoZeroPoints<Int4x2, float, int64_t>();
  Test_GatherAxis0_NoZeroPoints<Int4x2, MLFloat16, int64_t>();
  Test_GatherAxis0_NoZeroPoints<uint8_t, float, int32_t>();
  // Test_GatherAxis0_NoZeroPoints<uint8_t, MLFloat16, int32_t>();
  // Test_GatherAxis0_NoZeroPoints<uint8_t, float, int64_t>(4);
  // Test_GatherAxis0_NoZeroPoints<uint8_t, MLFloat16, int64_t>(4);
  // Test_GatherAxis0_NoZeroPoints<uint8_t, float, int64_t>(8);
  // Test_GatherAxis0_NoZeroPoints<uint8_t, MLFloat16, int64_t>(8);
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis1_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {2, -3, 2};
  std::vector<int64_t> indices_shape = {1, 3};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {2, 1, 4};
  std::vector<int> zero_points = {-1, 1, 0, 0, 1, -1, 0, 0};
  std::vector<float> output = {1.f, 0.f, 2.f, 6.f,
                               -7.f, -16.f, -6.f, -10.f,
                               1.f, 0.f, 2.f, 6.f,
                               -5.f, -4.f, -2.f, -2.f,
                               3.f, 12.f, 6.f, 14.f,
                               -5.f, -4.f, -2.f, -2.f};
  std::vector<int64_t> output_shape = {2, 1, 3, 4};

  constexpr int64_t gather_axis = 1;
  constexpr int64_t quantize_axis = 1;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                -2, -2, block_size, bits, output, output_shape, true);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis1) {
  Test_GatherAxis1_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_GatherAxis1_WithZeroPoints<Int4x2, float, int32_t>();
  Test_GatherAxis1_WithZeroPoints<UInt4x2, MLFloat16, int32_t>();
  Test_GatherAxis1_WithZeroPoints<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis1_WithZeroPoints<UInt4x2, float, int64_t>();
  Test_GatherAxis1_WithZeroPoints<Int4x2, float, int64_t>();
  Test_GatherAxis1_WithZeroPoints<UInt4x2, MLFloat16, int64_t>();
  Test_GatherAxis1_WithZeroPoints<Int4x2, MLFloat16, int64_t>();
}

template <typename T1, typename T2, typename Tind>
void Test_GatherAxis2_WithZeroPoints() {
  std::vector<int> data = {-8, -7, -6, -5,
                           -4, -3, -2, -1,
                           0, 1, 2, 3,
                           4, 5, 6, 7,
                           4, 5, 6, 7,
                           -4, -3, -2, -1};
  std::vector<int64_t> data_shape = {2, 3, 4};
  std::vector<int> indices = {-2, 0};
  std::vector<int64_t> indices_shape = {2, 1};
  std::vector<float> scales = {1.0f, 2.0f, 1.0f, 2.0f,
                               1.0f, 2.0f, 1.0f, 2.0f,
                               1.0f, 2.0f, 1.0f, 2.0f};
  std::vector<int64_t> scales_shape = {1, 3, 4};
  std::vector<int> zero_points = {-1, 1, 0, 0,
                                  1, -1, 0, 0,
                                  0, 0, 1, -1};
  std::vector<float> output = {-6.f, -7.f, -2.f, -5.f, 1.f, 0.f,
                               6.f, 5.f, 6.f, 3.f, -3.f, -4.f};
  std::vector<int64_t> output_shape = {2, 3, 2, 1};

  constexpr int64_t gather_axis = 2;
  constexpr int64_t quantize_axis = 0;
  constexpr int64_t block_size = 16;
  constexpr int64_t bits = 4;
  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                gather_axis, quantize_axis, block_size, bits, output, output_shape, true);

  RunUnpackedData<T1, T2, Tind>(data, data_shape, indices, indices_shape, scales, scales_shape, zero_points,
                                -1, -3, block_size, bits, output, output_shape, true);
}

TEST(GatherBlockQuantizedOpTest, GatherAxis2) {
  Test_GatherAxis2_WithZeroPoints<UInt4x2, float, int32_t>();
  Test_GatherAxis2_WithZeroPoints<Int4x2, float, int32_t>();
  Test_GatherAxis2_WithZeroPoints<UInt4x2, MLFloat16, int32_t>();
  Test_GatherAxis2_WithZeroPoints<Int4x2, MLFloat16, int32_t>();
  Test_GatherAxis2_WithZeroPoints<UInt4x2, float, int64_t>();
  Test_GatherAxis2_WithZeroPoints<Int4x2, float, int64_t>();
  Test_GatherAxis2_WithZeroPoints<UInt4x2, MLFloat16, int64_t>();
  Test_GatherAxis2_WithZeroPoints<Int4x2, MLFloat16, int64_t>();
}

}  // namespace test
}  // namespace onnxruntime
