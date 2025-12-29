// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT License.

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/functional.h>

#include "core/mlas/inc/mlas_q4.h"
#include "contrib_ops/cpu/quantization/dequantize_blockwise_bnb4.h"
#include "core/util/thread_utils.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.h"
#include "contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors.h"
#endif
#include <stdexcept>
#include <memory>

namespace pybind11 {
namespace detail {
// python3 -c 'import numpy as np; print(np.dtype(np.float16).num)'
constexpr int NPY_FLOAT16 = 23;
template <>
struct npy_format_descriptor<onnxruntime::MLFloat16> {
  static constexpr auto name = _("float16");
  static pybind11::dtype dtype() {
    handle ptr = npy_api::get().PyArray_DescrFromType_(NPY_FLOAT16);
    return reinterpret_borrow<pybind11::dtype>(ptr);
  }
  static std::string format() {
    // following: https://docs.python.org/3/library/struct.html#format-characters
    return "e";
  }
};
}  // namespace detail
}  // namespace pybind11

namespace onnxruntime {
namespace python {

namespace py = pybind11;
using namespace onnxruntime;

template <typename T, int qbits>
void QuantizeMatMulNBitsBlockwise(
    py::array_t<uint8_t> dst,          // shape: [ N, block_per_K, block_blob_size ]
    py::array_t<T> src,                // shape: [K, N]
    py::array_t<T> scale,              // shape: [N, block_per_K]
    py::array_t<uint8_t> zero_points,  // shape: [N, block_per_K] if bits > 4 else [N, (block_per_K + 1) / 2]
    int32_t block_size,
    int32_t N,
    int32_t K,
    bool is_symmetric) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  py::buffer_info dst_buf = dst.request();
  py::buffer_info src_buf = src.request();
  py::buffer_info scale_buf = scale.request();
  py::buffer_info zp_buf = zero_points.request();

  MlasQuantizeBlockwise<T, qbits>(
      reinterpret_cast<uint8_t*>(dst_buf.ptr),
      reinterpret_cast<T*>(scale_buf.ptr),
      is_symmetric ? nullptr : reinterpret_cast<uint8_t*>(zp_buf.ptr),
      reinterpret_cast<const T*>(src_buf.ptr),
      block_size,
      true,
      K,
      N,
      N,
      tp.get());
}

template <typename T>
bool QuantizeQDQMatMul4BitsBlockwise(
    py::array_t<uint8_t> dst,          // shape: [K, N / 2]
    py::array_t<T> src,                // shape: [K, N]
    py::array_t<T> scale,              // shape: [block_per_K, N]
    py::array_t<uint8_t> zero_points,  // shape: [block_per_K, N / 2]
    int32_t quant_block_size,
    int32_t N,
    int32_t K,
    bool is_symmetric) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  py::buffer_info dst_buf = dst.request();
  py::buffer_info src_buf = src.request();
  py::buffer_info scale_buf = scale.request();
  py::buffer_info zp_buf = zero_points.request();

  return MlasQDQQuantizeBlockwise<T, 4>(
      reinterpret_cast<const T*>(src_buf.ptr),
      reinterpret_cast<T*>(scale_buf.ptr),
      is_symmetric ? nullptr : reinterpret_cast<uint8_t*>(zp_buf.ptr),
      reinterpret_cast<uint8_t*>(dst_buf.ptr),
      true,
      K,
      N,
      quant_block_size,
      tp.get());
}

template <typename T>
void QuantizeMatMulBnb4Blockwise(
    py::array_t<uint8_t> dst,
    py::array_t<T> src,
    py::array_t<T> absmax,
    int32_t block_size,
    int32_t quant_type,
    int32_t N,
    int32_t K) {
  OrtThreadPoolParams to;
  auto tp = concurrency::CreateThreadPool(&onnxruntime::Env::Default(), to,
                                          concurrency::ThreadPoolType::INTRA_OP);

  py::buffer_info dst_buf = dst.request();
  py::buffer_info src_buf = src.request();
  py::buffer_info absmax_buf = absmax.request();

  contrib::QuantizeBlockwiseBnb4<T>(
      static_cast<uint8_t*>(dst_buf.ptr),
      static_cast<const T*>(src_buf.ptr),
      static_cast<T*>(absmax_buf.ptr),
      block_size,
      quant_type,
      N,
      K,
      tp.get());
}

#ifdef USE_CUDA
namespace cuda {
struct CudaDeleter {
  void operator()(void* p) const {
    if (p) cudaFree(p);
  }
};

using CudaPtr = std::unique_ptr<void, CudaDeleter>;

// Preprocess quantized weights for CUDA mixed-precision GEMM kernels (FpA_IntB format).
//
// MatMulNBits/QMoE stores quantized weights in (N, K) layout:
//   - N = number of output channels (columns in weight matrix W)
//   - K = number of input features (rows in weight matrix W)
//   - For 4-bit: shape is (N, K/2) bytes where each byte packs 2 elements
//   - For 8-bit: shape is (N, K) bytes
//
// FpA_IntB GEMM kernels expect weights in (K, N) layout (transposed) for efficient
// memory access during matrix multiplication. This function:
//   1. Transposes from (N, K) to (K, N) layout
//   2. Converts unsigned quantized values to signed int8 with zero-point adjustment
//      - 4-bit: uint4 -> int8 with zero_point=8 (range [0,15] -> [-8,7])
//      - 8-bit: uint8 -> int8 with zero_point=128 (range [0,255] -> [-128,127])
//   3. Applies architecture-specific row permutation for optimized tensor core access
//
// Input:  q_weights - Quantized weights from MatMulNBits in (N, K) layout
// Output: Preprocessed weights in (K, N) layout ready for fpA_intB GEMM kernels
py::array_t<int8_t> PackWeightsForMixedGemm(
    py::array_t<uint8_t> q_weights,
    int32_t N,
    int32_t K,
    int32_t bits) {
  py::buffer_info q_weights_buf = q_weights.request();

  size_t n = static_cast<size_t>(N);
  size_t k = static_cast<size_t>(K);

  size_t packed_weight_bytes = n * k / (8 / bits);
  py::array_t<int8_t> processed_weights({static_cast<pybind11::ssize_t>(packed_weight_bytes)});
  py::buffer_info processed_weights_buf = processed_weights.request();

  auto make_cuda_ptr = [](size_t bytes) -> CudaPtr {
    void* p = nullptr;
    if (cudaMalloc(&p, bytes) != cudaSuccess) {
      throw std::runtime_error("cudaMalloc failed");
    }
    return CudaPtr(p);
  };

  auto packed_transposed_weight_space = make_cuda_ptr(packed_weight_bytes);
  int8_t* packed_transposed_weight = reinterpret_cast<int8_t*>(packed_transposed_weight_space.get());

  auto fpA_intB_weight_buffer_ = make_cuda_ptr(packed_weight_bytes);
  int8_t* preprocessed_weight = reinterpret_cast<int8_t*>(fpA_intB_weight_buffer_.get());

  const uint8_t* blob_data_cpu = static_cast<const uint8_t*>(q_weights_buf.ptr);

  auto blob_data_gpu_buf = make_cuda_ptr(packed_weight_bytes);
  uint8_t* blob_data_gpu = reinterpret_cast<uint8_t*>(blob_data_gpu_buf.get());

  cudaStream_t stream = cudaStreamLegacy;
  cudaMemcpyAsync(blob_data_gpu, blob_data_cpu, packed_weight_bytes, cudaMemcpyHostToDevice, stream);

  if (bits == 4) {
    ::onnxruntime::llm::kernels::fpA_intB_gemv::unpack_uint4_transposed_to_int8_direct_cuda(
        stream, packed_transposed_weight, blob_data_gpu, n, k);
  } else {
    // 8 bits
    ::onnxruntime::llm::kernels::fpA_intB_gemv::transpose_uint8_matrix_and_convert_to_int8(
        stream, packed_transposed_weight, blob_data_gpu, n, k);
  }

  using ::onnxruntime::llm::kernels::weight_only::QuantType;
  QuantType quant_type = bits == 4 ? QuantType::W4_A16 : QuantType::W8_A16;

  int sm = 0;
  int device_id = 0;
  cudaGetDevice(&device_id);
  cudaDeviceProp device_prop;
  cudaGetDeviceProperties(&device_prop, device_id);
  sm = device_prop.major * 10 + device_prop.minor;

  auto permutation_map_buffer = make_cuda_ptr(32 * sizeof(int32_t));

  ::onnxruntime::llm::kernels::weight_only::preprocess_weights_for_mixed_gemm_cuda(
      stream,
      sm,
      preprocessed_weight,
      packed_transposed_weight,
      reinterpret_cast<int32_t*>(permutation_map_buffer.get()),
      {static_cast<size_t>(k), static_cast<size_t>(n)},
      quant_type);

  cudaMemcpyAsync(processed_weights_buf.ptr, preprocessed_weight, packed_weight_bytes, cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  return processed_weights;
}
}  // namespace cuda
#endif

void CreateQuantPybindModule(py::module& m) {
  m.def("quantize_matmul_2bits", &QuantizeMatMulNBitsBlockwise<float, 2>);
  m.def("quantize_matmul_2bits", &QuantizeMatMulNBitsBlockwise<MLFloat16, 2>);
  m.def("quantize_matmul_4bits", &QuantizeMatMulNBitsBlockwise<float, 4>);
  m.def("quantize_matmul_4bits", &QuantizeMatMulNBitsBlockwise<MLFloat16, 4>);
  m.def("quantize_matmul_8bits", &QuantizeMatMulNBitsBlockwise<float, 8>);
  m.def("quantize_matmul_8bits", &QuantizeMatMulNBitsBlockwise<MLFloat16, 8>);
  m.def("quantize_matmul_bnb4", &QuantizeMatMulBnb4Blockwise<float>);
  m.def("quantize_matmul_bnb4", &QuantizeMatMulBnb4Blockwise<MLFloat16>);
  m.def("quantize_qdq_matmul_4bits", &QuantizeQDQMatMul4BitsBlockwise<float>);
  m.def("quantize_qdq_matmul_4bits", &QuantizeQDQMatMul4BitsBlockwise<MLFloat16>);
#ifdef USE_CUDA
  m.def("pack_weights_for_cuda_mixed_gemm", &cuda::PackWeightsForMixedGemm,
        "Pack quantized weights for CUDA mixed-precision GEMM (FpA_IntB format)");
#endif
}

}  // namespace python
}  // namespace onnxruntime
