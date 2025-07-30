import torch

def quant_dequant_torch(weights: torch.Tensor, is_4_bit_quantization: bool):
    """
    Performs symmetric per-column quantization and dequantization on a weight tensor.

    This implementation is a pure PyTorch replacement for the original function that
    relied on a custom tensorrt_llm operator. It supports both 8-bit (int8) and
    4-bit (quint4x2 style) quantization.

    Args:
        weights (torch.Tensor): The input weight tensor to be quantized.
        is_4_bit_quantization (bool): If True, performs 4-bit quantization. If False,
                                    performs 8-bit quantization.

    Returns:
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
            - scales (torch.float16): The quantization scales for each column.
            - processed_q_weight (torch.int8): The packed quantized weights. For
              4-bit mode, two 4-bit values are packed into a single int8. For
              8-bit mode, this is the standard int8 quantized tensor. It is
              transposed relative to the input weights' shape.
            - dequantized_weights (torch.Tensor): The weights after being dequantized,
              restored to the original dtype and device.
    """
    # Determine quantization bits and range based on the mode
    if is_4_bit_quantization:
        # 4-bit symmetric quantization path
        q_bits = 4
        q_max = 2 ** (q_bits - 1) - 1  # 7
        q_min = -(2 ** (q_bits - 1))  # -8

        max_abs_val = torch.max(torch.abs(weights), dim=0, keepdim=True).values
        max_abs_val[max_abs_val == 0] = 1.0
        scales = max_abs_val / q_max

        quant_weights = torch.round(weights / scales).clamp(q_min, q_max).to(torch.int8)

        # Pack two 4-bit integers into a single int8
        q_weights_t = quant_weights.T.contiguous()
        shape = q_weights_t.shape
        q_weights_t_reshaped = q_weights_t.view(shape[0], shape[1] // 2, 2)
        lower_nibble = q_weights_t_reshaped[..., 0]
        upper_nibble = q_weights_t_reshaped[..., 1]
        processed_q_weight = (lower_nibble & 0x0F) | (upper_nibble << 4)

    else:
        # 8-bit symmetric quantization path
        q_bits = 8
        q_max = 2 ** (q_bits - 1) - 1  # 127
        q_min = -(2 ** (q_bits - 1))  # -128

        max_abs_val = torch.max(torch.abs(weights), dim=0, keepdim=True).values
        max_abs_val[max_abs_val == 0] = 1.0
        scales = max_abs_val / q_max

        quant_weights = torch.round(weights / scales).clamp(q_min, q_max).to(torch.int8)

        # For 8-bit, the processed weights are just the transposed quantized weights (no packing)
        processed_q_weight = quant_weights.T.contiguous()

    # Dequantize the weights to verify and return for PyTorch-side parity check
    dequantized_weights = quant_weights.to(weights.dtype) * scales.to(weights.dtype)

    return (scales.squeeze(0).to(torch.float16), processed_q_weight, dequantized_weights.to(device=weights.device))


def quant_dequant_trt(weights, is_4_bit_quantization: bool = True):
    # use the test version `_symmetric_...` to get the non-interleaved weights
    type = torch.quint4x2 if is_4_bit_quantization else torch.int8
    # This import is needed to use torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix()
    # Comment out this line for passing the lintrunner check in the CI.
    import tensorrt_llm

    """ Here is C++ code for _symmetric_quantize_last_axis_of_batched_matrix:
    std::vector<Tensor> symmetric_quantize_helper(
        Tensor weight, torch::ScalarType quant_type, bool return_unprocessed_quantized_tensor)
    {
        CHECK_CPU(weight);
        CHECK_CONTIGUOUS(weight);
        TORCH_CHECK(weight.numel() != 0, "weight should not be empty tensor");
        TORCH_CHECK(weight.dim() == 2 || weight.dim() == 3, "Invalid dim. The dim of weight should be 2 or 3");

        auto _st = weight.scalar_type();
        TORCH_CHECK(_st == torch::kFloat32 || _st == torch::kFloat16 || _st == torch::kBFloat16,
            "Invalid datatype. Weight must be FP16 or BF16");
        check_quant_type_allowed(quant_type);
        QuantType ft_quant_type = get_ft_quant_type(quant_type);

        const size_t num_experts = weight.dim() == 2 ? 1 : weight.size(0);
        const size_t num_rows = weight.size(-2);
        const size_t num_cols = weight.size(-1);

        const size_t bits_in_type = get_weight_quant_bits(ft_quant_type);
        const size_t bytes_per_out_col = num_cols * bits_in_type / 8;

        std::vector<int64_t> quantized_weight_shape;
        std::vector<int64_t> scale_shape;
        if (weight.dim() == 2)
        {
            quantized_weight_shape = {int64_t(num_rows), int64_t(bytes_per_out_col)};
            scale_shape = {int64_t(num_cols)};
        }
        else if (weight.dim() == 3)
        {
            quantized_weight_shape = {int64_t(num_experts), int64_t(num_rows), int64_t(bytes_per_out_col)};
            scale_shape = {int64_t(num_experts), int64_t(num_cols)};
        }
        else
        {
            TORCH_CHECK(false, "Invalid weight dimension. Weight must have dim 2 or 3");
        }

        Tensor unprocessed_quantized_weight
            = torch::empty(quantized_weight_shape, torch::dtype(torch::kInt8).device(torch::kCPU).requires_grad(false));

        Tensor processed_quantized_weight = torch::empty_like(unprocessed_quantized_weight);

        Tensor scales = torch::empty(scale_shape, torch::dtype(weight.dtype()).device(torch::kCPU).requires_grad(false));

        int8_t* unprocessed_quantized_weight_ptr = get_ptr<int8_t>(unprocessed_quantized_weight);
        int8_t* processed_quantized_weight_ptr = get_ptr<int8_t>(processed_quantized_weight);

        // TODO This should be removed if Grouped GEMM is updated to not need interleaved input
        bool force_interleave = weight.dim() == 3;

        if (weight.scalar_type() == at::ScalarType::Float)
        {
            symmetric_quantize<float, float>(processed_quantized_weight_ptr, unprocessed_quantized_weight_ptr,
                get_ptr<float>(scales), get_ptr<float const>(weight), {num_experts, num_rows, num_cols}, ft_quant_type,
                force_interleave);
        }
        else if (weight.scalar_type() == at::ScalarType::Half)
        {
            symmetric_quantize<half, half>(processed_quantized_weight_ptr, unprocessed_quantized_weight_ptr,
                get_ptr<half>(scales), get_ptr<half const>(weight), {num_experts, num_rows, num_cols}, ft_quant_type,
                force_interleave);
        }
    #ifdef ENABLE_BF16
        else if (weight.scalar_type() == at::ScalarType::BFloat16)
        {
            symmetric_quantize<__nv_bfloat16, __nv_bfloat16>(processed_quantized_weight_ptr,
                unprocessed_quantized_weight_ptr, get_ptr<__nv_bfloat16>(scales), get_ptr<__nv_bfloat16 const>(weight),
                {num_experts, num_rows, num_cols}, ft_quant_type, force_interleave);
        }
    #endif
        else
        {
            TORCH_CHECK(false, "Invalid datatype. Weight must be BF16/FP16");
        }

        if (return_unprocessed_quantized_tensor)
        {
            return std::vector<Tensor>{unprocessed_quantized_weight, processed_quantized_weight, scales};
        }

        return std::vector<Tensor>{processed_quantized_weight, scales};
    }



    // Same as symmetric_quantize_last_axis_of_batched_matrix but returns a tuple of:
    // (unprocessed_quantized_weights, preprocessed_quantized_weights, scales)
    // Exposed mainly for testing, so that the unprocessed weights can be passed to torch functions.
    std::vector<Tensor> _symmetric_quantize_last_axis_of_batched_matrix(Tensor weight, torch::ScalarType quant_type)
    {
        return symmetric_quantize_helper(weight, quant_type, true);
    }
    """
    quant_weights, processed_q_weight, torch_weight_scales = (
        torch.ops.trtllm._symmetric_quantize_last_axis_of_batched_matrix(weights.T.cpu().contiguous(), type)
    )

    # Unpack the int4s int int8s
    if is_4_bit_quantization:
        upper = quant_weights >> 4
        lower = (quant_weights << 4) >> 4  # Arithmetic right shift sign extends
        quant_weights = torch.stack((lower, upper), dim=2).view(weights.T.shape)

    quant_weights = quant_weights.to(dtype=weights.dtype)
    result = torch.multiply(quant_weights, torch_weight_scales.unsqueeze(0)).T.contiguous()
    return torch_weight_scales.to(torch.float16), processed_q_weight, result.to(device=weights.device)


def run_test(in_features, out_features, is_4_bit):
    """
    Runs a single test case, comparing the outputs of the two functions.
    """
    print(f"Running test for in_features={in_features}, out_features={out_features}, is_4_bit={is_4_bit}")

    # Create a random weight tensor
    weights = torch.randn(in_features, out_features, dtype=torch.float16)

    # Get the outputs from the PyTorch implementation
    scales_torch, processed_q_weight_torch, dequantized_torch = quant_dequant_torch(weights, is_4_bit)

    # Get the outputs from the TRT implementation
    scales_trt, processed_q_weight_trt, dequantized_trt = quant_dequant_trt(weights, is_4_bit)

    print(f"{weights.shape=} {in_features=} {out_features=} {scales_torch.shape=} {scales_trt.shape=} {processed_q_weight_torch.shape=} {processed_q_weight_trt.shape=} {dequantized_torch.shape=} {dequantized_trt.shape=}")

    print("scales_torch", scales_torch)
    print("scales_trt", scales_trt)

    print("processed_q_weight_torch", processed_q_weight_torch)
    print("processed_q_weight_trt", processed_q_weight_trt)

    print("dequantized_torch", dequantized_torch)
    print("dequantized_trt", dequantized_trt)

    # Compare the processed quantized weights
    processed_q_weight_all_close = torch.allclose(processed_q_weight_torch, processed_q_weight_trt)
    print(f"  Processed quantized weights are close: {processed_q_weight_all_close}")

    # Compare the dequantized weights
    dequantized_all_close = torch.allclose(dequantized_torch, dequantized_trt, atol=1e-2)
    print(f"  Dequantized weights are close: {dequantized_all_close}")

    # Compare the scales
    scales_all_close = torch.allclose(scales_torch, scales_trt, atol=1e-2)
    print(f"  Scales are close: {scales_all_close}")

    print("-" * 30)


if __name__ == "__main__":
    # Test with 4-bit quantization
    # run_test(64, 256, is_4_bit=True)

    # Test with 8-bit quantization
    run_test(64, 256, is_4_bit=False)
