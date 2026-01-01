# Related Document:
MoE and QMoE Technical Documentation at
onnxruntime/contrib_ops/cuda/moe/README.md

Our kernel is based on TensorRT-LLM cutlass kernel, and the documents for the cutlass kernel are at
onnxruntime/contrib_ops/cuda/moe/moe_cutlass_kernel_int4_groupwise_quant_format.md
onnxruntime/contrib_ops/cuda/moe/moe_cutlass_kernel_int8_groupwise_quant_format.md

# MOE Operator Specification

constexpr const char* MoE_ver1_doc = R"DOC(
      Mixture of experts. Examples: Switch transformer(https://arxiv.org/pdf/2101.03961.pdf) use top 1,
      GLaM(https://arxiv.org/abs/2112.06905) activates top 2 FFN, Vision MOE(https://arxiv.org/pdf/2106.05974.pdf)
      usually uses top 32 experts and Mixtral(https://huggingface.co/blog/mixtral).

      The SwiGLU (Swish-Gated Linear Unit) activation function is like:
         g = xW + b
         l = xV + c
         G = clamp(g, max=limit)
         L = clamp(l, min=-limit, max=limit)
         swiglu = G * sigmoid(alpha * G) * (L + beta)
      where x is the input, W and V are weight matrices, b and c are bias vectors, and alpha, beta and limit are constant float parameters.
      When swiglu_fusion=0, two GEMMs are not fused, and they are FC1 and FC3 in the inputs.
      When swiglu_fusion=1, two GEMMs are fused so that g and l are computed in a single GEMM (FC1), and g and l are interleaved on each row of size 2 * inter_size.
      When swiglu_fusion=2, two GEMMs are fused, and g and l are concatenated on each row.
      )DOC";

ONNX_MS_OPERATOR_SET_SCHEMA(
    MoE, 1,
    OpSchema()
        .SetDoc(MoE_ver1_doc)
        .Attr("activation_type", "Activation function to use. Choose from relu, gelu, silu, swiglu and identity. Default is relu", AttributeProto::STRING, std::string("relu"))
        .Attr("swiglu_fusion", "0: not fused, 1: fused and interleaved. 2: fused and not interleaved.", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("swiglu_limit", "The limit used to clamp in SwiGLU. No clamp when limit is not provided.", AttributeProto::FLOAT, OPTIONAL_VALUE)
        .Attr("activation_alpha", "Alpha parameter used in activation function.", AttributeProto::FLOAT, 1.0f)
        .Attr("activation_beta", "Beta parameter used in activation function.", AttributeProto::FLOAT, 0.0f)
        .Attr("k", "Number of top experts to select from expert pool", AttributeProto::INT, static_cast<int64_t>(1))
        .Attr("normalize_routing_weights", "Whether to normalize routing weights", AttributeProto::INT, static_cast<int64_t>(0))
        .Attr("use_sparse_mixer", "Whether to use sparse mixer", AttributeProto::INT, static_cast<int64_t>(0))
        .Input(0, "input", "2D input tensor with shape (num_tokens, hidden_size) or 3D input tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .Input(1, "router_probs", "2D input tensor with shape (num_tokens, num_experts)", "T")
        .Input(2, "fc1_experts_weights", "3D input tensor with shape (num_experts, fusion_size * inter_size, hidden_size), where fusion_size is 2 for fused swiglu, and 1 otherwise", "T")
        .Input(3, "fc1_experts_bias", "2D optional input tensor with shape (num_experts, fusion_size * inter_size)", "T", OpSchema::Optional)
        .Input(4, "fc2_experts_weights", "3D input tensor with shape (num_experts, hidden_size, inter_size)", "T")
        .Input(5, "fc2_experts_bias", "2D optional input tensor with shape (num_experts, hidden_size)", "T", OpSchema::Optional)
        .Input(6, "fc3_experts_weights", "3D optional input tensor with shape (num_experts, inter_size, hidden_size)", "T", OpSchema::Optional)
        .Input(7, "fc3_experts_bias", "2D optional input tensor with shape (num_experts, inter_size)", "T", OpSchema::Optional)
        .Output(0, "output", "2D input tensor with shape (num_tokens, hidden_size) or 3D input tensor with shape (batch_size, sequence_length, hidden_size)", "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

constexpr const char* qMoE_ver1_doc = R"DOC(
      Quantized mixture of experts (MoE).

      The quantized weights are stored in column major order per expert.
      The quantization block size can be specified. If not provided, column wise quantization is used.

      The formula of linear dequantization of the quantized weights using scale and (optionally) zero-point is:
        dequantized_weight = (quantized_weight - zero_point) * scale
      When zero_point is not provided, the default value is 2^(bits-1): 8 for 4 bits, 128 for 8 bits.

      If block_size is provided, both hidden_size and inter_size must be divisible by the block size, and
      the dequantization is performed per block of size block_size along the K (input feature) dimension.

      If block_size and zero_point are provided, both hidden_size and inter_size must be divisible by block_size * pack_size,
      where pack_size = 8 / expert_weight_bits.

      The SwiGLU (Swish-Gated Linear Unit) activation function is like:
         g = xW + b
         l = xV + c
         G = clamp(g, max=limit)
         L = clamp(l, min=-limit, max=limit)
         swiglu = G * sigmoid(alpha * G) * (L + beta)
      where x is the input, W and V are weight matrices, b and c are bias vectors, and alpha, beta and limit are constant float parameters.
      When swiglu_fusion=0, two GEMMs are not fused, and they are FC1 and FC3 in the inputs.
      When swiglu_fusion=1, two GEMMs are fused so that g and l are computed in a single GEMM (FC1), and g and l are interleaved on each row of size 2 * inter_size.
      When swiglu_fusion=2, two GEMMs are fused, and g and l are concatenated on each row.
      )DOC";

# QMoE Operator Specification

ONNX_MS_OPERATOR_SET_SCHEMA(
    QMoE, 1,
    OpSchema()
        .SetDoc(qMoE_ver1_doc)
        .Attr("activation_type",
              "Activation function to use. Choose from relu, gelu, silu, swiglu and identity. Default is relu",
              AttributeProto::STRING,
              std::string("relu"))
        .Attr("k",
              "Number of top experts to select from expert pool",
              AttributeProto::INT,
              static_cast<int64_t>(1))
        .Attr("normalize_routing_weights",
              "Whether to normalize routing weights",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("use_sparse_mixer",
              "Whether to use sparse mixer",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("expert_weight_bits",
              "Number of bits used in quantized weights. Default is 4 bits",
              AttributeProto::INT,
              static_cast<int64_t>(4))
        .Attr("swiglu_fusion",
              "0: not fused, 1: fused and interleaved. 2: fused and not interleaved.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Attr("swiglu_limit",
              "The limit used to clamp inputs in SwiGLU. It is infinite when limit is not provided.",
              AttributeProto::FLOAT,
              OPTIONAL_VALUE)
        .Attr("activation_alpha",
              "Alpha parameter used in activation function.",
              AttributeProto::FLOAT, 1.0f)
        .Attr("activation_beta",
              "Beta parameter used in activation function.",
              AttributeProto::FLOAT, 0.0f)
        .Attr("block_size",
              "Size of each quantization block along the K (input feature) dimension. "
              "Must be power of two and ≥ 16 (e.g., 16, 32, 64, 128). "
              "If provided, both hidden_size and inter_size must be divisible by the block size. "
              "Otherwise, there is no blocking and a whole column shares one scaling factor. ",
              AttributeProto::INT,
              OPTIONAL_VALUE)
        .Attr("swiglu_interleaved",
              "Whether to use interleaved SwiGLU.",
              AttributeProto::INT,
              static_cast<int64_t>(0))
        .Input(0,
               "input",
               "2D tensor with shape (num_tokens, hidden_size), or "
               "3D tensor with shape (batch_size, sequence_length, hidden_size)",
               "T")
        .Input(1,
               "router_probs",
               "2D tensor with shape (num_tokens, num_experts)",
               "T")
        .Input(2,
               "fc1_experts_weights",
               "3D tensor with shape (num_experts, fusion_size * inter_size, hidden_size / pack_size), "
               "The fusion_size is 2 for fused swiglu, or 1 otherwise. The pack_size is 8 / expert_weight_bits.",
               "T1")
        .Input(3,
               "fc1_scales",
               "2D tensor with shape (num_experts, fusion_size * inter_size), or "
               "3D tensor with shape (num_experts, fusion_size * inter_size, hidden_size / block_size) when block_size is provided.",
               "T2")
        .Input(4,
               "fc1_experts_bias",
               "2D optional tensor with shape (num_experts, fusion_size * inter_size)", "T", OpSchema::Optional)
        .Input(5,
               "fc2_experts_weights",
               "3D tensor with shape (num_experts, hidden_size, inter_size / pack_size)",
               "T1")
        .Input(6,
               "fc2_scales",
               "2D tensor with shape (num_experts, hidden_size), or "
               "3D tensor with shape (num_experts, hidden_size, inter_size / block_size) when block_size is provided.",
               "T2")
        .Input(7,
               "fc2_experts_bias",
               "2D optional tensor with shape (num_experts, hidden_size)",
               "T",
               OpSchema::Optional)
        .Input(8,
               "fc3_experts_weights",
               "3D optional tensor with shape (num_experts, inter_size, hidden_size / pack_size)",
               "T1",
               OpSchema::Optional)
        .Input(9,
               "fc3_scales",
               "2D optional tensor with shape (num_experts, inter_size), or "
               "3D optional tensor with shape (num_experts, inter_size, hidden_size / block_size) when block_size is provided.",
               "T2",
               OpSchema::Optional)
        .Input(10,
               "fc3_experts_bias",
               "2D optional tensor with shape (num_experts, inter_size)",
               "T",
               OpSchema::Optional)
        .Input(11,
               "fc1_zero_points",
               "2D tensor with shape (num_experts, fusion_size * inter_size / pack_size), or "
               "3D tensor with shape (num_experts, fusion_size * inter_size, hidden_size / block_size / pack_size) when block_size is provided.",
               "T1",
               OpSchema::Optional)
        .Input(12,
               "fc2_zero_points",
               "2D tensor with shape (num_experts, hidden_size / pack_size), or "
               "3D tensor with shape (num_experts, hidden_size, inter_size / block_size / pack_size) when block_size is provided.",
               "T1",
               OpSchema::Optional)
        .Input(13,
               "fc3_zero_points",
               "2D optional tensor with shape (num_experts, inter_size / pack_size), or "
               "3D optional tensor with shape (num_experts, inter_size, hidden_size / block_size / pack_size) when block_size is provided.",
               "T1",
               OpSchema::Optional)
        .Output(0,
                "output",
                "output tensor with same shape of input",
                "T")
        .TypeConstraint("T", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain input and output types to float tensors.")
        .TypeConstraint("T1", {"tensor(uint8)"}, "Constrain weights type to uint8 tensors.")
        .TypeConstraint("T2", {"tensor(float)", "tensor(float16)", "tensor(bfloat16)"}, "Constrain scales type to float tensors.")
        .TypeAndShapeInferenceFunction(ONNX_NAMESPACE::propagateShapeAndTypeFromFirstInput));

# Source Code Location
The following path is related to /home/tlwu/onnxruntime/

Tensor Shape Checking: onnxruntime/contrib_ops/cpu/moe/moe_helper.h
MOE CUDA implementation: onnxruntime/contrib_ops/cuda/moe/moe.cc
Quantized MOE CUDA implementation: onnxruntime/contrib_ops/cuda/moe/moe_quantization.cc
Quantized MOE cutlass Kernel: onnxruntime/contrib_ops/cuda/llm/moe_gemm/
                              onnxruntime/contrib_ops/cuda/llm/cutlass_extensions/
                              onnxruntime/contrib_ops/cuda/llm/common/
Tests: onnxruntime/test/python/transformers/test_moe_cuda.py
       onnxruntime/test/python/transformers/test_qmoe_cuda.py

# Related Code for Reference
The following path is related to /home/tlwu/onnxruntime/

Quantized MOE CPU implementation: onnxruntime/contrib_ops/cpu/moe/moe_quantization_cpu.cc
TensorRT-LLM MOE plugin: TensorRT-LLM/cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.cpp

Our code of cutlass kernel is from older version of TensorRT LLM.Here is newer version of cutlass kernel:
TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/moe_gemm
TensorRT-LLM/cpp/tensorrt_llm/cutlass_extensions

# Format Code

Please run the following command to format code:
lintrunner -a
It will automatically fix most of the warnings.

You can run it again to verify that all warnings are fixed. If not, please modify the code to fix the warnings until
`lintrunner -a` command does not show any warnings.

# How to build and install wheel

It is recommended to use run.sh to build and install wheel:
./run.sh --build --install

Or the following to clean moe related build artifacts and then build and install wheel:
./run.sh --build --install --clean_moe

If you need clean all build artifacts (rm -rf build) and then build and install wheel:
./run.sh --clean --build --install

If you want to enable debug node inputs and outputs (You may add --clean if previous build has no --dump option):
./run.sh --build --install --dump

# How to test
run test_moe_cuda.py:
./run.sh --test_moe

run test_qmoe_cuda.py:
./run.sh --test_qmoe

If you need to run a specific test case, you can use --test_moe_case or --test_qmoe_case option like:
./run.sh --test_qmoe_case TestSwigluQMoE.test_swiglu_qmoe_blockwise_parity_3

If you want to enable debug node inputs and outputs (Current build need to be built with --dump option):
./run.sh --test_qmoe_case TestSwigluQMoE.test_swiglu_qmoe_blockwise_parity_3 --dump
