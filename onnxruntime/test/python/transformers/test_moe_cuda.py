import os
import sys
import unittest

import numpy as np
import torch
import torch.nn as nn
from onnx import TensorProto, helper, numpy_helper

import onnxruntime

# Add the directory containing test_qmoe_cuda.py to path to reuse MoE reference if needed
sys.path.append(os.path.dirname(__file__))


class MoE(nn.Module):
    def __init__(self, num_experts, num_rows, hidden_size, inter_size, topk, activation_type="swiglu"):
        super().__init__()
        self.num_experts = num_experts
        self.num_rows = num_rows
        self.hidden_size = hidden_size
        self.inter_size = inter_size
        self.topk = topk
        self.activation_type = activation_type

        # Experts
        self.fc1 = nn.Parameter(torch.empty(num_experts, hidden_size, inter_size))
        self.fc2 = nn.Parameter(torch.empty(num_experts, inter_size, hidden_size))
        self.fc3 = nn.Parameter(torch.empty(num_experts, hidden_size, inter_size))  # For SwiGLU

        nn.init.uniform_(self.fc1, -0.1, 0.1)
        nn.init.uniform_(self.fc2, -0.1, 0.1)
        nn.init.uniform_(self.fc3, -0.1, 0.1)

        # Biases (optional in ORT, but lets stick to simple first)
        self.use_bias = False

    def forward(self, x, router_probs):
        # x: [num_rows, hidden_size]
        # router_probs: [num_rows, num_experts]

        # Simple reference implementation:
        # 1. Routing (ORT applies softmax internally)
        # 2. Expert computation
        # 3. Aggregation

        # ORT treats router_probs as logits and applies softmax internally
        router_softmax = torch.softmax(router_probs.float(), dim=1).to(router_probs.dtype)

        # Select topk from softmax probabilities
        topk_probs, topk_indices = torch.topk(router_softmax, self.topk, dim=1)
        topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)  # normalize_routing_weights=1

        # Compute

        # Naive implementation loop over samples
        # Ideally vectorised but loop is fine for reference correctness

        # Or loop over experts (easier)
        # Using a mask approach

        final_output = torch.zeros_like(x)

        for k in range(self.topk):
            expert_idx = topk_indices[:, k]  # [num_rows]
            prob = topk_probs[:, k].unsqueeze(1)  # [num_rows, 1]

            # For each active expert
            # Only process rows that selected this expert?
            # Or simpler: process all samples with expert E if E is in topk[sample]

            # Let's iterate over ALL experts
            for e in range(self.num_experts):
                mask = expert_idx == e
                if mask.sum() == 0:
                    continue

                inp = x[mask]

                # Forward pass for expert e
                w1 = self.fc1[e]
                w2 = self.fc2[e]

                h1 = torch.matmul(inp, w1)

                if self.activation_type == "swiglu":
                    # SwiGLU: SiLU(x @ W1) * (x @ W3) @ W2
                    # W1 is Gate, W3 is Up
                    w3 = self.fc3[e]
                    h3 = torch.matmul(inp, w3)
                    act = torch.nn.functional.silu(h1) * h3
                elif self.activation_type == "identity":
                    act = h1
                else:
                    raise ValueError(f"Unsupported activation: {self.activation_type}")

                h2 = torch.matmul(act, w2)

                final_output[mask] += h2 * prob[mask]

        return final_output


class TestMoECuda(unittest.TestCase):
    def test_moe_cuda_swiglu(self):
        num_experts = 4
        num_rows = 16
        hidden_size = 128
        inter_size = 256
        topk = 2

        # Inputs
        input_data = torch.randn(num_rows, hidden_size).cuda().half()
        router_probs = torch.randn(num_rows, num_experts).cuda().half()

        # Model with SwiGLU
        model = MoE(num_experts, num_rows, hidden_size, inter_size, topk, activation_type="swiglu").cuda().half()

        # Reference output
        ref_out = model(input_data, router_probs)

        onnx_input = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [num_rows, hidden_size])
        onnx_router = helper.make_tensor_value_info("router_probs", TensorProto.FLOAT16, [num_rows, num_experts])

        # Prepare weights for SwiGLU (Interleaved: swiglu_fusion=1)
        # CUDA kernel expects interleaved weights: Gate at even indices, Linear at odd indices
        # fc1 has shape [E, H, I] (gate), fc3 has shape [E, H, I] (linear)
        fc1_np = model.fc1.detach().cpu().numpy().astype(np.float16)  # [E, H, I] (gate)
        fc2_np = model.fc2.detach().cpu().numpy().astype(np.float16)  # [E, I, H]
        fc3_np = model.fc3.detach().cpu().numpy().astype(np.float16)  # [E, H, I] (up/linear)

        # Interleave weights: [Gate, Linear, Gate, Linear, ...]
        E, H, I = fc1_np.shape
        fc1_fused_np = np.empty((E, H, 2 * I), dtype=np.float16)
        fc1_fused_np[:, :, 0::2] = fc1_np  # Gate
        fc1_fused_np[:, :, 1::2] = fc3_np  # Linear

        # Schema expects FC1 as [E, fusion_size * inter_size, hidden_size]
        # Current fc1_fused_np is [E, H, 2*I]. Need to transpose last two dims.
        fc1_tensor_np = fc1_fused_np.transpose(0, 2, 1)  # [E, 2*I, H]
        # Schema expects FC2 as [E, hidden_size, inter_size]
        # Current fc2_np is [E, I, H] (from torch reference). Need to transpose.
        fc2_tensor_np = fc2_np.transpose(0, 2, 1)

        fc1_tensor = numpy_helper.from_array(fc1_tensor_np, name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_tensor_np, name="fc2")

        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="swiglu",
            activation_alpha=1.0,
            activation_beta=0.0,
            swiglu_fusion=1,  # Fused interleaved
            normalize_routing_weights=1,
        )

        graph = helper.make_graph(
            [node],
            "moe_test_swiglu",
            [onnx_input, onnx_router],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [num_rows, hidden_size])],
            [fc1_tensor, fc2_tensor],
        )

        op = helper.make_opsetid("com.microsoft", 1)
        model_proto = helper.make_model(graph, opset_imports=[op, helper.make_opsetid("", 14)])

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 0
        providers = ["CUDAExecutionProvider"]

        session = onnxruntime.InferenceSession(model_proto.SerializeToString(), sess_options, providers=providers)

        inputs = {
            "input": input_data.cpu().numpy().astype(np.float16),
            "router_probs": router_probs.cpu().numpy().astype(np.float16),
        }

        output = session.run(["output"], inputs)[0]

        # Verify
        print("Comparing SwiGLU outputs...", flush=True)
        ref_np = ref_out.detach().cpu().numpy()

        np.testing.assert_allclose(output, ref_np, rtol=1e-2, atol=1e-2)
        print("SwiGLU Success!", flush=True)

    def test_moe_cuda_topk1(self):
        # Isolate routing logic for k=1
        num_experts = 4
        num_rows = 16
        hidden_size = 128
        inter_size = 256
        topk = 1

        # Inputs
        input_data = torch.randn(num_rows, hidden_size).cuda().half()
        router_probs = torch.randn(num_rows, num_experts).cuda().half()

        # Model with SwiGLU
        model = MoE(num_experts, num_rows, hidden_size, inter_size, topk, activation_type="swiglu").cuda().half()

        # Reference output
        ref_out = model(input_data, router_probs)

        onnx_input = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [num_rows, hidden_size])
        onnx_router = helper.make_tensor_value_info("router_probs", TensorProto.FLOAT16, [num_rows, num_experts])

        fc1_np = model.fc1.detach().cpu().numpy().astype(np.float16)  # [E, H, I] (gate)
        fc2_np = model.fc2.detach().cpu().numpy().astype(np.float16)  # [E, I, H]
        fc3_np = model.fc3.detach().cpu().numpy().astype(np.float16)  # [E, H, I] (up/linear)

        # Interleave weights: [Gate, Linear, Gate, Linear, ...]
        E, H, I = fc1_np.shape
        fc1_fused_np = np.empty((E, H, 2 * I), dtype=np.float16)
        fc1_fused_np[:, :, 0::2] = fc1_np  # Gate
        fc1_fused_np[:, :, 1::2] = fc3_np  # Linear

        # Transpose to match Schema (E, 2*I, H) and (E, H, I)
        fc1_tensor_np = fc1_fused_np.transpose(0, 2, 1)
        fc2_tensor_np = fc2_np.transpose(0, 2, 1)

        fc1_tensor = numpy_helper.from_array(fc1_tensor_np, name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_tensor_np, name="fc2")

        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="swiglu",
            swiglu_fusion=1,
            normalize_routing_weights=1,
        )

        graph = helper.make_graph(
            [node],
            "moe_test_topk1",
            [onnx_input, onnx_router],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [num_rows, hidden_size])],
            [fc1_tensor, fc2_tensor],
        )

        op = helper.make_opsetid("com.microsoft", 1)
        model_proto = helper.make_model(graph, opset_imports=[op, helper.make_opsetid("", 14)])

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 0
        providers = ["CUDAExecutionProvider"]

        session = onnxruntime.InferenceSession(model_proto.SerializeToString(), sess_options, providers=providers)

        inputs = {
            "input": input_data.cpu().numpy().astype(np.float16),
            "router_probs": router_probs.cpu().numpy().astype(np.float16),
        }

        output = session.run(["output"], inputs)[0]

        # Verify
        print("Comparing TopK=1 outputs...", flush=True)
        ref_np = ref_out.detach().cpu().numpy()

        np.testing.assert_allclose(output, ref_np, rtol=1e-2, atol=1e-2)
        print("TopK=1 Success!", flush=True)

    def test_moe_cuda(self):
        self._test_moe_cuda_generic(dtype=np.float16)

    def test_moe_cuda_float32(self):
        self._test_moe_cuda_generic(dtype=np.float32)

    def _test_moe_cuda_generic(self, dtype):
        num_experts = 4
        num_rows = 16
        hidden_size = 128
        inter_size = 256
        topk = 2

        onnx_dtype = TensorProto.FLOAT if dtype == np.float32 else TensorProto.FLOAT16
        torch_dtype = torch.float32 if dtype == np.float32 else torch.float16

        np.random.seed(0)
        torch.manual_seed(0)
        input_data = torch.randn(num_rows, hidden_size).cuda().to(torch_dtype)
        router_probs = torch.randn(num_rows, num_experts).cuda().to(torch_dtype)

        # Naive Reference
        # Note: using default MoE attributes (identity implied if activation_type default or handled)
        # The original code passed no activation_type, so it uses default.
        # Actually MoE class defaults to 'relu' if not specified? Let's check imports.
        # Wait, the MoE class definition in this file?
        # No, it is imported or defined above. Assuming standard definition.
        # We will use explicit 'identity' for clarity if possible, but strict match to old test used none.
        model = (
            MoE(num_experts, num_rows, hidden_size, inter_size, topk, activation_type="identity").cuda().to(torch_dtype)
        )

        ref_out = model(input_data, router_probs)

        fc1_np = model.fc1.detach().cpu().numpy().astype(dtype)
        fc2_np = model.fc2.detach().cpu().numpy().astype(dtype)

        # Transpose to match Schema (E, I, H) and (E, H, I) - wait, FC2 schema is (E, H, I)?
        # Let's recheck schema.
        # Input(2, fc1): [E, fusion*I, H].
        # Input(4, fc2): [E, H, I].
        # In generic test, activation='identity', so fusion=1.
        # FC1 PyTorch: [E, H, I]. Need [E, I, H]. Transpose.
        # FC2 PyTorch: [E, I, H]. Need [E, H, I].
        # Wait. In generic test:
        # self.fc1 = [E, H, I]. matmul(x, w1). w1=[H, I].
        # self.fc2 = [E, I, H]. matmul(act, w2). w2=[I, H].
        # So FC2 PyTorch is [E, I, H].
        # Schema for FC2 is [E, H, I].
        # So BOTH need transpose.

        fc1_tensor = numpy_helper.from_array(fc1_np.transpose(0, 2, 1), name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_np.transpose(0, 2, 1), name="fc2")

        onnx_input = helper.make_tensor_value_info("input", onnx_dtype, [num_rows, hidden_size])
        onnx_router = helper.make_tensor_value_info("router_probs", onnx_dtype, [num_rows, num_experts])

        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="identity",
            normalize_routing_weights=1,
        )

        graph = helper.make_graph(
            [node],
            "moe_test",
            [onnx_input, onnx_router],
            [helper.make_tensor_value_info("output", onnx_dtype, [num_rows, hidden_size])],
            [fc1_tensor, fc2_tensor],
        )

        op = helper.make_opsetid("com.microsoft", 1)
        model_proto = helper.make_model(graph, opset_imports=[op, helper.make_opsetid("", 14)])

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 0
        providers = ["CUDAExecutionProvider"]

        session = onnxruntime.InferenceSession(model_proto.SerializeToString(), sess_options, providers=providers)
        print(f"Session Providers: {session.get_providers()}", flush=True)

        inputs = {
            "input": input_data.cpu().numpy().astype(dtype),
            "router_probs": router_probs.cpu().numpy().astype(dtype),
        }

        output = session.run(["output"], inputs)[0]

        print(f"Comparing outputs for {dtype}...", flush=True)
        ref_np = ref_out.detach().cpu().numpy()
        np.testing.assert_allclose(output, ref_np, rtol=1e-2, atol=1e-2)
        print("Success!", flush=True)

    def test_moe_cuda_debug(self):
        """Minimal test case for debugging weight layout.

        Uses:
        - 1 expert (simplest routing)
        - topk=1 (no mixing)
        - Small dimensions
        - Identity activation (no nonlinearity)
        - Identity-like weights for GEMM1 (to isolate FC2 behavior)
        """
        num_experts = 1
        num_rows = 2
        hidden_size = 8  # Must be multiple of 8 for FP16 alignment
        inter_size = 8
        topk = 1

        # Simple input: just ones
        input_data = torch.ones(num_rows, hidden_size).cuda().half()

        # Router probs: all go to expert 0
        router_probs = torch.ones(num_rows, num_experts).cuda().half()

        # FC1: Identity matrix [E=1, H=4, I=4] -> becomes [1, 4, 4]
        # After transpose for ORT: [1, I=4, H=4]
        fc1_torch = torch.eye(hidden_size, inter_size).unsqueeze(0).cuda().half()

        # FC2: Scale by 2 [E=1, I=4, H=4] -> after transpose [1, H=4, I=4]
        fc2_torch = (2 * torch.eye(inter_size, hidden_size)).unsqueeze(0).cuda().half()

        # Expected output with Identity activation:
        # x @ FC1 @ FC2 = x @ I @ 2I = 2x
        # With normalize_routing_weights=1 and topk=1, prob=1.0
        expected = 2 * input_data

        # --- ORT Setup ---
        # Schema expects: fc1=[E, I, H], fc2=[E, H, I]
        # PyTorch defines: fc1=[E, H, I], fc2=[E, I, H]
        # So we transpose last two dims
        fc1_np = fc1_torch.cpu().numpy().transpose(0, 2, 1).astype(np.float16)
        fc2_np = fc2_torch.cpu().numpy().transpose(0, 2, 1).astype(np.float16)

        print(f"FC1 shape (ORT): {fc1_np.shape}")  # Should be [1, 4, 4]
        print(f"FC2 shape (ORT): {fc2_np.shape}")  # Should be [1, 4, 4]
        print(f"FC1[0]:\n{fc1_np[0]}")
        print(f"FC2[0]:\n{fc2_np[0]}")

        fc1_tensor = numpy_helper.from_array(fc1_np, name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_np, name="fc2")

        onnx_input = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [num_rows, hidden_size])
        onnx_router = helper.make_tensor_value_info("router_probs", TensorProto.FLOAT16, [num_rows, num_experts])

        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="identity",
            normalize_routing_weights=1,
        )

        graph = helper.make_graph(
            [node],
            "moe_debug_test",
            [onnx_input, onnx_router],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [num_rows, hidden_size])],
            [fc1_tensor, fc2_tensor],
        )

        op = helper.make_opsetid("com.microsoft", 1)
        model_proto = helper.make_model(graph, opset_imports=[op, helper.make_opsetid("", 14)])

        sess_options = onnxruntime.SessionOptions()
        # Enable verbose logging
        sess_options.log_severity_level = 0
        providers = ["CUDAExecutionProvider"]

        session = onnxruntime.InferenceSession(model_proto.SerializeToString(), sess_options, providers=providers)
        print(f"Session Providers: {session.get_providers()}", flush=True)

        inputs = {
            "input": input_data.cpu().numpy().astype(np.float16),
            "router_probs": router_probs.cpu().numpy().astype(np.float16),
        }

        output = session.run(["output"], inputs)[0]

        print(f"Input:\n{input_data.cpu().numpy()}")
        print(f"Expected (2*input):\n{expected.cpu().numpy()}")
        print(f"ORT Output:\n{output}")
        print(f"Diff:\n{output - expected.cpu().numpy()}")

        np.testing.assert_allclose(output, expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
        print("Debug test passed!", flush=True)

    def test_moe_cuda_debug_random(self):
        """Debug test with random weights but still 1 expert.

        This isolates the weight layout issue from multi-expert routing.
        """
        num_experts = 1
        num_rows = 4
        hidden_size = 64
        inter_size = 128
        topk = 1

        torch.manual_seed(42)
        np.random.seed(42)

        # Random input
        input_data = torch.randn(num_rows, hidden_size).cuda().half()

        # All go to expert 0
        router_probs = torch.ones(num_rows, num_experts).cuda().half()

        # Random weights (PyTorch layout)
        # fc1: [E, H, I] - used as x @ W1
        # fc2: [E, I, H] - used as act @ W2
        fc1_torch = torch.randn(num_experts, hidden_size, inter_size).cuda().half() * 0.1
        fc2_torch = torch.randn(num_experts, inter_size, hidden_size).cuda().half() * 0.1

        # Expected output with Identity activation:
        # output = x @ FC1 @ FC2
        # Since topk=1 and only 1 expert, prob=1.0
        expected = input_data @ fc1_torch[0] @ fc2_torch[0]

        # --- ORT Setup ---
        # Schema expects: fc1=[E, I, H], fc2=[E, H, I]
        # We transpose last two dims: [E, H, I] -> [E, I, H]
        fc1_np = fc1_torch.cpu().numpy().transpose(0, 2, 1).astype(np.float16)
        fc2_np = fc2_torch.cpu().numpy().transpose(0, 2, 1).astype(np.float16)

        print(f"FC1 shape (PyTorch): {fc1_torch.shape} -> ORT: {fc1_np.shape}")
        print(f"FC2 shape (PyTorch): {fc2_torch.shape} -> ORT: {fc2_np.shape}")

        fc1_tensor = numpy_helper.from_array(fc1_np, name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_np, name="fc2")

        onnx_input = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [num_rows, hidden_size])
        onnx_router = helper.make_tensor_value_info("router_probs", TensorProto.FLOAT16, [num_rows, num_experts])

        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="identity",
            normalize_routing_weights=1,
        )

        graph = helper.make_graph(
            [node],
            "moe_debug_random_test",
            [onnx_input, onnx_router],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [num_rows, hidden_size])],
            [fc1_tensor, fc2_tensor],
        )

        op = helper.make_opsetid("com.microsoft", 1)
        model_proto = helper.make_model(graph, opset_imports=[op, helper.make_opsetid("", 14)])

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 4  # Reduce verbosity
        providers = ["CUDAExecutionProvider"]

        session = onnxruntime.InferenceSession(model_proto.SerializeToString(), sess_options, providers=providers)
        print(f"Session Providers: {session.get_providers()}", flush=True)

        inputs = {
            "input": input_data.cpu().numpy().astype(np.float16),
            "router_probs": router_probs.cpu().numpy().astype(np.float16),
        }

        output = session.run(["output"], inputs)[0]

        print(f"Expected[0,:8]: {expected.cpu().numpy()[0, :8]}")
        print(f"ORT Output[0,:8]: {output[0, :8]}")
        print(f"Diff[0,:8]: {(output - expected.cpu().numpy())[0, :8]}")

        # Check total mismatch
        diff = np.abs(output - expected.cpu().numpy())
        mismatch_ratio = np.sum(diff > 0.01) / diff.size
        print(f"Mismatch ratio: {mismatch_ratio:.1%}")
        print(f"Max diff: {diff.max():.4f}")

        np.testing.assert_allclose(output, expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
        print("Debug random test passed!", flush=True)

    def test_moe_cuda_debug_multiexpert(self):
        """Debug test with multiple experts and topk=1.

        Each row explicitly routed to a specific expert.
        """
        num_experts = 4
        num_rows = 4  # 1 row per expert for easy verification
        hidden_size = 64
        inter_size = 128
        topk = 1

        torch.manual_seed(42)
        np.random.seed(42)

        # Random input
        input_data = torch.randn(num_rows, hidden_size).cuda().half()

        # Explicit routing: row i goes to expert i
        router_probs = torch.zeros(num_rows, num_experts).cuda().half()
        for i in range(num_rows):
            router_probs[i, i % num_experts] = 1.0

        # Random weights (PyTorch layout)
        fc1_torch = torch.randn(num_experts, hidden_size, inter_size).cuda().half() * 0.1
        fc2_torch = torch.randn(num_experts, inter_size, hidden_size).cuda().half() * 0.1

        # Expected output with Identity activation:
        # Each row i uses expert i
        expected = torch.zeros_like(input_data)
        for i in range(num_rows):
            expert_idx = i % num_experts
            expected[i] = input_data[i] @ fc1_torch[expert_idx] @ fc2_torch[expert_idx]

        # --- ORT Setup ---
        fc1_np = fc1_torch.cpu().numpy().transpose(0, 2, 1).astype(np.float16)
        fc2_np = fc2_torch.cpu().numpy().transpose(0, 2, 1).astype(np.float16)

        fc1_tensor = numpy_helper.from_array(fc1_np, name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_np, name="fc2")

        onnx_input = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [num_rows, hidden_size])
        onnx_router = helper.make_tensor_value_info("router_probs", TensorProto.FLOAT16, [num_rows, num_experts])

        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="identity",
            normalize_routing_weights=1,
        )

        graph = helper.make_graph(
            [node],
            "moe_debug_multiexpert_test",
            [onnx_input, onnx_router],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [num_rows, hidden_size])],
            [fc1_tensor, fc2_tensor],
        )

        op = helper.make_opsetid("com.microsoft", 1)
        model_proto = helper.make_model(graph, opset_imports=[op, helper.make_opsetid("", 14)])

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 4
        providers = ["CUDAExecutionProvider"]

        session = onnxruntime.InferenceSession(model_proto.SerializeToString(), sess_options, providers=providers)
        print(f"Session Providers: {session.get_providers()}", flush=True)

        inputs = {
            "input": input_data.cpu().numpy().astype(np.float16),
            "router_probs": router_probs.cpu().numpy().astype(np.float16),
        }

        output = session.run(["output"], inputs)[0]

        print("Router probs:\n", router_probs.cpu().numpy())
        for i in range(num_rows):
            expert_idx = i % num_experts
            print(f"Row {i} -> Expert {expert_idx}:")
            print(f"  Expected[:8]: {expected.cpu().numpy()[i, :8]}")
            print(f"  ORT Out[:8]:  {output[i, :8]}")
            print(f"  Diff[:8]:     {(output - expected.cpu().numpy())[i, :8]}")

        diff = np.abs(output - expected.cpu().numpy())
        mismatch_ratio = np.sum(diff > 0.01) / diff.size
        print(f"Mismatch ratio: {mismatch_ratio:.1%}")
        print(f"Max diff: {diff.max():.4f}")

        np.testing.assert_allclose(output, expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
        print("Debug multiexpert test passed!", flush=True)

    def test_moe_cuda_debug_topk2(self):
        """Debug test with topk=2 to match original test configuration."""
        num_experts = 4
        num_rows = 8
        hidden_size = 64
        inter_size = 128
        topk = 2

        torch.manual_seed(42)
        np.random.seed(42)

        input_data = torch.randn(num_rows, hidden_size).cuda().half()
        router_probs = torch.randn(num_rows, num_experts).cuda().half()

        fc1_torch = torch.randn(num_experts, hidden_size, inter_size).cuda().half() * 0.1
        fc2_torch = torch.randn(num_experts, inter_size, hidden_size).cuda().half() * 0.1

        # ORT applies softmax to router_probs internally!
        # We need to do the same for the reference calculation.
        router_softmax = torch.softmax(router_probs.float(), dim=1).half()

        # Compute expected with same logic as ORT
        topk_probs, topk_indices = torch.topk(router_softmax, topk, dim=1)
        topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)  # normalize_routing_weights=1

        expected = torch.zeros_like(input_data)
        for k in range(topk):
            expert_idx = topk_indices[:, k]
            prob = topk_probs[:, k].unsqueeze(1)

            for e in range(num_experts):
                mask = expert_idx == e
                if mask.sum() == 0:
                    continue
                inp = input_data[mask]
                h1 = inp @ fc1_torch[e]  # Identity activation
                h2 = h1 @ fc2_torch[e]
                expected[mask] += h2 * prob[mask]

        # ORT setup
        fc1_np = fc1_torch.cpu().numpy().transpose(0, 2, 1).astype(np.float16)
        fc2_np = fc2_torch.cpu().numpy().transpose(0, 2, 1).astype(np.float16)

        fc1_tensor = numpy_helper.from_array(fc1_np, name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_np, name="fc2")

        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="identity",
            normalize_routing_weights=1,
        )

        graph = helper.make_graph(
            [node],
            "moe_debug_topk2_test",
            [
                helper.make_tensor_value_info("input", TensorProto.FLOAT16, [num_rows, hidden_size]),
                helper.make_tensor_value_info("router_probs", TensorProto.FLOAT16, [num_rows, num_experts]),
            ],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT16, [num_rows, hidden_size])],
            [fc1_tensor, fc2_tensor],
        )

        model_proto = helper.make_model(
            graph, opset_imports=[helper.make_opsetid("com.microsoft", 1), helper.make_opsetid("", 14)]
        )

        sess_options = onnxruntime.SessionOptions()
        sess_options.log_severity_level = 4
        session = onnxruntime.InferenceSession(
            model_proto.SerializeToString(), sess_options, providers=["CUDAExecutionProvider"]
        )

        print(f"Session Providers: {session.get_providers()}", flush=True)

        output = session.run(
            ["output"],
            {
                "input": input_data.cpu().numpy().astype(np.float16),
                "router_probs": router_probs.cpu().numpy().astype(np.float16),
            },
        )[0]

        print(f"TopK indices:\n{topk_indices.cpu().numpy()}")
        print(f"TopK probs (normalized):\n{topk_probs.cpu().numpy()}")
        print(f"Expected[0,:8]: {expected.cpu().numpy()[0, :8]}")
        print(f"ORT Output[0,:8]: {output[0, :8]}")

        diff = np.abs(output - expected.cpu().numpy())
        mismatch_ratio = np.sum(diff > 0.01) / diff.size
        print(f"Mismatch ratio: {mismatch_ratio:.1%}")
        print(f"Max diff: {diff.max():.4f}")

        np.testing.assert_allclose(output, expected.cpu().numpy(), rtol=1e-2, atol=1e-2)
        print("Debug topk2 test passed!", flush=True)


if __name__ == "__main__":
    unittest.main()
