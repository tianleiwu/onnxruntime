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
        # 1. Routing
        # 2. Expert computation
        # 3. Aggregation

        # Select topk
        topk_probs, topk_indices = torch.topk(router_probs, self.topk, dim=1)
        topk_probs = topk_probs / topk_probs.sum(dim=1, keepdim=True)  # Normalize

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

        # Prepare weights for SwiGLU
        # fc1: gate, fc3: up. Fused as [num_experts, hidden_size, 2*inter_size]
        fc1_np = model.fc1.detach().cpu().numpy().astype(np.float16)  # [E, H, I]
        fc2_np = model.fc2.detach().cpu().numpy().astype(np.float16)  # [E, I, H]
        fc3_np = model.fc3.detach().cpu().numpy().astype(np.float16)  # [E, H, I]

        # Concatenate fc1 and fc3 along last dim for ORT
        fc1_fused_np = np.concatenate([fc1_np, fc3_np], axis=-1)

        fc1_tensor = numpy_helper.from_array(fc1_fused_np, name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_np, name="fc2")

        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="swiglu",
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

        fc1_np = model.fc1.detach().cpu().numpy().astype(np.float16)
        fc2_np = model.fc2.detach().cpu().numpy().astype(np.float16)
        fc3_np = model.fc3.detach().cpu().numpy().astype(np.float16)

        # Pack fc1 and fc3 together: [E, H, 2*I]
        fc1_fused_np = np.concatenate([fc1_np, fc3_np], axis=-1)

        fc1_tensor = numpy_helper.from_array(fc1_fused_np, name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_np, name="fc2")

        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="swiglu",
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
        num_experts = 4
        num_rows = 16
        hidden_size = 128  # Must be >= TileK (64)
        inter_size = 256  # After swiglu halving (512/2=256), must meet TileN requirements
        topk = 2

        # Inputs
        input_data = torch.randn(num_rows, hidden_size).cuda().half()
        router_probs = torch.randn(num_rows, num_experts).cuda().half()

        # Model
        model = MoE(num_experts, num_rows, hidden_size, inter_size, topk).cuda().half()

        # Reference output
        ref_out = model(input_data, router_probs)

        # Create ONNX model
        # MoE input order:
        # input, router_probs, fc1, fc1_bias, fc2, fc2_bias, fc3, fc3_bias

        # Weights need to be passed as tensors
        # Shape expectations for ORT MoE?
        # Likely: [num_experts, hidden_size, inter_size] -> matches PyTorch logic approximately
        # But ORT often takes [num_experts * inter_size, hidden_size] or similar?
        # moe.cc CheckInputs will tell, but let's try assuming standard 3D or flattened 2D.
        # MoE usually expects: FC1 [num_experts, hidden_size, inter_size]

        onnx_input = helper.make_tensor_value_info("input", TensorProto.FLOAT16, [num_rows, hidden_size])
        onnx_router = helper.make_tensor_value_info("router_probs", TensorProto.FLOAT16, [num_rows, num_experts])

        # ORT Inputs for swiglu_fusion:
        # 0: input
        # 1: router_probs
        # 2: fc1 (fused gate+up projection) [num_experts, hidden_size, 2*inter_size]
        # 3: fc1_bias (optional)
        # 4: fc2
        # 5: fc2_bias (optional)
        # 6: fc3 - empty for swiglu_fusion
        # 7: fc3_bias - empty for swiglu_fusion

        # For non-fused path, use separate fc1 and fc2
        fc1_np = model.fc1.detach().cpu().numpy().astype(np.float16)
        fc2_np = model.fc2.detach().cpu().numpy().astype(np.float16)

        fc1_tensor = numpy_helper.from_array(fc1_np, name="fc1")
        fc2_tensor = numpy_helper.from_array(fc2_np, name="fc2")

        # Use basic MoE without gated activation for testing
        # swiglu_fusion=0 uses separate fc1/fc2 passes with default activation
        node = helper.make_node(
            "MoE",
            ["input", "router_probs", "fc1", "", "fc2", "", "", ""],
            ["output"],
            domain="com.microsoft",
            k=topk,
            activation_type="identity",  # No fused activation, just linear
            normalize_routing_weights=1,
        )

        graph = helper.make_graph(
            [node],
            "moe_test",
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

        print(f"Session Providers: {session.get_providers()}", flush=True)

        inputs = {
            "input": input_data.cpu().numpy().astype(np.float16),
            "router_probs": router_probs.cpu().numpy().astype(np.float16),
        }

        output = session.run(["output"], inputs)[0]

        # Verify
        print("Comparing outputs...", flush=True)
        ref_np = ref_out.detach().cpu().numpy()

        np.testing.assert_allclose(output, ref_np, rtol=1e-2, atol=1e-2)
        print("Success!", flush=True)


if __name__ == "__main__":
    unittest.main()
