# --------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
import unittest

import numpy
import torch
import torch.nn.functional as F
from onnx import TensorProto, helper
from parameterized import parameterized
from torch import nn

import onnxruntime

torch.manual_seed(42)
numpy.random.seed(42)

USE_QUANT = False
ORT_DTYPE = TensorProto.FLOAT16 if USE_QUANT else TensorProto.FLOAT
NP_TYPE = numpy.float16 if ORT_DTYPE == TensorProto.FLOAT16 else numpy.float32
THRESHOLD = 5e-1 if USE_QUANT else 1e-2


class SparseMoeBlockORTHelper(nn.Module):
    def __init__(self):
        super().__init__()

    def create_ort_session(self, moe_onnx_graph):
        from onnxruntime import InferenceSession, SessionOptions  # noqa: PLC0415

        sess_options = SessionOptions()

        cuda_providers = ["CUDAExecutionProvider"]
        if cuda_providers[0] not in onnxruntime.get_available_providers():
            return None

        sess_options.log_severity_level = 2
        ort_session = InferenceSession(moe_onnx_graph, sess_options, providers=["CUDAExecutionProvider"])

        return ort_session

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        pass

    def ort_forward(self, hidden_states: torch.Tensor, iobinding=False) -> torch.Tensor:
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        # router_logits: (batch * sequence_length, n_experts)
        router_logits = self.gate(hidden_states)

        ort_inputs = {
            "input": numpy.ascontiguousarray(hidden_states.detach().numpy().astype(NP_TYPE)),
            "router_probs": numpy.ascontiguousarray(router_logits.detach().numpy().astype(NP_TYPE)),
        }

        ort_output = None
        if self.ort_sess is not None:
            if not iobinding:
                ort_output = self.ort_sess.run(None, ort_inputs)
                return torch.tensor(ort_output).reshape(batch_size, sequence_length, -1)  # , router_logits
            else:
                self.ort_run_with_iobinding(ort_inputs)
                return None

        return None

    def ort_run_with_iobinding(self, ort_inputs, repeat=1000):
        iobinding = self.ort_sess.io_binding()
        device_id = torch.cuda.current_device()

        iobinding.bind_input(
            name="input",
            device_type="cuda",
            device_id=device_id,
            element_type=NP_TYPE,
            shape=ort_inputs["input"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(ort_inputs["input"], "cuda", device_id).data_ptr(),
        )

        iobinding.bind_input(
            name="router_probs",
            device_type="cuda",
            device_id=device_id,
            element_type=NP_TYPE,
            shape=ort_inputs["router_probs"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(
                ort_inputs["router_probs"], "cuda", device_id
            ).data_ptr(),
        )

        iobinding.bind_output(
            name="output",
            device_type="cuda",
            device_id=device_id,
            element_type=NP_TYPE,
            shape=ort_inputs["input"].shape,
            buffer_ptr=onnxruntime.OrtValue.ortvalue_from_numpy(
                numpy.zeros(ort_inputs["input"].shape), "cuda", device_id
            ).data_ptr(),
        )

        # warm up
        for _ in range(5):
            iobinding.synchronize_inputs()
            self.ort_sess.run_with_iobinding(iobinding)
            iobinding.synchronize_outputs()

        import time  # noqa: PLC0415

        s = time.time()
        for _ in range(repeat):
            iobinding.synchronize_inputs()
            self.ort_sess.run_with_iobinding(iobinding)
            iobinding.synchronize_outputs()
        e = time.time()
        print(f"MoE cuda kernel time: {(e - s) / repeat * 1000} ms")

    def parity_check(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        torch_output = self.forward(hidden_state)
        ort_output = self.ort_forward(hidden_state)
        if ort_output is not None:
            print(
                "name:",
                self.__class__.__name__,
                " batch_size:",
                self.batch_size,
                " sequence_length:",
                self.sequence_length,
                " max_diff:",
                (torch_output - ort_output).abs().max(),
            )
            torch.testing.assert_close(ort_output.to(torch.float32), torch_output, rtol=THRESHOLD, atol=THRESHOLD)

    def benchmark_ort(self):
        hidden_state = torch.randn(self.batch_size, self.sequence_length, self.hidden_dim)
        self.ort_forward(hidden_state, iobinding=True)


def small_test_cases():
    for batch_size in [1, 4, 16]:
        for sequence_length in [128, 512, 1024]:
            yield batch_size, sequence_length


# ---------------------------------------------
# The following test are for swiglu activation
# ---------------------------------------------
class SwigluMoeConfig:
    def __init__(
        self,
        hidden_size=2048,
        intermediate_size=2048,
        num_experts_per_token=2,
        num_local_experts=8,
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts_per_token = num_experts_per_token
        self.num_local_experts = num_local_experts


class SwigluMlp(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.intermediate_size = config.intermediate_size
        self.hidden_dim = config.hidden_size
        self.w1 = nn.Linear(self.hidden_dim, 2 * self.intermediate_size, bias=True)
        self.w2 = nn.Linear(self.intermediate_size, self.hidden_dim, bias=True)

    def swiglu(self, x: torch.Tensor):
        dim = x.shape[-1]
        x = x.view(-1, dim // 2, 2)
        x_glu, x_linear = x[..., 0], x[..., 1]
        y = x_glu * torch.sigmoid(1.702 * x_glu) * (x_linear + 1)
        return y

    def forward(self, x):
        y = self.swiglu(self.w1(x))
        y = self.w2(y)
        return y


def create_swiglu_moe_onnx_graph(
    num_tokens,
    num_experts,
    hidden_size,
    inter_size,
    fc1_experts_weights,
    fc1_experts_bias,
    fc2_experts_weights,
    fc2_experts_bias,
    topk,
):
    nodes = [
        helper.make_node(
            "MoE",
            [
                "input",
                "router_probs",
                "fc1_experts_weights",
                "fc1_experts_bias",
                "fc2_experts_weights",
                "fc2_experts_bias",
            ],
            ["output"],
            "MoE_0",
            k=topk,
            normalize_routing_weights=1,
            activation_type="swiglu",
            domain="com.microsoft",
        ),
    ]

    fc1_weight_shape = [num_experts, hidden_size, 2 * inter_size]
    fc1_bias_shape = [num_experts, 2 * inter_size]

    fc2_weight_shape = [num_experts, inter_size, hidden_size]
    fc2_bias_shape = [num_experts, hidden_size]

    torch_type = torch.float16 if ORT_DTYPE == TensorProto.FLOAT16 else torch.float32

    initializers = [
        helper.make_tensor(
            "fc1_experts_weights",
            ORT_DTYPE,
            fc1_weight_shape,
            fc1_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc1_experts_bias",
            ORT_DTYPE,
            fc1_bias_shape,
            fc1_experts_bias.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_weights",
            ORT_DTYPE,
            fc2_weight_shape,
            fc2_experts_weights.to(torch_type).flatten().tolist(),
            raw=False,
        ),
        helper.make_tensor(
            "fc2_experts_bias",
            ORT_DTYPE,
            fc2_bias_shape,
            fc2_experts_bias.to(torch_type).flatten().tolist(),
            raw=False,
        ),
    ]

    graph_inputs = [
        helper.make_tensor_value_info("input", ORT_DTYPE, [num_tokens, hidden_size]),
    ]

    graph_inputs.append(
        helper.make_tensor_value_info(
            "router_probs",
            ORT_DTYPE,
            [num_tokens, num_experts],
        )
    )

    graph_outputs = [
        helper.make_tensor_value_info("output", ORT_DTYPE, [num_tokens, hidden_size]),
    ]

    graph = helper.make_graph(
        nodes,
        "MoE_Graph",
        graph_inputs,
        graph_outputs,
        initializers,
    )

    model = helper.make_model(graph)
    return model.SerializeToString()


class SwigluMoEBlock(SparseMoeBlockORTHelper):
    def __init__(self, config: SwigluMoeConfig, batch_size: int, sequence_length: int):
        super().__init__()
        self.hidden_dim = config.hidden_size
        self.ffn_dim = config.intermediate_size
        self.num_experts = config.num_local_experts
        self.top_k = config.num_experts_per_token

        self.gate = nn.Linear(self.hidden_dim, self.num_experts, bias=True)

        self.experts = nn.ModuleList([SwigluMlp(config) for _ in range(self.num_experts)])

        weight_1_list = []
        weight_2_list = []
        bias_1_list = []
        bias_2_list = []
        for i in range(self.num_experts):
            weight_1_list.append(self.experts[i].w1.weight)
            weight_2_list.append(self.experts[i].w2.weight)
            bias_1_list.append(self.experts[i].w1.bias)
            bias_2_list.append(self.experts[i].w2.bias)

        self.moe_experts_weight1 = torch.stack(weight_1_list, dim=0)
        self.moe_experts_weight2 = torch.stack(weight_2_list, dim=0)

        self.moe_experts_bias1 = torch.stack(bias_1_list, dim=0)
        self.moe_experts_bias2 = torch.stack(bias_2_list, dim=0)

        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.moe_onnx_graph = create_swiglu_moe_onnx_graph(
            num_tokens=self.batch_size * self.sequence_length,
            num_experts=self.num_experts,
            hidden_size=self.hidden_dim,
            inter_size=self.ffn_dim,
            fc1_experts_weights=self.moe_experts_weight1,
            fc1_experts_bias=self.moe_experts_bias1,
            fc2_experts_weights=self.moe_experts_weight2,
            fc2_experts_bias=self.moe_experts_bias2,
            topk=self.top_k,
        )

        self.ort_sess = self.create_ort_session(self.moe_onnx_graph)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """ """
        batch_size, sequence_length, hidden_dim = hidden_states.shape
        hidden_states = hidden_states.view(-1, hidden_dim)
        router_logits = self.gate(hidden_states)  # router_logits shape is (batch * sequence_length, num_experts)
        routing_weights, selected_experts = torch.topk(router_logits, self.top_k, dim=-1)

        routing_weights = F.softmax(routing_weights, dim=1, dtype=torch.float)

        # we cast back to the input dtype
        routing_weights = routing_weights.to(hidden_states.dtype)

        final_hidden_states = torch.zeros(
            (batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype, device=hidden_states.device
        )

        # One hot encode the selected experts to create an expert mask
        # this will be used to easily index which expert is going to be sollicitated
        expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)

        # Loop over all available experts in the model and perform the computation on each expert
        for expert_idx in range(self.num_experts):
            expert_layer = self.experts[expert_idx]
            idx, top_x = torch.where(expert_mask[expert_idx])

            if top_x.shape[0] == 0:
                continue

            # Index the correct hidden states and compute the expert hidden state for
            # the current expert. We need to make sure to multiply the output hidden
            # states by `routing_weights` on the corresponding tokens (top-1 and top-2)
            current_state = hidden_states[None, top_x].reshape(-1, hidden_dim)
            current_hidden_states = expert_layer(current_state) * routing_weights[top_x, idx, None]

            # However `index_add_` only support torch tensors for indexing so we'll use
            # the `top_x` tensor here.
            final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
        final_hidden_states = final_hidden_states.reshape(batch_size, sequence_length, hidden_dim)
        return final_hidden_states


class TestSwigluMoE(unittest.TestCase):
    @parameterized.expand(small_test_cases())
    def test_swiglu_moe_parity(self, batch_size, sequence_length):
        config = SwigluMoeConfig(hidden_size=1024, intermediate_size=1024)
        moe = SwigluMoEBlock(config, batch_size, sequence_length)
        moe.parity_check()


if __name__ == "__main__":
    unittest.main()
