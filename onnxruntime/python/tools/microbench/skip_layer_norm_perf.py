# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------
"""
Comprehensive benchmark for SkipLayerNormalization kernel across different
hidden sizes, batch sizes, and sequence lengths. Generates ONNX models on the
fly so it doesn't depend on pre-built model files.

Usage:
    python skip_layer_norm_perf.py --provider cuda --precision fp16
    python skip_layer_norm_perf.py --provider cuda --precision fp32
"""

import argparse
import os
import tempfile
import time

import numpy as np
import onnx
from onnx import TensorProto, helper

import onnxruntime as ort


def create_skip_layer_norm_model(batch_size, seq_len, hidden_size, data_type):
    """Create an ONNX model with a single SkipLayerNormalization op."""
    onnx_type = TensorProto.FLOAT16 if data_type == np.float16 else TensorProto.FLOAT

    input_tensor = helper.make_tensor_value_info("INPUT", onnx_type, [batch_size, seq_len, hidden_size])
    skip_tensor = helper.make_tensor_value_info("SKIP", onnx_type, [batch_size, seq_len, hidden_size])
    gamma_tensor = helper.make_tensor_value_info("GAMMA", onnx_type, [hidden_size])
    beta_tensor = helper.make_tensor_value_info("BETA", onnx_type, [hidden_size])
    bias_tensor = helper.make_tensor_value_info("BIAS", onnx_type, [hidden_size])

    output_tensor = helper.make_tensor_value_info("OUTPUT", onnx_type, [batch_size, seq_len, hidden_size])

    node = helper.make_node(
        "SkipLayerNormalization",
        inputs=["INPUT", "SKIP", "GAMMA", "BETA", "BIAS"],
        outputs=["OUTPUT", "", "", ""],
        domain="com.microsoft",
        epsilon=1e-5,
    )

    graph = helper.make_graph(
        [node],
        "skip_layer_norm_benchmark",
        [input_tensor, skip_tensor, gamma_tensor, beta_tensor, bias_tensor],
        [output_tensor],
    )

    opset_imports = [
        helper.make_opsetid("", 17),
        helper.make_opsetid("com.microsoft", 1),
    ]

    model = helper.make_model(graph, opset_imports=opset_imports)
    model.ir_version = 7
    return model


def run_benchmark(model_path, inputs, provider, warmup=10, iterations=100):
    """Run a benchmark and return average time in milliseconds."""
    sess_opt = ort.SessionOptions()
    sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_DISABLE_ALL
    sess = ort.InferenceSession(model_path, sess_options=sess_opt, providers=[provider])

    # Warm up
    for _ in range(warmup):
        sess.run(None, inputs)

    # Measure
    start_time = time.time()
    for _ in range(iterations):
        sess.run(None, inputs)

    elapsed_time = (time.time() - start_time) * 1000 / iterations
    return elapsed_time


def main():
    parser = argparse.ArgumentParser(description="SkipLayerNormalization kernel benchmark")
    parser.add_argument("--provider", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp16")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    provider = "CUDAExecutionProvider" if args.provider == "cuda" else "CPUExecutionProvider"
    data_type = np.float16 if args.precision == "fp16" else np.float32

    # Test configurations: (batch_size, seq_len, hidden_size)
    configs = [
        # Small hidden sizes (uses vec4 kernel path)
        (1, 384, 128),
        # Common BERT/GPT sizes (uses vec8 kernel path)
        (1, 384, 768),
        (1, 384, 1024),
        (1, 128, 2048),
        (1, 128, 4096),
        (1, 128, 5120),
        # Larger batch sizes
        (4, 128, 1024),
        (8, 512, 1024),
        (4, 128, 4096),
        # LLM-scale
        (1, 2048, 4096),
        (1, 2048, 5120),
        (1, 4096, 8192),
    ]

    print(f"SkipLayerNormalization Benchmark ({args.precision}, {args.provider})")
    print(f"{'Config (B, S, H)':<25} {'Time (ms)':>10} {'Throughput (GB/s)':>18}")
    print("-" * 60)

    with tempfile.TemporaryDirectory() as tmpdir:
        for batch_size, seq_len, hidden_size in configs:
            # Create model
            model = create_skip_layer_norm_model(batch_size, seq_len, hidden_size, data_type)
            model_path = os.path.join(tmpdir, f"skip_ln_{batch_size}_{seq_len}_{hidden_size}.onnx")
            onnx.save(model, model_path)

            # Create inputs
            np.random.seed(0)
            inputs = {
                "INPUT": np.random.rand(batch_size, seq_len, hidden_size).astype(data_type),
                "SKIP": np.random.rand(batch_size, seq_len, hidden_size).astype(data_type),
                "GAMMA": np.random.rand(hidden_size).astype(data_type),
                "BETA": np.random.rand(hidden_size).astype(data_type),
                "BIAS": np.random.rand(hidden_size).astype(data_type),
            }

            # Run benchmark
            elapsed = run_benchmark(model_path, inputs, provider, args.warmup, args.iterations)

            # Calculate throughput: read input + skip + bias + gamma + beta + write output + sum_output
            # Approximate: 2 reads (input, skip) + 3 param reads (gamma, beta, bias) + 2 writes (output, sum_output)
            # Main data: 4 * batch * seq * hidden (2 reads + 2 writes of main tensors)
            elem_size = 2 if data_type == np.float16 else 4
            total_elements = batch_size * seq_len * hidden_size
            # input read + skip read + output write + sum_output write = 4 * total_elements
            # gamma/beta/bias are small (hidden_size), negligible
            bytes_transferred = 4 * total_elements * elem_size
            throughput_gbps = bytes_transferred / (elapsed * 1e-3) / 1e9

            config_str = f"({batch_size}, {seq_len}, {hidden_size})"
            print(f"{config_str:<25} {elapsed:>10.4f} {throughput_gbps:>18.2f}")

    print("-" * 60)
    print("Done.")


if __name__ == "__main__":
    main()
