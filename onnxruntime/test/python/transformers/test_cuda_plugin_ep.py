# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.

import os
import sys

import numpy as np

import onnxruntime as onnxrt


def create_model(model_path):
    import onnx
    from onnx import TensorProto, helper

    # Create a simple Add model: Y = A + B
    node_def = helper.make_node(
        "Add",
        ["A", "B"],
        ["Y"],
    )

    graph_def = helper.make_graph(
        [node_def],
        "test-model",
        [
            helper.make_tensor_value_info("A", TensorProto.FLOAT, [3, 2]),
            helper.make_tensor_value_info("B", TensorProto.FLOAT, [3, 2]),
        ],
        [helper.make_tensor_value_info("Y", TensorProto.FLOAT, [3, 2])],
    )

    model_def = helper.make_model(graph_def, producer_name="onnx-example")
    onnx.save(model_def, model_path)


def test_cuda_plugin_registration():
    # Priority for finding the plugin library:
    # 1. Environment variable ORT_CUDA_PLUGIN_PATH
    # 2. Known build location relative to this script
    ep_lib_path = os.environ.get("ORT_CUDA_PLUGIN_PATH")
    if not ep_lib_path:
        # Assuming we are in <repo>/onnxruntime/test/python/transformers/
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        ep_lib_path = os.path.join(base_dir, "build", "cuda", "Release", "libonnxruntime_providers_cuda_plugin.so")

    if not os.path.exists(ep_lib_path):
        print(f"Error: Plugin library not found at: {ep_lib_path}")
        print("Set ORT_CUDA_PLUGIN_PATH to point to the .so file.")
        sys.exit(1)

    # Use the name the EP reports internally to avoid any confusion
    ep_name = "CudaPluginExecutionProvider"

    print(f"Attempting to register plugin from: {ep_lib_path}", flush=True)

    try:
        onnxrt.register_execution_provider_library(ep_name, ep_lib_path)
        print(f"onnxrt.register_execution_provider_library('{ep_name}') returned success", flush=True)
    except Exception as e:
        print(f"Registration failed: {e}", flush=True)
        return

    print("Checking available providers...", flush=True)
    available = onnxrt.get_available_providers()
    print(f"Available providers: {available}", flush=True)

    plugin_devices = []
    if hasattr(onnxrt, "get_ep_devices"):
        print("Calling get_ep_devices()...", flush=True)
        try:
            devices = onnxrt.get_ep_devices()
            print(f"get_ep_devices() returned {len(devices)} devices", flush=True)
            for d in devices:
                print(f"  Device: {d.ep_name}, Vendor: {d.ep_vendor}", flush=True)
                if d.ep_name == ep_name:
                    plugin_devices.append(d)
        except Exception as e:
            print(f"get_ep_devices() failed: {e}", flush=True)

    if not plugin_devices:
        print("Error: No devices found for our plugin!", flush=True)
        sys.exit(1)

    print(f"Found {len(plugin_devices)} plugin devices. Selecting the first one.", flush=True)
    target_device = plugin_devices[0]

    print("Creating session with add_provider_for_devices...", flush=True)
    model_path = "dummy.onnx"
    create_model(model_path)
    try:
        sess_options = onnxrt.SessionOptions()
        sess_options.add_provider_for_devices([target_device], {"prefer_nhwc": "1"})

        # Note: If we use add_provider_for_devices, we don't need to pass providers to InferenceSession
        sess = onnxrt.InferenceSession(model_path, sess_options=sess_options)
        active_providers = sess.get_providers()
        print(f"Session created. Active providers: {active_providers}", flush=True)

        if ep_name in active_providers:
            print(f"SUCCESS: {ep_name} is active in the session!", flush=True)
        else:
            print(
                f"FAILURE: {ep_name} is NOT active in the session. Only {active_providers} were selected.", flush=True
            )
            sys.exit(1)

        # Run inference
        print("Running inference...", flush=True)
        a = np.random.rand(3, 2).astype(np.float32)
        b = np.random.rand(3, 2).astype(np.float32)
        res = sess.run(None, {"A": a, "B": b})
        np.testing.assert_allclose(res[0], a + b, rtol=1e-5)
        print("Inference successful!", flush=True)

    except Exception as e:
        print(f"Test failed: {e}", flush=True)
        sys.exit(1)
    finally:
        if os.path.exists(model_path):
            os.remove(model_path)

    print("Test finished successfully.", flush=True)


if __name__ == "__main__":
    test_cuda_plugin_registration()
