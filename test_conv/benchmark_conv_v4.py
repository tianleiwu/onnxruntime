# /home/tlwu/cuda12.8/bin/nsys profile  --force-overwrite true     --sample process-tree --backtrace fp --stats true     -t cuda,cudnn,cublas,osrt,nvtx     --cuda-memory-usage true --cudabacktrace all    
# -o "conv_profile_v4_all" $CONDA_PREFIX/bin/python3 benchmark_conv_v4.py
# benchmark_conv_v3_nsys_ready.py
import onnx
import onnx.helper as helper
import onnx.numpy_helper as numpy_helper
import numpy as np
import os
import onnxruntime as ort
from onnxruntime.transformers.io_binding_helper import CudaSession, GpuBinding
import torch
import time
import nvtx

# NEW: Added a dtype parameter to control precision
def create_conv_model(model_path: str, dtype: str):
    if dtype == 'float32':
        onnx_dtype = onnx.TensorProto.FLOAT
        np_dtype = np.float32
    elif dtype == 'float16':
        onnx_dtype = onnx.TensorProto.FLOAT16
        np_dtype = np.float16
    else:
        raise ValueError("Unsupported dtype. Please use 'float32' or 'float16'.")

    input_tensor = helper.make_tensor_value_info(
        "input", onnx_dtype, [198, 1, "S", 768]
    )
    output_tensor = helper.make_tensor_value_info(
        "output", onnx_dtype, None
    )

    transpose_node = helper.make_node(
        "Transpose",
        inputs=["input"],
        outputs=["transposed_output"],
        perm=[1, 0, 2, 3],
        name="TransposeInput",
    )

    data_path = f"{model_path}.data"
    W_arr = np.random.randn(178200, 1, 5, 768).astype(np_dtype)
    with open(data_path, 'wb') as f:
        f.write(W_arr.tobytes())

    W_init = onnx.TensorProto()
    W_init.name = "W"
    W_init.data_type = onnx_dtype
    W_init.dims.extend(W_arr.shape)
    W_init.data_location = onnx.TensorProto.EXTERNAL
    location_entry = W_init.external_data.add()
    location_entry.key = "location"
    location_entry.value = os.path.basename(data_path)

    B_arr = np.random.randn(178200).astype(np_dtype)
    B_init = numpy_helper.from_array(B_arr, name="B")

    conv_node = helper.make_node(
        "Conv",
        inputs=["transposed_output", "W", "B"],
        outputs=["output"],
        dilations=[1, 1],
        group=198,
        kernel_shape=[5, 768],
        pads=[4, 0, 4, 0],
        strides=[1, 1],
    )

    graph = helper.make_graph(
        [transpose_node, conv_node],
        "ConvGraph",
        [input_tensor],
        [output_tensor],
        initializer=[W_init, B_init],
    )

    opset_version = 18
    model = helper.make_model(
        graph,
        producer_name="onnx-benchmark",
        opset_imports=[helper.make_opsetid("", opset_version)]
    )

    onnx.save(model, model_path)


def create_session(model_path: str, session_options=None,  device_id:int=0, enable_cuda_graph:bool = False, use_tf32:bool=False, prefer_nhwc:bool=False) -> ort.InferenceSession:
    provider_options = CudaSession.get_cuda_provider_options(device_id, enable_cuda_graph)
    provider_options["use_tf32"] = int(use_tf32)
    if prefer_nhwc:
        provider_options["prefer_nhwc"] = 1
    providers = [("CUDAExecutionProvider", provider_options), "CPUExecutionProvider"]
    ort_session = ort.InferenceSession(model_path, session_options, providers=providers)
    device= torch.device(f"cuda:{device_id}")
    cuda_session = CudaSession(ort_session, device, enable_cuda_graph)
    return cuda_session



def benchmark(model_path: str, seq_len: int, dtype: str, 
              enable_cuda_graph=True, use_tf32=True, prefer_nhwc=True,
              warmup: int = 10, runs: int = 50, debug:bool = False):
    if dtype == 'float32':
        np_dtype = np.float32
    elif dtype == 'float16':
        np_dtype = np.float16
    else:
        raise ValueError("Unsupported dtype. Please use 'float32' or 'float16'.")

    so = ort.SessionOptions()
    if debug:
        so.log_severity_level = 0
        so.log_verbosity_level = 4
    cuda_session = create_session(model_path, so, device_id=0, enable_cuda_graph=enable_cuda_graph, use_tf32=use_tf32, prefer_nhwc=prefer_nhwc)

    device = torch.device("cuda:0")

    input_shape = (198, 1, seq_len, 768)
    output_shape = (1, 178200, seq_len + 4, 1)
    shape_dict = {"input": input_shape, "output": output_shape}

    cuda_session.allocate_buffers(shape_dict)

    inp_np = np.random.randn(*input_shape).astype(np_dtype)
    inp_torch = torch.from_numpy(inp_np).to(device)
    feed_dict = {"input": inp_torch}

    # warmup (synchronous to be safe)
    for _ in range(warmup):
        cuda_session.infer(feed_dict)
        torch.cuda.synchronize()

    # Place a *single* NVTX region around entire benchmark batch
    with nvtx.annotate("one_run_all"):
        start = time.time()
        for _ in range(runs):
            cuda_session.infer(feed_dict)
            # ensure kernels complete while still inside NVTX region
            torch.cuda.synchronize()
        end = time.time()

    total_time = end - start
    avg_latency = total_time / runs * 1000  # ms
    throughput = runs / total_time

    print(f"[CUDA EP - {dtype.upper()}] seq_len={seq_len}")
    print(f"Avg Latency: {avg_latency:.2f} ms")
    print(f"Throughput: {throughput:.2f} runs/s")
    print("-" * 30)

def test(enable_cuda_graph:bool, use_tf32:bool, prefer_nhwc:bool, dtype:str):
    model_path = f"conv_benchmark_{dtype}.onnx"
    print(f"=== Enable CUDA Graph: {enable_cuda_graph}, Use TF32: {use_tf32}, Prefer NHWC: {prefer_nhwc}, DType: {dtype.upper()} ===")
    create_conv_model(model_path, dtype=dtype)
    for seq_len in [8]:  # [4 , 8, 12, 16]:
        benchmark(model_path, seq_len=seq_len, dtype=dtype)
    print("\n")

if __name__ == "__main__":
    # for dtype in ['float32' , 'float16']:
    #     for enable_cuda_graph in [True, False]:
    #         for use_tf32 in [True, False]:
    #             for prefer_nhwc in [True, False]:
    #                 test(enable_cuda_graph, use_tf32, prefer_nhwc, dtype)
    test(enable_cuda_graph=True, use_tf32=True, prefer_nhwc=True, dtype='float32')