# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.  See License.txt in the project root for
# license information.
# --------------------------------------------------------------------------
"""Decode-phase benchmark comparing Lean Attention against Flash Attention.

This driver answers a single question: for the token-generation (decode) phase
(sequence_length == 1, non-empty KV cache), does the Lean Attention kernel beat
the Flash Attention split-KV ("flash decoding") kernel, and in which region of
(batch * num_heads) x past_sequence_length?

Two matched sweeps are produced:
  A) MultiHeadAttention: direct Lean-vs-Flash comparison. Lean is only wired into
     the contrib MultiHeadAttention op, so this is the only apples-to-apples test.
     Reuses benchmark_mha.run_tflops_test which pins the kernel via the
     sdpa_kernel provider option AND verifies the *actually routed* kernel through
     ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO -- so a "lean" row is guaranteed to be
     real Lean (it is dropped if the shape falls back to Flash).
  B) GroupQueryAttention: Lean is NOT wired into GQA today. GQA already has its own
     flash "fast decode" (split-KV by kv_num_heads), plus XQA and cuDNN decode
     routes. This sweep measures GQA's best existing decode kernel on matched
     shapes to quantify the headroom a GQA-capable Lean would have to beat.

Usage (activate the CUDA venv and set CUDA/CUDNN env first):
    python benchmark_lean_decode.py --op mha  --repeats 2000
    python benchmark_lean_decode.py --op gqa  --repeats 2000
    python benchmark_lean_decode.py --op both --quick
Then summarise:
    python benchmark_lean_decode.py --analyze benchmark_lean_mha_*.csv
"""

from __future__ import annotations

import argparse
import csv
import os
from datetime import datetime

import torch

import benchmark_mha as bm
from benchmark_mha import (
    InputFormats,
    MultiHeadAttentionConfig,
    SdpaKernel,
    create_session,
    flops,
    get_compute_capability,
    sdpa_kernel_from_debug_info,
    tflops_per_second,
)
from onnxruntime import SessionOptions


def device_time(session, feed, repeats):
    """Average GPU-side latency (seconds) per infer, isolated from CPU launch
    overhead by queueing `repeats` infers back-to-back between two CUDA events."""
    for _ in range(10):  # warm up
        session.infer(feed, synchronize=False)
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        session.infer(feed, synchronize=False)
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / 1000.0 / repeats  # ms -> s per infer

# ------------------------------------------------------------------ shape grids

# Lean's target regime is low occupancy (small batch*heads) with a long KV cache.
FULL_BATCHES = [1, 2, 4, 8, 16, 32, 64]
FULL_HEADS = [8, 16, 32, 64]
FULL_HEAD_SIZES = [64, 128]
FULL_PASTS = [128, 512, 1024, 2048, 4096, 8192, 16384, 32768]

QUICK_BATCHES = [1, 4, 16]
QUICK_HEADS = [8, 32]
QUICK_HEAD_SIZES = [64, 128]
QUICK_PASTS = [512, 2048, 8192, 32768]

# GQA ratios num_heads / kv_num_heads. 32/32 is the MHA-equivalent control so the
# MHA lean numbers can be overlaid at ratio 1.
FULL_GQA_RATIOS = [(32, 32), (32, 8), (64, 8), (32, 4)]
QUICK_GQA_RATIOS = [(32, 32), (32, 8), (64, 8)]

CSV_COLUMNS = [
    "op",
    "use_gpu",
    "enable_cuda_graph",
    "format",
    "causal",
    "batch_size",
    "sequence_length",
    "past_sequence_length",
    "num_heads",
    "kv_num_heads",
    "head_size",
    "has_attn_bias",
    "broadcast_attn_bias_dim_0",
    "broadcast_attn_bias_dim_1",
    "intra_op_num_threads",
    "average_latency",
    "tflops",
    "request_kernel",
    "kernel",
]


def run_mha_sweep(csv_writer, batches, heads, head_sizes, pasts, repeats):
    sm = get_compute_capability()
    device = torch.device("cuda", torch.cuda.current_device())
    print(f"# MHA decode sweep (sm={sm}); device-timed Flash vs Lean vs cuDNN vs Efficient.")
    print("op\tbatch\theads\th_dim\tpast\tkernel\tus\tTFLOPS")

    kernels = [
        SdpaKernel.FLASH_ATTENTION,
        SdpaKernel.LEAN_ATTENTION,
        SdpaKernel.CUDNN_FLASH_ATTENTION,
        SdpaKernel.EFFICIENT_ATTENTION,
    ]
    for head_size in head_sizes:
        for num_heads in heads:
            for batch_size in batches:
                for past in pasts:
                    total = past + 1
                    config = MultiHeadAttentionConfig(
                        batch_size=batch_size,
                        sequence_length=1,  # decode: single new token
                        num_heads=num_heads,
                        head_size=head_size,
                        causal=True,
                        use_kv_cache=True,
                        past_sequence_length=past,
                        max_cache_sequence_length=None,
                        kv_sequence_length=None,
                        provider="CUDAExecutionProvider",
                        enable_cuda_graph=False,
                        device=device,
                        dtype=torch.float16,
                        share_past_present_buffer=False,
                        input_format=InputFormats.Q_K_V_BSNH_BSNH_BSNH,
                        has_past_input=True,
                    )
                    for kernel in kernels:
                        sess_options = SessionOptions()
                        actual = sdpa_kernel_from_debug_info(config, kernel, sess_options)
                        request = bm.get_gpu_kernel_name(kernel)
                        if actual is None:
                            continue
                        # Only keep rows where the requested kernel actually ran (so a
                        # "lean" row is guaranteed real Lean, not a silent Flash fallback).
                        if actual != request:
                            continue
                        try:
                            session = create_session(config, sess_options, attention_kernel=kernel)
                            avg = device_time(session, config.random_inputs(), repeats)
                        except Exception as e:  # noqa: BLE001
                            print(f"  MHA failed {request} b{batch_size} h{num_heads} p{past}: {e}")
                            continue
                        del session
                        speed = tflops_per_second(flops(batch_size, 1, total, head_size, num_heads, True), avg)
                        csv_writer.writerow(
                            {
                                "op": "mha",
                                "use_gpu": True,
                                "enable_cuda_graph": False,
                                "format": "Q,K,V",
                                "causal": True,
                                "batch_size": batch_size,
                                "sequence_length": 1,
                                "past_sequence_length": past,
                                "num_heads": num_heads,
                                "kv_num_heads": num_heads,
                                "head_size": head_size,
                                "has_attn_bias": False,
                                "broadcast_attn_bias_dim_0": False,
                                "broadcast_attn_bias_dim_1": False,
                                "intra_op_num_threads": 0,
                                "average_latency": avg,
                                "tflops": speed,
                                "request_kernel": request,
                                "kernel": actual,
                            }
                        )
                        s = f"{speed:.3f}" if speed is not None else "NA"
                        print(f"mha\t{batch_size}\t{num_heads}\t{head_size}\t{past}\t{actual}\t{avg * 1e6:.2f}\t{s}")


# ---------------------------------------------------------------------- GQA path

def _make_gqa_session(config, attention_kernel, enable_cuda_graph=False):
    """Like gqa_test_helper.create_gqa_ort_session but pins sdpa_kernel so we can
    force Flash / cuDNN paths. Returns a CudaSession."""
    from gqa_test_helper import create_group_query_attention_onnx_model
    from onnxruntime import InferenceSession
    from onnxruntime.transformers.io_binding_helper import CudaSession

    onnx_model_str = create_group_query_attention_onnx_model(config)
    device_id = torch.cuda.current_device() if isinstance(config.device, str) else config.device.index
    provider_options = CudaSession.get_cuda_provider_options(
        device_id, enable_cuda_graph=enable_cuda_graph, stream=torch.cuda.current_stream().cuda_stream
    )
    provider_options["sdpa_kernel"] = int(attention_kernel)
    providers = [(config.provider, provider_options), "CPUExecutionProvider"]
    ort_session = InferenceSession(onnx_model_str, None, providers=providers)
    cuda_session = CudaSession(ort_session, config.device, enable_cuda_graph=enable_cuda_graph)
    cuda_session.allocate_buffers(config.shape_dict())
    for i, o in {"past_key": "present_key", "past_value": "present_value"}.items():
        cuda_session.set_buffer_sharing(i, o)
    return cuda_session


class _gqa_env:
    """Set ORT_ENABLE_XQA / ORT_DISABLE_FLASH_DECODE for the duration of a block
    and restore the prior values afterwards. Both are read at kernel construction
    time, so the session must be created inside this context."""

    def __init__(self, enable_xqa, disable_flash_decode):
        self._want = {
            "ORT_ENABLE_XQA": "1" if enable_xqa else "0",
            "ORT_DISABLE_FLASH_DECODE": "1" if disable_flash_decode else "0",
        }
        self._prev = {}

    def __enter__(self):
        for k, v in self._want.items():
            self._prev[k] = os.environ.get(k)
            os.environ[k] = v
        return self

    def __exit__(self, *exc):
        for k, prev in self._prev.items():
            if prev is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = prev
        return False


def _gqa_actual_kernel(config, attention_kernel, enable_xqa, disable_flash_decode):
    """Return the SdpaKernel name GQA actually routed to for this shape/config."""
    import re

    from benchmark_mha import CaptureStdout

    os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "1"
    text = None
    try:
        with _gqa_env(enable_xqa, disable_flash_decode), CaptureStdout() as cap:
            sess = _make_gqa_session(config, attention_kernel)
            sess.infer(config.random_inputs())
        text = cap.output.decode(errors="replace")
    except Exception as e:  # noqa: BLE001
        print(f"  GQA route probe failed ({attention_kernel}): {e}")
    os.environ["ORT_ENABLE_ATTENTION_KERNEL_DEBUG_INFO"] = "0"
    if text:
        m = re.search(r"SdpaKernel=(?P<k>[A-Z_]+)", text)
        if m:
            return m.group("k")
    return None


def run_gqa_sweep(csv_writer, batches, ratios, head_sizes, pasts, repeats):
    from gqa_test_helper import GroupQueryAttentionConfig

    sm = get_compute_capability()
    device = torch.device("cuda", torch.cuda.current_device())
    print(f"# GQA decode sweep (sm={sm}); measures GQA's best existing decode kernel per shape.")

    # (request_kernel, enable_xqa, disable_flash_decode, label)
    # XQA is GQA's default decode path on sm90 and outranks sdpa_kernel, so it must
    # be disabled (ORT_ENABLE_XQA=0) to measure the flash / cuDNN decode routes.
    variants = [
        (SdpaKernel.DEFAULT, True, False, "xqa"),
        (SdpaKernel.FLASH_ATTENTION, False, False, "flash_decode"),
        (SdpaKernel.FLASH_ATTENTION, False, True, "flash_nosplit"),
        (SdpaKernel.CUDNN_FLASH_ATTENTION, False, False, "cudnn"),
    ]

    for head_size in head_sizes:
        for num_heads, kv_num_heads in ratios:
            for batch_size in batches:
                for past in pasts:
                    total = past + 1
                    config = GroupQueryAttentionConfig(
                        batch_size=batch_size,
                        sequence_length=1,
                        max_sequence_length=total,
                        past_sequence_length=past,
                        num_heads=num_heads,
                        kv_num_heads=kv_num_heads,
                        head_size=head_size,
                        provider="CUDAExecutionProvider",
                        device=device,
                        dtype=torch.float16,
                        local_window_size=-1,
                        max_cache_sequence_length=total,
                    )
                    for req_kernel, enable_xqa, disable_fd, label in variants:
                        actual = _gqa_actual_kernel(config, req_kernel, enable_xqa, disable_fd)
                        if actual is None:
                            continue
                        with _gqa_env(enable_xqa, disable_fd):
                            try:
                                sess = _make_gqa_session(config, req_kernel)
                            except Exception as e:  # noqa: BLE001
                                print(f"  GQA session failed b{batch_size} h{num_heads}/{kv_num_heads} p{past}: {e}")
                                continue
                            avg = device_time(sess, config.random_inputs(), repeats)
                        del sess
                        speed = tflops_per_second(
                            flops(batch_size, 1, total, head_size, num_heads, True), avg
                        )
                        req_label = f"{label}:{actual.lower()}"
                        csv_writer.writerow(
                            {
                                "op": "gqa",
                                "use_gpu": True,
                                "enable_cuda_graph": False,
                                "format": "Q,K,V",
                                "causal": True,
                                "batch_size": batch_size,
                                "sequence_length": 1,
                                "past_sequence_length": past,
                                "num_heads": num_heads,
                                "kv_num_heads": kv_num_heads,
                                "head_size": head_size,
                                "has_attn_bias": False,
                                "broadcast_attn_bias_dim_0": False,
                                "broadcast_attn_bias_dim_1": False,
                                "intra_op_num_threads": 0,
                                "average_latency": avg,
                                "tflops": speed,
                                "request_kernel": req_label,
                                "kernel": actual.lower(),
                            }
                        )
                        s = f"{speed:.3f}" if speed is not None else "NA"
                        print(
                            f"gqa\tb{batch_size}\th{num_heads}/{kv_num_heads}\td{head_size}\tp{past}\t"
                            f"{avg * 1e6:.1f}us\t{s}TFLOPS\t{label}->{actual.lower()}"
                        )


# ------------------------------------------------------------------- analysis

def analyze(csv_paths):
    rows = []
    for p in csv_paths:
        with open(p, newline="") as f:
            rows.extend(list(csv.DictReader(f)))

    def key(r):
        return (r["op"], r["batch_size"], r["num_heads"], r["kv_num_heads"], r["head_size"], r["past_sequence_length"])

    # MHA: lean vs flash speedup. Sub-key by kernel for MHA (ort:flash / ort:lean),
    # but by request_kernel for GQA so flash_decode vs flash_nosplit vs xqa vs cudnn
    # stay distinct (several GQA variants route to the same underlying kernel name).
    shapes = {}
    for r in rows:
        label = r["kernel"] if r["op"] == "mha" else r["request_kernel"]
        shapes.setdefault(key(r), {})[label] = float(r["average_latency"])

    print("\n=== MHA lean-vs-flash speedup (flash_latency / lean_latency, >1 = lean faster) ===")
    print("batch\theads\th_dim\tpast\tflash_us\tlean_us\tspeedup")
    wins = 0
    total = 0
    for (op, b, nh, kvh, hd, past), lat in sorted(shapes.items()):
        if op != "mha":
            continue
        flash = lat.get("ort:flash")
        lean = lat.get("ort:lean")
        if flash and lean:
            total += 1
            sp = flash / lean
            wins += sp > 1.0
            print(f"{b}\t{nh}\t{hd}\t{past}\t{flash * 1e6:.1f}\t{lean * 1e6:.1f}\t{sp:.3f}")
    if total:
        print(f"\nLean faster in {wins}/{total} MHA shapes.")

    # GQA: best kernel per shape.
    print("\n=== GQA best decode kernel per shape ===")
    print("batch\theads/kv\th_dim\tpast\tbest_kernel\tbest_us")
    for (op, b, nh, kvh, hd, past), lat in sorted(shapes.items()):
        if op != "gqa":
            continue
        best_k = min(lat, key=lat.get)
        print(f"{b}\t{nh}/{kvh}\t{hd}\t{past}\t{best_k}\t{lat[best_k] * 1e6:.1f}")


# ----------------------------------------------------------------------- main

def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--op", choices=["mha", "gqa", "both"], default="both")
    parser.add_argument("--repeats", type=int, default=2000)
    parser.add_argument("--quick", action="store_true", help="Use the compact shape grid.")
    parser.add_argument("--csv", type=str, default=None, help="Output CSV path (default: timestamped).")
    parser.add_argument("--analyze", nargs="+", default=None, help="Analyze existing CSV file(s) and exit.")
    args = parser.parse_args()

    if args.analyze:
        analyze(args.analyze)
        return

    assert torch.cuda.is_available(), "CUDA device required"
    sm = get_compute_capability()
    assert sm >= 80, f"Lean/Flash require sm>=80, got {sm}"

    batches = QUICK_BATCHES if args.quick else FULL_BATCHES
    heads = QUICK_HEADS if args.quick else FULL_HEADS
    head_sizes = QUICK_HEAD_SIZES if args.quick else FULL_HEAD_SIZES
    pasts = QUICK_PASTS if args.quick else FULL_PASTS
    ratios = QUICK_GQA_RATIOS if args.quick else FULL_GQA_RATIOS

    csv_path = args.csv or "benchmark_lean_{}_{}.csv".format(args.op, datetime.now().strftime("%Y%m%d-%H%M%S"))
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        if args.op in ("mha", "both"):
            run_mha_sweep(writer, batches, heads, head_sizes, pasts, args.repeats)
        if args.op in ("gqa", "both"):
            run_gqa_sweep(writer, batches, ratios, head_sizes, pasts, args.repeats)
    print(f"\nWrote {csv_path}")
    analyze([csv_path])


if __name__ == "__main__":
    main()
