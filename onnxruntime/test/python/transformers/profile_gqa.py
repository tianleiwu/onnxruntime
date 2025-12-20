# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

"""
Simple profiling script for GroupQueryAttention with quantized KV cache.

Usage:
  # Run directly to measure timing
  conda activate py312
  cd /onnxruntime/test/python/transformers
  python profile_gqa.py

  # Profile with Nsight Compute (kernel-level analysis)
  ncu --set full -o gqa_fp16 python profile_gqa.py --mode fp16 --warmup 5 --repeat 1
  ncu --set full -o gqa_int8 python profile_gqa.py --mode int8 --warmup 5 --repeat 1

  # Profile with Nsight Systems (timeline analysis)
  nsys profile -o gqa_fp16 python profile_gqa.py --mode fp16
  nsys profile -o gqa_int8 python profile_gqa.py --mode int8
"""

import argparse
import time
import torch
from test_sparse_attention import GroupQueryAttentionConfig, OrtGroupQueryAttention


def create_gqa_config(
    mode: str = "fp16",
    batch_size: int = 1,
    sequence_length: int = 1,
    past_sequence_length: int = 2048,
    max_sequence_length: int = 4096,
    num_heads: int = 32,
    kv_num_heads: int = 8,
    head_size: int = 128,
    device: str = "cuda",
) -> GroupQueryAttentionConfig:
    """Create a GQA config based on the mode (fp16, int8, or int4)."""
    if mode == "fp16":
        k_quant_type = "NONE"
        v_quant_type = "NONE"
        kv_cache_type = "float16"
    elif mode == "int8":
        k_quant_type = "PER_TENSOR"
        v_quant_type = "PER_TENSOR"
        kv_cache_type = "int8"
    elif mode == "int4":
        k_quant_type = "PER_CHANNEL"
        v_quant_type = "PER_CHANNEL"
        kv_cache_type = "int4"
    else:
        raise ValueError(f"Unknown mode: {mode}")

    config = GroupQueryAttentionConfig(
        batch_size=batch_size,
        sequence_length=sequence_length,
        max_sequence_length=max_sequence_length,
        past_sequence_length=past_sequence_length,
        num_heads=num_heads,
        kv_num_heads=kv_num_heads,
        head_size=head_size,
        local_window_size=-1,
        do_rotary=True,
        is_packed_qkv=False,
        use_smooth_softmax=False,
        device=device,
        k_quant_type=k_quant_type,
        v_quant_type=v_quant_type,
        kv_cache_type=kv_cache_type,
    )
    return config


def benchmark_gqa(config: GroupQueryAttentionConfig, warmup: int = 50, repeat: int = 100):
    """Run benchmark and return average time in ms."""
    obj = OrtGroupQueryAttention(config)

    # Warmup
    for _ in range(warmup):
        obj.infer()
    torch.cuda.synchronize()

    # Benchmark
    start = time.perf_counter()
    for _ in range(repeat):
        obj.infer()
    torch.cuda.synchronize()
    end = time.perf_counter()

    avg_ms = (end - start) * 1000 / repeat
    return avg_ms


def run_comparison(args):
    """Compare FP16 vs quantized performance."""
    print(f"\n{'='*70}")
    print(f"GQA Performance Comparison")
    print(f"{'='*70}")
    print(f"Config: batch={args.batch_size}, seq_len={args.sequence_length}, "
          f"past_seq={args.past_sequence_length}")
    print(f"        num_heads={args.num_heads}, kv_heads={args.kv_num_heads}, "
          f"head_size={args.head_size}")
    print(f"        warmup={args.warmup}, repeat={args.repeat}")
    print(f"{'='*70}\n")

    modes = ["fp16", "int8", "int4"] if args.mode == "all" else [args.mode]
    results = {}

    for mode in modes:
        config = create_gqa_config(
            mode=mode,
            batch_size=args.batch_size,
            sequence_length=args.sequence_length,
            past_sequence_length=args.past_sequence_length,
            max_sequence_length=args.max_sequence_length,
            num_heads=args.num_heads,
            kv_num_heads=args.kv_num_heads,
            head_size=args.head_size,
        )
        avg_ms = benchmark_gqa(config, warmup=args.warmup, repeat=args.repeat)
        results[mode] = avg_ms
        print(f"  {mode.upper():6s}: {avg_ms:.4f} ms")

    # Print comparison if we have baseline
    if "fp16" in results and len(results) > 1:
        print(f"\n  Relative to FP16:")
        for mode, ms in results.items():
            if mode != "fp16":
                ratio = ms / results["fp16"]
                print(f"    {mode.upper()}: {ratio:.2f}x slower")


def main():
    parser = argparse.ArgumentParser(description="Profile GQA with quantized KV cache")
    parser.add_argument("--mode", choices=["fp16", "int8", "int4", "all"], default="all",
                        help="Quantization mode to test")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size")
    parser.add_argument("--sequence-length", type=int, default=1,
                        help="Query sequence length (1 for token generation)")
    parser.add_argument("--past-sequence-length", type=int, default=2048,
                        help="Past KV cache sequence length")
    parser.add_argument("--max-sequence-length", type=int, default=4096,
                        help="Max sequence length for KV cache buffer")
    parser.add_argument("--num-heads", type=int, default=32, help="Number of query heads")
    parser.add_argument("--kv-num-heads", type=int, default=8, help="Number of KV heads")
    parser.add_argument("--head-size", type=int, default=128, help="Head dimension")
    parser.add_argument("--warmup", type=int, default=50, help="Warmup iterations")
    parser.add_argument("--repeat", type=int, default=100, help="Benchmark iterations")

    args = parser.parse_args()

    # Check CUDA
    if not torch.cuda.is_available():
        print("CUDA not available!")
        return

    major, minor = torch.cuda.get_device_capability()
    print(f"GPU: {torch.cuda.get_device_name()} (SM{major}{minor})")

    with torch.cuda.stream(torch.cuda.Stream()), torch.no_grad():
        run_comparison(args)


if __name__ == "__main__":
    main()
