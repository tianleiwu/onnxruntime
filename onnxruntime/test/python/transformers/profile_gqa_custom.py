
import torch
import triton
from test_sparse_attention import GroupQueryAttentionConfig, OrtGroupQueryAttention

def run_benchmark():
    batch_size = 1
    num_heads = 32
    kv_num_heads = 8
    head_size = 128
    device = "cuda"
    dtype = "float16"

    # Shapes to benchmark
    # (sequence_length, past_sequence_length, "Label")
    scenarios = [
        (512, 0, "Prefill 512"),
        (1024, 0, "Prefill 1024"),
        (1, 512, "Decode 1+512"),
        (1, 1024, "Decode 1+1024"),
        (1, 2048, "Decode 1+2048"),
    ]

    print(f"{'Scenario':<20} | {'Latency (ms)':<15}")
    print("-" * 40)

    for seq_len, past_seq_len, label in scenarios:
        config = GroupQueryAttentionConfig(
            batch_size=batch_size,
            sequence_length=seq_len,
            max_sequence_length=4096, # Sufficient max
            past_sequence_length=past_seq_len,
            num_heads=num_heads,
            kv_num_heads=kv_num_heads,
            head_size=head_size,
            do_rotary=True,
            device=device,
            dtype=torch.float16,
            is_packed_qkv=True # Assume packed for standard benchmarking
        )

        obj = OrtGroupQueryAttention(config)

        # Warmup
        for _ in range(10):
            obj.infer()

        # Benchmark using Triton for accurate timing
        ms = triton.testing.do_bench(obj.infer, warmup=10, rep=100)

        print(f"{label:<20} | {ms:<15.4f}")

if __name__ == "__main__":
    run_benchmark()
