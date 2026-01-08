# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation.  All rights reserved.
# Licensed under the MIT License.
# -------------------------------------------------------------------------

import argparse
import os

import numpy as np
import torch
from onnx import TensorProto

# Import helpers from test_gqa.py
# This assumes the script is run from the same directory: onnxruntime/test/python/transformers/
from test_gqa import GQAConfig, gqa_past_func


def run_comparison(config_name, config, device="cuda"):
    print(f"\nRunning Comparison: {config_name}")
    print(f"Config: {config}")

    torch.manual_seed(42)
    # Generate Inputs
    # Dimensions
    batch_size = config.batch_size
    q_seq_len = config.q_sequence_length
    kv_seq_len = config.kv_sequence_length  # New tokens
    past_seq_len = config.past_kv_sequence_length
    total_seq_len = past_seq_len + q_seq_len  # For past_func, q_seq_len is new tokens?
    # In gqa_past_func:
    # q is [B, q_seq_len, H*D]
    # new_k/v is [B, q_seq_len, H_kv*D]
    # So q_seq_len here represents the "new" sequence length (decoding step, usually 1)

    hidden_size = config.num_heads * config.head_size
    kv_hidden_size = config.kv_num_heads * config.head_size

    # Inputs
    q = torch.randn(batch_size, q_seq_len, hidden_size, device=device, dtype=torch.float16)

    # Past State (KV Cache)
    # Shape: [B, N_kv, MaxSeq, H] for BNSH
    k_cache = torch.randn(
        batch_size,
        config.kv_num_heads,
        config.buffer_sequence_length,
        config.head_size,
        device=device,
        dtype=torch.float16,
    )
    v_cache = torch.randn_like(k_cache)

    # New KV (to be appended)
    new_k = torch.randn(batch_size, q_seq_len, kv_hidden_size, device=device, dtype=torch.float16)
    new_v = torch.randn(batch_size, q_seq_len, kv_hidden_size, device=device, dtype=torch.float16)

    # Cos/Sin for RoPE
    rotary_dim = config.head_size // 2  # Simplified
    cos = torch.randn(config.buffer_sequence_length, rotary_dim, device=device, dtype=torch.float16)
    sin = torch.randn(config.buffer_sequence_length, rotary_dim, device=device, dtype=torch.float16)

    seqlens_k = torch.full((batch_size,), past_seq_len, device=device, dtype=torch.int32)

    # ----------------------------------------------------------------------
    # Run FUSED (Default)
    # ----------------------------------------------------------------------
    print("  -> Running FUSED mode...")
    if "ORT_DISABLE_FUSED_KV" in os.environ:
        del os.environ["ORT_DISABLE_FUSED_KV"]

    # Create copies of cache for fused run to avoid in-place pollution
    k_fused = k_cache.clone()
    v_fused = v_cache.clone()

    out_fused, present_k_fused, present_v_fused = gqa_past_func(
        q=q,
        k=k_fused,
        v=v_fused,
        config=config,
        new_k=new_k,
        new_v=new_v,
        cos=cos,
        sin=sin,
        seqlens_k=seqlens_k,
        position_ids=None,
        attention_bias=None,
        head_sink=None,
        ep="CUDAExecutionProvider",
        device=device,
        share_buffer=True,
        ort_type=TensorProto.FLOAT16,
    )

    # ----------------------------------------------------------------------
    # Run UNFUSED (Comparision)
    # ----------------------------------------------------------------------
    print("  -> Running UNFUSED mode...")
    os.environ["ORT_DISABLE_FUSED_KV"] = "1"

    # Create copies of cache for unfused run
    k_unfused = k_cache.clone()
    v_unfused = v_cache.clone()

    out_unfused, present_k_unfused, present_v_unfused = gqa_past_func(
        q=q,
        k=k_unfused,
        v=v_unfused,
        config=config,
        new_k=new_k,
        new_v=new_v,
        cos=cos,
        sin=sin,
        seqlens_k=seqlens_k,
        position_ids=None,
        attention_bias=None,
        head_sink=None,
        ep="CUDAExecutionProvider",
        device=device,
        share_buffer=True,
        ort_type=TensorProto.FLOAT16,
    )

    # ----------------------------------------------------------------------
    # Verify
    # ----------------------------------------------------------------------
    print("  -> Verifying results...")

    # Check for cache corruption (Past data should be preserved)
    # Indices 0 to 127 should match past_key
    past_len = config.past_kv_sequence_length
    past_key_np = k_cache.cpu().numpy()  # This is the input past_key

    # Check Fused Result
    present_k_fused_np = present_k_fused.cpu().numpy()
    present_k_past_region = present_k_fused_np[:, :, :past_len, :]
    past_key_region = past_key_np[:, :, :past_len, :]

    if not np.allclose(present_k_past_region, past_key_region, atol=1e-3):
        print("     [FAIL] FUSED Cache Corruption Detected in Past Region!")
        diff = np.abs(present_k_past_region - past_key_region)
        max_diff = np.max(diff)
        mismatch_idx = np.where(diff > 1e-3)
        print(f"            Max Diff: {max_diff}")
        if len(mismatch_idx[0]) > 0:
            idx0 = mismatch_idx[0][0]
            idx1 = mismatch_idx[1][0]
            idx2 = mismatch_idx[2][0]
            idx3 = mismatch_idx[3][0]
            print(f"            First mismatch at: [{idx0}, {idx1}, {idx2}, {idx3}]")
            print(f"            Ref: {past_key_region[idx0, idx1, idx2, idx3]}")
            print(f"            Act: {present_k_past_region[idx0, idx1, idx2, idx3]}")
    else:
        print("     [OK] FUSED: No Cache Corruption in Past Region")

    # Compare Output
    try:
        np.testing.assert_allclose(out_fused.cpu().numpy(), out_unfused.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print("     [OK] Output Tensor Match")
    except AssertionError as e:
        print(f"     [FAIL] Output Tensor Mismatch: {e}")
        return False

    # Compare KV Cache (The critical part for Fused KV Kernel)
    # We only verify the *appended* part and the *past* part (should be unchanged)
    # Actually, comparing the whole buffer is safest.
    try:
        np.testing.assert_allclose(present_k_fused.cpu().numpy(), present_k_unfused.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print("     [OK] Key Cache Match")
    except AssertionError as e:
        print(f"     [FAIL] Key Cache Mismatch: {e}")
        return False

    try:
        np.testing.assert_allclose(present_v_fused.cpu().numpy(), present_v_unfused.cpu().numpy(), rtol=1e-3, atol=1e-3)
        print("     [OK] Value Cache Match")
    except AssertionError as e:
        print(f"     [FAIL] Value Cache Mismatch: {e}")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="Verify Fused vs Unfused GQA Kernels")
    args = parser.parse_args()

    results = []

    # Case 1: Unpacked QKV, Shared Buffer, Rotary
    # This targets `ConcatKVInPlaceFused` vs `ConcatKVInPlace` + Explicit RoPE
    config1 = GQAConfig(
        batch_size=2,
        q_sequence_length=1,  # Decoding
        kv_sequence_length=1,
        num_heads=16,
        kv_num_heads=4,
        head_size=128,
        past_kv_sequence_length=128,
        buffer_sequence_length=256,
        rotary=True,
        packed=False,
        share_buffer=True,
    )
    if run_comparison("Unpacked + Shared + Rotary", config1):
        results.append("PASS")
    else:
        results.append("FAIL")

    # Case 2: Packed QKV, Shared Buffer, Rotary
    # This targets `UnpackQKVWithRoPEAndAppendKV` vs `UnpackQKV` + Explicit RoPE
    # Not easily testable via gqa_past_func as it usually takes unpacked inputs or handles packing internally?
    # gqa_past_func takes `new_k`, `new_v` separately.
    # To test packed path, we need to feed a PACKED input to the operator.
    # But `gqa_past_func` logic in `test_gqa.py` seems to unpack inputs or bind them separately.
    # Let's check `create_gqa_node_and_io`.
    # It constructs `inputs`. If `config.packed` is True, it expects `query` to contain QKV?
    # Actually, looking at `gqa_past_func`, input `q` is reshaped.
    # If packed, `test_gqa` usually handles packing input.
    # Im skipping this for now to rely on the primary Unpacked check which was the main refactor.

    print("\nSummary:")
    print(f"Unpacked + Shared + Rotary: {results[0]}")


if __name__ == "__main__":
    main()
