# TensorRT-LLM MoE Plugin: INT8 Group-wise Quantization Format

This document describes the tensor formats for **float16 input with INT8 group-wise quantization** in the TensorRT-LLM Mixture of Experts (MoE) plugin.

## GEMM Dimensions

For MoE GEMM1 (first fully-connected layer):

| Dimension | Description | Example |
|-----------|-------------|---------|
| **M** | Number of tokens assigned to expert | Dynamic |
| **K** | `hidden_size` (input dimension) | 4096 |
| **N** | `inter_size × 2` (for gated activations like SwiGLU) | 28672 |

---

## Weight Tensor

### Logical Shape
```
[num_experts, K, N]
```
Example: `[8, 4096, 28672]` for 8 experts

### Storage Format
Weights are stored as **int8** after preprocessing, but viewed as **float16** for TensorRT compatibility:

```python
cuda_q_weight = preprocessor(weight, torch.int8, torch.float16).view(torch.float16)
```

### Preprocessing Pipeline

The weights undergo several transformations in `cutlass_preprocessors.cpp`:

1. **Row Permutation** (for LDSM on Turing+)
   - Groups of 16 rows are permuted using:
   ```
   [0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15]
   ```

2. **Subbyte Transpose**
   - Row-major `[K, N]` → Column-major `[N, K]`

3. **Column Interleaving** (Ampere/sm80)
   - Uses `ColumnMajorTileInterleave` layout

4. **Bias Addition + Register Interleaving**
   - Bias of +128 added: shifts signed `[-128, 127]` → unsigned `[0, 255]`
   - Register layout: `[elt_3, elt_1, elt_2, elt_0]` per 32-bit word

---

## Scale Tensor

### Shape
```
[num_experts, K // group_size, N]
```
Example: `[8, 32, 28672]` for `group_size=128`

### Memory Layout
- **Row-major** (standard contiguous layout)
- Per-group, per-output-channel scaling

### Data Type
- `float16` or `bfloat16` (matches activation dtype)

---

## Zero-Point Tensor

### Shape
```
[num_experts, K // group_size, N]
```
Same shape as scale tensor.

### Memory Layout
- **Row-major** (matches scale layout)
- Per-group, per-output-channel zero-point

### Data Type
- `float16` or `bfloat16` (**same as scale and activation**)

> [!NOTE]
> For W4A8 (FP8 activation), scales and zeros are always `float16` regardless of activation dtype.

---

## Quantization Modes

TensorRT-LLM supports two zero-point modes:

| Mode | Zero-Point | Flag | Use Case |
|------|------------|------|----------|
| **Symmetric** | Hard-coded +128 bias | Default | Simple weight-only quant |
| **Asymmetric** | Custom tensor | `GroupwiseQuantAlgo::ZERO` | AWQ, GPTQ |

---

## Dequantization Formula

### Symmetric Mode (default)
```cpp
float dequant_weight = (float)(quant_weight - 128) * scale;
```

### Asymmetric Mode (with zero tensor)
```cpp
float dequant_weight = (float)quant_weight * scale + zero;
```

Python reference:
```python
# Scale is broadcast from [E, num_groups, N] to [E, K, N]
scale_expanded = scale.repeat_interleave(group_size, dim=1)[:, :K, :]
zero_expanded = zero.repeat_interleave(group_size, dim=1)[:, :K, :]

dequant_weight = quant_weight * scale_expanded + zero_expanded
```

---

## Summary Table

| Tensor | Shape | Data Type | Layout |
|--------|-------|-----------|--------|
| **Input Activation** | `[M, K]` | float16 / bfloat16 | Row-major |
| **Weight** | `[E, K, N]` | int8 (viewed as fp16) | Preprocessed CUTLASS layout |
| **Scale** | `[E, K // group_size, N]` | float16 / bfloat16 | Row-major |
| **Zero-Point** | `[E, K // group_size, N]` | float16 / bfloat16 | Row-major |
| **Output** | `[M, N]` | float16 / bfloat16 | Row-major |

Where:
- `E` = num_experts
- `K` = hidden_size
- `N` = inter_size × 2 (for gated activation) or inter_size
- `M` = number of tokens
- `group_size` = typically 64 or 128

---

## Code References

- Weight preprocessing: [`cutlass_preprocessors.cpp`](file:///home/tlwu/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/cutlass_preprocessors.cpp)
- QuantParams struct: [`moe_kernels.h`](file:///home/tlwu/TensorRT-LLM/cpp/tensorrt_llm/kernels/cutlass_kernels/include/moe_kernels.h#L242-L419)
- MoE Plugin: [`mixtureOfExpertsPlugin.h`](file:///home/tlwu/TensorRT-LLM/cpp/tensorrt_llm/plugins/mixtureOfExperts/mixtureOfExpertsPlugin.h)
- Test reference: [`test_weight_only_groupwise_quant_matmul.py`](file:///home/tlwu/TensorRT-LLM/tests/unittest/trt/quantization/test_weight_only_groupwise_quant_matmul.py)
