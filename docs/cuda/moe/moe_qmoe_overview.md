
# MoE and QMoE Technical Documentation

This document describes the technical implementation details of the Mixture of Experts (MoE) and Quantized Mixture of Experts (QMoE) operators in ONNX Runtime, specifically targeting the CUDA execution provider.

## 1. Data Layouts

The implementation relies on strict memory layouts for weights and quantization parameters to interact correctly with the underlying Cutlass GEMM kernels.

### 1.1 Weights (QMoE)
For Group-wise Quantized MoE (e.g., 4-bit or 8-bit), the weights are **not** standard linear layers. They must be pre-packed to match the specific interleaving requirements of the Cutlass Mixed Input GEMM kernel.

*   **Logical Shape**: `[NumExperts, HiddenSize, InterSize]` (Note: ONNX uses `[E, In, Out]` convention for MoE, but PyTorch Linear is `[Out, In]`. Effectively `[E, K, N]`).
    *   *Clarification*: This shape corresponds to the `MatMulNBits` standard (`[Experts, N, K_Blocks] `) *before* architecture-specific packing.
*   **Physical Layout (Storage)**: Opaque Blob.
    *   The actual stored data is **no longer** in the logical shape. It is a packed, transposed, and interleaved blob formatted specifically for the Cutlass kernel.
    *   **Packing**: The `pack_weights_for_cuda_mixed_gemm` function transforms the logical `[E, N, K]` weights into this architecture-dependent blob.
    *   *Note*: Do not attempt to interpret the raw bytes of this tensor as a standard array; it must be treated as a blob passed directly to the kernel.
*   **Packing Tool**: The weights are packed using `pack_weights_for_cuda_mixed_gemm` (exposed via `onnxruntime.quantization.matmul_4bits`). This function rearranges the elements into the column-major interleaved format expected by the Cutlass kernel.
*   **Input Index**:
    *   `fc1_experts_weights`: Input 2
    *   `fc2_experts_weights`: Input 5
    *   `fc3_experts_weights`: Input 8 (Optional)

> **FC3 Note**: The underlying CUTLASS `CutlassMoeFCRunner` has no separate FC3 GEMM — its `runMoe()` interface only accepts fc1 and fc2 weight pointers. This mirrors TensorRT-LLM's design, which also has no FC3 concept. For gated activations (SwiGLU), FC1 and FC3 (gate + up projection) weights must be **pre-concatenated** into the fc1 weight tensor with doubled output dimension (`[Experts, 2 × InterSize, HiddenSize / pack_size]`). The `fc3_experts_weights` ONNX input (index 8) is validated for shape consistency in `CheckInputs` but is **not consumed** by the QMoE CUDA compute path. The `has_fc3_` field stored in the runner is never read after construction.

### 1.2 Scales (QMoE)
Scaling factors for dequantization.

*   **Shape**: `[NumExperts, OutputChannels, InputChannels / BlockSize]`.
    *   Example: For `Hidden=4096`, `Block=64`, `Scale` dim is `4096/64 = 64`.
    *   `fc1_scales` (Hidden -> Inter): `[NumExperts, InterSize, HiddenSize // BlockSize]`.
    *   `fc2_scales` (Inter -> Hidden): `[NumExperts, HiddenSize, InterSize // BlockSize]`.
    *   *Reference*: This layout corresponds to the ONNX interface `[NumExperts, Output, Input_Blocks]`.
    *   *Note*: The underlying Cutlass kernel (and TensorRT-LLM specifications) describes a transposed order `[Groups, N]`. The ONNX Runtime implementation adapts this to ensure the correct kernel execution.
*   **Type**: `float16` (MLFloat16).
*   **Input Index**: 3, 6, 9.

### 1.3 Zero Points (QMoE)
Zero points for asymmetric quantization.

*   **Shape**: Same as Scales: `[NumExperts, OutputChannels, InputChannels / BlockSize]`.
*   **Type**: `uint8` (TensorProto.UINT8).
*   **Input Index**: 11, 12, 13.
*   **Important**: The runtime treats these as **Additive Biases** during `PrePack`. See Section 2.

## 2. PrePack Transformations

The `QMoE` operator implements a `PrePack` method (`moe_quantization.cc`) to optimize quantization parameters before execution.

### 2.1 Weight Packing
*   **Runtime Action**: None. The `QMoE::PrePack` implementation currently **skips** weights. It assumes that weights provided to the ONNX model have *already* been packed using the offline `pack_weights_for_cuda_mixed_gemm` tool.

### 2.2 Scale & Zero-Point Processing
To maximize performance and compatibility with Cutlass kernels, the `PrePack` step adapts the data based on the quantization type:

1.  **Capture Scales**: Copies `fc_scales` from CPU to GPU.
2.  **Zero-Point / Bias Conversion**:
    *   **8-bit Weights**: Weights are shifted by -128 (uint8 -> int8). We compute a bias to compensate:
        ```cpp
        Bias = (128 - ZeroPoint) * Scale
        ```
        *   This effectively treats the calculation as `(W_stored + 128 - ZP) * Scale`.
    *   **4-bit Weights**: Zero Points are unpacked from nibbles (2 per byte) and converted to **Scaled Biases**:
        ```cpp
        Bias = (8 - ZeroPoint) * Scale
        ```
        *   This transformation allows the kernel to handle asymmetric quantization by adding a pre-calculated bias term, effectively computing `W * Scale + Bias` which is equivalent to `(W - (ZeroPoint - 8)) * Scale`.
    *   **Symmetric**: Bias is 0.

    *   This conversion happens via generic kernels (`LaunchQMoEPrePackOffsetBias`, `LaunchQMoEPrePackPacked4BitZPKernel`).
    *   The resulting buffer (`packed_bias`) is stored in `float16/bfloat16/float`, matching the Scale type.

## 3. Cutlass Kernel Expectations

The backend uses `CutlassMoeFCRunner` to dispatch kernels.

### 3.1 Kernel Signature
The kernels are typically instantiated as:
```cpp
MoeGemmRunner<half, uint8_t, half>  // InputType, WeightType, OutputType
```

### 3.2 Layout Requirements
*   **Weights**: Column-Major Interleaved (Packed).
*   **Scales & Bias**: Row-Major `[Output, Input/Block]`. (Note: Since we process one expert at a time or grouped experts, the `NumExperts` dimension is the batch).
*   **Bias/ZP Type Constraint**:
    *   For Symmetric Quantization (`Bias=0`), the kernel works seamlessly.
    *   For Asymmetric Quantization, the kernel expects `ElementZero` (the bias/ZP buffer type) to match `ElementScale` (`half`, `bfloat16`, or `float`).
        *   **4-bit**: Stores Pre-calculated Bias.
        *   **8-bit**: Stores Pre-calculated Bias.
    *   *Note*: `DefaultGemmGrouped` may infer `ElementZero` from `ElementB` (Weights, `uint8`). This implementation ensures `ElementZero` matches `ElementScale` (Floating Point) to support asymmetric quantization correctly.

## 4. Testing Infrastructure

The primary test scripts are located in `onnxruntime/test/python/transformers/`.

### 4.1 Test Scripts
1.  **`test_qmoe_cuda.py`**:
    *   **Purpose**: Tests the Quantized MoE (QMoE) operator.
    *   **Features**:
        *   Generates random data and quantizes it using `quant_dequant_blockwise`/`quantize_matmul_4bits`/`8bits`.
        *   Constructs an ONNX graph with `QMoE` node.
        *   Packs weights using `pack_weights_for_cuda_mixed_gemm`.
        *   Compares ORT output against a PyTorch reference implementation (Float16 matmul using dequantized weights).
    *   **Usage**:
        ```bash
        python onnxruntime/test/python/transformers/test_qmoe_cuda.py
        # Run specific test
        python onnxruntime/test/python/transformers/test_qmoe_cuda.py -k "test_swiglu_qmoe_blockwise_parity_cpu_4"
        ```

2.  **`test_moe_cuda.py`**:
    *   **Purpose**: Tests the standard (non-quantized) `MoE` operator.
    *   **Features**: Tests FP16/BF16 functionality, routing logic, and standard GEMM parity.

### 4.2 Quantization & Reference
*   **Quantization**: Done in Python using numpy/torch helpers (e.g., `_quantize.quantize_matmul_4bits`).
*   **Reference**: The "Ground Truth" is calculated by dequantizing the weights back to FP16 in Python:
    ```python
    dequantized = (q_weight - zero_point) * scale
    reference = input @ dequantized.T
    ```
    This ensures we verify the numerical correctness of the *dequantization* fusion.

## 5. Technical Details & Nuances

### 5.1 Architecture Support

The MoE/QMoE operators dispatch between two CUTLASS kernel families at runtime, selected by `CutlassMoeFCRunner::supportsTmaWarpSpecialized()`:

| Path | Arch | Condition |
|------|------|-----------|
| **TMA Warp-Specialized** (SM90+) | SM90 (Hopper) | `T == WeightType` (fp16×fp16, bf16×bf16), plus W4A16 FP4 dispatch when enabled |
|  | SM100–119 (Blackwell) | Valid Blackwell MoE specialisation |
|  | SM120–121 | FP4×FP4 only (`isValidSM120MOESpecialisation`) |
| **Ampere GemmGrouped** (fallback) | SM80/86/89 | All types |
|  | SM90 | INT4/INT8 weights (mixed-type) |
|  | SM120 | All non-FP4 types (fp16, bf16, INT4, INT8) |
|  | Any SM | `float32` (forced to SM80 behavior) |

**Minimum dimension constraint (`min_dim`)**:
- Both `hidden_size` and `inter_size` must be ≥ 16.
- TMA WS path: smallest tile is 128×16×128B (N=16 for FP16). K residues handled by TMA.
- Ampere GemmGrouped path: smallest instantiated tile N=128, but CUTLASS predicates N < tile_N.
- Alignment to 128 bits is enforced separately (e.g., dimensions must be multiples of 8 for FP16).

**Target GPUs**: RTX 3090 (SM86), RTX 4090 (SM89), RTX 5090 (SM120), H200 (SM90 dev).

*   **Data Types**: Supports `float16` and `bfloat16` for inputs and outputs. `float32` is supported but always uses SM80 (Ampere) kernel path.

### 5.2 SwiGLU Fusion
*   **Interleaved**: The operator supports `swiglu_fusion=1`. In this mode, the weights for the Gating and Value projections are interleaved in the `fc1` tensor.
    *   Shape: `[Experts, 2 * InterSize, HiddenSize]`.
    *   The kernel computes the GEMM, then applies SwiGLU activation + gating in the epilogue.
*   **FC3 handling**: Regardless of the `swiglu_fusion` attribute value, the CUTLASS runner always expects gate and up projection weights to be fused into `fc1` (i.e., `is_fused_swiglu = (activation_type == Swiglu)` is always true in the QMoE compute path). There is no separate FC3 GEMM dispatch. This is consistent with TensorRT-LLM, which validates `w1.N == inter_size * 2` for gated activations and has zero FC3 references in its kernel code.

### 5.3 Memory Management
*   **Workspace**: The operator requires a workspace for intermediate results (sorting indices, permuted rows).
*   **Pre-allocated Buffers**: `PrePack` allocates GPU memory for Scales and Biases. These persist for the lifetime of the session, reducing overhead per inference step.

> **Note**: The general quantization format and kernel expectations align with TensorRT-LLM specifications.

### 5.4 Limitations
*   **Row-wise Quantization**: Row-wise quantization (`block_size <= 0`) does not currently support zero points in the QMoE operator.
*   **Block Size**: Asymmetric zero points are currently supported only when `block_size >= 64`.
*   **Minimum Dimension**: `hidden_size` and `inter_size` must be ≥ 16. Alignment to 128 bits is enforced separately. See Section 5.1.
*   **FP4 W4A16 availability**: QMoE FP4 (`quant_type="fp4"`) requires CUDA 12.8+ (`ENABLE_FP4`) and SM90+. The current build excludes the full SM90 mixed-input FP4 launcher and links stub instantiations that throw if the W4A16 path reaches them.
*   **Float32**: Always forced to SM80 kernel path regardless of actual hardware SM version.

**Weight Conversion**:
When exporting a model to ONNX with SwiGLU, the weights for the Gate and Value projections (typically FC1 and Gate_Proj) are often merged. To use **Interleaved** mode, these weights must be interleaved at the output channel dimension during the export/packing phase so that the GEMM output naturally results in `[G, V, G, V...]`.

## 6. Weight Packing Details (`pack_weights_for_cuda_mixed_gemm`)

The `pack_weights_for_cuda_mixed_gemm` function is a critical offline preprocessing step required to format weights for Cutlass Mixed Input GEMM kernels. The source code is distributed across:
*   **Python Binding**: `onnxruntime/python/onnxruntime_pybind_quant.cc`
*   **Kernel Adaptor**: `onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemm_adaptor.cu`
*   **Implementation**: `onnxruntime/contrib_ops/cuda/llm/fpA_intB_gemm_preprocessors_impl.cu` (and `.h`)

### 6.1 Processing Pipeline
The function transforms standard linear weights into a hardware-optimized format through the following stages:

1.  **Input Layout**: Accepts weights in `[N, K]` layout, corresponding to `(Out_Features, In_Features)`.
    *   For 4-bit, data is packed (2 elements per byte).
2.  **Transpose & Signed Conversion**:
    *   Transposes to `[K, N]` layout.
    *   Converts unsigned data to **Signed Int8** intermediate representation.
        *   **4-bit**: Unpacks `uint4` [0, 15], subtracts 8 -> `int8` [-8, 7].
        *   **8-bit**: Subtracts 128 from `uint8` [0, 255] -> `int8` [-128, 127].
3.  **Row Permutation (LDSM Optimization)** (SM80+):
    *   Reorders rows within small tiles to align with the **Load Shared Memory (LDSM)** instruction requirements of Ampere+ Tensor Cores.
    *   **W8_A16 (8-bit)**: Permutes every 16 rows using map `{0, 1, 8, 9, 2, 3, 10, 11, 4, 5, 12, 13, 6, 7, 14, 15}`.
    *   **W4_A16 (4-bit)**: Permutes every 32 rows using map `{0, 1, 8, 9, 16, 17...}`.
    *   *Source*: `kPerm_W8_A16` / `kPerm_W4_A16` in `fpA_intB_gemm_preprocessors_impl.h`.
4.  **Interleaving**:
    *   Applies column interleaving (e.g., `ColumnMajorTileInterleave`) if required by the specific Cutlass kernel layout for the target architecture.
5.  **Final Register Layout Adjustment**:
    *   **Add Bias**: Adds 128 to shift values back to alignment with `uint8` storage if needed (effectively negating the earlier subtraction for storage, or adjusting for kernel expectations).
    *   **Swap**: Performs sub-register swapping (e.g., `[0, 1, 2, 3] -> [0, 2, 1, 3]`) for specific register file layouts.
    *   *Kernel*: `add_bias_and_interleave_int8s_inplace_kernel`.

### 6.2 Interleaving Details
The packing process applies specific interleaving patterns to align data with Cutlass kernel expectations (`ColumnMajorTileInterleave`). The parameters depend on the quantization type:

| Quantization Type | Cutlass Kernel Layout | Interleaving Parameters |
| :--- | :--- | :--- |
| **W8_A16** (8-bit) | `ColumnMajorTileInterleave<64, 2>` | `RowsPerTile=64`, `ColumnsInterleaved=2` |
| **W4_A16** (4-bit) | `ColumnMajorTileInterleave<64, 4>` | `RowsPerTile=64`, `ColumnsInterleaved=4` |

*Note*: `RowsPerTile` refers to the K-dimension tile size. `ColumnsInterleaved` indicates how many N-dimension columns are interleaved together.

#### Visualizing `ColumnMajorTileInterleave<Rows, Cols>`
To answer the specific layout question: **Yes, the tile is effectively stored in Column-Major format.**

For `<64, 2>` (Interleaving 2 columns for every 64 rows):
1.  The logical block is `64 Rows x 2 Columns`.
2.  **Storage Order**:
    *   Store all 64 elements of **Column 0**.
    *   Followed immediately by all 64 elements of **Column 1**.
3.  **Global Structure**: The entire matrix is composed of these `(64x2)` tiles stacked vertically (down K) and then horizontally (across N). Memory is a linear sequence of these tiles.

### 6.3 Cross-Architecture Compatibility
While weight packing is architecture-aware, many architectures share the same layout format. The following table summarizes compatibility:

| Target Architecture | Compatible Packed Weights From... | details |
| :--- | :--- | :--- |
| **SM70 (Volta)* | *Not Supported* | (Requires Tensor Cores with Int8 support in this specific layout) |
| **SM75 (Turing)** | SM75, SM80, SM86, SM89, SM100+ | Uses LDSM Permutation + Interleaving. |
| **SM80 (Ampere)** | SM75, SM80, SM86, SM89, SM100+ | Same as above. |
| **SM86/89 (Ada/Lovelace)** | SM75, SM80, SM86, SM89, SM100+ | Same as above. |
| **SM90 (Hopper)** | **SM90 Only** | **Incompatible**. SM90 skips the interleaving step (`arch != 90`), using a Permuted-Linear layout. |
| **SM100/120 (Blackwell)** | SM75, SM80, SM86, SM89, SM100+ | Falls back to SM80 behavior (LDSM Permutation + Interleaving). INT4/INT8 weight packing compatible with Group A. |

**Summary**:
*   **Group A (Universal)**: SM75, SM80, SM86, SM89, SM100, SM120. Weights packed on any of these can be used on any other in this group.
*   **Group B (Hopper)**: SM90. Weights packed for SM90 are unique to SM90.
*   **FP4 (MXFP4)**: FP4 weights do not use `pack_weights_for_cuda_mixed_gemm`; they use a separate format (see [qmoe_fp4.md](qmoe_fp4.md)).

## 7. FP4 (MXFP4) Quantization Support

The QMoE operator has been extended to describe **MXFP4 quantized weights** (W4A16: FP4 weights + FP16/BF16 activations) via the `quant_type="fp4"` attribute. The intended execution path is the mixed-input TMA warp-specialized CUTLASS kernel path on SM90+ with CUDA 12.8+.

Current build caveat: the full SM90 mixed-input FP4 launcher (`moe_gemm_tma_ws_sm90_mixed_fp4.generated.cu`) is excluded from the build because it is incompatible with the bundled CUTLASS 4.4.2 mainloop. The build uses `moe_gemm_tma_ws_sm90_mixed_fp4_stub.cu` for link completeness; those stubs throw if reached at runtime.

Key additions:
- **`quant_type` attribute**: `"int"` (default, backward compatible) or `"fp4"` for MXFP4 mode
- **New inputs (indices 15-20)**: FP4 block scales (`uint8`, `float_ue8m0_t`/ue8m0 encoded) and per-expert global scales (`float`) for FC1/FC2/FC3
- **Template instantiations**: `MoeGemmRunner<half, __nv_fp4_e2m1, half>` and `MoeGemmRunner<__nv_bfloat16, __nv_fp4_e2m1, __nv_bfloat16>`
- **CUTLASS generalization**: Group size is type-dependent (32 for MXFP4 vs 128 for INT4), scale element type is `float_ue8m0_t` for FP4

For full design details, implementation status, and remaining work, see [qmoe_fp4.md](qmoe_fp4.md).

## 8. SwiGLU Details

The operator supports **SwiGLU** activation with support for interleaved inputs, which is critical for performance in certain model architectures (e.g., GPT-OSS).

### 8.1 Formula
The SwiGLU activation is computed as:
```
SwiGLU(x) = Gate * Sigmoid(alpha * Gate) * (Value + beta)
```
Where the input `x` contains both `Gate` and `Value` components.

### 8.2 MoE (Float16/BFloat16) Runtime Fusion
For the standard **MoE** operator (non-quantized), the `Compute` method includes logic to automatically fuse split weight tensors at runtime.

*   **Trigger**: If the optional `fc3_experts_weights` input is provided.
*   **Behavior**:
    *   The operator allocates a temporary buffer.
    *   It manually concatenates `fc1` (Gate) and `fc3` (Value) for each expert.
    *   **Resulting Layout**: `[Expert0: FC1|FC3, Expert1: FC1|FC3, ...]`.
    *   This fused buffer is then passed to the kernel, simulating `swiglu_fusion=2` (Block Fusion).
*   **Activation Check**: This path is taken implicitly when `fc3` is present, typically used with Gated activations like `SiLU` (Mixtral) or `SwiGLU`.

> **Note**: This runtime packing is specific to **standard MoE**. The **QMoE** operator does **not** perform runtime fusion; correct packing must be done offline (see Section 6).

> **FC3 in QMoE**: Because the CUTLASS `runMoe()` interface only accepts fc1 and fc2 weights (no fc3 parameter), QMoE requires that gate and up projection weights are always pre-concatenated into the fc1 tensor before model export. The `fc3_experts_weights` ONNX input exists for shape validation and backward compatibility with the op schema, but its data is not read during QMoE inference. For FP4/FP8 quantization modes, the same rule applies — block scales and global scales for the fused fc1 tensor cover both gate and up projections, and the fc3-specific scale/zero-point inputs (indices 9, 13, 17) are unused in the compute path.

### 8.3 Fusion Modes (`swiglu_fusion`)
The operator handles three distinct modes for SwiGLU, controlled by the `swiglu_fusion` attribute:

1.  **No Fusion (`swiglu_fusion=0`)**:
    *   **Inputs**: 3 distinct weight tensors (`fc1`, `fc2`, `fc3`).
    *   **Logic**: `fc1` (Gate) and `fc3` (Value) are provided separately. The kernel handles the computation and activation conceptually as if they were separate comparisons.

2.  **Interleaved Fusion (`swiglu_fusion=1`)**:
    *   **Inputs**: 2 distinct weight tensors (`fc1`, `fc2`).
        *   `fc1` contains *both* Gate and Value weights fused.
    *   **Memory Layout**: `[Gate_0, Value_0, Gate_1, Value_1, ..., Gate_N, Value_N]`.
    *   **Cutlass Requirements**: The kernel expects adjacent elements in the GEMM output to correspond to Gate and Value.
    *   **Usage**: Recommended for optimal performance on newer architectures as it aligns with interleaved GEMM optimizations.

3.  **Block Fusion (`swiglu_fusion=2`)**:
    *   **Inputs**: 2 distinct weight tensors (`fc1`, `fc2`).
    *   **Memory Layout**: `[Gate_0 ... Gate_N | Value_0 ... Value_N]` (Concatenated).
    *   **Logic**: The kernel processes the first half as Gate and the second half as Value.

## 9. Summary of Kernel Changes from TensorRT-LLM

The Cutlass kernels in this implementation are derived from TensorRT-LLM (cutlass 4.4.2, commit `346018db87`) but have been significantly modified.

### Key Modifications:

1.  **Pre-Packed ZP/Bias Optimization**:
    *   Implemented `PrePack` logic to pre-process Zero Points offline (or at initialization).
    *   **8-bit**: Pre-calculates `Bias = (128 - ZP) * Scale` to handle the weight shift efficiently.
    *   **4-bit**: Unpacks and stores unscaled ZPs to match Cutlass kernel requirements.

2.  **SwiGLU Interleaving**:
    *   Enhanced activation kernels to support **Interleaved** SwiGLU (as described in Section 8), allowing direct compatibility with weights packed for interleaved output.

3.  **Sparse Mixer Support**:
    *   Added support for Sparse Mixer architectures (controlled by `use_sparse_mixer`).

4.  **`supportsTmaWarpSpecialized()` interface**:
    *   Exposed on `CutlassMoeFCRunnerInterface` to allow dynamic `min_dim` selection without knowing the concrete template type at call sites.

5.  **Backported bug fix (TRT-LLM `603ec03f`)**:
    *   Moved `griddepcontrol.launch_dependents` to after `computeTmaWarpSpecializedInputPointers` in `computeStridesTmaWarpSpecializedKernel` to fix potential preexit race condition.

### Cleanup (2026-04-30):

The following TRT-LLM features were **removed** as not needed for MoE/QMoE:
- LoRA parameters (`use_lora`, `LoraParams`)
- Min-latency mode (`MoeMinLatencyParams`)
- AllToAll MoE paths (`enable_alltoall`)
- DeepSeek FP8 block-scale mode (`use_deepseek_fp8_block_scale`, `BlockScaleParams`)
- FP8 W4A8 / W8A8 kernel instantiations (to be restored in a follow-up PR)
- Deep Gemm, FP4 standalone gemm, FP8 blockscale gemm, fused gated gemm directories

## 10. Test Status
As of the current branch (`tlwu/20260503/qmoe_fp4`):
*   **MoE** (`test_moe_cuda.py`): existing CUDA MoE coverage should be used for FP16/BF16 SiLU/GeLU/SwiGLU regression.
*   **QMoE INT** (`test_qmoe_cuda.py`): existing INT4/INT8 QMoE coverage should remain the primary regression signal for the production QMoE path.
*   **QMoE FP4** (`test_qmoe_fp4_cuda.py`): the test file covers MXFP4 quantization utilities, packing, model construction, FP16/BF16, SiLU/SwiGLU, top-k, and expert-count variants. End-to-end runtime execution is not currently available because the SM90 mixed-input FP4 launcher is stubbed; tests that hit an unavailable FP4 build/path skip or raise a clear FP4/launcher error.
