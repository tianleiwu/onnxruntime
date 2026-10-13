# CUDA Expert Offload, ExpertStore, and GenAI Integration

Status: design + experimental implementation report (not an upstream contract).
Scope: ONNX Runtime (ORT) CUDA execution provider and ONNX Runtime GenAI.
Worked example: Qwen3.8 Flash-Next INT4 (48 MoE layers, 512 routed experts, top-10) under a 24 GiB GPU budget.

## 0. How to read this document

Every claim is labelled by maturity, because most of the machinery below exists only in private experimental
branches and some of it only on paper.

| Label | Meaning |
| --- | --- |
| **Implemented** | Present in the private ORT tree (`5fc711d515`) or private GenAI tree (`e634a664`) and exercised by tests. |
| **Measured** | Backed by a strict (fresh, rotated, audited) benchmark or canonical quality run. Pilots are called pilots. |
| **Branch-only** | Implemented only on the experimental ORT branch `qmoe/expertstore-candidate` (`32a01968b1`, runtime stage43). |
| **Proposed** | A design recommendation of this document; not implemented. |
| **Upstream** | Already in upstream ORT `main` / GenAI `main` (`cf827033`). |

Numbers come from the experiment ledger kept in the `dev` repository (see [Appendix C](#appendix-c-evidence-index)).
Benchmark numbers are for one machine (H200 class GPU, 96-core 2-socket Xeon, 1.8 TB RAM), one model and mostly
greedy decoding at 2K/8K/32K context with a batch of one. They illustrate the design trade-offs; they are not
general performance claims and no configuration described here is a default.

Throughout, the model-agnostic design is described first and Qwen3.8 Flash-Next is used as the running example.
"Qwen-specific" is called out explicitly.

## 1. Problem, goals and non-goals

### 1.1 Problem

Large Mixture-of-Experts (MoE) models hold most of their parameters in routed experts, but each token activates only
a few of them. When the whole model does not fit the GPU, the experts can live in a lower tier (host RAM, later SSD),
and each token's routed experts are served from wherever they are:

* **GPU-resident** experts execute on the GPU (fast path).
* **Cold** experts either execute on the **CPU** (weights never cross PCIe; only activations do), or are
  **streamed** to the GPU for execution.

Example (Qwen3.8 Flash-Next, INT4 weights, block size 32):

| Quantity | Value |
| --- | ---: |
| Routed experts | 48 layers x 512 = 24,576 |
| Experts used per token per layer (top-k) | 10 (480 routed experts / token) |
| Bytes per expert (weights + scales + biases) | 2,772,480 |
| All routed experts | 68,136,468,480 B (63.457 GiB) |
| Non-expert decoder weights | 2,476,191,768 B (2.31 GiB) |
| GPU budget (hard, incl. startup peaks) | 24 GiB (25,769,803,776 B) |
| All-miss expert traffic per token | 480 x 2,772,480 B = 1.239 GiB |

The "all-miss" row is the key number for every storage decision below: serving a token entirely from outside the GPU
moves 1.24 GiB, i.e. ~78 GB/s at 59 tok/s. Offload only works if most routed experts are **not** missed, or if misses
are served by compute placed next to the data (CPU experts).

### 1.2 Goals

1. Run MoE models whose routed-expert weights exceed the device budget, with decode throughput within reach of
   dedicated offload systems (the project compared against Strata) and without changing model numerics beyond
   documented, tested boundaries.
2. Hold a **hard** device budget (here 24 GiB), enforced inside ORT rather than by convention.
3. Make the storage tier an explicit, replaceable abstraction (RAM today; SSD or remote later) with leases,
   budgets and same-request error semantics.
4. Keep GenAI (the serving runtime) in charge of *policy* (admission, speculation, placement inputs) and ORT in charge
   of *mechanism* (layout, execution, memory).
5. Stage the work as small, independently reviewable PRs whose contracts survive the final design.

### 1.3 Non-goals (for this document)

* Multi-GPU expert parallelism (the location map is designed to be device-keyed; nothing multi-device is verified).
* Training, sampling-quality studies and non-greedy decoding (only greedy decoding is qualified).
* CUDA-graph capture of the offload operator (explicitly rejected today; see [Section 9.5](#95-cuda-graphs)).
* Making any experimental policy a default.

## 2. Architecture overview

```text
          model artifact (ONNX + external data + genai_config.json)
                                       |
                                       v
+------------------------------ GenAI (policy) ------------------------------+
| Config / loader       Budget / admission        Engine                     |
|  model.memory.*        device/host/pinned        dynamic batching          |
|  shared initializers   KV blocks, fixed state    phase/stats plumbing      |
|                                                  MTP / n-gram / hybrid     |
+----------------------------------------------------------------------------+
                                       |
                                       | gpu_device_mem_limit, token_phase, packed rows
                                       v
+------------------------- ORT CUDA EP (mechanism) --------------------------+
| Device budget ledger  QMoEOffload op            CPU expert kernels         |
|  reserve/commit/free   router top-k (GPU)        SharedExpertBlockV1       |
|                        resident slab (GPU hits)  AVX-512 + spin pool       |
|                        prefill streaming ring    cold experts run here     |
|                        cold dispatch ------->    (activations only)        |
+----------------------------------------------------------------------------+
                                       ^
                                       | expert bytes via leases (Branch-only seam)
                                       v
+-------------------- ExpertStore (RAM today, SSD later) --------------------+
| TensorExpertStore  zero-copy RAM views     (Branch-only, measured)         |
| FileExpertStore    bounded file-backed     (prototype, rejected as is)     |
+----------------------------------------------------------------------------+
```

Storage hierarchy (hot to cold):

| Tier | Holds | Accessed by | Today |
| --- | --- | --- | --- |
| GPU resident slab (per layer) | hot experts in SM80 W4A16 layout | grouped CUTLASS MoE runner | Implemented |
| GPU landing ring | experts streamed for a prefill tile | grouped runner, then optional D2D refill of the slab | Implemented |
| Pinned host | in-place registered expert blocks + staging/leases | DMA engine | Implemented (opt-in) |
| Pageable host RAM | all experts, one immutable initializer per node | CPU expert kernels, H2D source | Implemented |
| SSD / remote | experts as files | ExpertStore reads into pageable cache | Prototype only; not viable for decode misses ([Section 11](#11-ssd-and-tiered-storage-extension)) |

### 2.1 Responsibility split

| Concern | Owner | Why |
| --- | --- | --- |
| Expert byte layout and its versioning | ORT | The same bytes must be consumable by CPU and CUDA kernels; layout is part of the op contract. |
| Router, top-k, combine, numerics | ORT op | Numerics must be bit-reproducible against the unfused graph. |
| Which experts are resident (placement/epoch policy) | **Designed**: GenAI; **Implemented**: inside each op node (local frequency scores) | A global view needs request/phase knowledge GenAI has. Local scoring was enough for every measured result. |
| Device/host/pinned budgets | ORT ledger for allocations; GenAI for admission | The allocator must refuse over-budget allocations; admission needs to know KV/state demand. |
| Speculation (MTP, n-gram lookup, hybrid) | GenAI Engine | Needs transactional verify/rollback of KV and recurrent state. |
| Model export / conversion to shared layout | Offline converter (builder) | Weights are repacked offline; runtime never repacks. |

### 2.2 What is model-agnostic and what is Qwen-specific

| Area | Model-agnostic | Qwen3.8 Flash-Next specific |
| --- | --- | --- |
| `QMoEOffload` mechanism (top-k on GPU, cold CPU, resident slab, streaming) | Yes, for SwiGLU INT4 block-32 experts | Fixed `activation=swiglu`, `swiglu_fusion=1`, INT4 only today (a restriction, not a design limit). |
| `SharedExpertBlockV1` | Parameterised by `H`, `I`, block size, interleave; `H`,`I` multiples of 64 | Concrete stride 2,772,480 B at `H=2560`, `I=640`. |
| ExpertStore | Yes (opaque key + bytes) | - |
| Budget ledger | Yes (CUDA EP feature) | - |
| GenAI `model.memory.*`, `token_phase`, stats outputs | Yes | - |
| MTP head isolation, graph-query capacity | Yes (hosted one-layer head) | Head width 10,240 hidden handoff; builder emitters exist only for Qwen. |
| Hyperconnection ops, gated `ScaledSiLU`, `SplitScaledSiLUFusion` | No | Needed by the Flash-Next graph. |
| MatMulNBits tail kernel (`N % 8 == 4`) | Kernel is general | Motivated by `N=324, K=10240` hyperconnection projections. |

## 3. Expert storage: the ExpertStore seam

### 3.1 Why a seam

Today's shared-offload node consumes one **whole CPU initializer** (`expert_blocks[E, stride]`): GenAI reads the
entire declared range into one immutable allocation, ORT borrows it, CPU kernels read it in place and H2D copies read
from it. That model has three structural limits:

* It cannot hold fewer bytes in RAM than the model (no bounded-RAM mode, no SSD tier).
* Lifetime is implicit: a reader just dereferences the buffer. Any tier that can *evict* needs an explicit
  acquisition/release protocol.
* Errors have nowhere to go: a read cannot fail, so a future storage tier has no way to fail the owning request.

The ExpertStore seam makes "give me the bytes of expert *e* in layer *l*" an operation with an identity, a lease and a
failure mode, without changing arithmetic.

### 3.2 Interface (prototype, `dev/prototypes/expert_store`; Branch-only integration)

Identity (immutable):

```text
ExpertKey = (weights_namespace, layer, expert, layout_version, revision)
```

`revision` lets a refreshed weight set coexist with in-flight readers; `layout_version` is the
`SharedExpertBlockV1` version.

Operations:

| Operation | Contract |
| --- | --- |
| `Acquire(key) -> Lease` | Returns a ready view (pointer, size, alignment) or an error. While the lease lives the entry cannot be evicted. Blocks only for stores that fetch on demand. |
| `Prefetch(keys, generation)` | Deduplicated, bounded asynchronous fetch. Late completions of a cancelled generation are discarded. |
| `Wait(key)` / `Inspect(key)` | Observe the state machine below. |
| `GetBudgetSnapshot()` / `GetDiagnostics()` | Pageable cache, pinned staging, in-flight IO bytes and peaks; hit/miss/eviction counters. |

State machine per entry:

```text
 Missing --Prefetch/Acquire--> Fetching --ok--> Ready --evict (no leases)--> Missing
                                  |                |
                                  +-- short/corrupt/missing read --> Failed (request-level error, no partial data visible)
```

Invariants (all unit-tested in the prototype under ASan/UBSan; TSan could not start on the test host):

1. A leased entry is never evicted; capacity pressure fails the acquire instead.
2. Concurrent prefetches of one key share one read.
3. Aggregate pageable cache + pinned staging + in-flight IO never exceeds the configured limits.
4. Cancellation uses generations: cache contents may outlive a request, unfinished operations may not.
5. Short, corrupt (checksum) or unavailable reads fail the *owning request*; no stale or partial bytes become visible.
6. No blocking disk IO and no CUDA-dependent waits inside CUDA host callbacks or the token-commit path.
7. Storage availability must never silently switch arithmetic policy (e.g. CPU vs GPU execution of a cold expert).

### 3.3 Implementations

| Store | Behaviour | Status |
| --- | --- | --- |
| `TensorExpertStore` | Zero-copy ready views over the existing immutable initializer; `Acquire` never blocks or fails; supports the async host-callback path. | Branch-only, **Measured**: token-identical to the direct-pointer path, 60.69 vs 59.09 tok/s at 2K (+2.7%, within run-to-run noise). |
| `FileExpertStore` | Background IO workers, bounded pageable cache, pinned staging and in-flight budgets, per-expert files or per-layer containers (`layerNN.xptpack` + TSV manifest). Not compatible with the async callback or in-place pinning paths (fails closed). | Prototype + Branch-only integration; **rejected in its current form** (Section 11). |

### 3.4 How the operator uses it (Branch-only)

Selected by `ORT_QMOE_SHARED_EXPERT_STORE=tensor|file` (unset keeps the legacy direct-pointer hot loop, bit-for-bit
the old path). Before each CPU cold-expert phase the op batches `Prefetch` for the routed cold experts and then
`Acquire`s a lease per expert and runs the existing CPU kernel on the leased bytes (file mode waits up to 5 minutes for
prefetches, in batches bounded by the in-flight IO limit). Lease release timing relative to the dependent H2D copy is
part of the open lifetime work in Section 12. Configuration (`ORT_QMOE_SHARED_EXPERT_{NAMESPACE,REVISION,FILE,CACHE_MB,IO_THREADS}`) fails
closed on invalid values. File-mode cache bytes are tracked in the store's own budget snapshot, **not** the CUDA device
ledger (host memory); a production version must register them in the host-side ledger (Section 8).

### 3.5 Proposed production shape

* The store is an ORT **resource keyed by (session or device, namespace)**, not a process-global singleton and not
  an environment-variable switch. Selection should be an op attribute / session option so it is visible in the model
  and tests.
* Leases are tied to the **request** that acquired them and are released only after the last consumer (CPU kernel,
  H2D copy) completes, using the same completion token as the pinned staging lease.
* Expert bytes are never copied by the store when the backing is RAM (preserves the measured zero-cost seam).
* The seam is internal. It should not become a public API until a second backing (file) has proven correct end to end.

## 4. Shared expert layout (`SharedExpertBlockV1`)

### 4.1 Idea and alternatives

All experts of a layer live in one `[E, stride]` byte tensor. The **same bytes** feed CPU kernels and the CUDA grouped
runner, so an expert can move between tiers by copying `stride` bytes and no repacking is ever done at run time.

| Alternative | Verdict |
| --- | --- |
| Keep stock `QMoE` weights (prepacked for CUDA) and a second CPU copy | Doubles host memory (63 GiB -> 127 GiB for Qwen); rejected. |
| Repack at model load (`PrePack`) per tier | Slow load, extra temporary memory, two formats to validate; rejected. |
| **One offline-prepacked layout consumed directly by CPU and CUDA** | Chosen. CPU reorders the *activation* row once per call and decodes nibbles on the fly instead of rewriting weights. |

### 4.2 Layout contract (v1)

Generic parameters: hidden size `H`, intermediate size `I`, quantization block 32, 4-bit codes stored as `q+8`,
CUTLASS SM80 W4A16 K-permutation and column interleave (`ColumnMajorTileInterleave<64,4>`), 256-byte bank alignment,
FP16 scales/biases. Per-expert banks in order: FC1 weights, FC1 scales, FC1 bias, FC2 weights, FC2 scales, FC2 bias.

Qwen3.8 (`H=2560`, `I=640`):

| Bank | Formula | Bytes | Offset |
| --- | --- | ---: | ---: |
| FC1 weights | `2I*H/2` | 1,638,400 | 0 |
| FC1 scales | `(H/32)*2I*2` | 204,800 | 1,638,400 |
| FC1 bias | `2I*2` | 2,560 | 1,843,200 |
| FC2 weights | `H*I/2` | 819,200 | 1,845,760 |
| FC2 scales | `(I/32)*H*2` | 102,400 | 2,664,960 |
| FC2 bias | `H*2` | 5,120 | 2,767,360 |
| **Stride** | aligned | **2,772,480** | |

Acceptance contract: `weights_prepacked=1`, `block_size=32`, `layout_version=1`, rank/stride/alignment checks. The
runtime never calls CPU or CUDA `PrePack()` for this layout. The manifest id `ort.qmoe.w4a16.block32.shared.v1` is
recorded offline; at runtime only `layout_version` is enforced (see Section 12, defects).

### 4.3 Conversion

* **Implemented (test/helper code):** `assemble_blocks_from_prepacked()` converts legacy CUDA-prepacked banks into
  shared blocks (weight bytes copied verbatim, scales transposed to block-major) with golden tests against real
  artifact layers (0/17/47) proving the CPU kernel matches the CUDA op.
* **Branch-only:** `qmoe_export_expert_store.py` writes per-layer `.xptpack` containers + manifest for the file store.
* **Missing:** a supported, versioned converter in the GenAI model builder (Section 9.2).

### 4.4 Versioning rules (Proposed)

A layout change is a new `layout_version` plus a new namespace/revision in the ExpertStore key; v1 stays readable.
Layout identity must be carried by the op (attribute) **and** the manifest string and both must be verified.

## 5. The `QMoEOffload` operator

`com.microsoft::QMoEOffload` (private; upstream has only the stock `QMoE`). It reuses the stock router
(`LaunchSoftmaxTopK`), the CUTLASS grouped MoE runner and the gather/scatter/finalize path, and adds cold-expert
execution, residency and phase/statistics plumbing.

### 5.1 Interface: current schema and what changes

The private schema is a superset of stock `QMoE` with 26 input slots (0..25) and 4 outputs, because it was grown from a
legacy six-bank prototype and keeps stock `QMoE` slot numbering. It has two mutually exclusive modes: **legacy**
(six CPU weight/scale/bias banks, bounded eager staging to a GPU scratch, no CPU compute, selected by omitting
`expert_blocks`) and **shared** (`expert_blocks`, CPU cold experts, residency, streaming). Shared mode is the
one all measurements use; legacy mode was the first prototype (4.09 tok/s at 2K) and is **proposed for removal**.

The disposition column states what happens to each slot in the proposed upstream operator:
**Keep**, **Remove (legacy)** = only used by the six-bank mode, **Remove (placeholder)** = stock-`QMoE` slot that
`QMoEOffload` must reject, **Defer** = schema-present but runtime-rejected, to be re-added additively when implemented.

Inputs:

| Idx | Name | Type / memory | Current status | Proposed |
| ---: | --- | --- | --- | --- |
| 0 | `input` | fp16, CUDA, `[T,H]` (legacy also `[B,S,H]`) | Implemented; shared mode rejects 3D | **Keep** (packed `[T,H]` only) |
| 1 | `router_probs` | fp16, CUDA, `[T,E]` | Implemented; top-k stays on GPU | **Keep** |
| 2 | `fc1_experts_weights` | uint8, CPU, `[E,H,I]` | Legacy bank 1 of 6 | **Remove (legacy)** |
| 3 | `fc1_scales` | fp16, CPU | Legacy bank 2 | **Remove (legacy)** |
| 4 | `fc1_experts_bias` | fp16, CPU | Legacy bank 3 | **Remove (legacy)** |
| 5 | `fc2_experts_weights` | uint8, CPU | Legacy bank 4 | **Remove (legacy)** |
| 6 | `fc2_scales` | fp16, CPU | Legacy bank 5 | **Remove (legacy)** |
| 7 | `fc2_experts_bias` | fp16, CPU | Legacy bank 6 | **Remove (legacy)** |
| 8-10 | `fc3_experts_{weights,scales,bias}` | - | Must be omitted | **Remove (placeholder)** |
| 11-13 | `fc{1,2,3}_zero_points` | - | Must be omitted (symmetric INT4 only) | **Remove (placeholder)** |
| 14 | `router_weights` | - | Must be omitted | **Remove (placeholder)** |
| 15-20 | `fc{1,2}_global_scale`, `fc{1,2}_act_scale`, `fc{1,2}_act_block_scale` | - | Must be omitted (FP4/FP8 slots) | **Remove (placeholder)** |
| 21 | `expert_blocks` | uint8, **CPU**, `[E,stride]` | Implemented; immutable, borrowed pageable memory (or a store view) | **Keep**, becomes the weight input (renumbered 2) |
| 22 | `slab` | uint8, CUDA | Schema-present, **runtime-rejected** | **Defer** (re-add as an optional trailing input with the placement manager) |
| 23 | `slot_tables` | int32, CUDA `[L,E]` | Schema-present, **runtime-rejected**; paired with `slab` | **Defer** (same) |
| 24 | `fc2_bias` | fp16, CUDA, `[E,H]` | Implemented, required in shared mode; applied once at combine | **Keep** (required, renumbered 3) |
| 25 | `token_phase` | int8, CUDA, `[T]` | Implemented, required in shared mode; phase 0..4 (Section 9.3) | **Keep** (required, renumbered 4) |

Outputs (no output is specific to the six-bank mode):

| Idx | Name | Type / memory | Current status | Proposed |
| ---: | --- | --- | --- | --- |
| 0 | `output` | fp16, CUDA, same shape as input | Implemented | **Keep** |
| 1 | `route_rows` | int32, CPU, `[5,E]` | Schema-optional but runtime-required in shared mode | **Keep**, make required |
| 2 | `route_calls` | int32, CPU, `[5,E]` (0/1 "expert active this invocation") | Same | **Keep**, make required |
| 3 | `call_counts` | int32, CPU, `[6]` | Same; index 5 ("staged expert uploads") is documented but never incremented | **Keep**, make required; either implement index 5 or shrink to `[5]` |

Resulting target signature (shared-only operator, inputs renumbered; placement inputs appended later):

```text
QMoEOffload(input[T,H], router_probs[T,E], expert_blocks[E,stride] (CPU), fc2_bias[E,H], token_phase[T])
    -> (output[T,H], route_rows[5,E] (CPU), route_calls[5,E] (CPU), call_counts[5] (CPU))
    [+ optional slab, slot_tables  (Deferred; trailing)]
```

This removes 19 of the 26 input slots (2-20) and defers 2 more (22-23), leaving 5 inputs, so the signature matches what is
actually executed. Because `QMoEOffload` has never shipped, no compatibility shim is needed.

Attributes:

| Attribute | Current | Proposed |
| --- | --- | --- |
| `activation_type` (swiglu), `swiglu_fusion` (1), `k`, `normalize_routing_weights`, `activation_alpha/beta` | Validated/fixed | **Keep** (validate; extend later for other activations) |
| `quant_type` (`int`), `expert_weight_bits` (4), `block_size` (32) | Validated/fixed | **Keep** (needed for future 2/8-bit and block sizes via layout versions) |
| `layout_version`, `hidden_size`, `inter_size`, `layer_index` | Required in shared mode | **Keep**; add a verified `layout_id` string |
| `token_tile_size` (32), `gpu_working_bytes` (64 MiB), `pinned_staging_bytes` (8 MiB) | Implemented | **Keep** |
| `fc{1,2,3}_expert_weight_bits` | Per-FC overrides that must equal 4 | **Remove** |
| `expert_wave_size` | Legacy eager-staging wave size | **Remove (legacy)** (streaming wave stays an op/session option) |
| `packing_layout` (`cutlass_sm80_int4_v1`), `weights_prepacked` (`-1` or `1`) | Legacy bank packing identity; shared mode requires `weights_prepacked=1` | **Remove**; identity is carried by `layout_version`/`layout_id` |
| `expert_cache_bytes` (must be 0) | Placeholder | **Remove**; replaced by explicit resident-slot and store configuration |
| `accuracy_level` (must be 0), `use_sparse_mixer` (must be 0) | Placeholders | **Remove** |
| `swiglu_limit` | Optional clamp inherited from `QMoE` | **Verify** shared-mode support or remove |
| *(new)* resident slots, streaming/async selectors, store selector, namespace, revision | Today: process-wide `ORT_QMOE_SHARED_*` environment variables | **Add** as attributes/session options (Section 5.4) |

`normalize_routing_weights` is kept; the shared path uses the original full-pool top-k scores and never renormalizes
away cold experts.

### 5.2 Differences from stock `QMoE`

| Aspect | Stock `QMoE` | `QMoEOffload` |
| --- | --- | --- |
| Weights | CUDA tensors | One CPU `uint8` `[E,stride]` tensor |
| Cold experts | n/a (all resident) | CPU kernels or streamed H2D |
| Quantization | int/fp4/nvfp4/fp8/wfp4afp8 | INT4 block-32 only |
| Residency | n/a | Per-node resident slab (+ designed external slab/slot tables) |
| Extra outputs | none | Route/phase statistics on CPU |
| Capture | graph-capturable | **Rejected** (host callbacks, pinned leases) |

### 5.3 Execution modes (all Implemented)

| Mode | What runs where | Enabled by |
| --- | --- | --- |
| Zero-cache sync | GPU top-k -> D2H route readback -> CPU cold experts -> H2D contributions -> GPU combine | default shared mode |
| Resident decode | hot routes on GPU from the slab; cold on CPU | `ORT_QMOE_SHARED_RESIDENT_SLOTS>0` |
| Async decode | `cudaLaunchHostFunc` runs the CPU work; GPU stream keeps launching; optional aux stream runs resident hits during the callback | `ORT_QMOE_SHARED_ASYNC=1` (+aux) |
| Prefill streaming | heavily routed cold experts copied into a 2-bank landing ring, run on GPU in waves, refill slab via D2D | `ORT_QMOE_SHARED_STREAM_ROWS` |

### 5.4 Proposed schema cleanup before upstreaming

1. Replace behaviour-selecting environment variables with attributes/session options (they are process-wide today).
2. Make the schema match the runtime: drop the legacy six-bank inputs and stock-`QMoE` placeholders, defer `slab`/`slot_tables`
   until implemented, make outputs 1..3 required (Section 5.1).
3. Add an `expert_store` selector and namespace/revision attributes instead of `ORT_QMOE_SHARED_EXPERT_*`.
4. Carry `layout_id` (string) and verify it, not only `layout_version`.
5. Define request boundaries for ragged steps (or declare "one request per step" for the first version).

## 6. Execution design and its performance impact

### 6.1 Per-layer timeline (batch-one decode)

Synchronous shared path (what the baseline numbers measure):

1. GPU: router softmax + top-k.
2. D2H readback of indices/phase/activation tile; **one stream synchronize**.
3. Host classifies each route: resident hit vs cold; updates counters.
4. Resident hits: gather -> grouped runner against the slab -> scatter (GPU).
5. Cold: CPU FC1 (all cold experts) then FC2, each a parallel-for over the spin pool.
6. H2D of CPU contributions; GPU combine applies `fc2_bias` once and finalizes in rank order.
7. After the drain: promote the best-scoring cold experts into the slab.

Async path: step 2 and 5 move to a host callback on a callback stream; resident hits run on an aux stream while the
callback computes cold experts; the main stream merges (`LaunchQMoEOffloadMerge`). A second host callback releases
pinned leases after the contribution copy.

### 6.2 CPU expert kernel

* Direct consumption of the CUDA bytes; activations reordered once per row; FP32 accumulation; scalar and AVX-512
  (`avx512f/bw/vl + f16c`) paths; at most 8 rows per expert call.
* FC1 task width 64 columns and FC2 width 128 (`ORT_QMOE_SHARED_FC{1,2}_COLS`) replaced 32/32.
* Process-wide **generation-tagged spin pool** (`ORT_QMOE_SHARED_CPU_THREADS`, 40 default) with optional task batching
  (`ORT_QMOE_SHARED_POOL_TASK_BATCH`).
* Numeric contract: CPU contributions are FP16, combined in FP32 rank order on GPU. An opt-in FP16 boundary mode
  (`ORT_QMOE_SHARED_CPU_FP16_BOUNDARIES`) is available; it was not accepted (quality, below).

### 6.3 Resident cache and promotion

Per node (layer) slab of `ORT_QMOE_SHARED_RESIDENT_SLOTS` experts. Each routed expert accumulates a hotness score;
scores are halved every `RESIDENT_DECAY_ROUTES` (4096) routes. After a call the best cold candidate is promoted if
`score >= 2` and either a slot is free or `best >= lowest*margin/100 + 2`. Options: separate scores for prefill vs
non-prefill phases (`RESIDENT_PHASE_SCORES`), hysteresis margin, promotions per call (2; 48 for the streaming arm).
Promotion sources: CPU blocks over H2D (optionally from pinned-registered memory), or D2D reuse of the last two
streaming waves (`STREAM_PROMOTION_REUSE`, saves exactly 2,767,360 B of H2D per reused expert).

Design rationale: `calls` (not `rows`) is the hotness statistic, because a cold expert re-reads its 2.77 MB once per
call regardless of rows. Prior art (Strata) showed per-layer slabs matter: first-come admission gave 2.97% hits vs 21.4%
(8 slots/layer) and 70.4% (64 slots/layer).

### 6.4 Prefill streaming

For multi-tile calls, cold experts with at least `STREAM_MIN_ROWS` (16) rows are copied into the landing ring in waves
(`STREAM_WAVE`, 32) and run on the GPU instead of on the CPU; the last two waves can refill the resident slab.
Reason: prefill routes many rows per expert, so GPU execution amortizes the 2.77 MB copy, whereas decode routes ~1 row
per expert where CPU execution wins (Exp20: even ideal decode streaming would help only ~3%).

### 6.5 Measured impact of each design choice (Qwen3.8 Flash-Next, batch 1, greedy)

All rows are strict medians unless marked pilot. Quality column: `-` = no canonical run for that row.
Decode is tok/s; contexts 2K / 8K / 32K.

| # | Design choice (cumulative where stated) | Decode tok/s | TTFT s | VRAM peak | Verdict |
| --- | --- | --- | --- | --- | --- |
| 1 | Eager staging, repeated H2D every forward (prototype) | 4.09 (2K) | - | - | 135.9 ms exposed staging, 3,851 H2D, 7,030 launches per forward; rejected |
| 2 | Shared layout + CPU cold experts, pinned scratch reuse (smoke) | 16.8 (pilot) | 34.6 | 14.8 GiB | Proof of concept (5.0 -> 16.8) |
| 3 | Grouped CPU dispatch, 40 threads | 25.3 / 25.4 / 25.4 | 20 / 84 / 342 | 14.8 GiB | Flat in context: attention is not the limiter; GPU idle 72% |
| 4 | + async host-callback CPU experts | 27.6 / 27.4 / 27.4 | 21 / 87 / 352 | 14.8 GiB | +8-9% exact |
| 5 | + resident slab (80 slots/layer) + aux stream | 32.5 / 32.2 / 41.1 | 7.6 / 30.7 / 122 | 20.5 GiB | 48 and 64 slots weaker; 96 no better; 112/128 exceed quota |
| 6 | + MTP1 head (resident 48) | 45.1 / 41.1 / 56.6 | 10 / 42 / 168 | 20.1 GiB | Fewer forwards per token: the only big lever once host-bound |
| 7 | + AVX-512 conversions, widths 64/128 | 51.1 / 47.0 / 59.8 | - | 20-21 GiB | Exact +; callback 253.7 -> 227.7 us/layer at 8K |
| 8 | + prefill streaming (stream1024, phase scores, 48 promotions) | 56.3 / 47.4 / 59.0 | **1.8 / 8.8 / 37.7** | 20-21 GiB | ~4x TTFT win; canonical quality 142/200 (fail) |
| 9 | Lookup (n-gram) speculation, size 4, 6 chained drafts, capacity-8 graphs (the project's selected config) | 52.4 / 57.7 / **120.3** | 2.0 / 8.1 / 35.9 | <=20.8 GiB | Closed the *repetitive* 32K gap; 150/200 (fail) |
| 10 | Stage36: tail-safe INT4 small-M kernel + stream4 | **59.09** (2K) | 2.08 | - | +12.7% decode vs control; 153/200 (pass) |
| 11 | Stage34 8K: stream4 + intra8 | 91.4 (8K; 95.8 in a later paired triple) | 8.5 | - | Beat Strata decode (86.6); frozen 1024-cap gate 150/200 (fail), but passes the pre-registered 8192-cap side protocol (Section 10); TTFT gap to Strata (2.6 s) remains |

Reference comparator on matched synthetic prompts (Strata v0.1.42): 88.5 / 86.6 / 119.4 tok/s decode, 1.26 / 2.64 / 10.5 s
TTFT, 22.9 GiB. The final lookup-only ORT arm closes only the repetitive 32K *decode* gap; 2K/8K decode, TTFT and
request-throughput gaps remain, and representative (non-repetitive) 32K inputs reached only 41-42 tok/s.

Resident hit fractions (selected arm, share of routed experts served from the slab): 64.75% at 2K, 53.25% at 8K,
92.92% at 32K (repetitive). Decode speed tracks the *decode-phase* hit rate, which is why prefill streaming needed
separate phase scores and more promotions merely to recover decode.

### 6.6 Optimizations that did not pay off (record of negative results)

| Idea | Result |
| --- | --- |
| Resident48 alone, 2K | No isolated gain; helped only 8-way fast gate |
| Phase scores / hysteresis (Exp13/14) | Hits 38% -> 49-58%, promotions halved, throughput flat, outputs changed, no approval |
| Pin experts for promotions (Exp6) | Profiled memcpy 579 -> 46 ms, end-to-end +0-2% |
| Grouped QSA top-k merge (Exp10) | +0.5/+0.2/-0.6%: launch reduction off the critical path |
| Pad MatMulNBits `N` 324 -> 384 (Exp11) | -14%, slower generic path, token divergence |
| fpA_intB dispatch on 365 eligible projections | 44.2 vs 51.7 tok/s (slower) |
| `SplitScaledSiLU` fusion (exact, 96 rewrites) | +0.1..1.9%, no standalone case |
| Decode-time streaming of misses (Exp20) | +1.3% screen; model says <= ~3% |
| Fallback dequant cache | No end-to-end win; left off |
| Threads: `intra8` on 2K | Pilot win vanished under strict triples (52.2 vs 52.7) |

### 6.7 CPU pool batching and NUMA (stream D, CPU-only)

At 40 threads, requesting a batch >1 clamps to an effective chunk of `max(1, tasks/threads)` = 5 (200 tasks/phase ->
40 claims). Scheduling-only work gains 2-8%; real memory-bound FC1/FC2 work gains 10-28% depending on placement and
number of concurrent submitters. Interleaved or unpinned placement was most robust: pinning all work to node 0
collapsed from 2.49 M to 0.91 M tasks/s when another process loaded node 0 (p99 ~3.9 ms vs 0.2 ms). Conclusion: placement matters more than the
batch knob; default stays 1.

### 6.8 A first-order model for choosing a placement policy

Let `L` layers, `k` experts per token, `S` bytes per expert, `h` the resident hit rate, `R` decode tokens/s.

* CPU cold work per token = `L*k*(1-h)*S`. Qwen at `h=0.65`: ~169 cold experts, ~0.47 GB/token; at 59 tok/s ~28 GB/s of
  host memory reads (spread over ~40 threads, hence the NUMA sensitivity). *(derived arithmetic)*
* H2D streaming of the same bytes would take ~0.47 GB / 35 GB/s ~= 13 ms/token (measured effective H2D 35.3 GB/s), i.e.
  a ceiling of ~75 tok/s before any overlap loss, versus ~17 ms per token available at 59 tok/s. CPU execution moves only
  activations, which is why decode uses CPU experts and prefill (many rows per expert) uses streaming.
* SSD service of misses: required hit rate `h >= 1 - B/(R*L*k*S)` (Section 11).

## 7. Speculation as a throughput multiplier

Once decode is host-bound (launch/sync/callback overhead per forward), the largest lever is **fewer target forwards per
token**. This is a GenAI Engine feature that interacts with the operator only through `token_phase` and the packed `[T,H]`
interface (verification rows have `T>1`, which the operator treats as multi-row expert calls).

| Policy | Mechanism | Forwards / output token (target, head) |
| --- | --- | --- |
| Target-only | none | 1.00, 0 |
| MTP1/2/3 | hosted one-layer neural head drafts 1-3 tokens | 0.56, 0.56 / 0.56, 1.10 / 0.54, 1.61 |
| Lookup (n-gram) | committed-history n-gram proposer, default off, fixed width | 0.48, 0 |
| **Hybrid** | lookup first; head skipped on hits; one MTP draft on misses | 0.32, 0.21 |

Strict 2K medians on one workload (same base, same model), decode / request tok/s / TTFT s:
MTP1 50.7 / 43.6 / 1.69, lookup 59.2 / 48.1 / 1.92, **hybrid 70.9 / 56.5 / 1.85**
(hybrid +40% decode over MTP1, +20% over lookup). MTP2 and MTP3 pay more head work for less accepted work
(44.3 and 38.2 tok/s). Hybrid does not reduce RAM/VRAM footprint (68 GiB RAM, 23.5 GiB VRAM).

Quality: hybrid scored 148/200 on the canonical gate (reference 151, lookup 152-153). Investigation found the shortfall
consistent with truncation-boundary noise (8 reference-only cap flips and 0 completed-answer losses; hybrid not worse
on questions both completed; Engine argmax-vote audit symmetric, 21/17/2 vs control 19/21/0) but the frozen gate result is a FAIL, so hybrid
stays **experimental and default-off**. The offload operator itself is policy-neutral; hybrid on the tensor-store
runtime was token-identical to hybrid on the baseline runtime (69.84 vs 69.48 tok/s).

## 8. Memory budgeting

### 8.1 Device ledger (ORT, Implemented)

`CUDAExecutionProviderInfo::gpu_device_mem_limit` (0 = off) installs a per-device `CudaDeviceMemoryBudget {limit,
committed, pending, peak}`. `Reserve(bytes)` checks `bytes <= limit - committed - pending` *before* allocation;
`Commit()` moves pending to committed; `Release()` decrements; an uncommitted reservation aborts on destruction.
Concurrent reservations are tested (32 workers on a 1,024-byte budget: exactly 8 succeed). It is rejected with external
allocators or CUDA memory pools.

### 8.2 What the ledger does not yet cover (audited gaps)

| Allocation | Counted? |
| --- | --- |
| `CUDAAllocator` / `GetScratchBuffer()` / arena, resident slab, slot tables | Yes |
| Streaming landing ring (`cudaMalloc` in a process-global owner) | **No** |
| Streaming pinned ring (`cudaHostAlloc`) | **No** |
| `cudaHostRegister` of the expert initializer | **No** |
| Borrowed pageable expert backing | **No** (host memory) |
| ExpertStore pageable/pinned/in-flight (Branch-only) | Own snapshot, not the ledger |
| Plugin `Device*` stats through generic `AllocatorStats` | **Dropped** by the bridge (reproducer exists) |

The consequence is that the quota is enforced by an *external* wrapper (MPS memory limit + reserve) in all measurements,
and the ledger is a supporting guard, not the sole enforcer.

### 8.3 GenAI budget configuration and admission (Implemented, private)

| Field | Meaning |
| --- | --- |
| `model.memory.device_limit_bytes` | Hard device cap; GenAI forwards `limit - safety_margin` as `gpu_device_mem_limit` before allocator init. |
| `model.memory.safety_margin_bytes`, `scratch_reserve_bytes` | Reserve; scratch is subtracted before KV sizing. |
| `model.memory.host_weight_limit_bytes` | Cap on bytes of CPU-placed shared initializers (checked before load). |
| `model.memory.pinned_staging_bytes` | Required nonzero for offload. |
| `model.memory.expert_cache_bytes` | Must be 0 today (placement is local to ops). |
| `model.memory.offload_mode` | `disabled | gpu_stream | hybrid_frequency`; `hybrid_frequency` requires `token_phase` and all three statistics outputs. |
| `model.decoder.shared_initializers[].placement`, `.packing_layout` | `cpu|device`; layout identity used for validation and dedup. |

Offload checkpoints are gated to: CUDA graphs off, dynamic batching present, `max_batch_size == 1`, no `model.draft`, and MTP only with
an explicit opt-in. Paged KV is sized from the free bytes left after the scratch reserve and graph-buffer reservation,
with a 0.9 fragmentation factor.

### 8.4 Qwen3.8 budget walk-through

| Item | Bytes / GiB | Where |
| --- | --- | --- |
| All routed experts | 63.46 GiB | host RAM (pageable) |
| MTP head experts (1 layer x 512) | 1.32 GiB | host RAM; combined CPU backing 64.78 GiB |
| Non-expert decoder weights | 2.31 GiB | GPU |
| MTP head non-expert (shared LM head excluded) | 1.26 GiB | GPU |
| Paged KV (block 256, target+head) | 6,815,744 B/block | GPU |
| Resident slabs (80 slots/layer) + landing ring (1 GiB) + scratch | remainder | GPU |
| Measured peak VRAM | 19.8-23.5 GiB (arm dependent) | GPU |
| Measured peak host RAM | 68.0-68.6 GiB | host |

What remains for expert cache is a *residual* of everything fixed (weights, KV, recurrent/indexer state, activations,
workspaces, transfer tiles, MTP state, CUDA overhead). With zero cache the observed peak was 14.64 GiB, leaving at
most ~9.4 GiB (~3,600 experts, 14.7% of all experts) before shrinking KV.

## 9. GenAI changes

### 9.1 Configuration surface (`genai_config.json`)

Section 8.3 lists the memory fields. Decoder additions: optional `inputs.token_phase` (int8, one entry per packed row) and
optional `outputs.offload_route_rows/offload_route_calls/offload_call_counts` (per-layer int32 CPU outputs, shapes
`[6]` or `[5,N]`, all-or-nothing). Speculation: `speculative.ngram_size` (2..16), `ngram_chained_lookup`,
`ngram_mtp_fallback` (Engine-only; rejected by `Generator`/`MtpGenerator`). Runtime-only: `engine.max_graph_query_tokens_per_request`,
`engine.hidden_states_output_required`.

### 9.2 Loading and the builder

* **Implemented:** shared initializers can be `placement: cpu`; the loader sums CPU bytes against
  `host_weight_limit_bytes`, validates file bounds and exact `length == sizeof(dtype)*prod(shape)`, reads directly into the
  CPU allocation (device placement still stages through a temporary vector), and dedups by (file identity, offset, length,
  dtype, device, packing layout, shape). Budget forwarding happens before allocator creation, and a second session
  requesting a conflicting device or memory policy against the global allocator is rejected.
* **Missing in the checked-in GenAI builders:** emission of `placement`, `packing_layout`, QMoEOffload node attributes,
  shared-layout payloads, `token_phase` and statistics outputs. Today these come from an offline converter. A builder
  step that turns a stock `QMoE` model into shared-layout `QMoEOffload` is the main missing artifact path.
* **Qwen-only emitters exist** for the MTP head (`add_mtp_to_genai_config`, shared-initializer overlap with the target LM
  head, hidden handoff width 10,240).

### 9.3 Phase signalling and statistics

`token_phase` (per packed row): 0 prefill, 1 ordinary decode, 2 target verification (draft tokens present), 3 first
hosted-head pass, 4 chained head passes. It exists so the operator (and later a placement manager) can score hotness per
phase; it requires **eager** Engine execution. Statistics outputs return per-layer routed rows and calls per phase.
**No GenAI consumer reads them yet.**

### 9.4 Placement manager (Proposed; not implemented)

The design describes a GenAI-owned `ExpertCacheManager` that (a) is created before the Engine, (b) grants per-layer slab
capacity from the device budget, (c) consumes the statistics outputs per epoch, (d) publishes immutable `slot_tables`
snapshots with rotating spare slots, and (e) holds leases for in-flight invocations. Only a ranking object
(`ExpertCachePolicy`: phase weights `[0.1,1,1,0.25,1]`, half-lives, minimum support/residency, hysteresis) exists, with unit tests
and no runtime consumer. Every measured result used the **local per-node policy**. Reasons to still want a global
manager: namespace separation (target vs head), cross-layer budget sharing, profile seeding, fairness across
requests. Reasons to defer: no measured gain yet beyond local scoring, and it needs the keyed-owner and snapshot
protocols to be correct first (Section 12).

### 9.5 CUDA graphs

* Target and hosted-head decoders can use CUDA graphs in GenAI (graph id encodes query width, batch, block-table bucket,
  fixed-state binding). The head-only capture path was qualified on real hardware: 280 replays, widths 1/2, four graph
  ids, 16 warm-up + 512 measured tokens with eager-identical tokens and MTP counters; the first instantiation costs 2 MiB
  and capture windows retain 0 net bytes.
* **Offload and capture do not compose today.** `QMoEOffload` rejects capture (host callbacks, pinned leases) and GenAI
  rejects offload checkpoints with graphs on. Capturing the cold path would need stable addresses, leases that outlive the
  replay, and a callback-free CPU hand-off; none of that is designed in detail.

### 9.6 Engine integration for speculation (Implemented, private)

* Head isolation: the projected head clears target-only bindings (`logits_indices`, `token_phase`, offload outputs,
  offload mode) and resolves its disabled-indexer capacity from **actual head session metadata** (extracted as reviewable
  GA-01, draft PR microsoft/onnxruntime-genai#2708).
* Graph planning: query ceiling separated from rollback capability; `PlanGraphBuffers` overflow guards (GA-02, #2709).
* Lookup proposer (GA-03) and lookup-first hybrid (GA-04): lookup runs after target commit; a hit skips the head and drops
  any stale head shadow; an eligible miss consumes the committed suffix and prepares one MTP step; recoverable head
  failures commit target progress, count `standard_fallback_steps`/`mtp_failures` and disable MTP after three consecutive
  failures; rollback/cleanup failure is fatal. Tests cover hit->miss, rollback, rewind, cancel, mixed batches, remaining
  budgets (264 native tests pass); a post-commit allocation-failure test after a lookup->MTP transition is missing.

## 10. Quality methodology and numerics

* Gate: native thinking, greedy, 200 questions, 1,024-token output cap, explicit-final-v2 grader, **>= 151/200** correct
  among *completed* answers (capped outputs count as wrong). Frozen reference 151/200 (157 completed, 43 caps). v1 and v2
  scores are never compared.
* Outputs differ between numerically distinct but valid configurations: only 16-24 of 200 token streams are identical
  between any two runs. Completed-only scores of valid configurations over the project: 147 to 153 (SD ~2.1-2.8, 14 runs),
  because ~28-39 questions sit near the 1,024-token cap and flip between completed and capped.
* Cap-8192 side protocol (`explicit-final-v2 / cap8192`, pre-registered: completed-correct within 3 of a same-cap reference,
  paired McNemar p > 0.05, completed count within 5): the stage34 8K arm (stream4 + intra8) scored 173/188 completed
  vs the fresh reference 176/188 (12 caps each, McNemar p = 0.58, shared-completion accuracy 92.9% vs 93.4%): **pass**.
  The frozen 1024-cap result (150/200) is unchanged. Stacking the tail-safe kernel and hybrid on that arm scored
  higher (177/190) but lost throughput in a strict 8K triple (66.5 vs 95.8 tok/s decode), so it was not advanced.
* Consequence for process: a single-run pass/fail at +-3 points is not a reliable discriminator. Proposed for future
  near-threshold candidates (without changing the frozen gate retroactively): several numerically distinct runs, a paired
  shared-completion comparison, a cap-insensitive completed-answer metric, and a target-only vote check.
* Numerics to keep explicit in the op contract: CPU FP32 accumulation with FP16 result store, GPU FP16 resident path,
  rank-ordered FP32 combine, bias once. The CPU FP16-boundary mode and the `MatMulNBits` tail kernel are *not*
  bit-identical to the old paths; they are opt-in and validated end to end.

## 11. SSD and tiered storage extension

### 11.1 What was built and measured

A bounded `FileExpertStore` prototype (leases, states, dedup, cancellation, failure semantics; Section 3) passes release and
ASan/UBSan tests. Storage characterization on the ext4 `/home` disk (O_DIRECT, expert-sized random reads):

| Queue depth | 1 | 2 | 4 | 8 | 16 |
| --- | ---: | ---: | ---: | ---: | ---: |
| MiB/s | 455 | **946** | 596 | 490 | 474 |
| Avg latency ms | 5.8 | 5.5 | 17.6 | 42.9 | 82.5 |

Required hit rate: `h >= 1 - B / (R * L * k * S)`. With `B` = 946 MiB/s, `L*k*S` = 1.33 GB: at 52.2 tok/s `h >= 98.57%`, at
59.1 tok/s `h >= 98.74%`. Observed 2K hit rate is 64.75% (all-miss -> 22.8-25.8 GiB/s needed). Per-layer time budget is
0.35-0.40 ms versus ~5.5 ms read latency, so a miss needs 14-16 layers of prefetch lead to be hidden.

### 11.2 Integrated file-backed mode (Branch-only) - negative result

On the real model: tensor mode correct and fast (60.7 tok/s); file mode produced **wrong tokens even with the cache large
enough to hold everything** (and also differs from a synchronous no-store control, so it is not merely async-policy
drift), ran at ~9 tok/s (6.8x slower; TTFT ~10 s; 51 GiB read), and failed structurally at 512 MiB ("lease re-acquire after eviction")
and 256 MiB ("cannot publish without evictable space"). Verdict: stop the current file-backed design; keep the tensor seam.

### 11.3 Where SSD could still make sense

1. **Capacity tier, not a decode-miss path.** Use SSD only for experts that are essentially never hit (very cold tail), with the RAM tier holding
   the working set, and prove the working-set hit rate with a real workload first (the threshold is ~98.6%).
2. **Prefill/streaming tier.** Prefill is bandwidth-bound and tolerant of latency if layer-ahead prefetch is possible;
   whole-model prefetch of a layer's heavy experts overlaps with the previous layer.
3. **Cold start.** Lazy first-touch loading to shorten startup, since the model currently reads the full 63.5 GiB into RAM eagerly.

### 11.4 What would have to change (checklist)

| Layer | Required change |
| --- | --- |
| GenAI loader | New per-expert file source (`(namespace, layer, expert, layout_version, revision)`), no eager full-range read; config for store/manifest/cache. |
| GenAI admission | Account pageable cache + pinned staging + in-flight IO together with KV/state; refuse requests that cannot be served. |
| ORT op | Store handle as attribute/session option; keyed (not global) owner; leases that span CPU compute and the dependent H2D. |
| ORT budget | Register cache/pinned/in-flight bytes in a host-side ledger; fix the `Device*` stats bridge. |
| Execution | Layer-ahead prefetch from route predictions (e.g. previous-layer or previous-token routing); no disk IO in host callbacks; error surfaced before publishing results. |
| Correctness | Root-cause the file-mode token divergence at full budget before any further tuning (byte-equality of leased bytes against the tensor store, then arithmetic). |
| Validation | Byte-parity (file vs tensor) at every budget; fault injection (short/corrupt/missing/slow); concurrent sessions; teardown. |

## 12. Failure, ownership and concurrency model - known defects

These were found by a source audit of the private tree and must be fixed before any slice that depends on them is frozen
(ranked):

| Severity | Defect | Needed fix |
| --- | --- | --- |
| Critical | Async CPU-callback failures only set a latch checked on the *next* invocation; the failing request may already have copied results and finalized. | Surface failure before publishing the request's output (event/status handoff, poison the output). |
| High | Prefill streaming owner is a process-global singleton (one copy stream, one landing buffer, one pinned buffer): unsafe with two devices/sessions or concurrent runs. | Key by (session, device); owner lifetime tied to the session. |
| High | Streaming host/device budget checks wrong or missing (host bytes compared to the wrong limit; landing `cudaMalloc` outside the ledger). | Route through the ledger; correct the limit. |
| High | Device-budget bridge drops plugin `Device*` stats. | Fix `AllocatorStats` mapping; keep the reproducer test. |
| Medium | Schema/runtime mismatches (inputs 22/23, optional outputs required, `call_counts[5]` unused). | Section 5.4. |
| Medium | Layout identity only partly enforced. | Verify layout id string. |
| Medium | Destructor `Persist()` aborts on exception (diagnostic code). | Remove or make non-fatal. |

Ownership principles to freeze: (1) every async resource has a key and an owner whose lifetime is >= any in-flight
work; (2) a request's results are never published after a failure of work done on its behalf; (3) generation tags
protect callbacks against late completion; (4) teardown order is stream drain -> release leases -> unregister -> free;
(5) no CUDA-dependent waiting inside host callbacks.

## 13. Delivery plan and confidence

Slices are ordered by the probability that their contract survives the final design. "Experimental" slices need a design
experiment before a PR.

| Slice | Content | Confidence | Status |
| --- | --- | --- | --- |
| GA-01 | Head/target config isolation, head-local indexer metadata | High | Draft PR GenAI#2708 (166 CPU tests, plus real-model head replay) |
| GA-02 | Graph buffer overflow guards + all-width tests | High | Draft PR GenAI#2709 (167 CPU tests) |
| ORT-02 | Opt-in tail-safe INT4 small-M MatMulNBits (`N%8==4`, `M<=4`) | Medium | Draft PR ORT#33299 (144 byte-parity cases, memcheck/synccheck clean; M5..8 intentionally excluded because slower) |
| ORT-01 | Shared layout + conversion core + CPU kernel + **synchronous** `QMoEOffload` | Medium-high concept | Next; separable from async/resident/streaming |
| MEM-01 | Host/device/pinned budget accounting and bridge | Medium | Blocked on design (Section 8.2) |
| ASYNC-01 | Async/aux execution with keyed owner and same-request errors | Low-medium as designed | Blocked until Section 12 fixes |
| ORT-04 | Pool task batching | Medium | Evidence complete; default stays 1 |
| GA-03 / GA-04 | Lookup proposer; lookup-first hybrid | Medium-high / low-medium | Private; hybrid default-off |
| SSD-01 | File-backed ExpertStore | Experimental | Stopped (Section 11.2) |
| POLICY-01 | Adaptive/unified drafting, residency profiles | Experimental | Deferred |

Rules for every slice: record exact base and commit, keep behaviour default-off, show actual exercise evidence (counters
and receipts, not flags), pair every speed claim with fresh strict measurements, and keep quality (gate >= 151/200
with the paired side analysis above) separate from performance.

## 14. Open questions

1. Can the cold path ever be graph-captured, or should offload remain eager and amortize with speculation?
2. Is a GenAI-owned global placement manager worth its protocol cost over local per-node scoring? (No measured gain yet.)
3. What is the smallest ragged-step contract (request boundaries, phases) that keeps `QMoEOffload` correct for mixed
   prefill/decode/MTP batches?
4. Can routing prediction (previous token / previous layer) provide the layer-ahead prefetch an SSD tier needs, and what
   hit rate does it actually reach on representative (non-repetitive) workloads?
5. How should block sizes other than 32, 2/8-bit experts and non-SwiGLU activations enter the layout versioning?
6. Quality gating: adopt the proposed paired, cap-insensitive side analysis as a pre-registered criterion?
7. NUMA: should the pool honour a placement policy (interleave) by default on dual-socket hosts?
8. Layout v2: the per-expert FC2 bias region (5,120 B) is not applied by the CPU path (contributions are bias-free) and the
   GPU combine uses the separate `fc2_bias` input; can the in-block bias regions be dropped or made consistent (FC1 bias use on the
   CPU path still needs to be verified in source)?

## Appendix A. Environment knobs (private ORT; all default as shown)

| Knob | Default | Effect |
| --- | ---: | --- |
| `ORT_QMOE_SHARED_RESIDENT_SLOTS` | 0 | Resident slab slots per node (clamped 0..E). |
| `ORT_QMOE_SHARED_GROUPED_CALLS` | 1 | Expert calls fused per FC1/FC2 dispatch (1..64); 10 used in measurements. |
| `ORT_QMOE_SHARED_ASYNC` | 0 | Async CPU callback path (needs `PINNED_REUSE=1`, plugin EP). |
| `ORT_QMOE_SHARED_PINNED_REUSE` | 0 | Reusable pinned host leases. |
| `ORT_QMOE_SHARED_AUX_STREAM` | 0 | Resident hits on an aux stream during callback. |
| `ORT_QMOE_SHARED_CPU_THREADS` | 40 | CPU threads including callback thread. |
| `ORT_QMOE_SHARED_POOL_TASK_BATCH` | 1 | Task-claim batching 1..16 (>1 needs async). |
| `ORT_QMOE_SHARED_POOL_SPIN_US` | 2000 | Worker spin before sleep. |
| `ORT_QMOE_SHARED_FC1_COLS` / `FC2_COLS` | 64 / 128 | CPU task widths. |
| `ORT_QMOE_SHARED_CPU_FP16_BOUNDARIES` | 0 | Round FC1/SwiGLU boundaries to FP16 (not accepted). |
| `ORT_QMOE_SHARED_RESIDENT_PROMOTIONS` | 2 | Promotions per call. |
| `ORT_QMOE_SHARED_RESIDENT_DECAY_ROUTES` | 4096 | Score halving interval. |
| `ORT_QMOE_SHARED_RESIDENT_PHASE_SCORES` | 0 | Separate prefill/decode score tables. |
| `ORT_QMOE_SHARED_RESIDENT_MARGIN_PERCENT` | 100 | Hysteresis. |
| `ORT_QMOE_SHARED_PIN_EXPERTS` | 0 | `cudaHostRegister` the expert blocks in place. |
| `ORT_QMOE_SHARED_STREAM_ROWS` | 0 | Prefill streaming tile; needs slots > 0 and pinned experts. |
| `ORT_QMOE_SHARED_STREAM_MIN_ROWS` / `WAVE` / `PROMOTIONS` / `GPU_MIB` | 16 / 32 / 16 / 1024 | Streaming selection, wave size, promotions, extra GPU scratch. |
| `ORT_QMOE_SHARED_STREAM_PROMOTION_REUSE` | 0 | D2D refill from the last two landing waves. |
| `ORT_QMOE_SHARED_EXPERT_STORE` | unset | **Branch-only:** `tensor` or `file`. |
| `ORT_QMOE_SHARED_EXPERT_{NAMESPACE,REVISION,FILE,CACHE_MB,IO_THREADS}` | `qmoe.shared`,1,-,512,2 | **Branch-only** store config. |
| `ORT_MATMULNBITS_UNALIGNED4_FAST_PATH` | 0 | Opt-in tail INT4 kernel (draft ORT#33299). |

## Appendix B. Source map

| Area | Location (private ORT unless stated) |
| --- | --- |
| Op schema | `onnxruntime/core/graph/contrib_ops/contrib_defs.cc` (`QMoEOffload`) |
| Operator | `onnxruntime/contrib_ops/cuda/moe/qmoe_offload.{h,cc}`, `qmoe_kernels.{h,cu}` |
| Shared layout / CPU kernels | `onnxruntime/contrib_ops/cpu/moe/qmoe_shared_layout.{h,cc}` |
| Spin pool | `onnxruntime/contrib_ops/cuda/moe/qmoe_spin_pool.h`, `docs/qmoe_pool_task_batch.md` |
| Stream promotion reuse | `docs/qmoe_stream_promotion_reuse.md` |
| Device budget | `onnxruntime/core/providers/cuda/cuda_device_memory_budget.h` |
| ExpertStore (branch `32a01968b1`) | `onnxruntime/contrib_ops/cuda/moe/qmoe_expert_store.{h,cc}`, `onnxruntime/test/python/qmoe_export_expert_store.py` |
| Tests | `onnxruntime/test/python/test_qmoe_offload.py`, `.../transformers/test_qmoe_shared_layout.py` |
| GenAI config / loader | `src/config.cpp`, `src/models/model.cpp` (`AddSharedInitializers`) |
| GenAI phases / stats | `src/engine/decoders/varlen_decoder_io.cpp` (`GetExpertTokenPhases`, `PrepareExpert{Inputs,Outputs}`) |
| GenAI speculation | `src/engine/engine.cpp` (`ValidateEngineNGramConfig`, `PublishNGramDrafts`, `PublishNGramMtpFallback`), `docs/SpeculativeDecoding.md` |

## Appendix C. Evidence index

The experiment ledger and reports live in the `dev` repository (`~/git/dev`); paths below are relative to it.

| Topic | Document |
| --- | --- |
| Original design, layout, budget, placement manager | `docs/design/qwen38_flash_next_int4_24g_offload.md` |
| Experiment ledger (Exp0-Exp30), Strata comparison | `docs/design/qwen38_flash_next_int4_24g_offload_experiments.md` |
| Final gap results | `docs/design/qwen38_flash_next_int4_24g_gap_results_20261011.md` |
| 2K/8K short-context optimization, tail kernel | `docs/design/qwen38_flash_next_int4_24g_short_context.md` |
| Production PR ranking and plan | `docs/design/qwen38_production_pr_plan_20261012.md`, `docs/qmoe_offload_pr_plan.md` |
| ORT-01/MEM-01/ASYNC-01 audit | `docs/experiment/streamA_ort01_mem01_async01/report.md` |
| SSD prototype and feasibility | `docs/experiment/streamB_ssd_expertstore/report.md`, `prototypes/expert_store/` |
| MTP policy matrix | `docs/experiment/streamC_mtp_policy_matrix/report.md` |
| Pool batching / NUMA | `docs/experiment/streamD_ort04_pool_numa/report.md` |
| ExpertStore integration (stage43) | `docs/experiment/streamE_expertstore_integration/report.md` |
| Hybrid candidate and quality | `docs/experiment/streamH_hybrid_candidate/report.md`, `docs/experiment/streamQ_hybrid_quality/report.md` |
| Draft PRs | GenAI#2708, GenAI#2709, ORT#33299 |
