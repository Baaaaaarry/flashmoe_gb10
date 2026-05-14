# FlashMoE on vLLM: Integration Plan and Toggle Matrix

## 1. How to attach this project to vLLM

Use vLLM's out-of-tree plugin path, not an in-tree fork first.

vLLM documents two relevant extension points:

- plugin entry points via Python `entry_points`
- out-of-tree model registration with `ModelRegistry.register_model(...)`

References:

- vLLM plugin system: [docs](https://docs.vllm.ai/en/stable/design/plugin_system/)
- vLLM model registration: [docs](https://docs.vllm.ai/en/v0.11.1/contributing/model/registration/)
- vLLM custom ops / pluggable layers: [docs](https://docs.vllm.ai/en/latest/api/vllm/model_executor/custom_op/)

This repo now contains a package scaffold under:

- `python/pyproject.toml`
- `python/flashmoe_vllm_plugin/plugin.py`
- `python/flashmoe_vllm_plugin/vllm_adapter.py`
- `python/flashmoe_vllm_plugin/layer_replacement.py`
- `python/flashmoe_vllm_plugin/custom_op.py`

The intended hookup is:

1. Keep vLLM's tokenizer, scheduler, paged attention, sampling, and serving stack unchanged.
2. Register a custom `FlashMoEQwenForCausalLM` model wrapper.
3. Inside that wrapper, replace only the routed-expert execution path.
4. Dense attention and KV handling continue to use vLLM-native code.
5. Routed experts call into a FlashMoE runtime layer that decides:
   - cache hit or miss
   - prefetch target
   - parallel pread chunking
   - fused routed/shared expert kernel variant

## 1.1 What "real vLLM routed-MoE layer replacement" means

It does **not** mean registering an empty wrapper class and hoping vLLM will call us later.

It means:

1. Instantiate the original vLLM Qwen MoE model class.
2. Walk the module tree.
3. Detect MoE routed-expert layers by structure, not by top-level model name.
4. Replace those submodules with `FlashMoERoutedMoELayer`.
5. Leave attention, KV cache, scheduler, and sampling untouched.

That is what the updated `python/flashmoe_vllm_plugin/vllm_adapter.py` and `python/flashmoe_vllm_plugin/layer_replacement.py` now do.

The replacement layer:

- computes routing logits with the original gate
- performs `top-k`
- dispatches selected routed experts through a FlashMoE custom op
- optionally adds the shared expert output

So the swapped layer is no longer a placeholder. It is the real execution point for routed MoE.

## 2. Runtime split: what stays in vLLM and what moves to FlashMoE

Keep in vLLM:

- request scheduler
- continuous batching
- paged KV cache
- attention backend selection
- sampling / logits processors
- tensor parallel / pipeline parallel machinery

Move into FlashMoE runtime:

- expert file index
- expert hot-cache metadata
- replacement policy inference
- predictor inference for speculative prefetch
- layer-major expert file dispatch
- chunked parallel `pread`
- CUDA fused MoE kernel dispatch

## 3. Toggle design

All requested features are now represented as explicit flags in:

- `python/flashmoe_vllm_plugin/config.py`

Relevant toggles:

- `enable_predictor`
- `enable_cache_policy`
- `enable_sliding_window`
- `enable_row_column_bundling`
- `enable_dram_sparse_cache`
- `enable_expert_streaming`
- `enable_layer_major_scheduling`
- `enable_fused_moe_kernel`
- `enable_parallel_pread`
- `enable_expert_prefetch`
- `enable_chunked_pread`
- `enable_shared_expert_overlap`

Print the active matrix with:

```bash
python3 -m flashmoe_vllm_plugin.ablate \
  --config python/flashmoe_vllm_plugin/examples/gb10_qwen395_397b.json
```

## 4. Predictor training

The public repos log routing data with `--collect-routing`. Their shipped `train_predictor.py` only trains a routing predictor, not a cache-policy FFN.

This project now separates the two:

### Routing predictor

Files:

- `python/flashmoe_vllm_plugin/train_predictor.py`

Input:

- binary routing traces collected from runtime
- format matches the Apple repos: `layer_idx`, `K`, hidden state, expert indices

Training target:

- multi-label expert activation vector over `512` experts

Output:

- exported JSON weights for later runtime loading

Use:

```bash
flashmoe-train-predictor \
  --input routing_data.bin \
  --output predictor.json \
  --epochs 10
```

### Cache-policy FFN

Files:

- `python/flashmoe_vllm_plugin/build_cache_dataset.py`
- `python/flashmoe_vllm_plugin/train_policy.py`
- `python/flashmoe_vllm_plugin/policy.py`

Input CSV columns:

- `recency`
- `frequency`
- `reuse_distance`
- `size_ratio`
- `layer_pressure`
- `is_prefetched`
- `label`

Recommended label construction:

1. Build a compact text routing trace from decode runs.
2. Run `flashmoe-build-cache-dataset` to replay the trace and generate Belady-style labels.
3. For each eviction point, compute future distance for every resident expert.
4. Mark the optimal victim with label `1.0`, or regress future distance directly.
5. Train the FFN on those features.

Example:

```bash
flashmoe-build-cache-dataset \
  --trace routing_trace.txt \
  --output cache_policy_train.csv \
  --cache-entries 16384

flashmoe-train-policy \
  --input cache_policy_train.csv \
  --output cache_policy.json
```

Output:

- JSON MLP weights

Runtime load path:

- `FlashMoERuntime` loads the JSON and uses `python/flashmoe_vllm_plugin/policy.py` to score candidates even without torch.

This is the missing link in the public repos: training and runtime loading are now separate, explicit, and configurable.

## 5. Which paper / repo techniques have implementations?

### `sliding window`

- Paper/repo status: not a core FlashMoE implementation feature in the two public repos.
- vLLM status: native attention-side switch already exists.
- This project: `enable_sliding_window` toggle exists, but it should be treated as a vLLM attention experiment, not a routed-expert optimization.

### `row-column bundling`

- Paper status: mentioned as inspiration from earlier SSD-streaming work.
- Public repo status: no explicit implementation found in the Apple repos.
- This project: represented as `enable_row_column_bundling`, but still needs the actual on-disk packer update for bundled layouts.

### `dram sparse cache`

- Paper status: central idea.
- Public repo status: Metal LRU and malloc caches exist, but disabled by default on Apple.
- This project: implemented as the explicit GB10 design center:
  - C++ side: `src/expert_cache.cpp`
  - Python control plane: `enable_dram_sparse_cache`

### `expert-level stream`

- Paper status: yes.
- Public repo status: yes, strongly implemented.
- This project: preserved as `enable_expert_streaming`.

### `layer-major scheduling`

- Paper status: implied by per-layer expert fetch.
- Public repo status: yes, via `packed_experts/layer_XX.bin`.
- This project: explicit toggle `enable_layer_major_scheduling`.

### `fused kernel`

- Public repo status: yes, especially on Metal:
  - routed expert matvec
  - shared expert overlap
  - combine + residual + norm fusions
- This project: now has a minimal CUDA custom op backed by `cuBLASLt`:
  - `python/flashmoe_vllm_plugin/ops.py`
  - `python/flashmoe_vllm_plugin/custom_op.py`
  - `python/csrc/flashmoe_ops.cpp`
  - `python/csrc/flashmoe_ops_cuda.cu`

The minimal kernel path is:

1. group tokens by routed expert
2. `cuBLASLt` GEMM for `gate_proj`
3. `cuBLASLt` GEMM for `up_proj`
4. `SiLU(gate) * up`
5. `cuBLASLt` GEMM for `down_proj`
6. weighted scatter-add back to output

This is intentionally minimal and runnable. It is not yet the final GB10-optimized streamed-expert kernel.

### `parallel pread`

- Public repo status: yes, and Anemll improves it further with persistent workers and split fanout.
- This project: directly implemented in C++ as `src/io_pool.cpp` and exposed with:
  - `enable_parallel_pread`
  - `enable_chunked_pread`

## 6. Recommended GB10 validation order

Use these toggles one by one:

1. `expert_streaming=on`, all others off except `layer_major_scheduling`
2. add `parallel_pread`
3. add `chunked_pread`
4. add `dram_sparse_cache`
5. add `cache_policy`
6. add `shared_expert_overlap`
7. add `fused_moe_kernel`
8. only then test `predictor`

That order keeps attribution clean and makes the throughput deltas interpretable.

## 7. GB10 scripts

Prepared scripts:

- `scripts/gb10_create_env.sh`
- `scripts/gb10_build_extension.sh`
- `scripts/gb10_benchmark_custom_op.sh`
- `scripts/gb10_export_qwen_experts.sh`
- `scripts/gb10_export_qwen_experts_chunked.sh`
- `scripts/gb10_generate_synthetic_trace.sh`
- `scripts/gb10_init_config.sh`
- `scripts/gb10_build_expert_manifest.sh`
- `scripts/gb10_replay_streaming.sh`
- `scripts/gb10_train_predictor.sh`
- `scripts/gb10_train_cache_policy.sh`
- `scripts/gb10_serve_vllm.sh`

These scripts assume:

- DGX Spark Ubuntu/DGX OS
- CUDA already installed with the platform image
- system build prerequisites already installed

Recommended OS packages before running the scripts:

```bash
sudo apt-get update
sudo apt-get install -y \
  build-essential \
  git \
  cmake \
  ninja-build \
  python3.12-dev
```

Default behavior:

- `gb10_create_env.sh` now installs `vllm` from PyPI by default
- set `VLLM_SOURCE=source` only if you explicitly want to build `vLLM` from `~/src/vllm`

Recommended benchmark order:

```bash
./scripts/gb10_benchmark_custom_op.sh --backend torch --tokens 16 --experts 64
./scripts/gb10_benchmark_custom_op.sh --backend vllm --tokens 16 --experts 64
./scripts/gb10_benchmark_custom_op.sh --backend flashmoe --tokens 16 --experts 64
./scripts/gb10_benchmark_custom_op.sh --backend all --tokens 16 --experts 64
```

Interpretation:

- `torch`: eager PyTorch reference path
- `vllm`: stock vLLM `fused_experts(...)` path when available
- `flashmoe`: our custom CUDA op
- `est_payload_gbps`: estimated algorithmic tensor payload divided by average latency; this is a comparative throughput proxy, not a hardware counter
- `active_experts`: number of experts touched by the sampled route pattern
- `routes_per_active_expert`: route fanout pressure after dispatch
- `cuda_peak_alloc_gb` / `cuda_peak_reserved_gb`: peak CUDA allocator footprint during the benchmark
- `process_rss_gb` / `system_used_gb`: process and system memory usage from `/proc`; on GB10 UMA this is the most practical DDR occupancy proxy from user space

Streaming validation commands:

```bash
EXPORT_FORMAT=q3like ./scripts/gb10_export_qwen_experts.sh Qwen/Qwen3.5-397B-A17B /path/to/packed_experts_q3like /path/to/hf_cache
EXPORT_FORMAT=q3like ./scripts/gb10_export_qwen_experts_chunked.sh /local/model_dir /path/to/packed_experts_q3like
./scripts/gb10_init_config.sh /local/model_dir /path/to/packed_experts_q3like /path/to/flashmoe_config.json
./scripts/gb10_generate_synthetic_trace.sh /path/to/packed_experts_q3like/expert_manifest.json /path/to/routing_trace.txt
./scripts/gb10_build_expert_manifest.sh /path/to/packed_experts_q3 /path/to/expert_manifest.json
./scripts/gb10_replay_streaming.sh /path/to/flashmoe_config.json routing_trace.txt
```

If the GB10 host cannot reach Hugging Face:

- download the model on another machine first
- copy the full local model directory onto GB10
- then call:

```bash
./scripts/gb10_export_qwen_experts.sh /local/copied/model_dir /path/to/packed_experts_bf16
```

Use this replay path before full weight-to-CUDA integration to answer the real GB10 question:

- how much routed expert traffic spills to SSD
- how much DRAM hot cache is needed
- whether hit rate is high enough for single-node `397B`

Expert export formats:

- `dense`
  - compatibility-first
  - largest SSD traffic
- `q3like`
  - 3-bit packed weights with per-row scales
  - intended to reduce replay and deployment SSD load before custom CUDA decode is added

Trace options:

- Synthetic trace:
  - use `gb10_generate_synthetic_trace.sh`
  - good for smoke test, cache sizing, and replay validation
- Real trace:
  - set `routing_trace_path` in `flashmoe_config.json`
  - run the FlashMoE routed layer path
  - each forward appends `layer_id:expert0,expert1,...` lines for replay reuse

Relevant references:

- vLLM install docs: [latest](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/)
- DGX Spark hardware and software docs: [hardware](https://docs.nvidia.com/dgx/dgx-spark/hardware.html), [overview](https://docs.nvidia.com/dgx/dgx-spark/system-overview.html), [porting guide](https://docs.nvidia.com/dgx/dgx-spark-porting-guide/overview.html)
