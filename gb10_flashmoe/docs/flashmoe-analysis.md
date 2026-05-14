# FlashMoE Paper vs Open-Source Repos vs GB10 Implementation

## 1. Paper claim vs repo reality

Paper: [FlashMoE arXiv 2601.17063v1](https://arxiv.org/html/2601.17063v1)

- The paper's core system claim is two-part:
  - separate non-experts from experts and stream experts from SSD
  - use an ML-based cache replacement policy trained against Belady-style oracle targets
- The evaluation hardware in the paper is not Apple Silicon. It is a desktop with `RTX 5070 Ti`, `64 GB DDR5`, and a `PCIe 5.0 NVMe` SSD.

Open-source repo reality:

- [`danveloper/flash-moe`](https://github.com/danveloper/flash-moe)
  - product direction moved away from the paper's ML cache
  - the README explicitly says the winning configuration is `Trust the OS`
  - `--cache-entries` and `--malloc-cache` still exist in `metal_infer/infer.m`, but default to disabled
  - `train_predictor.py` trains a top-K routing predictor, not the paper's eviction FFN
- [`Anemll/flash-moe`](https://github.com/Anemll/flash-moe)
  - keeps the same default stance on software cache
  - extends the engine with `Q3` experts, GGUF overlays, persistent I/O pool, and `--cache-io-split`
  - still does not implement the paper's per-layer eviction model as the default fast path

Conclusion:

- the SSD streaming idea is reproduced
- the "store experts separately and load on demand" part is reproduced
- the paper's **ML-based eviction policy is not reproduced as the production path** in either repo
- the performance-winning path in both repos is dominated by expert layout, quantization, and I/O path work, not ML eviction

## 2. Expert storage layout

### Paper

The paper states that experts are stored per layer and per expert, while non-experts are loaded first and kept resident. The intent is to minimize startup time and allow on-demand expert fetch.

### `danveloper/flash-moe`

`repack_experts.py` converts scattered safetensors into one contiguous file per layer:

- `layer_XX.bin`
- `512` experts per layer
- fixed-size expert records
- expert offset is `expert_idx * expert_size`

This is a major engineering divergence from the paper text:

- the paper says "individual `.pt` files according to each layer and expert index"
- the repo uses **contiguous packed layer files**, which is better for `pread()` locality and avoids per-file metadata overhead

Inside each expert block the components are laid out in fixed order:

- `gate_proj.weight`
- `gate_proj.scales`
- `gate_proj.biases`
- `up_proj.weight`
- `up_proj.scales`
- `up_proj.biases`
- `down_proj.weight`
- `down_proj.scales`
- `down_proj.biases`

That gives constant-time offset computation and removes tensor-name lookup from the critical decode path.

### `Anemll/flash-moe`

The same packed layout is retained, then extended with:

- `packed_experts_2bit/`
- `packed_experts_Q3/`

The fork's real performance gain comes from reducing per-expert bytes:

- original 4-bit experts: about `6.75 MB`
- Q3 routed experts: about `5.44 MB`
- 2-bit experts: about `3.75 MB`

For decode throughput this matters more than the paper's eviction model because per-token expert bytes directly scale SSD pressure.

## 3. Expert cache and replacement

### Paper

The paper proposes:

- recency score
- frequency score
- score normalization
- a small FFN trained with Belady-derived labels
- per-step eviction of the expert with the largest predicted future distance

This is a true software-managed expert cache. The paper is explicit that better eviction reduces SSD misses.

### `danveloper/flash-moe`

The source still contains two cache implementations in `metal_infer/infer.m`:

- Metal-buffer LRU cache
- malloc-backed LRU cache with zero-copy Metal wrappers

Important details:

- lookup table is `entry_idx[layer][expert]`
- eviction is strict LRU based on `last_used`
- telemetry tracks cold misses, eviction misses, reuse distance, and evictions
- cache population happens by reading misses directly into the chosen cache slot

What is missing relative to the paper:

- no learned eviction FFN
- no normalized recency/frequency input path inside the decode loop
- no per-layer trained replacement model
- no Belady-supervised eviction model loading

There is also a prediction path:

- `--predict`
- previous-token temporal routing reuse
- optional async predicted expert preads into buffer set `B`

But that is **routing prediction**, not cache replacement. The repo's own docs say it lost to the baseline because the hit rate was too low.

### `Anemll/flash-moe`

The fork keeps the same cache structures and still defaults to cache disabled. The main improvement is not a smarter eviction policy. It is:

- faster warm-page-cache servicing through a persistent worker pool
- splitting each expert read into page-aligned chunks with `--cache-io-split`

So the fork improves the miss path and page-cache utilization, not the replacement algorithm itself.

### GB10 implication

This is the biggest place where GB10 changes the answer:

- Apple M3 48 GB: explicit expert cache competed with GPU memory pressure and often lost
- GB10 128 GB UMA: explicit cache is attractive again because the platform can afford a large hot set

For `Qwen3.5-397B-A17B` on a single DGX Spark:

- dense resident weights are on the order of single-digit GB
- Q3 routed experts are around `163 GB` on disk in the Anemll measurements
- a `88-96 GB` explicit expert cache can hold roughly `54-59%` of the routed expert corpus

That is enough capacity that replacement policy quality matters again. So for GB10 the right move is to restore a real software cache instead of inheriting the Apple default blindly.

## 4. I/O and compute overlap

### Paper

The paper claims:

- replacement-policy computation is asynchronous
- its overhead is hidden behind expert loading latency
- expert loading dominates FFN compute

This is a light form of overlap argument. The paper mostly overlaps policy bookkeeping with I/O, not a detailed hardware pipeline.

### `danveloper/flash-moe`

The repo implements a much more concrete overlap model:

- `CMD1`: projections and attention work
- `CMD2`: `o_proj + residual + norm + routing + shared expert prep`
- CPU computes softmax and top-K
- async `pread()` starts immediately after routing
- `CMD3` for routed experts is committed without waiting
- the next layer's `CMD1` is allowed to queue behind the previous layer's `CMD3`

Key engineering trick:

- `CMD3` optionally performs combine + residual + norm on GPU
- if that happens, the next layer can consume `buf_input` directly
- this removes a CPU round-trip and turns the GPU queue itself into the dependency chain

That is a real decode pipeline, not just a conceptual overlap statement.

### `Anemll/flash-moe`

The fork deepens the I/O side:

- GCD async preads are replaced by a persistent worker pool
- `--cache-io-split N` divides one expert blob into `N` page-aligned tasks
- the async pread state tracks `num_experts * chunks_per_expert`
- completion validity is reconstructed by summing chunk results per expert

This matters because the repo authors found that warm-page-cache reads scale with concurrency even when total bytes stay fixed.

The standalone `cachebench` showed:

- `8` workers + `split=4` outperformed `split=1`
- the gain survives into end-to-end decode

### Apple-specific conclusion in the repos

The original repo's own experiments found that background SSD DMA hurt GPU latency badly on Apple UMA. That is why the README concludes the serial `GPU -> SSD -> GPU` pipeline is already optimal there.

### GB10-specific conclusion

GB10 should not inherit that conclusion unmodified:

- NVIDIA documents `273 GB/s` unified memory bandwidth and `2` copy engines on DGX Spark
- the SSD bandwidth is much smaller than memory bandwidth
- the CUDA stack gives us better control over stream ordering and prefetch hints than the Metal path used in the Apple repos

So GB10 should be tuned around **partial overlap**, not "never overlap".

The implementation in this directory therefore assumes:

- dense compute can overlap part of expert pread time
- we should measure an overlap factor, not hard-code it to zero
- the pipeline should keep a software expert cache in RAM and let Linux page cache serve the colder tail

## 5. What this implementation adds

Files:

- `CMakeLists.txt`
- `include/flashmoe/cache_policy.h`
- `include/flashmoe/expert_cache.h`
- `include/flashmoe/io_pool.h`
- `src/expert_cache.cpp`
- `src/io_pool.cpp`
- `src/trace_sim.cpp`
- `src/pipeline_model.cpp`

### `ExpertCache`

This is the missing GB10 piece:

- byte-budgeted explicit software cache
- per-expert metadata:
  - `last_touch`
  - `access_count`
  - `layer_resident_count_snapshot`
- pluggable eviction policy

The included `recency_frequency` policy is a lightweight stand-in for the paper's learned policy:

- higher recency score means "older, more evictable"
- higher frequency score means "hotter, less evictable"
- layer-balance penalty discourages one layer from dominating all residency

This is not yet the paper's FFN, but it restores the part of the design that both public repos backed away from.

### `ChunkedPreadPool`

This captures the Anemll lesson in a runtime-neutral form:

- persistent thread pool
- page-aligned chunk fanout
- one completion callback for the aggregated expert read

This is the right shape for a Linux/CUDA port because it separates:

- storage fetch scheduling
- later compute submission

### `flashmoe_trace_sim`

This is the offline policy tuner:

- replay a real routing trace
- compare `LRU`, `LFU`, `recency_frequency`, and `oracle`
- size the explicit cache for GB10 before wiring it into the full inference engine

That gives a direct way to answer:

- how much benefit does `96 GB` of expert residency buy
- whether the recency/frequency policy is good enough
- how far it still is from oracle on real routing traces

### `pipeline_model`

This avoids copying the Apple assumption that overlap is worthless:

- it models serial layer time
- it models partial overlap
- it can be calibrated later from Nsight Systems traces on real GB10 runs

## 6. Recommended GB10 inference plan

For a real `Qwen3.5-397B-A17B` bring-up on single DGX Spark:

1. Keep dense weights resident.
2. Use `Q3` routed experts first; keep a 4-bit fallback path for sensitive layers if quality regresses.
3. Reserve roughly:
   - `8-10 GB` dense weights
   - `12-20 GB` KV/state/workspace depending on target context
   - `88-96 GB` explicit expert cache
   - leave the remainder to Linux page cache and runtime slack
4. Use persistent `pread()` workers with `split=4` page-aligned chunking as the first GB10 baseline.
5. Start with recency+frequency eviction and replay real routing traces offline.
6. If the oracle gap remains large, train the paper-style FFN and replace the linear score path in `cache_policy.h`.
7. Only after cache hit-rate is stabilized should CUDA kernel work be the next bottleneck target.

## 7. The important take-away

The paper, the original repo, and the Anemll fork are not saying the same thing anymore:

- the paper says the ML eviction policy is central
- the original repo says the OS page cache beat custom caching on Apple
- the Anemll fork says better expert formats and better cached `pread()` fanout matter more than fancy eviction on Apple

For DGX Spark GB10, the correct synthesis is:

- keep the repo-side packed expert format and async fanout ideas
- reject the Apple-specific conclusion that explicit caching is a net loss
- restore a large software-managed expert cache, because 128 GB UMA changes the feasible hot-set size
- treat I/O/compute overlap as a tunable parameter, not as impossible
