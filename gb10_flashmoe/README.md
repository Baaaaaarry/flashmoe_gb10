# GB10 FlashMoE Runtime

This directory contains a DGX Spark GB10 oriented runtime skeleton for streamed MoE inference. It does not replace the full inference engine from the Apple/Metal forks. It isolates the three hardware-sensitive pieces that need to change for GB10:

- explicit expert residency management in 128 GB unified memory
- page-aligned, chunked `pread()` fanout for SSD-backed experts
- a scheduling model that assumes partial I/O and compute overlap is worthwhile on GB10, unlike the original M3/M5 Apple path
- expert-manifest based streaming replay so finite-cache `397B` validation can happen before full kernel wiring is complete
- local-or-Hugging-Face expert export path for building routed expert files on GB10
- model-spec aware runtime planning so the online path is not hard-coded to `Qwen3.5-397B`

## Why this exists

The two public `flash-moe` repositories are optimized for Apple Silicon:

- the original repo converged on `trust the OS page cache`
- the Anemll fork kept that default, then added better expert formats and page-cache fanout

DGX Spark is materially different:

- 128 GB coherent unified memory instead of 48 GB
- documented 273 GB/s memory bandwidth
- integrated Blackwell GPU with CUDA/Tensor Core stack
- NVMe storage plus copy engines, making partial I/O / compute overlap a realistic target

That shifts the best design point from "almost no explicit cache" to "large explicit expert cache plus Linux page cache for the cold tail".

## Build

```bash
cmake -S . -B build
cmake --build build -j
```

## Binaries

`flashmoe_trace_sim`

- replays a routing trace
- compares `lru`, `lfu`, `recency_frequency`, and `oracle`
- uses a byte budget that matches GB10 expert-cache planning

Trace format:

```text
token layer expert0 expert1 expert2 expert3
```

Example:

```bash
./build/flashmoe_trace_sim trace.txt 96 5.44
```

`flashmoe_pipeline_demo`

- compares the Apple-style "overlap barely helps" regime with a GB10 target regime
- prints an estimated steady-state decode throughput

`flashmoe_runtime_plan`

- prints the recommended dedicated-runtime plan for a model family
- keeps the runtime design compatible with both `Qwen3.5-397B-A17B` and a future switch to `DeepSeek-V4-Flash`

Examples:

```bash
./build/flashmoe_runtime_plan qwen3.5-397b-a17b
./build/flashmoe_runtime_plan deepseek-v4-flash
./scripts/gb10_runtime_plan.sh qwen3.5-397b-a17b
```

`flashmoe_online_runtime_demo`

- validates the first dedicated online-runtime control-plane components
- consumes a real `expert_manifest.json` and routing trace
- reports:
  - dense resident budget
  - hot-cache budget and usage
  - routed requests
  - unique experts touched
  - cache hits / misses / evictions
  - cold bytes touched by the online scheduler

Example:

```bash
./scripts/gb10_online_runtime_demo.sh \
  qwen3.5-397b-a17b \
  /path/to/expert_manifest.json \
  /path/to/routing_trace.txt \
  96 \
  4096
```

`flashmoe_decode_harness`

- estimates the first decode-stage timing breakdown for the dedicated runtime
- compares `host unpack` vs `gpu unpack` device-path profiles on the same manifest + trace
- reports:
  - route time
  - cold-load time
  - unpack-to-slot time
  - grouped-compute time
  - combine time
  - estimated tok/s

Example:

```bash
./scripts/gb10_decode_harness.sh \
  qwen3.5-397b-a17b \
  /path/to/expert_manifest.json \
  /path/to/routing_trace.txt \
  4096
```

To replace the estimated `compute_ms` with a measured grouped-compute profile from the CUDA benchmark:

```bash
./scripts/gb10_generate_compute_profile.sh \
  /path/to/compute_profile.json \
  1 4096 1536 64 4 bfloat16 200 1,2,4

./scripts/gb10_decode_harness.sh \
  qwen3.5-397b-a17b \
  /path/to/expert_manifest.json \
  /path/to/routing_trace.txt \
  4096 \
  /path/to/compute_profile.json
```

`flashmoe_streamed_runtime`

- runs the dedicated runtime mainline instead of only a standalone harness
- consumes:
  - model family
  - real `expert_manifest.json`
  - real routing trace
  - optional measured CUDA compute profile
- emits integrated runtime-stage statistics for the GPU-unpack path

Example:

```bash
./scripts/gb10_streamed_runtime.sh \
  qwen3.5-397b-a17b \
  /path/to/expert_manifest.json \
  /path/to/routing_trace.txt \
  4096 \
  /path/to/compute_profile.json
```

`flashmoe_streamed_service`

- provides a thin narrow-engine service layer on top of the dedicated streamed runtime
- exposes:
  - `GET /healthz`
  - `GET /v1/models`
  - `POST /v1/sessions`
  - `GET /v1/sessions/<id>`
  - `POST /v1/chat/completions`
- supports:
  - persistent streamed sessions
  - session reuse via `session_id`
  - `stream=true` SSE responses
- keeps routed-expert slot cache state inside a streamed session

Example:

```bash
./scripts/gb10_streamed_service.sh \
  qwen3.5-397b-a17b \
  /path/to/expert_manifest.json \
  /path/to/routing_trace.txt \
  /path/to/compute_profile.json \
  8080 \
  24
```

Current limitation:

- this is a narrow deployment scaffold, not yet the final full-operator inference engine
- it already supports end-to-end session flow, prompt-prefill accounting, runtime budgeting, thin service integration, and a minimal `KV cache + dense decode chain`
- the service runtime payload now exposes per-request timing breakdown for:
  - `embed`
  - `attention`
  - `norm/router`
  - `lm_head`
  - `route`
  - `load`
  - `unpack`
  - `compute`
  - `combine`
- the narrow engine now maintains a minimal unified decode state across prefill and decode and returns sampled `token_ids` alongside generated text
- the service can now run in a `runtime_router` mode, where expert routing is generated inside the engine rather than replayed from an external trace file
- in `runtime_router` mode, the engine now derives router decisions and token sampling from a minimal dense hidden-state/logit chain instead of pure hash-based rules
- the streamed service now has a switchable expert execution backend:
  - CPU reference backend for local correctness and fallback
  - CUDA expert backend when `CUDAToolkit` is available at build time
- it still uses the routed-trace driven expert core rather than a full real-router + logits execution stack

`flashmoe_streamed_cli`

- provides a direct CLI prompt entrypoint on top of the same streamed engine
- supports:
  - one-shot prompt generation
  - `--prompt-stdin`
  - `--interactive` multi-turn session reuse
  - the same `dense_artifact` / `tokenizer_artifact` / runtime-router options as the HTTP service

Example:

```bash
DENSE_ARTIFACT_PATH=/path/to/dense_runtime_artifact.json \
TOKENIZER_ARTIFACT_PATH=/path/to/tokenizer_artifact.json \
USE_RUNTIME_ROUTER=1 \
./scripts/gb10_streamed_cli.sh \
  qwen3.5-397b-a17b \
  /path/to/expert_manifest.json \
  "" \
  /path/to/compute_profile.json \
  24 \
  --prompt "Explain FlashMoE in one sentence." \
  --max-tokens 16
```

## Recommended GB10 starting point

- Routed expert format: `Q3` class experts first, 4-bit fallback for accuracy-sensitive layers
- Dense weights: fully resident
- Explicit hot cache budget: `24 GB` starting point
- Remaining RAM: explicit `KV cache + workspace + safety` budget before considering any cache increase
- I/O pool: `8-12` workers
- Routed expert fanout: `split=4` as the first setting to validate
- Replacement policy: recency + frequency by default, oracle replay for offline tuning

## Updated online direction

The current repository has validated that the `vLLM + PyTorch wrapper` online path is not a deployable target for `Qwen3.5-397B` on GB10. The recommended next online implementation is:

- keep this repo for export, replay, cache sizing, and expert-format tooling
- build a dedicated streamed-MoE runtime outside the framework parameter tree
- keep routed experts out of the model object graph entirely
- use model-spec based planning so a later switch from `Qwen3.5-397B` to `DeepSeek-V4-Flash` does not require redesigning the runtime core

See `docs/current-status-onepager.md` for the validated failure mode of the wrapper path and `flashmoe_runtime_plan` for the model-family aware deployment recommendation.

See `docs/flashmoe-analysis.md` for the full paper-to-code comparison and GB10 design rationale.
