# Dedicated Streamed-MoE Runtime Blueprint

## Goal

Build the next online runtime as a **dedicated streamed-MoE engine**, not as a `vLLM + PyTorch` wrapper.

The current repository already validated:

- expert export works
- `Q3-like` expert storage works
- manifest generation works
- routing replay works
- explicit cache sizing works

It also validated that the current online wrapper path is not deployable for `Qwen3.5-397B` on GB10.

## Runtime split

Keep the current repository for:

- expert export
- expert compression
- manifest generation
- routing replay
- cache policy / predictor tooling

Build the next runtime around these modules:

1. `model_spec`
   - model-family specific structural facts
   - layer count
   - top-k
   - expert count
   - shared-expert behavior
   - MLA / KV path differences

2. `dense_resident_loader`
   - loads only dense layers, router, shared experts, embeddings, norms, lm head
   - routed experts never become framework parameters

3. `expert_store`
   - layer-major expert bundles on SSD
   - manifest entry -> offset / size / format

4. `hot_expert_cache`
   - compute-ready cache slots in FP8 or BF16
   - eviction policy managed independently from the host framework

5. `decode_scheduler`
   - route
   - cache lookup
   - miss read
   - unpack into cache slot
   - grouped expert compute
   - combine and emit next-token logits

6. `server`
   - narrow OpenAI-compatible interface
   - no generic framework parameter lifecycle

## Compatibility strategy

The runtime core should not hard-code either Qwen or DeepSeek assumptions.

### Qwen3.5-397B-A17B

- source precision: `FP8`
- practical cold storage: `Q3-like`
- practical hot-cache compute: `FP8`
- top-k: `4`
- routed experts per layer: `512`

### DeepSeek-V4-Flash

- source precision: `FP4 + FP8`
- practical cold storage: `MXFP4` or `Q3-like`
- practical hot-cache compute: `FP8`
- top-k: `6`
- routed experts per layer: `256`
- special note: preserve CSA/HCA and any hash-router prefill behavior

The shared runtime should depend only on a `ModelSpec`, not on per-model hard-coded module trees.

## Phase order

### Phase 1

- model-spec layer
- dedicated resident dense-loader design
- expert-store interface
- hot-cache slot allocator
- grouped decode micro-kernel harness

### Phase 2

- single-token decode path
- async pread
- miss-to-cache-slot path
- FP8 compute-ready expert cache

### Phase 3

- predictor-driven prefetch
- GPU-side unpack
- CUDA Graph decode capture
- service API

## Current implementation support

The repository already contains the first reusable planning pieces for this direction:

- `include/flashmoe/model_spec.h`
- `include/flashmoe/runtime_plan.h`
- `src/model_spec.cpp`
- `src/runtime_plan.cpp`
- `src/runtime_plan_demo.cpp`
- `src/online_runtime.cpp`
- `src/decode_harness.cpp`
- `src/streamed_runtime.cpp`

Use `flashmoe_runtime_plan` to print the recommended runtime direction for:

- `qwen3.5-397b-a17b`
- `deepseek-v4-flash`

The repository now also contains a first integrated runtime executable:

- `flashmoe_streamed_runtime`
- `flashmoe_streamed_service`

This is not yet the final deployable server, but it already merges:

- manifest-driven routed-expert control
- compute-ready slot cache accounting
- GPU-unpack mainline device profile
- optional measured CUDA compute-profile injection
- narrow-engine style `engine / session / service` layering
- thin HTTP service shell

So the next online work should continue inside that runtime executable rather than by adding more disconnected microbenchmarks.
