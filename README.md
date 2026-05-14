# flashMoE Workspace

This repository keeps two codebases side by side:

- `gb10_flashmoe/`: the earlier from-scratch FlashMoE prototype and tooling.
- `ds4_gb10/`: the upstream `ds4` codebase, kept close to upstream for GB10/CUDA work.

Branch policy:

- `main`: preserves both directories without FlashMoE integration changes inside `ds4_gb10/`.
- `flashmoe_integration`: based on upstream `ds4_gb10/` and used for FlashMoE integration work.

The intent is to keep `ds4_gb10/` easy to sync with upstream while preserving the earlier `gb10_flashmoe/` implementation in the same workspace.
