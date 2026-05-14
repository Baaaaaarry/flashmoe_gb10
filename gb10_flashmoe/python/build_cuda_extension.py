from __future__ import annotations

from flashmoe_vllm_plugin.ops import load_flashmoe_cuda_extension


def main() -> None:
    load_flashmoe_cuda_extension(verbose=True)
    print("flashmoe_cuda_ops build complete")


if __name__ == "__main__":
    main()
