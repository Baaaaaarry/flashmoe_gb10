from __future__ import annotations

import argparse
from pathlib import Path

try:
    from .trace_utils import generate_synthetic_trace
except ImportError:
    import sys
    from pathlib import Path as _Path

    current_dir = _Path(__file__).resolve().parent
    package_root = current_dir.parent
    sys.path.append(str(package_root))
    sys.path.append(str(current_dir))
    try:
        from flashmoe_vllm_plugin.trace_utils import generate_synthetic_trace
    except ImportError:
        from trace_utils import generate_synthetic_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate a synthetic routing trace from expert manifest.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--locality", type=float, default=0.85)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    generate_synthetic_trace(
        manifest_path=args.manifest,
        output_path=args.output,
        steps=args.steps,
        top_k=args.top_k,
        locality=args.locality,
        seed=args.seed,
    )
    print(f"wrote {args.output.resolve()}")


if __name__ == "__main__":
    main()
