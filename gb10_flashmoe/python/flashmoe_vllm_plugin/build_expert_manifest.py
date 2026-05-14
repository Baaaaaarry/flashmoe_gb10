from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PATTERN = re.compile(r".*layer[_-]?(\d+).*expert[_-]?(\d+).*", re.IGNORECASE)


def build_manifest(root: Path) -> dict[str, list[dict[str, int | str]]]:
    entries: list[dict[str, int | str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        match = PATTERN.match(path.name)
        if not match:
            continue
        layer_id = int(match.group(1))
        expert_id = int(match.group(2))
        entries.append({
            "layer_id": layer_id,
            "expert_id": expert_id,
            "path": str(path.resolve()),
            "offset": 0,
            "size_bytes": path.stat().st_size,
        })
    return {"entries": entries}


def main() -> None:
    parser = argparse.ArgumentParser(description="Build FlashMoE expert manifest from expert files.")
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    manifest = build_manifest(args.root)
    args.output.write_text(json.dumps(manifest, indent=2, sort_keys=True))
    print(f"wrote {len(manifest['entries'])} entries to {args.output}")


if __name__ == "__main__":
    main()
