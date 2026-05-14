from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .export_qwen_experts import require_deps, resolve_model_dir
except ImportError:
    import sys
    from pathlib import Path as _Path

    current_dir = _Path(__file__).resolve().parent
    package_root = current_dir.parent
    sys.path.append(str(package_root))
    sys.path.append(str(current_dir))
    try:
        from flashmoe_vllm_plugin.export_qwen_experts import require_deps, resolve_model_dir
    except ImportError:
        from export_qwen_experts import require_deps, resolve_model_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Export tokenizer runtime artifact for streamed service.")
    parser.add_argument("--model", required=True, help="Local model path or Hugging Face repo id.")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cache-dir", default="", help="Optional HF cache dir.")
    parser.add_argument("--runtime-vocab-size", type=int, default=32)
    args = parser.parse_args()

    _, snapshot_download, LocalEntryNotFoundError, HfHubHTTPError, _, _ = require_deps()
    model_dir = resolve_model_dir(
        args.model,
        args.cache_dir or None,
        snapshot_download,
        LocalEntryNotFoundError,
        HfHubHTTPError,
    )

    tokenizer_json = model_dir / "tokenizer.json"
    if not tokenizer_json.exists():
        raise SystemExit(f"tokenizer.json not found under {model_dir}")
    tokenizer_config_json = model_dir / "tokenizer_config.json"
    special_tokens_map_json = model_dir / "special_tokens_map.json"

    raw = json.loads(tokenizer_json.read_text())
    vocab = raw.get("model", {}).get("vocab")
    if not isinstance(vocab, dict) or not vocab:
        raise SystemExit("tokenizer.json does not contain model.vocab")

    size = max(args.runtime_vocab_size, max(int(v) for v in vocab.values()) + 1)
    tokens = ["<unk>"] * size
    for token, idx in vocab.items():
        idx = int(idx)
        if idx < size:
            tokens[idx] = token

    tokenizer_config = json.loads(tokenizer_config_json.read_text()) if tokenizer_config_json.exists() else {}
    special_tokens_map = json.loads(special_tokens_map_json.read_text()) if special_tokens_map_json.exists() else {}

    def resolve_special_id(name: str):
        token = tokenizer_config.get(name)
        if token is None:
            token = special_tokens_map.get(name)
        if isinstance(token, dict):
            token = token.get("content")
        if isinstance(token, str) and token in vocab:
            return int(vocab[token])
        return None

    output = args.output.resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "tokens": tokens[: args.runtime_vocab_size],
                "bos_token_id": resolve_special_id("bos_token"),
                "eos_token_id": resolve_special_id("eos_token"),
                "unk_token_id": resolve_special_id("unk_token"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    print(f"model_dir={model_dir}")
    print(f"tokenizer_artifact={output}")
    print(f"runtime_vocab_size={min(args.runtime_vocab_size, len(tokens))}")


if __name__ == "__main__":
    main()
