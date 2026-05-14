from __future__ import annotations

import argparse
import json

from transformers import AutoTokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Tokenizer bridge for streamed runtime.")
    parser.add_argument("--model", required=True)
    parser.add_argument("--encode", default=None)
    parser.add_argument("--decode-ids", default=None)
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    if args.encode is not None:
        ids = tok.encode(args.encode, add_special_tokens=True)
        print(json.dumps(ids, ensure_ascii=False))
        return

    if args.decode_ids is not None:
        ids = json.loads(args.decode_ids)
        if not isinstance(ids, list):
            raise SystemExit("decode ids must be a JSON list")
        text = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
        print(text)
        return

    raise SystemExit("either --encode or --decode-ids is required")


if __name__ == "__main__":
    main()
