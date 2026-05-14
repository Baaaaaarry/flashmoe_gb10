from __future__ import annotations

import argparse
import sys
import time


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Correctness-first CLI using the official Hugging Face model execution path."
    )
    parser.add_argument("--model", required=True, help="Local model directory.")
    parser.add_argument("--prompt", default="", help="Prompt text.")
    parser.add_argument("--prompt-stdin", action="store_true", help="Read prompt text from stdin.")
    parser.add_argument("--interactive", action="store_true", help="Interactive chat loop.")
    parser.add_argument("--max-tokens", type=int, default=32, help="Max new tokens.")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature; 0 means greedy.")
    parser.add_argument("--top-p", type=float, default=1.0, help="Top-p for sampling when temperature > 0.")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--device-map", default="cuda:0")
    parser.add_argument("--torch-dtype", default="auto", choices=["auto", "float16", "bfloat16", "float32"])
    return parser


def resolve_dtype(name: str):
    import torch

    if name == "float16":
        return torch.float16
    if name == "bfloat16":
        return torch.bfloat16
    if name == "float32":
        return torch.float32
    return "auto"


def load_backend(model_path: str, device_map: str, torch_dtype: str, trust_remote_code: bool):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=trust_remote_code)
    resolved_device_map = device_map
    if device_map in {"cuda", "cuda:0", "gpu", "gpu:0"}:
        resolved_device_map = {"": 0}
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=trust_remote_code,
        device_map=resolved_device_map,
        torch_dtype=resolve_dtype(torch_dtype),
    )
    model.eval()
    return torch, tokenizer, model


def generate_once(torch, tokenizer, model, prompt: str, max_tokens: int, temperature: float, top_p: float):
    encoded = tokenizer(prompt, return_tensors="pt")
    inputs = {k: v.to(model.device) for k, v in encoded.items()}
    do_sample = temperature > 0.0
    start = time.time()
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=do_sample,
            temperature=max(temperature, 1e-5) if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - start
    prompt_len = inputs["input_ids"].shape[1]
    generated = output[0, prompt_len:]
    text = tokenizer.decode(generated, skip_special_tokens=True)
    tok_per_s = (generated.numel() / elapsed) if elapsed > 0 and generated.numel() > 0 else 0.0
    return text, int(prompt_len), int(generated.numel()), elapsed, tok_per_s


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    try:
        torch, tokenizer, model = load_backend(
            args.model,
            args.device_map,
            args.torch_dtype,
            args.trust_remote_code,
        )
    except Exception as exc:  # pragma: no cover
        print(f"reference cli failed during model load: {exc}", file=sys.stderr)
        return 1

    if args.prompt_stdin:
        args.prompt = sys.stdin.read()

    if args.interactive:
        history: list[dict[str, str]] = []
        if args.prompt:
            history.append({"role": "user", "content": args.prompt})
            prompt = args.prompt
            try:
                text, prompt_tokens, completion_tokens, elapsed, tok_per_s = generate_once(
                    torch, tokenizer, model, prompt, args.max_tokens, args.temperature, args.top_p
                )
            except Exception as exc:  # pragma: no cover
                print(f"reference cli failed during generation: {exc}", file=sys.stderr)
                return 1
            print(text)
            print(
                f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} elapsed_s={elapsed:.3f} tok_per_s={tok_per_s:.3f}",
                file=sys.stderr,
            )
            history.append({"role": "assistant", "content": text})
        while True:
            try:
                line = input("> ")
            except EOFError:
                break
            if line in {"/exit", "/quit"}:
                break
            history.append({"role": "user", "content": line})
            prompt = "\n".join(f"{m['role']}: {m['content']}" for m in history) + "\nassistant:"
            try:
                text, prompt_tokens, completion_tokens, elapsed, tok_per_s = generate_once(
                    torch, tokenizer, model, prompt, args.max_tokens, args.temperature, args.top_p
                )
            except Exception as exc:  # pragma: no cover
                print(f"reference cli failed during generation: {exc}", file=sys.stderr)
                return 1
            print(text)
            print(
                f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} elapsed_s={elapsed:.3f} tok_per_s={tok_per_s:.3f}",
                file=sys.stderr,
            )
            history.append({"role": "assistant", "content": text})
        return 0

    try:
        text, prompt_tokens, completion_tokens, elapsed, tok_per_s = generate_once(
            torch, tokenizer, model, args.prompt, args.max_tokens, args.temperature, args.top_p
        )
    except Exception as exc:  # pragma: no cover
        print(f"reference cli failed during generation: {exc}", file=sys.stderr)
        return 1

    print(text)
    print(
        f"prompt_tokens={prompt_tokens} completion_tokens={completion_tokens} elapsed_s={elapsed:.3f} tok_per_s={tok_per_s:.3f}",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
