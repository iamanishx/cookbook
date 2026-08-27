import argparse

import torch

from llama3.model import LLaMA, LLaMAConfig
from llama3.tokenizer import BPETokenizer


def prefill_stage(
    model: LLaMA, prompt_tokens: torch.Tensor
) -> tuple[torch.Tensor, dict]:
    print("[Prefill Stage] Processing prompt...")
    model.reset_cache()
    logits = model.forward(prompt_tokens, start_pos=0, use_cache=True)
    last_token_logits = logits[:, -1, :]
    kv_cache = {
        "seq_len": prompt_tokens.shape[1],
        "prompt_len": prompt_tokens.shape[1],
    }
    return last_token_logits, kv_cache


def decode_stage(
    model: LLaMA,
    token_logits: torch.Tensor,
    temperature: float = 1.0,
    top_k: int | None = None,
) -> torch.Tensor:
    if temperature > 0:
        logits = token_logits / temperature
        if top_k is not None:
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
    else:
        next_token = torch.argmax(token_logits, dim=-1, keepdim=True)

    return next_token


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained LLaMA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/llama3.pt")
    parser.add_argument("--prompt", type=str, default="The ")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    device = torch.device(args.device)

    tokenizer = BPETokenizer("cl100k_base")

    config = LLaMAConfig(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        hidden_dim=512,
        max_seq_len=512,
    )

    model = LLaMA(config).to(device)
    model.load_state_dict(
        torch.load(args.checkpoint, map_location=device, weights_only=True)
    )
    model.eval()

    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    logits, kv_cache = prefill_stage(model, prompt_tensor)
    generated = prompt_tensor.clone()

    print(f"[Decode Stage] Generating {args.max_new_tokens} tokens...")
    with torch.inference_mode():
        for step in range(args.max_new_tokens):
            next_token = decode_stage(
                model,
                logits,
                temperature=args.temperature,
                top_k=args.top_k,
            )

            generated = torch.cat([generated, next_token], dim=1)
            
            start_pos = kv_cache["seq_len"]
            kv_cache["seq_len"] += 1

            if kv_cache["seq_len"] > model.config.max_seq_len:
                generated = generated[:, -model.config.max_seq_len :]
                kv_cache["seq_len"] = model.config.max_seq_len

            logits = model.forward(
                next_token,
                start_pos=start_pos,
                use_cache=True,
            )[:, -1, :]

            if (step + 1) % 20 == 0:
                print(f"  Generated {step + 1}/{args.max_new_tokens} tokens")

    output_text = tokenizer.decode(generated[0].tolist())
    print(f"\n--- Generated Text ---\n{output_text}\n--- End ---")


if __name__ == "__main__":
    main()
