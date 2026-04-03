import argparse

import torch

from llama3.model import LLaMA, LLaMAConfig
from llama3.tokenizer import BPETokenizer


def main():
    parser = argparse.ArgumentParser(description="Generate text with trained LLaMA")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/llama3.pt")
    parser.add_argument("--prompt", type=str, default="The ")
    parser.add_argument("--max_new_tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
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
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model.eval()

    prompt_tokens = tokenizer.encode(args.prompt)
    prompt_tensor = torch.tensor([prompt_tokens], dtype=torch.long).to(device)

    generated = model.generate(
        prompt_tensor,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
    )

    output_text = tokenizer.decode(generated[0].tolist())
    print(f"\n--- Generated Text ---\n{output_text}\n--- End ---")


if __name__ == "__main__":
    main()
