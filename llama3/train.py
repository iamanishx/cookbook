import os
import math
import json
import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from llama3.model import LLaMA, LLaMAConfig
from llama3.tokenizer import BPETokenizer


class TokenDataset(Dataset):
    """Dataset that loads tokenized text chunks.

    Splits a flat list of token IDs into fixed-length chunks
    for next-token prediction training.

    Args:
        token_ids: Flat list of all token IDs
        seq_len: Sequence length for each chunk
    """

    def __init__(self, token_ids: list[int], seq_len: int):
        self.seq_len = seq_len
        self.data = []
        for i in range(0, len(token_ids) - seq_len, seq_len):
            chunk = token_ids[i : i + seq_len + 1]
            if len(chunk) == seq_len + 1:
                self.data.append(chunk)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        chunk = self.data[idx]
        x = torch.tensor(chunk[:-1], dtype=torch.long)
        y = torch.tensor(chunk[1:], dtype=torch.long)
        return x, y


def load_text_file(path: str) -> str:
    """Load a plain text file."""
    return Path(path).read_text(encoding="utf-8")


def load_jsonl_file(path: str, text_field: str = "text") -> str:
    """Load a JSONL file and concatenate the text field."""
    texts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data = json.loads(line)
                if text_field in data:
                    texts.append(data[text_field])
    return "\n".join(texts)


def load_data(path: str, text_field: str = "text") -> str:
    """Load data from a text or JSONL file."""
    ext = Path(path).suffix.lower()
    if ext == ".jsonl":
        return load_jsonl_file(path, text_field)
    return load_text_file(path)


def train(
    model: LLaMA,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LambdaLR | None,
    device: torch.device,
    epochs: int,
    max_grad_norm: float = 1.0,
    log_every: int = 10,
):
    """Training loop with gradient clipping and optional LR scheduling."""
    model.train()
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0.0
        n_batches = 0

        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_x, batch_y in pbar:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad()
            logits = model(batch_x)

            loss = criterion(logits.view(-1, logits.size(-1)), batch_y.view(-1))
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()

            if scheduler is not None:
                scheduler.step()

            total_loss += loss.item()
            n_batches += 1

            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_loss = total_loss / n_batches
        if (epoch + 1) % log_every == 0:
            print(f"Epoch {epoch+1}/{epochs} | Avg Loss: {avg_loss:.4f}")


def main():
    parser = argparse.ArgumentParser(description="Train LLaMA on BPE-tokenized text")
    parser.add_argument("--data", type=str, required=True, help="Path to .txt or .jsonl data file")
    parser.add_argument("--text_field", type=str, default="text", help="JSONL field name (if using .jsonl)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--seq_len", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output_dir", type=str, default="checkpoints")
    args = parser.parse_args()

    device = torch.device(args.device)

    print(f"Loading data from {args.data}...")
    text = load_data(args.data, args.text_field)
    print(f"Loaded {len(text):,} characters")

    tokenizer = BPETokenizer("cl100k_base")
    print(f"Tokenizing with BPE (vocab size: {tokenizer.vocab_size:,})...")
    token_ids = tokenizer.encode(text)
    print(f"Total tokens: {len(token_ids):,}")

    dataset = TokenDataset(token_ids, args.seq_len)
    print(f"Training samples: {len(dataset):,}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    config = LLaMAConfig(
        vocab_size=tokenizer.vocab_size,
        dim=256,
        n_layers=6,
        n_heads=4,
        n_kv_heads=2,
        head_dim=64,
        hidden_dim=512,
        max_seq_len=args.seq_len,
    )

    model = LLaMA(config).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    total_steps = len(dataloader) * args.epochs
    warmup_steps = int(0.1 * total_steps)

    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        return 0.5 * (1 + math.cos(math.pi * (step - warmup_steps) / (total_steps - warmup_steps)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    train(model, dataloader, optimizer, scheduler, device, args.epochs)

    os.makedirs(args.output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(args.output_dir, "llama3.pt"))
    print(f"Model saved to {args.output_dir}/llama3.pt")


if __name__ == "__main__":
    main()
