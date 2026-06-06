"""
Training script for MiniGPT.

Usage:
    python train.py

This trains the model on the built-in corpus and saves a checkpoint.
Training takes ~1-2 minutes on CPU.
"""

import os
import time

import torch
from torch.utils.data import DataLoader

from data import CORPUS, TextDataset
from model import MiniGPT
from tokenizer import CharTokenizer

# --- Hyperparameters ---
BLOCK_SIZE = 128    # context window (max sequence length)
N_EMBD = 128        # embedding dimension
N_HEAD = 4          # number of attention heads
N_LAYER = 4         # number of transformer blocks
BATCH_SIZE = 32
LEARNING_RATE = 3e-4
EPOCHS = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")


def train():
    print(f"Device: {DEVICE}")
    print(f"Training on {len(CORPUS)} characters of text\n")

    # 1. Build tokenizer
    tokenizer = CharTokenizer()
    tokenizer.fit(CORPUS)
    print(f"Vocabulary size: {tokenizer.vocab_size} characters")
    print(f"Vocab: {''.join(tokenizer.id_to_char[i] for i in range(tokenizer.vocab_size))!r}\n")

    # 2. Tokenize corpus
    token_ids = tokenizer.encode(CORPUS)

    # 3. Create dataset & dataloader
    dataset = TextDataset(token_ids, BLOCK_SIZE)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    print(f"Dataset: {len(dataset)} samples, {len(loader)} batches/epoch\n")

    # 4. Create model
    model = MiniGPT(
        vocab_size=tokenizer.vocab_size,
        block_size=BLOCK_SIZE,
        n_embd=N_EMBD,
        n_head=N_HEAD,
        n_layer=N_LAYER,
    ).to(DEVICE)
    print(f"Model parameters: {model.count_parameters():,}\n")

    # 5. Optimizer (AdamW, same as used for GPT training)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

    # 6. Training loop
    model.train()
    start = time.time()

    for epoch in range(1, EPOCHS + 1):
        total_loss = 0.0
        for x_batch, y_batch in loader:
            x_batch, y_batch = x_batch.to(DEVICE), y_batch.to(DEVICE)

            logits = model(x_batch)  # (B, T, vocab_size)

            # Cross-entropy loss: how well does the model predict the next token?
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                y_batch.view(-1),
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        if epoch % 10 == 0 or epoch == 1:
            elapsed = time.time() - start
            print(f"Epoch {epoch:4d}/{EPOCHS} | loss: {avg_loss:.4f} | time: {elapsed:.1f}s")

    print(f"\nTraining complete in {time.time() - start:.1f}s")

    # 7. Save checkpoint
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    ckpt_path = os.path.join(CHECKPOINT_DIR, "minigpt.pt")
    torch.save(
        {
            "model_state": model.state_dict(),
            "tokenizer_chars": tokenizer.char_to_id,
            "config": {
                "vocab_size": tokenizer.vocab_size,
                "block_size": BLOCK_SIZE,
                "n_embd": N_EMBD,
                "n_head": N_HEAD,
                "n_layer": N_LAYER,
            },
        },
        ckpt_path,
    )
    print(f"Checkpoint saved to {ckpt_path}")

    # 8. Quick sample to verify
    print("\n--- Sample generation ---")
    model.eval()
    prompt = "Human: What is"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)
    out = model.generate(ids, max_new_tokens=150, temperature=0.7)
    print(tokenizer.decode(out[0].tolist()))


if __name__ == "__main__":
    train()
