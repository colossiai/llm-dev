"""
Interactive chat with MiniGPT.

Usage:
    python chat.py

Loads a trained checkpoint and lets you chat with the model.
The model generates text by predicting one character at a time,
just like GPT — but much smaller.
"""

import os
import sys

import torch

from model import MiniGPT
from tokenizer import CharTokenizer

CHECKPOINT_PATH = os.path.join(os.path.dirname(__file__), "checkpoints", "minigpt.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"


def load_model() -> tuple[MiniGPT, CharTokenizer]:
    if not os.path.exists(CHECKPOINT_PATH):
        print("No checkpoint found. Run `python train.py` first.")
        sys.exit(1)

    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE, weights_only=True)
    config = ckpt["config"]

    # Rebuild tokenizer
    tokenizer = CharTokenizer()
    tokenizer.char_to_id = ckpt["tokenizer_chars"]
    tokenizer.id_to_char = {v: k for k, v in tokenizer.char_to_id.items()}
    tokenizer.vocab_size = config["vocab_size"]

    # Rebuild model
    model = MiniGPT(**config).to(DEVICE)
    model.load_state_dict(ckpt["model_state"])
    model.eval()

    return model, tokenizer


def generate_response(model: MiniGPT, tokenizer: CharTokenizer, user_input: str) -> str:
    """Format as conversation, generate until we see a double newline or hit the limit."""
    prompt = f"Human: {user_input}\nAssistant:"
    ids = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=DEVICE)

    output = model.generate(ids, max_new_tokens=300, temperature=0.7, top_k=30)
    full_text = tokenizer.decode(output[0].tolist())

    # Extract just the assistant's response
    response = full_text[len(prompt):]

    # Stop at the next "Human:" or double newline (end of turn)
    for stop in ["\nHuman:", "\n\nHuman:", "\n\n"]:
        if stop in response:
            response = response[: response.index(stop)]

    return response.strip()


def main():
    print("=" * 50)
    print("  MiniGPT — Educational Language Model Chat")
    print("=" * 50)
    print(f"  Device: {DEVICE}")
    print(f"  Type 'quit' or Ctrl+C to exit")
    print("=" * 50)
    print()

    model, tokenizer = load_model()
    print(f"Model loaded: {model.count_parameters():,} parameters\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nBye!")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit"):
            print("Bye!")
            break

        response = generate_response(model, tokenizer, user_input)
        print(f"MiniGPT: {response}\n")


if __name__ == "__main__":
    main()
