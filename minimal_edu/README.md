# MiniGPT — Educational Language Model

A minimal GPT-style language model built from scratch in PyTorch (~200 lines of model code).

## Architecture

Same architecture as GPT-2/ChatGPT, just tiny:

```
Input tokens
  → Token Embedding + Positional Embedding
  → 4x Transformer Blocks
      → LayerNorm → Multi-Head Self-Attention (causal mask)
      → LayerNorm → Feed-Forward MLP
  → LayerNorm → Linear → next-token logits
```

**~400K parameters** | Character-level tokenization | 128-token context window

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (~1-2 min on CPU)
python train.py

# Chat with it
python chat.py
```

## Files

| File | What it does |
|------|-------------|
| `model.py` | The transformer model (attention, FFN, embeddings) |
| `tokenizer.py` | Character-level tokenizer |
| `data.py` | Training corpus and dataset class |
| `train.py` | Training loop with AdamW optimizer |
| `chat.py` | Interactive chat interface |

## How It Works

1. **Tokenization**: Text is split into characters (real LLMs use subword tokens)
2. **Embedding**: Each character becomes a 128-dim vector
3. **Transformer blocks**: Self-attention lets each character "look at" previous characters
4. **Prediction**: The model outputs a probability distribution over the next character
5. **Generation**: Sample from that distribution, append the character, repeat

## Limitations

This is for **learning**, not production:
- Tiny training corpus (~3KB of text)
- Character-level tokenization (vs. BPE in real models)
- ~400K parameters (GPT-3 has 175 billion)
- No RLHF or instruction tuning
