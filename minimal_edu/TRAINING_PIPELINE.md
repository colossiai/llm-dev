```
⏺ Here's a summary of the training pipeline:

  Input:
  - A hardcoded CORPUS string (~20 Q&A pairs about AI/ML topics in Human:/Assistant: conversational format) defined in data.py
  - The text is character-tokenized (each unique character gets an ID), then chopped into overlapping sequences of length 128 (BLOCK_SIZE)
  - Each training sample is a pair: x = text[i:i+128] (input) and y = text[i+1:i+129] (target shifted by 1 character) — classic next-character prediction

  Output:
  - A checkpoint file saved to minimal_edu/checkpoints/minigpt.pt containing:
    - Model weights (model_state)
    - Tokenizer character mapping (tokenizer_chars)
    - Model config (vocab_size, block_size, n_embd=128, n_head=4, n_layer=4)
  - A sample generation printed to stdout using the prompt "Human: What is" (150 new tokens at temperature 0.7)

  It's a tiny 4-layer character-level GPT trained for 200 epochs with AdamW, designed to run in ~1-2 min on CPU.

```