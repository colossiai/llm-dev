"""
LLM embedding visualization demo.

In an LLM, every token (roughly: a word or sub-word) is represented as a vector
of numbers called an "embedding". All those vectors stacked together form the
"embedding matrix" of shape (vocab_size, embed_dim). The model learns these
numbers during training so that tokens with similar meaning end up with
similar vectors.

This script:
  1. Builds a tiny untrained embedding matrix with PyTorch's nn.Embedding.
  2. Shows the raw matrix as a heatmap (what the numbers literally look like).
  3. Squashes each vector from embed_dim down to 2D with PCA so we can plot
     tokens as points on a 2D map.
"""


"""

  - What an embedding matrix is (token → vector lookup table) in the module docstring
  - vocab_size vs embed_dim with real-LLM scale comparisons (30k–100k tokens, 768–12288 dims)
  - What nn.Embedding.weight actually represents and why ours is random (no training)
  - Why we need PCA — can't plot 64-D directly, so project to 2 axes of max variation
  - How to read each panel — heatmap rows/columns vs scatter dot distances
  
"""

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from sklearn.decomposition import PCA


def main():
    # The "vocabulary": the set of tokens the model knows about.
    # A real LLM has 30k-100k+ tokens; we use 20 words for the demo.
    vocab = [
        "king", "queen", "man", "woman",
        "cat", "dog", "kitten", "puppy",
        "apple", "banana", "orange", "grape",
        "car", "truck", "bike", "plane",
        "happy", "sad", "angry", "calm",
    ]
    vocab_size = len(vocab)      # how many tokens we have
    embed_dim = 64               # length of each token's vector (real LLMs: 768-12288)

    # Seed so the random numbers are the same every run (reproducible plots).
    torch.manual_seed(0)

    # nn.Embedding is essentially a lookup table: row i holds the vector for token i.
    # Its `.weight` IS the embedding matrix, shape (vocab_size, embed_dim).
    # Note: this matrix is RANDOM here because we haven't trained anything.
    # In a real LLM, training would shape these numbers so related words cluster.
    embedding = nn.Embedding(vocab_size, embed_dim)
    weights = embedding.weight.detach().numpy()  # detach: drop gradient tracking
    print(f"Embedding matrix shape: {weights.shape}")

    # PCA finds the 2 directions in 64-D space with the most variation,
    # then projects every vector onto those 2 axes. Result: an (N, 2) array
    # we can scatter-plot. This is the standard trick for visualizing high-dim
    # data — t-SNE and UMAP are popular alternatives.
    coords = PCA(n_components=2).fit_transform(weights)

    # Two side-by-side plots.
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- Left panel: raw matrix as a heatmap -----------------------------
    # Each row = one token, each column = one of the 64 embedding dimensions.
    # Color = the numeric value at that (token, dim) cell.
    im = axes[0].imshow(weights, aspect="auto", cmap="viridis")
    axes[0].set_title(f"Raw embedding matrix ({vocab_size} x {embed_dim})")
    axes[0].set_xlabel("embedding dim")
    axes[0].set_ylabel("token")
    axes[0].set_yticks(range(vocab_size))
    axes[0].set_yticklabels(vocab, fontsize=8)
    fig.colorbar(im, ax=axes[0])

    # --- Right panel: 2D scatter after PCA -------------------------------
    # Each dot is one token. Distance between dots ≈ similarity in the
    # original 64-D space. (With random weights, expect no real structure.)
    axes[1].scatter(coords[:, 0], coords[:, 1], s=80, alpha=0.7, c="tab:blue")
    for i, word in enumerate(vocab):
        axes[1].annotate(
            word, (coords[i, 0], coords[i, 1]),
            xytext=(6, 4), textcoords="offset points", fontsize=9,
        )
    axes[1].set_title("PCA projection (2D)")
    axes[1].set_xlabel("PC1")  # first principal component (most variation)
    axes[1].set_ylabel("PC2")  # second principal component
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("embeddings.png", dpi=120)
    print("Saved plot to embeddings.png")
    plt.show()


if __name__ == "__main__":
    main()
