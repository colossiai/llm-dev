"""
Training data utilities.

For this educational demo we bundle a small text corpus inline.
Real LLMs train on billions of tokens from the internet.
"""

import torch
from torch.utils.data import Dataset

# A small corpus of conversational Q&A to give the model something to learn.
CORPUS = (
    "Human: What is machine learning?\n"
    "Assistant: Machine learning is a branch of artificial intelligence where computers learn patterns from data instead of being explicitly programmed.\n\n"
    "Human: What is a neural network?\n"
    "Assistant: A neural network is a computing system inspired by the brain. It consists of layers of connected nodes that process information and learn to recognize patterns.\n\n"
    "Human: What is deep learning?\n"
    "Assistant: Deep learning is a subset of machine learning that uses neural networks with many layers. It excels at tasks like image recognition, language understanding, and game playing.\n\n"
    "Human: How does a transformer work?\n"
    "Assistant: A transformer uses self-attention to process all tokens in parallel. Each token attends to every other token, learning which parts of the input are relevant to each other.\n\n"
    "Human: What is attention?\n"
    "Assistant: Attention is a mechanism that lets a model focus on relevant parts of the input. It computes a weighted sum of values, where the weights are based on how relevant each value is to a query.\n\n"
    "Human: What is GPT?\n"
    "Assistant: GPT stands for Generative Pre-trained Transformer. It is a language model that predicts the next token in a sequence, trained on large amounts of text data.\n\n"
    "Human: What is a token?\n"
    "Assistant: A token is a unit of text that a language model processes. It can be a word, part of a word, or even a single character depending on the tokenizer used.\n\n"
    "Human: What is training?\n"
    "Assistant: Training is the process of teaching a model by showing it examples. The model adjusts its parameters to minimize errors in its predictions over many iterations.\n\n"
    "Human: What is a loss function?\n"
    "Assistant: A loss function measures how wrong the model's predictions are. During training, the model tries to minimize this value. For language models, cross-entropy loss is commonly used.\n\n"
    "Human: What is backpropagation?\n"
    "Assistant: Backpropagation is an algorithm that computes gradients of the loss with respect to each parameter. These gradients tell the model how to adjust its weights to reduce the loss.\n\n"
    "Human: What is an embedding?\n"
    "Assistant: An embedding is a dense vector representation of a token. It maps discrete tokens into a continuous space where similar meanings are close together.\n\n"
    "Human: What is overfitting?\n"
    "Assistant: Overfitting happens when a model memorizes the training data too well and fails to generalize to new data. Techniques like dropout and regularization help prevent it.\n\n"
    "Human: What is a parameter?\n"
    "Assistant: A parameter is a learnable value in a neural network, like a weight or bias. Large language models have billions of parameters that are adjusted during training.\n\n"
    "Human: What is inference?\n"
    "Assistant: Inference is using a trained model to make predictions on new data. For a language model, inference means generating text one token at a time.\n\n"
    "Human: What is temperature in sampling?\n"
    "Assistant: Temperature controls the randomness of text generation. A low temperature makes the model more confident and deterministic. A high temperature makes it more creative and random.\n\n"
    "Human: Hello!\n"
    "Assistant: Hello! I am a minimal language model built for educational purposes. I can answer basic questions about AI and machine learning.\n\n"
    "Human: Who are you?\n"
    "Assistant: I am MiniGPT, a tiny transformer language model. I was built to demonstrate how models like ChatGPT work at a fundamental level.\n\n"
    "Human: What is artificial intelligence?\n"
    "Assistant: Artificial intelligence is the field of computer science focused on creating systems that can perform tasks that normally require human intelligence, such as understanding language and recognizing images.\n\n"
    "Human: What is natural language processing?\n"
    "Assistant: Natural language processing is a field of AI that focuses on enabling computers to understand, interpret, and generate human language. Language models are a key part of NLP.\n\n"
    "Human: What is a large language model?\n"
    "Assistant: A large language model is a neural network trained on vast amounts of text. It learns to predict the next word and can generate coherent text, answer questions, and perform many language tasks.\n\n"
)


class TextDataset(Dataset):
    """
    Chops text into overlapping sequences of length `block_size`.
    Each sample: input = text[i:i+block_size], target = text[i+1:i+block_size+1]
    The model learns to predict the next character at every position.
    """

    def __init__(self, token_ids: list[int], block_size: int):
        self.data = torch.tensor(token_ids, dtype=torch.long)
        self.block_size = block_size

    def __len__(self):
        return max(0, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = chunk[:-1]  # input
        y = chunk[1:]   # target (shifted by 1)
        return x, y
