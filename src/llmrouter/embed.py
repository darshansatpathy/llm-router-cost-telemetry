"""Prompt embeddings. MiniLM for routing and the semantic cache; a hashing embedder for fast tests."""

from __future__ import annotations

import hashlib
import itertools
import re
from functools import lru_cache
from typing import Protocol

import numpy as np

DIM = 384


class Embedder(Protocol):
    name: str
    dim: int

    def embed(self, texts: list[str]) -> np.ndarray: ...


class HashEmbedder:
    name = "hash-384"
    dim = DIM

    def embed(self, texts: list[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for i, t in enumerate(texts):
            words = re.findall(r"[a-z0-9]+", t.lower())
            for feat in words + [a + "_" + b for a, b in itertools.pairwise(words)]:
                h = int.from_bytes(hashlib.blake2b(feat.encode(), digest_size=8).digest(), "little")
                out[i, h % self.dim] += 1.0 if (h >> 63) & 1 else -1.0
        n = np.linalg.norm(out, axis=1, keepdims=True)
        return out / np.where(n == 0, 1, n)


class MiniLM:
    name = "all-MiniLM-L6-v2"
    dim = DIM

    def __init__(self, model: str = "sentence-transformers/all-MiniLM-L6-v2", batch: int = 64, max_length: int = 256):
        import torch
        from transformers import AutoModel, AutoTokenizer

        self.torch, self.batch, self.max_length = torch, batch, max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model)
        self.model = AutoModel.from_pretrained(model).eval()

    def embed(self, texts: list[str]) -> np.ndarray:
        torch = self.torch
        chunks = []
        for i in range(0, len(texts), self.batch):
            enc = self.tokenizer(texts[i : i + self.batch], padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
            with torch.inference_mode():
                h = self.model(**enc).last_hidden_state
            m = enc["attention_mask"].unsqueeze(-1).to(h.dtype)
            chunks.append(torch.nn.functional.normalize((h * m).sum(1) / m.sum(1).clamp(min=1e-9), dim=1).numpy())
        return np.vstack(chunks).astype(np.float32) if chunks else np.zeros((0, self.dim), np.float32)


@lru_cache(maxsize=2)
def get(name: str) -> Embedder:
    return HashEmbedder() if name.startswith("hash") else MiniLM()
