from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import numpy.typing as npt

from pedolgov.manifest.config import RetrievalConfig

FloatArray = npt.NDArray[np.float64]


class Encoder(Protocol):
    def encode(self, texts: list[str]) -> FloatArray: ...


@dataclass(frozen=True)
class Chunker:
    chunk_size: int
    overlap: int

    def split(self, tokens: list[str]) -> list[list[str]]:
        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        step = max(1, self.chunk_size - self.overlap)
        chunks: list[list[str]] = []
        for start in range(0, max(1, len(tokens)), step):
            window = tokens[start : start + self.chunk_size]
            if window:
                chunks.append(window)
            if start + self.chunk_size >= len(tokens):
                break
        return chunks


class HashEncoder:
    def __init__(self, dim: int) -> None:
        self.dim = dim

    def encode(self, texts: list[str]) -> FloatArray:
        out = np.zeros((len(texts), self.dim), dtype=np.float64)
        for row, text in enumerate(texts):
            for token in text.lower().split():
                digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
                idx = int.from_bytes(digest[:4], "little") % self.dim
                sign = 1.0 if digest[4] & 1 else -1.0
                out[row, idx] += sign
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0.0] = 1.0
        return np.asarray(out / norms, dtype=np.float64)


class FaissIndex:
    def __init__(self, vectors: FloatArray) -> None:
        try:
            import faiss
        except ImportError as exc:
            raise RuntimeError("faiss is not installed; install the 'retrieval' extra") from exc
        self._index = faiss.IndexFlatIP(vectors.shape[1])
        self._index.add(vectors.astype(np.float32))

    def search(self, query: FloatArray, top_k: int) -> list[int]:
        _, idx = self._index.search(query.astype(np.float32), top_k)
        return [int(i) for i in idx[0] if i >= 0]


@dataclass
class Retriever:
    config: RetrievalConfig
    encoder: Encoder

    @classmethod
    def offline(cls, config: RetrievalConfig) -> Retriever:
        return cls(config=config, encoder=HashEncoder(config.dim))

    def rank(self, query: str, passages: list[str]) -> list[int]:
        if not passages:
            return []
        matrix = self.encoder.encode([query, *passages])
        scores = matrix[1:] @ matrix[0]
        order = np.argsort(-scores)
        return [int(i) for i in order[: self.config.top_k]]
