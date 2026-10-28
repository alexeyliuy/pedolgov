from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pedolgov.manifest.config import RetrievalConfig

FloatArray = npt.NDArray[np.float64]


class ContrieverRetriever:
    def __init__(self, config: RetrievalConfig) -> None:
        if config.chunk_size <= 0 or not 0 <= config.overlap < config.chunk_size:
            raise ValueError("retrieval overlap must be nonnegative and less than chunk size")
        if config.top_k < 1:
            raise ValueError("top_k must be positive")
        try:
            import faiss
            import torch
            from transformers import AutoModel, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError("Install the 'retrieval' extra for Contriever and FAISS") from exc
        self.faiss = faiss
        self.torch = torch
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.encoder)
        self.model = AutoModel.from_pretrained(config.encoder)
        self.model.eval()

    def chunks(self, texts: tuple[str, ...]) -> list[str]:
        chunks = []
        step = self.config.chunk_size - self.config.overlap
        for text in texts:
            tokens = self.tokenizer.encode(text, add_special_tokens=False)
            for start in range(0, len(tokens), step):
                window = tokens[start : start + self.config.chunk_size]
                if window:
                    chunks.append(self.tokenizer.decode(window, skip_special_tokens=True))
                if start + self.config.chunk_size >= len(tokens):
                    break
        return chunks

    def _embed(self, texts: list[str]) -> FloatArray:
        result = []
        for offset in range(0, len(texts), 32):
            encoded = self.tokenizer(
                texts[offset : offset + 32],
                padding=True,
                truncation=True,
                max_length=min(self.config.chunk_size, self.model.config.max_position_embeddings),
                return_tensors="pt",
            )
            with self.torch.inference_mode():
                hidden = self.model(**encoded).last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1)
                pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
            result.extend(pooled.cpu().numpy())
        return np.asarray(result, dtype=np.float32)

    def retrieve(self, query: str, texts: tuple[str, ...]) -> tuple[str, ...]:
        chunks = self.chunks(texts)
        if not chunks:
            raise ValueError("no label passages available for retrieval")
        vectors = self._embed(chunks)
        index = self.faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)
        _, indices = index.search(self._embed([query]), min(self.config.top_k, len(chunks)))
        return tuple(chunks[int(i)] for i in indices[0] if i >= 0)
