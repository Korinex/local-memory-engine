from functools import lru_cache

import numpy as np
from sentence_transformers import SentenceTransformer


class LocalEmbedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ) -> None:
        self.model = SentenceTransformer(model_name)

    def encode(self, text: str) -> np.ndarray:
        vector = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return np.asarray(vector, dtype=np.float32)


@lru_cache(maxsize=1)
def get_embedder() -> LocalEmbedder:
    return LocalEmbedder()
