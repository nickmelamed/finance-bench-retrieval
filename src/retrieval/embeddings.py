import os

from sentence_transformers import (
    SentenceTransformer,
)


class EmbeddingModel:

    def __init__(
        self,
        model_name: str | None = None,
    ):

        self.model_name = (
            model_name
            or os.getenv(
                "EMBEDDING_MODEL",
                "BAAI/bge-large-en-v1.5",
            )
        )

        self.model = SentenceTransformer(
            self.model_name
        )

    def embed(
        self,
        texts: list[str],
    ):

        return self.model.encode(
            texts,
            normalize_embeddings=True,
            show_progress_bar=False,
        )