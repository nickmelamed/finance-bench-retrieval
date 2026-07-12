from __future__ import annotations

import os
from typing import List

from dotenv import load_dotenv

from qdrant_client import QdrantClient

from qdrant_client.models import (
    Distance,
    PointStruct,
    VectorParams,
)

from finance_bench.types.schemas import (
    DocumentChunk,
)

load_dotenv()


COLLECTION_NAME = os.getenv(
    "COLLECTION_NAME",
    "financebench",
)


class QdrantManager:

    _client = None

    def __init__(self):

        self.collection_name = COLLECTION_NAME

        # Singleton Qdrant client

        if QdrantManager._client is None:

            QdrantManager._client = (
                QdrantClient(
                    url=os.getenv(
                        "QDRANT_URL"
                    ),
                    api_key=os.getenv(
                        "QDRANT_API_KEY"
                    ),
                )
            )

        self.client = (
            QdrantManager._client
        )

        # BGE-large dimension

        self.vector_size = 1024

    # collection management

    def ensure_collection(self):

        collections = (
            self.client
            .get_collections()
            .collections
        )

        existing = {
            c.name
            for c in collections
        }

        if self.collection_name in existing:

            print(
                f"Collection exists: "
                f"{self.collection_name}"
            )

            return

        print(
            f"Creating collection: "
            f"{self.collection_name}"
        )

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(
                size=self.vector_size,
                distance=Distance.COSINE,
            ),
        )

    # upload

    def upload_documents(
        self,
        chunks: List[DocumentChunk],
        embeddings: List[List[float]],
        batch_size: int = 64,
    ):

        if len(chunks) != len(embeddings):

            raise ValueError(
                "Chunks and embeddings "
                "must have same length."
            )

        points = []

        for chunk, vector in zip(
            chunks,
            embeddings,
        ):

            payload = {
                "chunk_id": chunk.chunk_id,
                "document_id": (
                    chunk.document_id
                ),
                "text": chunk.text,
                **chunk.metadata,
            }

            points.append(
                PointStruct(
                    id=chunk.chunk_id,
                    vector=vector,
                    payload=payload,
                )
            )

        print(
            f"Uploading {len(points)} "
            f"points to Qdrant..."
        )

        for i in range(
            0,
            len(points),
            batch_size,
        ):

            batch = points[
                i : i + batch_size
            ]

            self.client.upsert(
                collection_name=(
                    self.collection_name
                ),
                points=batch,
            )

        print(
            "Qdrant upload finished."
        )