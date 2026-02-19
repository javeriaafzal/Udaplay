from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.api.models.Collection import Collection
from chromadb.utils.embedding_functions import OpenAIEmbeddingFunction

from .models import GameRecord, RetrievalItem, RetrievalResult


class VectorStoreManager:
    """Reusable ChromaDB manager for ingesting and searching game data."""

    def __init__(
        self,
        persist_path: str = ".chroma",
        collection_name: str = "games",
        openai_api_key: str | None = None,
        embedding_model: str = "text-embedding-3-small",
    ) -> None:
        self.client = chromadb.PersistentClient(path=persist_path)
        embedding_function = None
        if openai_api_key:
            embedding_function = OpenAIEmbeddingFunction(
                api_key=openai_api_key,
                model_name=embedding_model,
            )

        self.collection: Collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function,
            metadata={"hnsw:space": "cosine"},
        )

    def ingest_json(self, json_path: str | Path) -> int:
        records = self._load_game_records(json_path)
        if not records:
            return 0

        ids = [r.id for r in records]
        documents = [self._to_document(r) for r in records]
        metadatas = [self._to_metadata(r) for r in records]

        self.collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
        return len(records)

    def semantic_search(self, query: str, top_k: int = 5) -> RetrievalResult:
        raw = self.collection.query(query_texts=[query], n_results=top_k)

        ids = raw.get("ids", [[]])[0]
        docs = raw.get("documents", [[]])[0]
        metadatas = raw.get("metadatas", [[]])[0]
        distances = raw.get("distances", [[]])[0]

        matches: list[RetrievalItem] = []
        for i, game_id in enumerate(ids):
            distance = distances[i] if i < len(distances) else 1.0
            score = max(0.0, 1.0 - float(distance))
            metadata = metadatas[i] if i < len(metadatas) else {}
            title = str(metadata.get("title", game_id))
            description = docs[i] if i < len(docs) else ""
            matches.append(
                RetrievalItem(
                    id=game_id,
                    title=title,
                    score=score,
                    metadata=metadata,
                    description=description,
                )
            )

        return RetrievalResult(query=query, matches=matches)

    @staticmethod
    def _to_document(game: GameRecord) -> str:
        return (
            f"Title: {game.title}\n"
            f"Description: {game.description}\n"
            f"Genre: {', '.join(game.genre)}\n"
            f"Publisher: {game.publisher or 'Unknown'}\n"
            f"Release Date: {game.release_date or 'Unknown'}\n"
            f"Platforms: {', '.join(game.platforms)}"
        )

    @staticmethod
    def _to_metadata(game: GameRecord) -> dict[str, str]:
        return {
            "title": game.title,
            "publisher": game.publisher or "Unknown",
            "release_date": game.release_date or "Unknown",
            "genre": ", ".join(game.genre),
            "platforms": ", ".join(game.platforms),
        }

    @staticmethod
    def _load_game_records(json_path: str | Path) -> list[GameRecord]:
        path = Path(json_path)
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)

        rows: Iterable[dict] = payload if isinstance(payload, list) else payload.get("games", [])
        return [GameRecord.model_validate(row) for row in rows]
