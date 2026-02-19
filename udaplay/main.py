from __future__ import annotations

import argparse

from .agent import UdaPlayAgent
from .reporting import ReportFormatter
from .vector_store import VectorStoreManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UdaPlay video game research agent")
    parser.add_argument("question", help="Question about a video game")
    parser.add_argument("--data", default="data/games.json", help="Path to games JSON file")
    parser.add_argument("--persist-path", default=".chroma", help="Chroma persistence directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of local retrieval results")
    parser.add_argument("--rebuild", action="store_true", help="Re-ingest JSON before answering")
    return parser


def main() -> None:
    args = build_parser().parse_args()

    store = VectorStoreManager(persist_path=args.persist_path)
    if args.rebuild:
        ingested = store.ingest_json(args.data)
        print(f"Ingested {ingested} game records from {args.data}")

    agent = UdaPlayAgent(vector_store=store)
    answer = agent.answer(question=args.question, top_k=args.top_k)
    print(ReportFormatter.to_markdown(answer))


if __name__ == "__main__":
    main()
