from __future__ import annotations

import argparse
from pathlib import Path

from .agent import UdaPlayAgent
from .reporting import ReportFormatter
from .session_logs import SessionLogger
from .vector_store import VectorStoreManager


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="UdaPlay video game research agent")
    parser.add_argument("question", nargs="?", help="Question about a video game")
    parser.add_argument("--data", default="data/games.json", help="Path to games JSON file")
    parser.add_argument("--persist-path", default=".chroma", help="Chroma persistence directory")
    parser.add_argument("--top-k", type=int, default=5, help="Number of local retrieval results")
    parser.add_argument("--rebuild", action="store_true", help="Re-ingest JSON before answering")
    parser.add_argument(
        "--log-dir",
        default="logs/sessions",
        help="Directory where session history logs are written",
    )
    parser.add_argument(
        "--session",
        action="store_true",
        help="Run in interactive mode to ask multiple follow-up questions in one session",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()

    store = VectorStoreManager(persist_path=args.persist_path)
    if args.rebuild:
        ingested = store.ingest_json(args.data)
        print(f"Ingested {ingested} game records from {args.data}")

    agent = UdaPlayAgent(vector_store=store)
    logger = SessionLogger(log_dir=Path(args.log_dir))

    if args.session:
        print("UdaPlay interactive session started. Type 'exit' or 'quit' to end.")
        try:
            while True:
                question = input("\nYou: ").strip()
                if question.lower() in {"exit", "quit"}:
                    print("Ending session.")
                    break
                if not question:
                    continue
                answer = agent.answer(question=question, top_k=args.top_k)
                logger.log_turn(question=question, answer=answer)
                print(ReportFormatter.to_markdown(answer))
        finally:
            session_file = logger.finalize()
            print(f"Session log saved: {session_file}")
        return

    if not args.question:
        raise SystemExit("A question is required unless --session is used.")

    answer = agent.answer(question=args.question, top_k=args.top_k)
    logger.log_turn(question=args.question, answer=answer)
    session_file = logger.finalize()
    print(ReportFormatter.to_markdown(answer))
    print(f"Session log saved: {session_file}")


if __name__ == "__main__":
    main()
