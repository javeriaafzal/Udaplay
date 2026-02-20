from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from .models import AgentAnswer


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class SessionLogger:
    log_dir: Path
    session_id: str = field(default_factory=lambda: f"session-{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}-{uuid4().hex[:8]}")
    started_at: str = field(default_factory=_utc_now_iso)

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.history_file = self.log_dir / "sessions.jsonl"
        self.session_file = self.log_dir / f"{self.session_id}.json"
        self._events: list[dict[str, object]] = []

    def log_turn(self, question: str, answer: AgentAnswer) -> None:
        event = {
            "timestamp": _utc_now_iso(),
            "question": question,
            "answer": answer.answer,
            "confidence": answer.confidence,
            "used_web_fallback": answer.used_web_fallback,
            "citations": answer.citations,
        }
        self._events.append(event)

    def finalize(self) -> Path:
        session_record = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "ended_at": _utc_now_iso(),
            "turn_count": len(self._events),
            "events": self._events,
        }

        self.session_file.write_text(json.dumps(session_record, indent=2), encoding="utf-8")

        with self.history_file.open("a", encoding="utf-8") as f:
            f.write(json.dumps({
                "session_id": self.session_id,
                "started_at": self.started_at,
                "ended_at": session_record["ended_at"],
                "turn_count": session_record["turn_count"],
                "file": self.session_file.name,
            }))
            f.write("\n")

        return self.session_file
