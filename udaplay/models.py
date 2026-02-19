from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class GameRecord(BaseModel):
    id: str
    title: str
    description: str
    genre: list[str] = Field(default_factory=list)
    publisher: str | None = None
    release_date: str | None = None
    platforms: list[str] = Field(default_factory=list)


class RetrievalItem(BaseModel):
    id: str
    title: str
    score: float
    metadata: dict[str, Any]
    description: str


class RetrievalResult(BaseModel):
    query: str
    matches: list[RetrievalItem] = Field(default_factory=list)


class EvaluationResult(BaseModel):
    sufficiency: Literal["high", "medium", "low"]
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class WebSearchItem(BaseModel):
    title: str
    url: str
    content: str
    score: float | None = None


class AgentAnswer(BaseModel):
    question: str
    answer: str
    confidence: float = Field(ge=0.0, le=1.0)
    retrieval_evaluation: EvaluationResult
    citations: list[str] = Field(default_factory=list)
    used_web_fallback: bool = False
    execution_trace: list[str] = Field(default_factory=list)
    tool_usage: list[str] = Field(default_factory=list)
