from __future__ import annotations

import os
from enum import Enum

from dotenv import load_dotenv
from openai import OpenAI
from tavily import TavilyClient

from .models import AgentAnswer, EvaluationResult, RetrievalResult, WebSearchItem
from .vector_store import VectorStoreManager


class AgentState(str, Enum):
    RETRIEVE = "retrieve"
    EVALUATE = "evaluate"
    WEB_SEARCH = "web_search"
    RESPOND = "respond"


class UdaPlayAgent:
    """AI research agent for video game questions with RAG + web fallback."""

    def __init__(
        self,
        vector_store: VectorStoreManager,
        openai_api_key: str | None = None,
        tavily_api_key: str | None = None,
        model: str = "gpt-4.1-mini",
    ) -> None:
        load_dotenv()
        self.vector_store = vector_store
        self.model = model
        self.openai_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        self.tavily_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        self.openai_client = OpenAI(api_key=self.openai_key) if self.openai_key else None
        self.tavily_client = TavilyClient(api_key=self.tavily_key) if self.tavily_key else None

    def retrieve_game(self, question: str, top_k: int = 5) -> RetrievalResult:
        return self.vector_store.semantic_search(query=question, top_k=top_k)

    def evaluate_retrieval(self, retrieval: RetrievalResult) -> EvaluationResult:
        if not retrieval.matches:
            return EvaluationResult(
                sufficiency="low",
                confidence=0.0,
                rationale="No relevant internal matches were found.",
            )

        top = retrieval.matches[0].score
        avg = sum(item.score for item in retrieval.matches) / len(retrieval.matches)

        if top >= 0.82 and avg >= 0.70:
            return EvaluationResult(
                sufficiency="high",
                confidence=min(1.0, (top + avg) / 2),
                rationale="Top and average relevance indicate internal data is sufficient.",
            )
        if top >= 0.60:
            return EvaluationResult(
                sufficiency="medium",
                confidence=min(0.85, (top + avg) / 2),
                rationale="Internal matches are somewhat relevant but may miss details.",
            )
        return EvaluationResult(
            sufficiency="low",
            confidence=max(0.1, (top + avg) / 2),
            rationale="Retrieved results are too weak; web search likely needed.",
        )

    def game_web_search(self, question: str, max_results: int = 5) -> list[WebSearchItem]:
        if not self.tavily_client:
            return []

        response = self.tavily_client.search(
            query=f"video game research: {question}",
            search_depth="advanced",
            max_results=max_results,
        )
        items: list[WebSearchItem] = []
        for row in response.get("results", []):
            items.append(
                WebSearchItem(
                    title=row.get("title", "Untitled"),
                    url=row.get("url", ""),
                    content=row.get("content", ""),
                    score=row.get("score"),
                )
            )
        return items

    def answer(self, question: str, top_k: int = 5) -> AgentAnswer:
        state = AgentState.RETRIEVE
        retrieval = RetrievalResult(query=question, matches=[])
        evaluation = EvaluationResult(sufficiency="low", confidence=0.0, rationale="Not evaluated yet.")
        web_results: list[WebSearchItem] = []

        while True:
            if state == AgentState.RETRIEVE:
                retrieval = self.retrieve_game(question=question, top_k=top_k)
                state = AgentState.EVALUATE
                continue

            if state == AgentState.EVALUATE:
                evaluation = self.evaluate_retrieval(retrieval)
                state = AgentState.RESPOND if evaluation.sufficiency == "high" else AgentState.WEB_SEARCH
                continue

            if state == AgentState.WEB_SEARCH:
                web_results = self.game_web_search(question=question)
                state = AgentState.RESPOND
                continue

            if state == AgentState.RESPOND:
                return self._build_response(question, retrieval, evaluation, web_results)

    def _build_response(
        self,
        question: str,
        retrieval: RetrievalResult,
        evaluation: EvaluationResult,
        web_results: list[WebSearchItem],
    ) -> AgentAnswer:
        citations: list[str] = []
        context_blocks: list[str] = []

        for item in retrieval.matches:
            citations.append(f"local:{item.id}")
            context_blocks.append(item.description)

        for item in web_results:
            citations.append(item.url)
            context_blocks.append(f"Web Source: {item.title}\n{item.content}")

        context = "\n\n".join(context_blocks[:8]) or "No supporting context found."

        if self.openai_client:
            prompt = (
                "You are UdaPlay, a video game research assistant.\n"
                "Answer using the provided context, be explicit about uncertainty, and cite sources inline.\n"
                f"Question: {question}\n\n"
                f"Context:\n{context}"
            )
            completion = self.openai_client.responses.create(
                model=self.model,
                input=prompt,
                temperature=0.2,
            )
            answer_text = completion.output_text
        else:
            answer_text = (
                "UdaPlay could not access an LLM API key. Here is the retrieved context summary:\n\n"
                f"{context[:2000]}"
            )

        confidence = evaluation.confidence
        if web_results and evaluation.sufficiency != "high":
            confidence = min(0.95, max(confidence, 0.65))

        return AgentAnswer(
            question=question,
            answer=answer_text,
            confidence=confidence,
            retrieval_evaluation=evaluation,
            citations=sorted(set(citations)),
            used_web_fallback=bool(web_results),
        )
