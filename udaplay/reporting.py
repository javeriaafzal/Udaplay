from __future__ import annotations

from .models import AgentAnswer


class ReportFormatter:
    @staticmethod
    def to_markdown(answer: AgentAnswer) -> str:
        citations = "\n".join(f"- {c}" for c in answer.citations) if answer.citations else "- None"
        trace = "\n".join(f"- {line}" for line in answer.execution_trace) if answer.execution_trace else "- None"
        tools = "\n".join(f"- {tool}" for tool in answer.tool_usage) if answer.tool_usage else "- None"
        return (
            f"## UdaPlay Response\n\n"
            f"**Question:** {answer.question}\n\n"
            f"**Confidence:** {answer.confidence:.2f}\n"
            f"**Retrieval Sufficiency:** {answer.retrieval_evaluation.sufficiency}\n"
            f"**Evaluation Rationale:** {answer.retrieval_evaluation.rationale}\n"
            f"**Used Web Fallback:** {'Yes' if answer.used_web_fallback else 'No'}\n\n"
            f"### Reasoning Trace\n{trace}\n\n"
            f"### Tool Usage\n{tools}\n\n"
            f"### Answer\n{answer.answer}\n\n"
            f"### Sources\n{citations}\n"
        )
