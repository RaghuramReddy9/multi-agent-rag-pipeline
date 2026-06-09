"""Immutable or shallow-mutable data for LangGraph tools.

These represent domain-specific responses used when future
LangChain Deep Agents or tool-calling agents are integrated.
Kept for forward-compatibility even though the current
pipeline uses department agents directly.
"""

from __future__ import annotations


class ToolResponse:
    """A generic container for tool/agent responses."""

    def __init__(self, content: str, department: str, confidence: float = 1.0):
        self.content = content
        self.department = department
        self.confidence = confidence

    def __repr__(self) -> str:
        return f"ToolResponse(dept={self.department}, conf={self.confidence:.2f})"

    def to_dict(self) -> dict:
        return {
            "content": self.content,
            "department": self.department,
            "confidence": self.confidence,
        }


class DepartmentTools:
    """Callable wrappers so agents can invoke other departments if needed.

    In a full Deep Agents setup these become LangChain tools.
    Currently used for cross-department escalation.
    """

    @staticmethod
    def escalate_to_billing(query: str, context: str | None = None) -> ToolResponse:
        from agents.billing_agent import answer_billing_query
        answer = answer_billing_query(query, chat_history=context)
        return ToolResponse(answer, "billing")

    @staticmethod
    def escalate_to_technical(query: str, context: str | None = None) -> ToolResponse:
        from agents.tech_agent import answer_tech_query
        answer = answer_tech_query(query, chat_history=context)
        return ToolResponse(answer, "technical")

    @staticmethod
    def escalate_to_general(query: str, context: str | None = None) -> ToolResponse:
        from agents.general_agent import answer_general_query
        answer = answer_general_query(query, chat_history=context)
        return ToolResponse(answer, "general")
