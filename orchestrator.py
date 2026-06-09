"""LangGraph orchestration for the multi-agent RAG pipeline.

Replaces the simple if-else router in app.py with a proper state graph:
  classify → route → [billing | tech | general] → respond

Features:
- Typed state with MessagesState (conversation history built-in)
- Conditional routing based on classifier output
- Parallel agent execution support
- Persistent memory via checkpointer (SQLite)
- Structured error handling at every node
"""

from __future__ import annotations

import os
from typing import Annotated, Literal

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.graph.message import add_messages

from config import settings
from logger import get_logger

logger = get_logger(__name__)

# Try to import SQLite checkpointer for persistent memory
try:
    from langgraph.checkpoint.sqlite import SqliteSaver  # type: ignore[import-untyped]
    HAS_SQLITE = True
except ImportError:
    HAS_SQLITE = False


# ── State Definition ──────────────────────────────────────────────────────────

class AgentState(MessagesState):
    """State passed between nodes in the graph.

    Inherits `messages` with add_messages reducer from MessagesState.
    Adds department routing and error tracking.
    """

    department: str  # "billing" | "technical" | "general"
    error: str | None


# ── Router Node ───────────────────────────────────────────────────────────────

def router_node(state: AgentState) -> dict[str, str | None]:
    """Classify the latest user message and set the department."""
    from router import classify_query

    last_msg = state["messages"][-1]
    query = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)

    try:
        dept = classify_query(query)
        logger.info("Router classified as: %s", dept)
        return {"department": dept, "error": None}
    except Exception as exc:
        logger.error("Router error: %s", exc, exc_info=True)
        return {"department": "general", "error": str(exc)}


# ── Department Agent Nodes ────────────────────────────────────────────────────

def _run_department_agent(department: str, state: AgentState) -> dict:
    """Run the appropriate department agent and return the response."""
    last_msg = state["messages"][-1]
    query = last_msg.content if isinstance(last_msg, HumanMessage) else str(last_msg)

    # Build conversation history string from messages (last 6 turns)
    history_msgs = state["messages"][-12:]
    history_lines = []
    for m in history_msgs:
        role = "User" if isinstance(m, HumanMessage) else "Assistant"
        history_lines.append(f"{role}: {m.content}")
    chat_history = "\n".join(history_lines) if history_lines else None

    try:
        if department == "billing":
            from agents.billing_agent import answer_billing_query
            answer = answer_billing_query(query, chat_history=chat_history)
        elif department == "technical":
            from agents.tech_agent import answer_tech_query
            answer = answer_tech_query(query, chat_history=chat_history)
        else:
            from agents.general_agent import answer_general_query
            answer = answer_general_query(query, chat_history=chat_history)

        logger.info("%s agent responded: %.80s...", department, answer)
        return {
            "messages": [AIMessage(content=answer)],
            "error": None,
        }
    except Exception as exc:
        logger.error("%s agent error: %s", department, exc, exc_info=True)
        fallback = (
            f"I'm sorry, the {department} agent encountered an error. "
            "Please try again."
        )
        return {
            "messages": [AIMessage(content=fallback)],
            "error": str(exc),
        }


def billing_node(state: AgentState) -> dict:
    return _run_department_agent("billing", state)


def tech_node(state: AgentState) -> dict:
    return _run_department_agent("technical", state)


def general_node(state: AgentState) -> dict:
    return _run_department_agent("general", state)


# ── Routing Function ──────────────────────────────────────────────────────────

def route_by_department(state: AgentState) -> str:
    """Return the next node name based on the classified department."""
    dept = state.get("department", "general")
    if dept not in ("billing", "technical", "general"):
        dept = "general"
    return dept


# ── Graph Builder ─────────────────────────────────────────────────────────────

def build_graph(checkpoint_path: str | None = None):
    """Build and compile the LangGraph state graph.

    Args:
        checkpoint_path: Optional path to a SQLite database for persistent
                         conversation memory across sessions. If None, uses
                         in-memory storage (per-session only).

    Returns:
        A compiled LangGraph application.
    """
    builder = StateGraph(AgentState)

    # Nodes
    builder.add_node("router", router_node)
    builder.add_node("billing", billing_node)
    builder.add_node("tech", tech_node)
    builder.add_node("general", general_node)

    # Edges
    builder.add_edge(START, "router")
    builder.add_conditional_edges(
        "router",
        route_by_department,
        {
            "billing": "billing",
            "technical": "tech",
            "general": "general",
        },
    )
    builder.add_edge("billing", END)
    builder.add_edge("tech", END)
    builder.add_edge("general", END)

    # Checkpointer for persistent memory
    if checkpoint_path and HAS_SQLITE:
        os.makedirs(os.path.dirname(checkpoint_path) or ".", exist_ok=True)
        checkpointer = SqliteSaver.from_conn_string(checkpoint_path)
        logger.info("Using SQLite checkpointer at %s", checkpoint_path)
    else:
        checkpointer = InMemorySaver()
        logger.info("Using in-memory checkpointer (per-session)")

    graph = builder.compile(checkpointer=checkpointer)
    return graph


# Module-level cache
_graph_cache: dict[str | None, object] = {}


def get_graph(checkpoint_path: str | None = None):
    """Return a compiled graph (cached per checkpoint_path)."""
    if checkpoint_path not in _graph_cache:
        _graph_cache[checkpoint_path] = build_graph(checkpoint_path=checkpoint_path)
    return _graph_cache[checkpoint_path]
