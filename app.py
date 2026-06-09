"""Streamlit app — Customer Support Multi-Agent Assistant.

Uses LangGraph for orchestration:
  Router classifies → conditional edge → department agent → response

Supports conversation memory via LangGraph checkpointer.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from langchain_core.messages import AIMessage, HumanMessage
from logger import get_logger
from orchestrator import get_graph

logger = get_logger(__name__)

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=settings.app_title,
    page_icon=settings.app_icon,
    layout="wide",
)

st.title(f"{settings.app_icon} {settings.app_title}")
st.markdown(
    "Ask any question related to **billing**, **technical support**, or **general inquiries**."
)

# ── Session state ─────────────────────────────────────────────────────────────
if "thread_id" not in st.session_state:
    import uuid
    st.session_state.thread_id = str(uuid.uuid4())

# Build the graph with per-session thread ID for memory
graph = get_graph()  # in-memory checkpointer; swap to SQLite path for persistence
thread_id = st.session_state.thread_id

# ── Display previous messages (from LangGraph state) ─────────────────────────
# We reconstruct from the graph state on each rerun
try:
    state = graph.get_state(config={"configurable": {"thread_id": thread_id}})
    existing_messages = state.values.get("messages", []) if state.values else []
except Exception:
    existing_messages = []

for msg in existing_messages:
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# ── User input ────────────────────────────────────────────────────────────────
user_query = st.chat_input("Type your message here...")

if user_query:
    # Show user message immediately
    with st.chat_message("user"):
        st.markdown(user_query)

    # Run the graph
    with st.spinner("Processing..."):
        try:
            result = graph.invoke(
                {"messages": [HumanMessage(content=user_query)]},
                config={"configurable": {"thread_id": thread_id}},
            )
            # Get the last AI message
            ai_messages = [
                m for m in result.get("messages", []) if isinstance(m, AIMessage)
            ]
            if ai_messages:
                answer = ai_messages[-1].content
            else:
                answer = "No response generated."

            with st.chat_message("assistant"):
                st.markdown(answer)

            dept = result.get("department", "unknown")
            logger.info("Turn complete | thread=%s | dept=%s", thread_id[:8], dept)

        except Exception as exc:
            logger.error("App error: %s", exc, exc_info=True)
            st.error(f"An error occurred: {exc}")
