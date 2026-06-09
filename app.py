"""Streamlit app — Customer Support Multi-Agent Assistant.

Routes user queries to specialized agents (Billing, Technical, General)
using an LLM-powered router. Supports conversation memory within session.
"""

from __future__ import annotations

import streamlit as st

from config import settings
from router import classify_query
from agents.billing_agent import answer_billing_query
from agents.tech_agent import answer_tech_query
from agents.general_agent import answer_general_query
from logger import get_logger

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
if "messages" not in st.session_state:
    st.session_state.messages = []

if "chat_history_str" not in st.session_state:
    st.session_state.chat_history_str = ""

# ── Display previous messages ────────────────────────────────────────────────
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── User input ────────────────────────────────────────────────────────────────
user_query = st.chat_input("Type your message here...")

if user_query:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Route
    with st.spinner("Routing your question..."):
        department = classify_query(user_query)

    # Answer with conversation memory
    with st.spinner(f"{department.capitalize()} agent is preparing a response..."):
        history = st.session_state.chat_history_str
        if department == "billing":
            answer = answer_billing_query(user_query, chat_history=history)
        elif department == "technical":
            answer = answer_tech_query(user_query, chat_history=history)
        else:
            answer = answer_general_query(user_query, chat_history=history)

    # Display answer
    st.session_state.messages.append({"role": "assistant", "content": answer})
    with st.chat_message("assistant"):
        st.markdown(answer)

    # Update conversation memory (last 6 turns = 12 messages)
    recent = st.session_state.messages[-12:]
    history_lines = [
        f"{m['role'].capitalize()}: {m['content']}" for m in recent
    ]
    st.session_state.chat_history_str = "\n".join(history_lines)

    logger.info(
        "Turn complete | dept=%s | query=%.50s | answer=%.50s",
        department, user_query, answer,
    )
