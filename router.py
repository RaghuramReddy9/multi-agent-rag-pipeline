"""Router agent — classifies user queries into department categories.

Uses Groq LLM to determine whether a query should be handled by
the billing, technical, or general support agent.
"""

from __future__ import annotations

from groq import Groq

from config import settings
from logger import get_logger

logger = get_logger(__name__)

# Valid routing targets
DEPARTMENTS = ("billing", "technical", "general")

ROUTER_SYSTEM_PROMPT = (
    "You are a support-request classifier for a company. "
    "Read the user's message and choose exactly one label:\n"
    "1. billing – if the message mentions payment, invoice, refund, subscription, or cost.\n"
    "2. technical – if it mentions bugs, errors, login, setup, or performance issues.\n"
    "3. general – for all other human-resources, account, or general questions.\n"
    "Return only the label word."
)


def _get_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def classify_query(query: str) -> str:
    """Classify a user query into one of: 'billing', 'technical', 'general'.

    Falls back to 'general' if the LLM response is unexpected or an error occurs.

    Args:
        query: The user's support question.

    Returns:
        One of 'billing', 'technical', or 'general'.
    """
    try:
        logger.info("Router classifying query: %.80s...", query)
        client = _get_client()

        completion = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": ROUTER_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
            max_tokens=10,
        )

        raw = completion.choices[0].message.content.strip().lower()
        logger.info("Router raw output: %s", raw)

        # Match against valid departments
        for dept in DEPARTMENTS:
            if dept in raw:
                logger.info("Routed to: %s", dept)
                return dept

        # Fallback
        logger.warning("Router could not classify, falling back to 'general'. Raw: %s", raw)
        return "general"

    except Exception as exc:
        logger.error("Router error: %s — falling back to 'general'", exc, exc_info=True)
        return "general"
