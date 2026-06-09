"""Billing agent — handles billing, payment, refund, and subscription queries.

Uses OpenRouter LLM + Chroma RAG over billing_faq.txt.
"""

from __future__ import annotations

from openai import OpenAI
from langchain_chroma import Chroma

from config import settings
from logger import get_logger
from utils.vector_store import build_or_load_vectorstore

logger = get_logger(__name__)


def _get_client() -> OpenAI:
    return OpenAI(
        api_key=settings.openrouter_api_key,
        base_url=settings.openrouter_base_url,
    )


def answer_billing_query(query: str, chat_history: str | None = None) -> str:
    """Answer a billing-related query using RAG context from billing docs.

    Args:
        query: The user's question.
        chat_history: Optional formatted conversation history for context.

    Returns:
        The agent's response string.
    """
    try:
        logger.info("Billing agent processing query: %.80s...", query)
        client = _get_client()
        db: Chroma = build_or_load_vectorstore(
            index_path=settings.billing_index_path,
            data_path=settings.billing_data_path,
        )
        retriever = db.as_retriever(search_kwargs={"k": settings.retrieval_k})
        results = retriever.invoke(query)
        context = "\n".join(doc.page_content for doc in results)

        system_prompt = (
            "You are a helpful customer support assistant specializing in "
            "billing and payment questions. Use the following company policy "
            "information to answer accurately and clearly. If the information "
            "is not in the context, say you don't have that data."
        )

        user_content = f"Context:\n{context}\n\nQuestion: {query}"
        if chat_history:
            user_content = f"Previous conversation:\n{chat_history}\n\n{user_content}"

        completion = client.chat.completions.create(
            model=settings.llm_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ],
            temperature=settings.llm_temperature,
            max_tokens=settings.llm_max_tokens,
        )
        answer = completion.choices[0].message.content.strip()
        logger.info("Billing agent response: %.80s...", answer)
        return answer

    except Exception as exc:
        logger.error("Billing agent error: %s", exc, exc_info=True)
        return (
            "I'm sorry, I encountered an error processing your billing query. "
            "Please try again or contact support directly."
        )
