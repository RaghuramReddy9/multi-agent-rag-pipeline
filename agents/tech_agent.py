"""Technical support agent — handles bugs, errors, login, setup, and performance queries.

Uses Groq LLM + Chroma RAG over tech_faq.txt.
"""

from __future__ import annotations

from groq import Groq
from langchain_chroma import Chroma

from config import settings
from logger import get_logger
from utils.vector_store import build_or_load_vectorstore

logger = get_logger(__name__)


def _get_client() -> Groq:
    return Groq(api_key=settings.groq_api_key)


def answer_tech_query(query: str, chat_history: str | None = None) -> str:
    """Answer a technical support query using RAG context from tech docs.

    Args:
        query: The user's question.
        chat_history: Optional formatted conversation history for context.

    Returns:
        The agent's response string.
    """
    try:
        logger.info("Tech agent processing query: %.80s...", query)
        client = _get_client()
        db: Chroma = build_or_load_vectorstore(
            index_path=settings.tech_index_path,
            data_path=settings.tech_data_path,
        )
        retriever = db.as_retriever(search_kwargs={"k": settings.retrieval_k})
        results = retriever.invoke(query)
        context = "\n".join(doc.page_content for doc in results)

        system_prompt = (
            "You are a technical support assistant for a SaaS product. "
            "Use the context below to answer user questions precisely and clearly. "
            "If the information isn't found in the context, say you don't have that data."
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
        logger.info("Tech agent response: %.80s...", answer)
        return answer

    except Exception as exc:
        logger.error("Tech agent error: %s", exc, exc_info=True)
        return (
            "I'm sorry, I encountered an error processing your technical query. "
            "Please try again or contact support directly."
        )
