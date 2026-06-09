"""General support agent — handles HR, account, and general inquiries.

Uses Groq LLM + Chroma RAG over general_faq.txt.
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


def answer_general_query(query: str, chat_history: str | None = None) -> str:
    """Answer a general support query using RAG context from general docs.

    Args:
        query: The user's question.
        chat_history: Optional formatted conversation history for context.

    Returns:
        The agent's response string.
    """
    try:
        logger.info("General agent processing query: %.80s...", query)
        client = _get_client()
        db: Chroma = build_or_load_vectorstore(
            index_path=settings.general_index_path,
            data_path=settings.general_data_path,
        )
        retriever = db.as_retriever(search_kwargs={"k": settings.retrieval_k})
        results = retriever.invoke(query)
        context = "\n".join(doc.page_content for doc in results)

        system_prompt = (
            "You are a polite and concise customer-support assistant for "
            "general or HR-related queries. Use the provided context to answer "
            "accurately. If the information is missing, say you could not "
            "locate that policy."
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
        logger.info("General agent response: %.80s...", answer)
        return answer

    except Exception as exc:
        logger.error("General agent error: %s", exc, exc_info=True)
        return (
            "I'm sorry, I encountered an error processing your query. "
            "Please try again or contact support directly."
        )
