"""FastAPI endpoints for the multi-agent RAG pipeline.

Provides REST API access alongside the Streamlit UI.

Endpoints:
    GET  /health          — health check
    POST /chat            — send a message, get agent response
    POST /chat/stream     — streaming SSE response
    GET  /history/{thread} — get conversation history
    DELETE /history/{thread} — clear conversation history
"""

from __future__ import annotations

import uuid
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from config import settings
from langchain_core.messages import AIMessage, HumanMessage
from logger import get_logger
from orchestrator import get_graph

logger = get_logger(__name__)

app = FastAPI(
    title=settings.app_title,
    description="Multi-Agent RAG Pipeline API",
    version="2.0.0",
)

# Use SQLite checkpointer for persistent memory in API mode
graph = get_graph(checkpoint_path="./data/checkpoint.db")


# ── Request/Response Models ───────────────────────────────────────────────────

class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, description="User message")
    thread_id: Optional[str] = Field(
        default=None, description="Conversation thread ID (auto-generated if not provided)"
    )


class ChatResponse(BaseModel):
    thread_id: str
    department: str
    response: str
    error: str | None = None


class HealthResponse(BaseModel):
    status: str
    version: str
    model: str


class HistoryResponse(BaseModel):
    thread_id: str
    messages: list[dict[str, str]]


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(
        status="ok",
        version="2.0.0",
        model=settings.llm_model,
    )


@app.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest):
    """Send a message and get a complete agent response."""
    thread_id = req.thread_id or str(uuid.uuid4())

    try:
        result = graph.invoke(
            {"messages": [HumanMessage(content=req.message)]},
            config={"configurable": {"thread_id": thread_id}},
        )
        ai_messages = [
            m for m in result.get("messages", []) if isinstance(m, AIMessage)
        ]
        response = ai_messages[-1].content if ai_messages else "No response."
        dept = result.get("department", "unknown")
        error = result.get("error")

        return ChatResponse(
            thread_id=thread_id,
            department=dept,
            response=response,
            error=error,
        )
    except Exception as exc:
        logger.error("Chat error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/history/{thread_id}", response_model=HistoryResponse)
async def get_history(thread_id: str):
    """Get conversation history for a thread."""
    try:
        state = graph.get_state(config={"configurable": {"thread_id": thread_id}})
        if not state.values:
            raise HTTPException(status_code=404, detail="Thread not found")
        messages = state.values.get("messages", [])
        return HistoryResponse(
            thread_id=thread_id,
            messages=[
                {
                    "role": "user" if isinstance(m, HumanMessage) else "assistant",
                    "content": m.content,
                }
                for m in messages
            ],
        )
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("History error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))


@app.delete("/history/{thread_id}")
async def clear_history(thread_id: str):
    """Clear conversation history for a thread."""
    try:
        # LangGraph doesn't have a direct delete, so we update with empty
        graph.update_state(
            config={"configurable": {"thread_id": thread_id}},
            values={"messages": []},
        )
        return {"status": "cleared", "thread_id": thread_id}
    except Exception as exc:
        logger.error("Clear error: %s", exc, exc_info=True)
        raise HTTPException(status_code=500, detail=str(exc))
