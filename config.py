"""Configuration management using Pydantic Settings.

Loads settings from environment variables and .env file.
All hardcoded values (model names, chunk sizes, retrieval k, etc.)
are centralized here instead of scattered across agent files.
"""

from __future__ import annotations

from pydantic import Field
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application settings loaded from environment / .env."""

    # --- LLM (OpenRouter) ---
    openrouter_api_key: str = Field(..., description="OpenRouter API key")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        description="OpenRouter API base URL",
    )
    llm_model: str = Field(
        default="openrouter/auto",
        description="OpenRouter model name (e.g., openrouter/auto, openrouter/anthropic/claude-sonnet-4)",
    )
    llm_temperature: float = Field(default=0.2, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=512, ge=1)

    # --- Embeddings ---
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="HuggingFace embedding model name",
    )

    # --- Vector Store ---
    chunk_size: int = Field(default=800, ge=100)
    chunk_overlap: int = Field(default=200, ge=0)
    retrieval_k: int = Field(default=3, ge=1, description="Number of docs to retrieve")

    # --- Paths ---
    billing_data_path: str = Field(default="data/billing_faq.txt")
    tech_data_path: str = Field(default="data/tech_faq.txt")
    general_data_path: str = Field(default="data/general_faq.txt")
    billing_index_path: str = Field(default="billing_index")
    tech_index_path: str = Field(default="tech_index")
    general_index_path: str = Field(default="general_index")

    # --- App ---
    app_title: str = Field(default="Customer Support Multi-Agent")
    app_icon: str = Field(default="🤖")
    log_level: str = Field(default="INFO")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


# Singleton — import this everywhere
settings = Settings()
