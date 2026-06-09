# Multi-Agent RAG Pipeline

**A production-grade, multi-agent Retrieval-Augmented Generation system for intelligent customer support.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![LangChain](https://img.shields.io/badge/LangChain-0.3+-green.svg)](https://www.langchain.com)
[![OpenRouter](https://img.shields.io/badge/LLM-OpenRouter-blue.svg)](https://openrouter.ai)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## Overview

Multi-Agent RAG Pipeline is an AI customer support system where **specialized agents collaborate** to solve domain-specific queries using **Retrieval-Augmented Generation (RAG)**.

- **Router Agent** classifies incoming queries by department (Billing, Technical, General)
- **Department Agents** each have their own knowledge base and RAG workflow
- **Conversation memory** maintains context across turns within a session
- **Structured logging** and **error handling** throughout

Built with **OpenRouter** for flexible LLM access (supports Claude, GPT, LLaMA, and 100+ models), **LangChain 0.3+** for orchestration, **ChromaDB** for vector search, and **Streamlit** for the UI.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     User Query                               │
└──────────────────────────┬──────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────┐
│  LangGraph Orchestrator                                      │
│  ┌────────────┐    ┌──────────────────────────────────────┐  │
│  │  Router     │───▶│  Conditional Edge                    │  │
│  │  (OpenRouter)│   │  billing / technical / general       │  │
│  └────────────┘    └──────────┬───────────────────────────┘  │
│                               │                              │
│         ┌─────────────────────┼─────────────────────┐        │
│         ▼                     ▼                     ▼        │
│  ┌────────────┐        ┌────────────┐        ┌────────────┐  │
│  │  Billing   │        │  Technical │        │  General   │  │
│  │  Agent     │        │  Agent     │        │  Agent     │  │
│  │  RAG over  │        │  RAG over  │        │  RAG over  │  │
│  │  billing   │        │  tech FAQ  │        │  general   │  │
│  │  FAQ       │        │            │        │  FAQ       │  │
│  └─────┬──────┘        └─────┬──────┘        └─────┬──────┘  │
│        └─────────────────────┼─────────────────────┘        │
│                              ▼                               │
│  ┌────────────────────────────────────────────────────────┐  │
│  │  Checkpointer (SQLite / InMemory)                      │  │
│  │  Persistent conversation memory across sessions        │  │
│  └────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
     ┌──────────────┐          ┌──────────────┐
     │  Streamlit   │          │  FastAPI     │
     │  UI          │          │  REST API    │
     │  :8501       │          │  :8000       │
     └──────────────┘          └──────────────┘
```

---

## Features

- **LLM-powered routing** — Groq classifies queries to the right department
- **Department-specific RAG** — each agent has its own knowledge base
- **Conversation memory** — context persists across turns in a session
- **Error handling** — graceful fallbacks if LLM or vector store fails
- **Structured logging** — timestamped, level-based logging throughout
- **Centralized config** — Pydantic settings with `.env` support
- **Current LangChain API** — uses `langchain-huggingface`, `langchain-chroma`, `langchain-core`
- **Tested** — unit tests for router and agent error handling

---

## Tech Stack

| Component        | Technology                                          |
| ---------------- | --------------------------------------------------- |
| **LLM Provider** | [OpenRouter](https://openrouter.ai) (100+ models: Claude, GPT, LLaMA, etc.) |
| **Framework**    | [LangChain 0.3+](https://www.langchain.com)          |
| **Vector Store** | [ChromaDB](https://www.trychroma.com)                |
| **Embeddings**   | `sentence-transformers/all-MiniLM-L6-v2`             |
| **Frontend**     | [Streamlit](https://streamlit.io)                    |
| **Config**       | [Pydantic Settings](https://docs.pydantic.dev/)      |
| **Testing**      | [pytest](https://docs.pytest.org/)                   |
| **Environment**  | Python 3.10+                                         |

---

## Project Structure

```
multi-agent-rag-pipeline/
├── app.py                    # Streamlit entry point
├── api.py                    # FastAPI REST API
├── router.py                 # Query classification agent
├── config.py                 # Pydantic settings (loads from .env)
├── logger.py                 # Structured logging setup
├── orchestrator.py           # LangGraph state graph
├── tools.py                  # Tool wrappers for cross-department escalation
├── requirements.txt          # Python dependencies
├── .env.example              # Template for environment variables
├── .gitignore
├── .dockerignore
├── LICENSE
├── Dockerfile                # Streamlit container
├── Dockerfile.api            # API container
├── docker-compose.yml        # Multi-service orchestration
├── README.md
├── .github/
│   └── workflows/
│       └── ci.yml            # GitHub Actions CI/CD
├── agents/
│   ├── billing_agent.py      # Billing RAG agent
│   ├── tech_agent.py         # Technical support RAG agent
│   └── general_agent.py      # General inquiries RAG agent
├── utils/
│   ├── __init__.py
│   └── vector_store.py       # Shared Chroma vector store builder
├── data/
│   ├── billing_faq.txt       # Billing knowledge base
│   ├── tech_faq.txt          # Technical support knowledge base
│   └── general_faq.txt       # General inquiries knowledge base
└── tests/
    ├── __init__.py
    ├── test_router.py        # Router classification tests
    └── test_agents.py        # Agent error-handling tests
```

---

## Installation & Setup

### 1. Clone the repo

```bash
git clone https://github.com/RaghuramReddy9/multi-agent-rag-pipeline.git
cd multi-agent-rag-pipeline
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate     # or .venv\Scripts\activate on Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment

```bash
cp .env.example .env
# Edit .env and add your OpenRouter API key
```

### 5. Run the app

```bash
streamlit run app.py
```

### 6. Run tests

```bash
pytest tests/ -v
```

---

## Configuration

All settings are managed in `config.py` via Pydantic Settings. Override any value in `.env`:

| Variable              | Default                          | Description                    |
| --------------------- | -------------------------------- | ------------------------------ |
| `OPENROUTER_API_KEY`  | *(required)*                     | OpenRouter API key                 |
| `OPENROUTER_BASE_URL` | `https://openrouter.ai/api/v1`   | OpenRouter API base URL            |
| `LLM_MODEL`           | `openrouter/auto`                | OpenRouter model (e.g., `openrouter/anthropic/claude-sonnet-4`) |
| `LLM_TEMPERATURE`     | `0.2`                            | LLM temperature                |
| `LLM_MAX_TOKENS`      | `512`                            | Max tokens per response        |
| `EMBEDDING_MODEL`     | `sentence-transformers/all-MiniLM-L6-v2` | HuggingFace embedding model |
| `CHUNK_SIZE`          | `800`                            | Document chunk size            |
| `CHUNK_OVERLAP`       | `200`                            | Chunk overlap                  |
| `RETRIEVAL_K`         | `3`                              | Number of docs to retrieve     |
| `LOG_LEVEL`           | `INFO`                           | Logging level                  |

---

## Example Queries

| Query                                | Routed Agent |
| ------------------------------------ | ------------ |
| "I want a refund for my payment."    | Billing      |
| "My login page keeps crashing."      | Technical    |
| "What's the HR contact email?"       | General      |
| "How do I cancel my subscription?"   | Billing      |
| "The app is running slowly."         | Technical    |

---

## What Changed — v2.0 (Full Production Upgrade)

### Phase 1 — Foundation
1. **LangChain imports fixed** — uses current `langchain-huggingface`, `langchain-chroma`, `langchain-core`
2. **Centralized config** — Pydantic Settings replaces hardcoded values
3. **Error handling** — every agent has try/catch with user-friendly fallbacks
4. **Conversation memory** — session-level chat history (last 6 turns)
5. **Structured logging** — timestamped, level-based logging
6. **Shared vector store** — single `utils/vector_store.py` with caching
7. **Tests** — pytest unit tests for router and agents
8. **`.env.example`** — documents all env vars
9. **LICENSE** — MIT
10. **Updated `.gitignore`** — covers `.env`, caches, IDE files

### Phase 2 — Orchestration & API
11. **LLM provider: Groq → OpenRouter** — supports 100+ models with one key
12. **LangGraph orchestration** — proper state graph with typed state, conditional routing, checkpointer for persistent memory
13. **FastAPI REST API** — `/health`, `/chat`, `/history/{thread}`, streaming support
14. **Docker** — `Dockerfile` (Streamlit), `Dockerfile.api`, `docker-compose.yml`
15. **CI/CD** — GitHub Actions workflow: lint, type-check, test, Docker build
16. **Tool wrappers** — `tools.py` for cross-department escalation (Deep Agents ready)
---

## Future Roadmap

- [x] **LangGraph orchestration** — graph-based agent workflow with conditional routing
- [x] **Persistent memory** — SQLite checkpointer for cross-session conversation history
- [x] **CI/CD** — GitHub Actions for testing and linting
- [x] **Docker** — containerized deployment
- [x] **API mode** — FastAPI endpoints alongside Streamlit
- [ ] **Deep Agents** — multi-step planning, tool use, filesystem access
- [ ] **Evaluation framework** — automated testing of agent response quality
- [ ] **Authentication** — API key auth for FastAPI endpoints
- [ ] **Rate limiting** — protect API from abuse

---

## Author

**Raghuramreddy Thirumalareddy**
AI Engineer | Generative AI & RAG Systems

- GitHub: https://github.com/RaghuramReddy9
- LinkedIn: https://linkedin.com/in/raghuram-genai

---

## License

MIT License © 2025 Raghuramreddy Thirumalareddy
