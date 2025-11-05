# Multi-Agent RAG Pipeline

### A Groq-powered multi-agent Retrieval-Augmented Generation system for intelligent customer support.

---

## Overview

**Multi-Agent RAG Pipeline** is an advanced AI architecture where multiple specialized agents collaborate to solve domain-specific queries using **Retrieval-Augmented Generation (RAG)**.

The system uses **Groq's low-latency LPU models** for fast inference and **LangChain + Chroma** for document retrieval.  
Each agent — **Billing**, **Technical**, and **General Support** — has its own RAG workflow and knowledge base, coordinated by a **Router Agent**.

---

## Architecture

```mermaid
graph TD
    A[User Query] --> B[Router Agent (Groq - LLaMA3 8B)]
    B -->|Billing| C[Billing Agent - RAG over billing_faq.txt]
    B -->|Technical| D[Tech Agent - RAG over tech_faq.txt]
    B -->|General| E[General Agent - RAG over general_faq.txt]
    C --> F[Groq Response]
    D --> F
    E --> F
    F --> G[Streamlit UI Response]
```

---

## Features
- Groq-Powered Agents – blazing fast reasoning using `llama-3.1-8b-instant`

- Billing / Tech / General Departments – each has its own context-aware knowledge base

- RAG with ChromaDB – document retrieval + embedding search

- Router Agent – auto-classifies user queries to the right department

- Modular Design – ready for LangGraph or CrewAI orchestration

- Streamlit Interface – interactive multi-agent chat with agent labels

---

## Tech Stack

| Component        | Technology                               |
| ---------------- | ---------------------------------------- |
| **LLM Provider** | [Groq API](https://console.groq.com)     |
| **Framework**    | [LangChain](https://www.langchain.com)   |
| **Vector Store** | [ChromaDB](https://www.trychroma.com)    |
| **Embeddings**   | `sentence-transformers/all-MiniLM-L6-v2` |
| **Frontend**     | [Streamlit](https://streamlit.io)        |
| **Environment**  | Python 3.10+                             |
| **Architecture** | Modular Multi-Agent RAG                  |

---

## Folder Structure

multi-agent-rag-pipeline/
├── app.py
├── router.py
├── agents/
│   ├── billing_agent.py
│   ├── tech_agent.py
│   └── general_agent.py
├── utils/
│   ├── groq_client.py
│   ├── embeddings.py
│   └── retriever.py
├── data/
│   ├── billing_faq.txt
│   ├── tech_faq.txt
│   └── general_faq.txt
├── requirements.txt
├── .env.example
└── README.md

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
### 4. Set your environment variable
Create a .env file:
```ini
GROQ_API_KEY=your_groq_api_key_here
```
### 5. Run the app
```bash
streamlit run app.py
```

----

## Example Queries
| Query                             | Routed Agent |
| --------------------------------- | ------------ |
| “I want a refund for my payment.” | Billing      |
| “My login page keeps crashing.”   | Technical    |
| “What’s the HR contact email?”    | General      |

---

## Future Upgrades

This repo will evolve into a LangGraph-based orchestration pipeline in upcoming :

- Graph-based node orchestration (`LangGraph`)

- Parallel agent reasoning

- Persistent agent memory

- Structured logs for fine-tuning

Stay tuned — this repo is designed as the foundation for enterprise-grade multi-agent AI pipelines.

---

## Author

**RaghuramReddy Thirumalareddy**
AI Engineer | Generative AI & RAG Systems
https://github.com/RaghuramReddy9 | https://linkedin.com/in/raghuramreddy-ai

---

## License
MIT License © 2025 RaghuramReddy