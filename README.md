# Tiny RAG CLI (Postgres + pgvector + Local LLM)

A tiny Retrieval-Augmented Generation (RAG) command-line tool that:
- embeds text with **SentenceTransformers**,
- stores embeddings in **PostgreSQL + pgvector**,
- retrieves relevant chunks with cosine similarity,
- and asks a **local LLM** (e.g., DeepSeek via Ollama-style API) to answer with context.

> Core pieces live in `agent_runner.py` (retriever, agent, CLI) and `load_documents.py` (example loader). Requirements are in `requirements.txt`.

## Features
- Minimal, readable code
- pgvector cosine retrieval
- Pluggable LLM via simple REST `POST /api/generate`
- Works fully offline once models are local

---

## Quick Start

### 1) Install dependencies
```bash
python -m venv .venv && source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
