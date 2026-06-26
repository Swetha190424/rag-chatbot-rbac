# RAG Chatbot with Role-Based Access Control (RBAC)

A production-grade AI chatbot built with LangChain, Qdrant, and Groq LLM that enforces 
department-level document access based on user roles.

## Features
- Role-Based Access Control (RBAC) — HR, Finance, and CEO roles with different data access
- RAG (Retrieval Augmented Generation) using Qdrant vector database
- Input guardrails to filter irrelevant questions
- PII masking using Microsoft Presidio
- LangSmith monitoring for query tracking
- Streamlit UI with login system

## Tech Stack
- Python, LangChain, Streamlit
- Qdrant (vector database)
- Groq API (llama-3.3-70b-versatile)
- HuggingFace Embeddings
- Microsoft Presidio (PII detection)
- LangSmith (monitoring)

## How It Works
1. Documents are split and stored in Qdrant with department metadata
2. User logs in with role (hr/finance/ceo)
3. Questions are filtered by guardrails
4. Only department-authorized documents are retrieved
5. LLM generates answer from filtered context
6. PII is masked before returning response

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your API keys in `app.py`,`rag.py`, `rag_rbac.py`, `guardrails.py`
4. Run: `streamlit run app.py`
