# LangGraph PDF RAG Chatbot

An intelligent PDF-based Retrieval-Augmented Generation (RAG) chatbot built using **LangGraph**, **Groq LLMs**, **FAISS**, **Hugging Face Embeddings**, **LangSmith**, and **Streamlit**.

The chatbot allows users to upload PDF documents, ask context-aware questions, perform web-assisted searches, and maintain persistent conversations through a modern chat interface.

> **Note:** This is a lightweight Agentic AI project created for learning and portfolio purposes. It demonstrates RAG, tool calling, conversational memory, observability, and agent orchestration, but is not intended to handle highly complex autonomous tasks.

---

## Features

### Core Capabilities

* PDF-based Question Answering using RAG
* LangGraph-powered Agentic Workflow
* Persistent Conversation Memory with SQLite
* Automatic Chat Title Generation
* Real-Time Streaming Responses
* Interactive Streamlit Interface

### Available Tools

* **PDF Retrieval Tool** – Retrieves relevant information from uploaded documents.
* **Web Search Tool** – Fetches recent information from the web.
* **Calculator Tool** – Performs arithmetic calculations.
* **Stock Price Tool** – Retrieves real-time stock prices.

### Observability & Monitoring

* End-to-end workflow tracing using **LangSmith**.
* Monitor LLM calls, retrieval steps, tool executions, and agent decisions.
* Simplifies debugging and performance analysis of the chatbot workflow.

---

## Tech Stack

* **LangGraph** – Agent orchestration
* **LangChain** – LLM and tool integration
* **Groq (Llama 3.1 8B Instant)** – Response generation
* **Hugging Face Embeddings** – Semantic search
* **FAISS** – Vector database
* **PyPDFLoader** – PDF processing
* **SQLite** – Persistent memory and chat history
* **LangSmith** – Tracing, monitoring, and debugging
* **Streamlit** – Frontend UI

---

## Project Structure

```text
├── rag_backend.py      # Agent workflow, tools, memory, vector store
├── rag_frontend.py     # Streamlit interface
├── chatbot.db          # Persistent chat storage
├── requirements.txt
└── .env
```

---

## Installation

### Install Dependencies

```bash
pip install -r requirements.txt
```

If needed:

```bash
pip install faiss-cpu
```

### Configure Environment Variables

Create a `.env` file:

```env
# Required
GROQ_API_KEY=your_groq_api_key

# Optional - LangSmith Tracing
LANGSMITH_TRACING_V2=true
LANGSMITH_ENDPOINT=https://api.smith.langchain.com
LANGSMITH_API_KEY=your_langsmith_api_key
LANGSMITH_PROJECT=Chatbot
```

### Run the Application

```bash
streamlit run rag_frontend.py
```

---

## Limitations

* Supports a limited set of tools:

  * PDF Retrieval
  * Web Search
  * Calculator
  * Stock Price Lookup

* Designed as a mini Agentic AI project rather than a production-grade AI assistant.

* Does not support multi-agent collaboration or advanced workflow automation.

* Not intended for highly complex autonomous tasks.

* Performance depends on the selected LLM and the quality of retrieved document context.

---

## Learning Outcomes

This project demonstrates practical experience with:

* Retrieval-Augmented Generation (RAG)
* Agentic AI Workflows
* LangGraph
* LangChain
* Tool Calling
* Vector Databases (FAISS)
* LLM Integration
* Conversational Memory
* LangSmith Observability & Tracing
* Streamlit Application Development
