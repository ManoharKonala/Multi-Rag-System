# Multi-RAG: Advanced Multi-Container Retrieval-Augmented Generation

A multi-model RAG system designed for deployment-ready scalability, featuring real-time streaming updates and a robust containerized architecture.

## 🚀 Vision
To provide a production-grade blueprint for large-scale document intelligence systems. This project leverages PostgreSQL for metadata, Milvus for high-performance vector search, and OpenAI's latest models for intelligent synthesis.

## 🛠 Architecture
- **API**: Flask with SSE (Server-Sent Events) for real-time streaming.
- **Vector DB**: Milvus (Vector storage and retrieval).
- **Database**: PostgreSQL (Metadata and application state).
- **Frontend**: Clean, modern UI with real-time status updates via SSE.
- **Orchestration**: Docker Compose for multi-container synergy.
- **Reverse Proxy**: Nginx with rate limiting and security headers.

## 📋 Prerequisites
- Docker & Docker Compose
- OpenAI API Key

## ⚡ Quick Start
1. **Clone and Configure**:
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API Key
   ```
2. **Launch Infrastructure**:
   ```bash
   docker-compose up -d
   ```
3. **Access the App**:
   - Frontend: [http://localhost](http://localhost)
   - API Status: [http://localhost/health](http://localhost/health)

## 📁 Directory Structure
- `src/core/`: RAG logic, embeddings, and database models.
- `src/api/`: REST and streaming endpoints.
- `src/utils/`: Background processing and pipeline orchestration.
- `frontend/`: Modern web interface.
- `data/`: Local storage for uploads and processed artifacts.

---
Built with ❤️ for the future of AI.
