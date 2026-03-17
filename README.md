# Multi-RAG: Advanced Multi-Modal Agentic AI System

A production-grade, multi-modal Retrieval-Augmented Generation (RAG) system with **Agentic Reasoning** capabilities. This platform goes beyond simple search, employing autonomous query analysis and multi-step reasoning to synthesize insights from Text, Images, and Structured Tables.

## 🤖 Agentic AI Capabilities
The system's "Brain" (`AdvancedQueryProcessor`) performs autonomous decision-making to optimize retrieval:
- **Intent-Driven Decomposition:** Automatically breaks down complex, multi-part questions into atomic sub-queries for broader coverage.
- **Autonomous Strategy Selection:** Dynamically chooses between `basic_retrieval`, `multi_step_reasoning`, and `advanced_analytical` patterns based on query complexity.
- **Intent Classification:** Detects whether a query is Factual, Comparative, Analytical, or Procedural to tailor the synthesis logic.
- **Entity & Keyword Intelligence:** Extracts critical entities (dates, currencies, technical terms) to refine vector space searches.

## 🖼️ Multi-Modal Intelligence
- **Visual Context (CLIP):** Utilizes OpenAI's CLIP model to generate semantic embeddings for images, charts, and graphs within PDFs.
- **Structured Data Vectorization:** Employs a specialized `TableProcessor` to parse and summarize complex tabular data into vector-friendly representations.
- **Cross-Modal Hybrid Search:** Simultaneously searches across disparate vector spaces (Text, Image Descriptions, Table Summaries) for a unified context window.

## 🛠 Technical Architecture
- **Vector Core:** **Milvus** for high-dimensional similarity search.
- **Metadata Layer:** **PostgreSQL** for structured document state and relational metadata.
- **Performance Layer:** **Redis** for intelligent response caching and session management.
- **Object Store:** **MinIO** for scalable storage of source artifacts.
- **Deployment-Ready Optimization:** Custom Docker layering that **bakes-in heavy AI models** (CLIP, SentenceTransformers) to eliminate cold-start delays.

## 🚀 Quick Start
1. **Infrastructure Setup**:
   ```bash
   cp .env.example .env
   # Add your OPENAI_API_KEY. Default host port: 8090
   ```
2. **One-Command Launch**:
   ```bash
   docker compose -p multi-rag-system up -d --build
   ```
3. **Endpoints**:
   - **Frontend UI:** [http://localhost:8090](http://localhost:8090)
   - **System Health:** [http://localhost:8090/health](http://localhost:8090/health)

## 📁 Source Organization
- `src/core/`: Agentic logic, multi-modal embeddings, and RAG orchestrators.
- `src/api/`: Streaming (SSE) and RESTful interfaces.
- `nginx.conf`: Hardened reverse-proxy with rate-limiting and security headers.

---
Built with a focus on **Autonomous Reasoning**, **Multi-Modal Synthesis**, and **Enterprise Scalability**.
