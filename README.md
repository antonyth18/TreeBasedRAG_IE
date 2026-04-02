# EYWA-AI | Tree-Based RAG

A powerful Retrieval-Augmented Generation (RAG) system utilizing the **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) approach. This system builds a hierarchical tree of document summaries for multiscale retrieval, enabling better answers to both specific and broad/thematic queries.

## 🚀 Features

- **Hierarchical Tree Retrieval**: Uses RAPTOR to build and traverse a tree of summaries.
- **Persistent Storage**: Automatically saves and re-loads trees from disk on startup.
- **Smart Deduplication**: Content-based hashing ensures identical files aren't processed or stored twice.
- **Dynamic Frontend**: Modern React + TypeScript UI with real-time tree visualization and retrieval details.
- **Hybrid Querying**: Automatically routes queries between "Broad" (collapsed tree) and "Specific" (tree traversal) strategies.

---

## 🐳 Quick Start (Docker - Recommended)

The easiest way to run the system is using Docker Compose. This handles all dependencies, including the spaCy language models and environment configurations.

### 1. Prerequisites
- Docker & Docker Desktop installed.
- **Ollama** running on your host machine (Mac/Windows).

### 2. Launch
```bash
# Navigate to project root
cd TreeBasedRAG_IE

# Build and start in detached mode
docker compose up --build -d
```

### 3. Access
* **Frontend**: [http://localhost:5173](http://localhost:5173)
* **Backend API Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

> [!TIP]
> The system automatically bridges to your host machine's Ollama instance via `host.docker.internal`. Ensure Ollama is running and accessible.

---

## 🛠️ Manual Development Setup

If you prefer to run the services locally without Docker:

### Backend (FastAPI)
1. **Python 3.9+** is required.
2. **Install Dependencies**:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   pip install -r requirements.txt
   # Download the required spaCy model
   python -m spacy download en_core_web_sm
   ```
3. **Run Server**:
   ```bash
   export PYTHONPATH=$PYTHONPATH:.
   python3 -m uvicorn backend.main:app --reload
   ```

### Frontend (React + Vite)
1. **Node.js v18+** is required.
2. **Install & Run**:
   ```bash
   cd frontend
   npm install
   npm run dev
   ```
   Available at `http://localhost:5173`.

---

## 🛠️ Optimizing Ollama Concurrency

To speed up the RAPTOR tree-building process, set the `OLLAMA_NUM_PARALLEL` environment variable **before** starting your Ollama server. This allows concurrent summarization of document clusters.

- **8GB - 12GB RAM**: `OLLAMA_NUM_PARALLEL=2`
- **16GB - 24GB RAM**: `OLLAMA_NUM_PARALLEL=4`
- **32GB+ RAM**: `OLLAMA_NUM_PARALLEL=6`

**Mac Users**: 
Run `launchctl setenv OLLAMA_NUM_PARALLEL 2` and restart the Ollama app.

---

## 📂 Project Structure

- `backend/`: FastAPI application, routers, and services.
- `frontend/`: React + TypeScript source code.
- `tree/`: Core RAPTOR tree construction and serialization logic.
- `retrieval/`: Multi-strategy retrieval and query classification.
- `data/`: Processed document storage (deduplicated by hash).
- `my_tree/`: Serialized tree structures (persisted across restarts).

---

## 📝 Usage

1. **Upload**: Use the sidebar to upload a PDF. Its content is hashed and saved to `data/`.
2. **Build**: The system automatically builds a RAPTOR tree. This is saved to `my_tree/` and reloaded on future restarts.
3. **Query**: Ask questions! The system will classify your query and retrieve context from the appropriate layers of the tree.
4. **Visualize**: Check the "Tree Structure" and "Retrieved Nodes" panels to see exactly how the RAG engine picked its context.
