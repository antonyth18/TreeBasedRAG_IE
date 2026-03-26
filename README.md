# EYWA-AI | Tree-Based RAG

A powerful Retrieval-Augmented Generation (RAG) system utilizing the **RAPTOR** (Recursive Abstractive Processing for Tree-Organized Retrieval) approach. This system builds a hierarchical tree of document summaries for multiscale retrieval, enabling better answers to both specific and broad/thematic queries.

## 🚀 Features

- **Hierarchical Tree Retrieval**: Uses RAPTOR to build and traverse a tree of summaries.
- **Persistent Storage**: Automatically saves and re-loads trees from disk on startup.
- **Smart Deduplication**: Content-based hashing ensures identical files aren't processed or stored twice.
- **Dynamic Frontend**: Modern React + TypeScript UI with real-time tree visualization and retrieval details.
- **Hybrid Querying**: Automatically routes queries between "Broad" (collapsed tree) and "Specific" (tree traversal) strategies.

---

## 🛠️ Backend Setup (FastAPI)

### 1. Prerequisites
- Python 3.9+

### 2. Installation
```bash
# Navigate to project root
cd TreeBasedRAG_IE

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install python-multipart uvicorn fastapi pydantic
```

---

### 3. Running the Server
**Important**: You must include the current directory in your `PYTHONPATH`.
```bash
export PYTHONPATH=$PYTHONPATH:.
python3 -m uvicorn backend.main:app --reload
```
The backend will be available at `http://localhost:8000`.

---

### 4. Optimizing Ollama Concurrency
To speed up the RAPTOR tree-building process, the pipeline summarizes document clusters concurrently. You **must** configure Ollama to allow parallel requests.

Set the `OLLAMA_NUM_PARALLEL` environment variable based on your system's available Unified Memory (Mac) or GPU VRAM (Windows/Linux) before starting the Ollama server:

- **< 8GB VRAM / RAM:** `OLLAMA_NUM_PARALLEL=1` (Sequential fallback for smaller systems)
- **8GB - 12GB VRAM:** `OLLAMA_NUM_PARALLEL=2` (Recommended for RTX 3060/4060 or Apple M1/M2/M3 base models)
- **16GB - 24GB VRAM:** `OLLAMA_NUM_PARALLEL=4` (Recommended for RTX 3090/4090, Radeon RX 7900, or Apple Pro/Max chips)
- **32GB+ VRAM:** `OLLAMA_NUM_PARALLEL=6` or higher

**Mac / Linux Terminal:**
```bash
# Stop any existing Ollama instance first, then:
export OLLAMA_NUM_PARALLEL=2
ollama serve
```
* **MacOS App Users:** You can permanently set this globally by running `launchctl setenv OLLAMA_NUM_PARALLEL 2` in your terminal and restarting the Ollama Mac App.
* **Linux Service Users:** Run `systemctl edit ollama.service`, add `Environment="OLLAMA_NUM_PARALLEL=2"` under `[Service]`, and restart the service via `systemctl restart ollama`.

**Windows:**
```powershell
# PowerShell
$env:OLLAMA_NUM_PARALLEL="2"
ollama serve

# Command Prompt
set OLLAMA_NUM_PARALLEL=2
ollama serve
```
* **Windows App Users:** To set this globally, search for "Environment Variables" in your Windows Start Menu. Add a new System Variable named `OLLAMA_NUM_PARALLEL` with the value `2`, click OK, and completely restart the Ollama tray app.

---

## 💻 Frontend Setup (React + Vite)

### 1. Prerequisites
- Node.js (v18+)
- npm or yarn

### 2. Installation
```bash
# Navigate to frontend directory
cd frontend

# Install dependencies
npm install
```

### 3. Running the Client
```bash
npm run dev
```
The frontend will be available at `http://localhost:5173`.

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
