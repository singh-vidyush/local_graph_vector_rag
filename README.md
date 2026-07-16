# 🎓 Training Institute Knowledge Assistant

> An End-to-End Local Graph-Vector RAG Application built using Flask, LangChain, ChromaDB, NetworkX, Ollama, and Gemini.

![Python](https://shields.io)
![Flask](https://shields.io)
![LangChain](https://shields.io)
![ChromaDB](https://shields.io)
![NetworkX](https://shields.io)
![Ollama](https://shields.io)
![Gemini](https://shields.io)


---

# 📖 Overview

The **Training Institute Knowledge Assistant** is a Graph-Enhanced Retrieval-Augmented Generation (RAG) application designed to answer questions from a collection of institute-related documents.

The system combines:

- Semantic Search using ChromaDB
- Graph-Based Retrieval using NetworkX
- Local Embeddings using Hugging Face
- Local LLM Inference using Ollama
- Cloud-Based Generation using Gemini
- Flask-based Web Interface

Unlike traditional chatbots, this application answers questions **strictly from the provided documents**, ensuring grounded and reliable responses.

---

# 🎯 Project Objectives

The primary goals of this project are:

- Load information from multiple document formats.
- Apply semantic and overlapping chunking techniques.
- Generate dense vector embeddings locally.
- Store vectors in a persistent ChromaDB database.
- Build relationships between chunks using a NetworkX graph.
- Implement multiple retrieval strategies.
- Generate grounded responses with source attribution.
- Provide an intuitive web interface using Flask.

---

# 🏗️ System Architecture

```text
PDF / DOCX / TXT / JSON
            │
            ▼
    Document Loading
            │
            ▼
 Metadata Standardization
            │
            ▼
      Chunking Layer
            │
 ┌──────────┴──────────┐
 │                     │
 ▼                     ▼
Overlap            Semantic
Chunking           Chunking
            │
            ▼
 HuggingFace Embeddings
            │
 ┌──────────┴──────────┐
 │                     │
 ▼                     ▼
ChromaDB         NetworkX Graph
(Vector Store)   (Knowledge Graph)
            │
            ▼
   Retrieval Engine
            │
 ┌──────┬──────┬──────┬──────────┐
 │      │      │      │
 ▼      ▼      ▼      ▼
Dense  MMR    BM25  Graph-Hybrid
            │
            ▼
 Retrieved Context
            │
            ▼
  Ollama / Gemini
            │
            ▼
      Flask Web App
```

---

# ✨ Features

## 📂 Multi-Format Document Support

The system supports:

- PDF
- DOCX
- TXT
- JSON

All documents are loaded into a unified knowledge base with consistent metadata.

---

## ✂️ Intelligent Chunking

### Recursive Overlap Chunking

- Chunk Size: 700
- Chunk Overlap: 100
- Preserves contextual continuity

### Semantic Chunking

- Hugging Face Embeddings
- Context-aware boundaries
- Better retrieval accuracy

---

## 🧠 Local Embeddings

Embedding Model:

```text
sentence-transformers/all-MiniLM-L6-v2
```

Benefits:

- Lightweight
- Fast
- Runs locally
- No API cost

---

## 🗄️ Persistent ChromaDB

Features:

- Local vector storage
- Semantic search
- MMR retrieval
- Persistent indexing

Database Location:

```text
chroma_db/
```

Collection:

```text
training_institute_rag
```

---

## 🕸️ Knowledge Graph

Every chunk becomes a graph node.

### Sequential Relationships

```text
Chunk A ──► Chunk B
Chunk B ──► Chunk C
```

Preserves document flow.

### Semantic Relationships

```text
Chunk A ── similarity: 0.84 ── Chunk B
```

Connects semantically related information across files.

Graph Storage:

```text
graph_db/chunk_graph.json
```

---

# 🔍 Retrieval Techniques

The application supports four retrieval modes.

## 1. Dense Similarity Retrieval

Retrieves the top-k semantically similar chunks.

Best for:

- Natural language questions
- Contextual understanding

---

## 2. Maximum Marginal Relevance (MMR)

Returns:

- Relevant chunks
- Diverse information
- Reduced redundancy

Best for:

- Broad questions
- Multi-source answers

---

## 3. BM25 Retrieval

Keyword-based retrieval.

Best for:

- Course codes
- Trainer names
- Department names
- Exact policy terms

---

## 4. Graph-Hybrid Retrieval

Workflow:

```text
Question
    ↓
Dense Retrieval
    ↓
Top 2 Seed Chunks
    ↓
Graph Expansion
    ↓
Neighbour Retrieval
    ↓
Context Aggregation
```

Best for connected questions spanning multiple documents.

---

# 🤖 Supported LLMs

## Ollama

Model:

```text
llama3.2:3b
```

Advantages:

- Fully local
- No internet required
- Free to use

---

## Gemini

Model:

```text
gemini-2.5-flash
```

Advantages:

- Better reasoning capability
- High-quality answers
- Fast response generation

---

# 📂 Project Structure

```text
local_graph_vector_rag/
│
├── app.py
├── ingest.py
├── config.py
├── requirements.txt
├── README.md
├── .env
├── .env.example
│
├── data/
│   ├── course_catalog.pdf
│   ├── fee_and_refund_policy.docx
│   ├── placement_rules.txt
│   └── departments.json
│
├── src/
│   ├── document_loader.py
│   ├── chunking.py
│   ├── storage.py
│   ├── graph_builder.py
│   ├── retrieval.py
│   └── generation.py
│
├── templates/
│   └── index.html
│
├── static/
│   └── style.css
│
├── chroma_db/
│
└── graph_db/
    └── chunk_graph.json
```

---

# ⚙️ Installation

## Clone Repository

```bash
git clone <repository-url>

cd local_graph_vector_rag
```

## Create Virtual Environment

### Windows

```bash
python -m venv .venv

.venv\Scripts\activate
```

### Linux/macOS

```bash
python -m venv .venv

source .venv/bin/activate
```

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Environment Configuration

Create a `.env` file:

```env
GEMINI_API_KEY=

OLLAMA_MODEL=llama3.2:3b

GEMINI_MODEL=gemini-2.5-flash

CHUNKING_METHOD=semantic

TOP_K=4

SEMANTIC_THRESHOLD=0.75

CHROMA_DIRECTORY=chroma_db

GRAPH_PATH=graph_db/chunk_graph.json

COLLECTION_NAME=training_institute_rag
```

---

# 🚀 Ollama Setup

Install Ollama and pull the required model.

```bash
ollama pull llama3.2:3b
```

Verify installation:

```bash
ollama list
```

Run model:

```bash
ollama run llama3.2:3b
```

---

# 📥 Data Ingestion

Build the vector database and knowledge graph.

```bash
python ingest.py
```

Expected Output:

```text
Documents loaded successfully

Overlapping chunks created

Semantic chunks created

Embedding model loaded

ChromaDB created successfully

Graph database created successfully

Ingestion completed
```

---

# 🌐 Running the Application

Start Flask:

```bash
python app.py
```

Open:

```text
http://127.0.0.1:5000
```

---

# 🔍 Sample Questions

## Direct Questions

```text
What is the duration of the AI Engineering course?
```

```text
What is the fee structure for Data Analytics?
```

```text
What is the attendance requirement for placement eligibility?
```

---

## BM25-Oriented Questions

```text
Who is the coordinator of the Data Science department?
```

```text
What is course code AIE-101?
```

---

## Graph-Based Questions

```text
Who manages the department that offers AI Engineering?
```

```text
What refund condition applies to the course offered by the Data Science department?
```

---

## Missing Information Question

```text
Does the institute provide free international accommodation?
```

Expected Response:

```text
The answer is not available in the provided documents.
```

---

# 📊 Chunking Comparison

| Feature | Overlap Chunking | Semantic Chunking |
|----------|-----------------|-------------------|
| Chunk Count | Higher | Lower |
| Context Preservation | Good | Excellent |
| Semantic Meaning | Moderate | High |
| Retrieval Quality | Good | Better |
| Selected Method | ❌ | ✅ |

### Selected Approach

Semantic Chunking was selected because it generates context-aware chunks, improves retrieval quality, and reduces irrelevant retrieval noise.

---

# ✅ Error Handling

The application handles:

- Empty user questions
- Missing data folder
- Missing ChromaDB
- Missing graph database
- Missing Gemini API key
- Unsupported retrieval mode
- Ollama not running
- Missing Ollama model
- No relevant context found

---

# 📸 Screenshots

## Screenshot 1 – Successful Ingestion

> Add ingestion screenshot here.

---

## Screenshot 2 – Ollama Response

> Add Flask response screenshot here.

---

## Screenshot 3 – Graph-Hybrid Retrieval Response

> Add graph-hybrid screenshot here.

---

# 🚧 Known Limitations

- Graph construction becomes slower on larger datasets.
- One-hop graph expansion may miss distant relationships.
- Small local LLMs may generate less refined responses.
- Retrieval quality depends on document quality.
- Semantic graph generation increases ingestion time.
- Gemini requires internet connectivity.

---

# 🌟 Future Enhancements

- Two-hop graph retrieval
- Hybrid BM25 + Vector search
- Conversation memory
- File uploads from UI
- Graph visualization dashboard
- Retrieval score analysis
- User authentication
- Multi-user support

---

# 🛠️ Technologies Used

### Backend

- Python
- Flask
- LangChain

### Retrieval

- ChromaDB
- BM25
- MMR Retrieval

### Embeddings

- Sentence Transformers
- Hugging Face Embeddings

### Graph Processing

- NetworkX

### LLMs

- Ollama
- Gemini

### Frontend

- HTML
- CSS

---

# 👨‍💻 Author

**Vidyush Singh**

Graduate Apprentice

---

# 📜 Conclusion

This project demonstrates a complete Graph-Vector Retrieval-Augmented Generation (RAG) pipeline capable of loading documents, generating embeddings, constructing a knowledge graph, retrieving relevant context through multiple retrieval techniques, and generating grounded answers using local and cloud-based LLMs.

The application showcases how modern AI systems can combine semantic search, graph relationships, and large language models to build intelligent, reliable, and explainable knowledge assistants.