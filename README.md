# CampusGPT: Comparative Evaluation of Fine-Tuned vs. RAG-Based Language Models for University Q&A Systems

CampusGPT is an end-to-end institutional Question-Answering (Q&A) platform engineered for Algoma University. This repository contains the complete implementation, datasets, evaluation framework, and deployment code for a Master's thesis comparing two prominent Large Language Model (LLM) domain-adaptation architectures:

1. **Retrieval-Augmented Generation (RAG):** Dynamic, document-grounded retrieval using LangChain, ChromaDB, and a locally hosted Llama 3 model via Ollama.
2. **Quantized Low-Rank Adaptation (QLoRA):** Parameter-efficient fine-tuning of Llama 3 using Unsloth on Google Colab GPUs, internalizing institutional knowledge into targeted adapter weights.
3. **Unified Full-Stack Deployment:** A decoupled architecture featuring a high-performance FastAPI backend connected to an interactive React and Material UI (MUI) web client.

---

## Table of Contents

- [Demos & Interface Recordings](#demos--interface-recordings)
- [Project Overview](#project-overview)
- [System Architecture](#system-architecture)
- [Tech Stack](#tech-stack)
- [Repository Structure](#repository-structure)
- [Data Pipeline & Preparation](#data-pipeline--preparation)
- [Installation & Setup](#installation--setup)
  - [Prerequisites](#prerequisites)
  - [1. Backend & RAG Setup (Local)](#1-backend--rag-setup-local)
  - [2. QLoRA Model Setup (Google Colab / Cloud GPU)](#2-qlora-model-setup-google-colab--cloud-gpu)
  - [3. Frontend Setup (React)](#3-frontend-setup-react)
- [Evaluation & Benchmark Results](#evaluation--benchmark-results)
- [The Hybrid Paradigm (Future Work)](#the-hybrid-paradigm-future-work)
- [License & Acknowledgments](#license--acknowledgments)

---

## Demos & Interface Recordings

The frontend features a dual-engine toggle allowing real-time switching and side-by-side evaluation between the RAG and QLoRA models.

| RAG-Based CampusGPT (Local via Ollama) | QLoRA Fine-Tuned CampusGPT (Cloud via Colab) |
| :---: | :---: |
| ![RAG Demo Placeholder](docs/assets/rag-demo.gif) | ![QLoRA Demo Placeholder](docs/assets/qlora-demo.gif) |
| *Dynamic retrieval with context-grounded citations* | *Direct neural generation with internalized domain style* |

---

## Project Overview

Institutional data across university campuses is notoriously fragmented. Policies, degree requirements, financial aid deadlines, and faculty contacts are distributed across PDFs, student portals, and static web pages. General foundation models fail in this setting because:
- They lack private, institutional data.
- They exhibit high rates of hallucination when asked precise, local policy questions.
- Retraining entire frontier models is computationally prohibitive.

This project investigates whether it is better to provide an LLM with external search capabilities (RAG) or re-train a fraction of its parameters on curated institutional pairs (QLoRA).

---

## System Architecture

```text
               +-------------------------------------------------------+
               |              React + Material UI Frontend             |
               |     (Dual-Engine Chat UI / Real-Time Model Toggle)    |
               +-------------------------------------------------------+
                                           |
                                   (HTTP REST / JSON)
                                           v
               +-------------------------------------------------------+
               |               FastAPI Python Gateway                  |
               |     - Request routing & validation                    |
               |     - Dynamic latency & performance telemetry         |
               +-------------------------------------------------------+
                            /                             \
                           /                               \
        [ Route: /api/chat/rag ]               [ Route: /api/chat/qlora ]
                         /                                   \
                        v                                     v
       +-------------------------------+       +-------------------------------+
       |       RAG Pipeline (Local)    |       |   QLoRA Inference (Cloud)     |
       |-------------------------------|       |-------------------------------|
       | - Ingestion: LangChain        |       | - Host: Google Colab GPU      |
       | - Vector Store: ChromaDB      |       | - Base Model: Llama 3 (4-bit) |
       | - Embeddings: Sentence-Transf.|       | - Adapters: Unsloth LoRA      |
       | - Generator: Ollama (Llama 3) |       | - Weights: Frozen W + ΔW (BA) |
       +-------------------------------+       +-------------------------------+
```

---

## Tech Stack

### Core AI & Machine Learning
- **Foundation Model:** Meta Llama 3 (8B)
- **Fine-Tuning Acceleration:** Unsloth (optimized 4-bit normal float quantization, rank $r=16$, $\alpha=32$)
- **Deep Learning Framework:** PyTorch
- **RAG Orchestration:** LangChain
- **Vector Database:** ChromaDB

### Backend & Serving
- **API Framework:** FastAPI
- **Local Model Runtime:** Ollama (serving the RAG generator locally)
- **Remote Model Runtime:** Google Colab (serving the fine-tuned QLoRA weights via an exposed API tunnel)
- **Data Serialization:** Pydantic

### Frontend
- **Framework:** React.js
- **UI Component Library:** Material UI (MUI) & Emotion Icons
- **HTTP Client:** Axios

---

## Repository Structure

```text
CampusGPT/
├── backend/
│   ├── app/
│   │   ├── api/
│   │   │   └── routes.py         # Endpoints for /rag, /qlora, and health checks
│   │   ├── core/
│   │   │   └── config.py         # Environment configurations & endpoint URLs
│   │   ├── rag/
│   │   │   ├── chunking.py       # Recursive character text splitting
│   │   │   ├── embed.py          # Vector embedding generation
│   │   │   ├── ingest.py         # PDF & HTML data ingestion into ChromaDB
│   │   │   └── pipeline.py       # LangChain retrieval QA chain
│   │   └── schemas/
│   │       └── chat.py           # Pydantic request/response models
│   ├── main.py                   # FastAPI initialization
│   ├── requirements.txt
│   └── .env.example
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── components/
│   │   │   ├── ChatBubble.jsx    # Formatted chat messages & source citations
│   │   │   ├── ChatInput.jsx     # Query submission component
│   │   │   ├── ModelSelector.jsx # Switch between RAG and QLoRA
│   │   │   └── TopBar.jsx        # Navigation header
│   │   ├── App.jsx               # Root UI orchestration
│   │   ├── index.js
│   │   └── theme.js              # Algoma University theme styling
│   ├── package.json
│   └── .env.example
├── training/
│   ├── datasets/
│   │   ├── raw/                  # Scraped university policies, PDFs, catalogs
│   │   └── processed/
│   │       ├── chunks.json       # Document chunks for ChromaDB
│   │       └── qlora_pairs.json  # Instruction-input-output pairs for Unsloth
│   ├── unsloth_qlora_llama3.ipynb# Colab notebook for 4-bit fine-tuning
│   └── export_adapters/          # Saved LoRA adapter checkpoints
├── evaluation/
│   ├── test_benchmarks.json      # Hold-out question set with gold-standard answers
│   ├── evaluate_accuracy.py      # Automated LLM-as-a-judge / semantic similarity tests
│   └── latency_benchmark.py      # Time-to-first-token & generation speed analysis
├── docs/
│   └── assets/                   # Architecture diagrams, UI gifs, recordings
└── README.md
```

---

## Data Pipeline & Preparation

Both architectures were developed using a unified collection of raw Algoma University documentation, processed into two target formats:

```text
Algoma University Raw Data (PDFs, Web Pages, Course Catalogs)
                           │
       ┌───────────────────┴───────────────────┐
       ▼                                       ▼
 [ RAG Preprocessing ]                   [ QLoRA Preprocessing ]
 - Text extraction & cleaning            - Instruction-response generation
 - Recursive chunking (500 chars)        - Removal of rigid citation artifacts
 - ChromaDB vector indexing              - Alpaca formatted JSON dataset
```

1. **RAG Data Preparation:**
   - Raw documents were cleaned of layout artifacts, header noise, and tabular corruptions.
   - Text was split into chunks using a `RecursiveCharacterTextSplitter` (chunk size: 500 characters, overlap: 50 characters).
   - Embedded using dense vector embeddings and stored directly in a local persistent ChromaDB collection.

2. **QLoRA Data Preparation:**
   - Raw documentation was synthesized into over 2,000 domain-specific instruction-response pairs (`instruction`, `input`, `output`).
   - Responses were scrubbed of rigid legalistic phrasing (e.g., "as seen in section 4.2", "according to table 1") to ensure the model learned to deliver facts in a natural conversational persona.

---

## Installation & Setup

### Prerequisites
- Python 3.10+
- Node.js 18+ and npm
- [Ollama](https://ollama.ai/) installed and running locally
- Google Colab account with a GPU instance (T4 or A100)

---

### 1. Backend & RAG Setup (Local)

1. Clone the repository and navigate to the backend:
   ```bash
   git clone [https://github.com/your-username/CampusGPT.git](https://github.com/your-username/CampusGPT.git)
   cd CampusGPT/backend
   ```

2. Set up the virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Pull and start Llama 3 locally via Ollama:
   ```bash
   ollama pull llama3
   ollama serve
   ```

4. Ingest the Algoma University documents into ChromaDB:
   ```bash
   python -m app.rag.ingest --source_dir ../training/datasets/raw
   ```

5. Configure environment variables:
   ```bash
   cp .env.example .env
   ```
   Ensure `.env` contains:
   ```env
   OLLAMA_BASE_URL=http://localhost:11434
   CHROMA_PERSIST_DIRECTORY=./chroma_db
   QLORA_REMOTE_ENDPOINT=http://your-colab-tunnel-url/generate
   ```

6. Start the FastAPI server:
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

---

### 2. QLoRA Model Setup (Google Colab / Cloud GPU)

1. Open `training/unsloth_qlora_llama3.ipynb` inside Google Colab.
2. Select a GPU runtime (`Runtime` > `Change runtime type` > `T4 GPU` or `A100 GPU`).
3. Upload `training/datasets/processed/qlora_pairs.json`.
4. Run the notebook to:
   - Load `unsloth/llama-3-8b-bnb-4bit`.
   - Attach LoRA adapters to attention layers (`q_proj`, `k_proj`, `v_proj`, `o_proj`, etc.).
   - Train on the institutional dataset.
5. Execute the final cell to launch a lightweight inference server exposed via `ngrok` or `localtunnel`.
6. Copy the generated public URL and paste it as `QLORA_REMOTE_ENDPOINT` in `backend/.env`.

---

### 3. Frontend Setup (React)

1. Navigate to the frontend directory:
   ```bash
   cd ../frontend
   npm install
   ```

2. Configure frontend environment variables:
   ```bash
   cp .env.example .env
   ```
   Ensure `.env` matches your FastAPI backend:
   ```env
   REACT_APP_API_BASE_URL=http://localhost:8000
   ```

3. Run the development server:
   ```bash
   npm start
   ```
4. Access the web interface at `http://localhost:3000`.

---

## Evaluation & Benchmark Results

Both systems were benchmarked against a hold-out evaluation set of 100 domain-specific questions spanning academic regulations, campus life, program prerequisites, and fee policies.

### Comparative Summary

| Metric | RAG Pipeline (ChromaDB + Ollama) | QLoRA Fine-Tuned (Unsloth + Colab) |
| :--- | :--- | :--- |
| **Factual Accuracy** | **94.2%** (Grounding prevents errors) | **76.8%** (Hallucinations on specific edge cases) |
| **Hallucination Rate** | **Low (< 5%)** (Failures stem from retrieval) | **Moderate (~ 23%)** (Confident false assertions) |
| **Conversational Tone** | Rigid, document-like synthesis | **Natural, human-like, engaging** |
| **Traceability** | **Full** (Explicit document chunks & sources) | **None** (Implicit neural weights) |
| **Knowledge Update Speed** | **Immediate** (Re-index vector database) | **Slow/Costly** (Requires full retraining loop) |
| **Average Query Latency** | ~2.4 seconds (Retrieval overhead) | **~1.1 seconds** (Direct parameter generation) |
| **Hardware Footprint** | Low GPU load (CPU vector store + LLM) | High GPU load during training/serving |

### Core Findings
1. **RAG is the superior standalone architecture for university policy Q&A:** In an educational context where a wrong deadline or prerequisite can severely impact a student, verifiable factual grounding and rapid knowledge updates outweigh conversational tone.
2. **QLoRA excels at persona and syntax:** The fine-tuned model produced significantly more natural, empathetic, and structurally coherent answers, but proved unsafe when tasked with remembering exact dates, monetary fees, and policy numbers.

---

## The Hybrid Paradigm (Future Work)

Rather than treating RAG and QLoRA as competing architectures, the logical next step is **RAG-Tuning (Hybrid Integration)**:

```text
User Query ──► Semantic Retrieval (ChromaDB) ──► Raw Chunks
                                                     │
                                                     ▼
                                      [ QLoRA Fine-Tuned Reader ]
                                      - Trained to parse raw chunks
                                      - Drops robotic document phrasing
                                      - Understands university jargon
                                                     │
                                                     ▼
                                         Factually Grounded &
                                     Naturally Phrased Response
```

- **Stylistic Synthesis:** Using QLoRA to train a smaller model to read raw RAG chunks and synthesize them conversationally without sounding like a legal PDF reader.
- **Campus Jargon Fluency:** Adapting the model's tokenizer and internal weights to local university acronyms so it can better interpret the retrieved context.
- **Cost Reduction:** A fine-tuned 8B model paired with RAG matches the output quality of proprietary frontier models while remaining entirely self-hosted.

---

## License & Acknowledgments

- **Institution:** Algoma University
- **Base Model:** Meta AI Llama 3
- **Fine-Tuning Engine:** Unsloth AI
- **Frameworks:** LangChain, ChromaDB, FastAPI, React, Material UI
