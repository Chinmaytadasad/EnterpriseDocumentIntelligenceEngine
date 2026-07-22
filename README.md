# Enterprise Document Intelligence Engine (Local RAG)

![Python](https://img.shields.io/badge/Python-3.10-blue.svg)
![Docker](https://img.shields.io/badge/Docker-Enabled-2496ED.svg)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-PGVector-336791.svg)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green.svg)
![Ollama](https://img.shields.io/badge/Ollama-Llama_3.2-black.svg)

A fully air-gapped, highly secure **Retrieval-Augmented Generation (RAG)** pipeline designed for enterprise environments with strict data privacy requirements. This system ingests disparate unstructured data (CSVs, PDFs, Linux logs) and allows users to query it naturally using a local LLM, completely bypassing external APIs like OpenAI.

## 🌟 Key Features

* **100% Air-Gapped & Local:** No external API dependencies. All inference runs locally via Ollama (Llama 3.2), ensuring sensitive corporate data never leaves the host machine.
* **Production-Grade Vector Search:** Utilizes a containerized **PostgreSQL** database with the **PGVector** extension for enterprise-scale similarity search, moving beyond toy in-memory vector stores.
* **Decoupled Architecture:** Built as a scalable microservice architecture. The Streamlit presentation layer is entirely decoupled from the vector storage and retrieval backend.
* **Multi-Modal Data Ingestion:** Includes robust ETL pipelines (`ingest.py`) that parse and chunk raw text logs, PDF documents, and structured tabular CSVs (via Pandas) into unified mathematical representations.
* **Verifiable Citations:** Defeats AI hallucinations by enforcing exact source tracing. The UI explicitly returns the File Name, Chunk ID, and exact Row/Page number where the AI found the information.
* **OOM-Safe Ingestion:** Uses optimized batched insertion and lightweight CPU-only HuggingFace Transformers (`nomic-embed-text`) to safely process thousands of chunks on standard consumer hardware.

## 🏗️ Architecture

1. **Frontend:** Streamlit Container
2. **Database:** PostgreSQL + PGVector Container
3. **LLM Engine:** Local host Ollama server running `llama3.2`
4. **Embeddings:** `HuggingFaceEmbeddings` processing locally inside the Docker network.

## 🚀 Getting Started

### Prerequisites
* Docker & Docker Compose
* [Ollama](https://ollama.com/) installed on the host machine.

### 1. Setup Ollama
Ensure your local Ollama server is running and pull the required model:
```bash
ollama pull llama3.2
```

### 2. Build & Launch Containers
Clone the repository and spin up the decoupled architecture:
```bash
docker-compose up --build -d
```
*This will initialize the PostgreSQL/PGVector database and build the Streamlit web application environment.*

### 3. Ingest Data
Once the containers are running, execute the ETL pipeline to parse your raw data, generate embeddings, and populate the PGVector database:
```bash
docker exec fde_streamlit_interface python ingest.py
```
*Note: Depending on your CPU, embedding thousands of chunks locally may take 10-20 minutes.*

### 4. Run the Interface
Navigate to your local browser to access the Document Intelligence Engine:
**[http://localhost:8501](http://localhost:8501)**

## 📊 Example Usage

Once the UI is loaded, you can ask cross-modal questions about your proprietary data. 

**Query:** *"Can you check the customer support tickets and tell me about any issues with the LG Smart TV?"*

The system will retrieve the exact matching CSV rows, generate a synthesized response using Llama 3.2, and provide a dropdown containing the verified PostgreSQL Row/Chunk IDs for auditing.

---
*Developed as part of a Forward Deployed Engineering (FDE) Portfolio Project.*
