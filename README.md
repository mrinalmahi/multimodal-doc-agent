# multimodal-doc-agent

Multimodal agent that answers questions over scientific PDFs by routing queries across text, image, and table tools. Includes a CI/CD pipeline that gates merges on tool-selection and answer accuracy.

## What it does

- Ingests any PDF: extracts text, renders figures, crops tables, builds a searchable index
- Routes each question to the correct tool (text search, table image, or figure image) via agentic tool selection
- Reads tables and figures visually rather than parsing table structure, avoiding failure on borderless or vector-graphic tables
- Supports multiple documents in one conversation, with document resolution handled per-question
- Maintains conversation history for follow-up questions
- Detects duplicate uploads via content hashing, skipping redundant ingestion
- Ships with an automated eval suite that gates CI/CD on tool-selection and answer accuracy

## Architecture

**Ingestion (`src/ingestion/`)**
- Column-aware PDF text extraction (PyMuPDF), with automatic header/footer stripping via cross-page repetition detection
- Caption detection for figures/tables via regex, tolerant of format variants (`Fig.`/`Figure`, period/space delimiters, case, multi-page `(Continued)` tables)
- Full-page figure rendering; table regions cropped via caption-anchored block traversal (column detection, gap-based boundary detection, straddle detection for full-width tables), with automatic stitching for tables spanning multiple pages
- Chunking + embedding (`intfloat/e5-base-v2`) into a single Chroma collection, namespaced by `document_id` metadata rather than per-document collections
- SHA-256 content hashing for duplicate-upload detection

**Agent (`src/agent.py`, `src/agent_tools.py`)**
LangChain/LangGraph agent on Gemini, five tools:

| Tool | Function |
|---|---|
| `search_text` | Hybrid search over document prose (e5 embeddings + BM25, merged with reciprocal rank fusion) |
| `list_tables` / `list_figures` | Deterministic index lookup (not inferred from retrieval) |
| `ask_about_table` | Vision read of a cropped table image |
| `ask_about_figure` | Vision read of a figure's page image |

Each tool takes `document_id` explicitly, supporting multi-document conversations with document resolution handled via the system prompt.

**API / frontend (`src/api.py`, `index.html`)**
FastAPI service (`/documents`, `/ask`, `/health`); browser frontend maintains conversation history and renders any table/figure image the agent used.

**Evaluation (`eval/`)**
18-question fixed suite covering all five tools, scored on tool-call correctness and answer-content correctness. Rate-limit-aware execution, non-zero exit on threshold failure.

**CI/CD**
GitHub Actions runs the eval suite against a fresh ingestion of a pinned test PDF on every PR. Branch protection on `main` requires this check to pass before merge.

## Setup

```bash
git clone https://github.com/<your-username>/multimodal-doc-agent.git
cd multimodal-doc-agent
conda create -n multimodal-doc-agent python=3.11 -y
conda activate multimodal-doc-agent
pip install -r requirements.txt
echo "GOOGLE_API_KEY=your_key_here" > .env
uvicorn src.api:app --reload
```

Open `index.html` in a browser (points to `http://127.0.0.1:8000`).

**Docker:**
```bash
docker build -t multimodal-doc-agent .
docker run -p 8000:8000 --env-file .env -v ${PWD}/data:/app/data multimodal-doc-agent
```

**Eval suite:**
```bash
python -m eval.ingest_test_document
python -m eval.run_eval
```

**Ingest a PDF from the command line:**
```bash
python -m src.ingestion.pipeline path/to/paper.pdf
```

## Structure

    src/
      ingestion/          # PDF -> text, figures, tables, embeddings
      agent.py            # agent construction, system prompt
      agent_tools.py      # tool definitions
      api.py              # FastAPI app
    eval/
      eval_questions.json
      run_eval.py
      fixtures/
    .github/workflows/
    index.html
    Dockerfile

## Known limitations

- No dedicated cross-document visual comparison tool; multi-document questions are handled via repeated single-document tool calls
- Table-crop gap thresholds tuned on layouts observed in testing; atypical spacing may require adjustment
- No authentication or per-user data isolation (single-user/trusted-shared scope)
- Ingestion runs synchronously; no background task queue

## Stack

Python · FastAPI · LangChain / LangGraph · Google Gemini · ChromaDB · sentence-transformers · PyMuPDF · Docker · GitHub Actions
