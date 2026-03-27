# AI Research Assistant

An AI-powered research assistant built with **Retrieval-Augmented Generation (RAG)** that lets you have intelligent conversations about any research paper. Upload one or more PDFs, ask questions, and get cited answers — all through a clean streaming chat interface.

---

## Features

- **Multi-PDF Upload** — Upload multiple research papers at once; previously ingested papers load automatically on startup
- **RAG Pipeline** — Retrieves the most relevant context chunks before generating an answer
- **Streaming Responses** — Answers stream token-by-token for a responsive chat experience
- **Query Classifier** — Detects greetings and small talk to skip unnecessary retrieval and API cost
- **Source Citations** — Every answer includes deduplicated, page-level citations from the retrieved chunks
- **Duplicate Detection** — Re-uploading a paper that's already in the database is safely skipped
- **Conversational Memory** — Maintains chat history across turns within a session
- **Query Metrics** — Logs every query with response time, query type, and source count to `data/metrics.json`
- **Gradio UI** — Clean, browser-based interface built with `gr.Blocks()`

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                         │
│                                                                     │
│  PDF Upload ──► PyPDFLoader (+ pypdf fallback)                      │
│                         │                                           │
│                         ▼                                           │
│              RecursiveCharacterTextSplitter                         │
│              (chunk_size=1500, overlap=300)                         │
│                         │                                           │
│                         ▼                                           │
│                  OpenAIEmbeddings                                   │
│              (text-embedding-3-small, 1536 dims)                    │
│                         │                                           │
│                         ▼                                           │
│                ChromaDB (SQLite backend)                            │
│                  ./data/vector_db/                                  │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│                         INFERENCE PIPELINE                          │
│                                                                     │
│  User Query                                                         │
│      │                                                              │
│      ▼                                                              │
│  ┌──────────────────────────────────────┐                           │
│  │   Layer 1: Query Classifier          │                           │
│  │   classify_query(question)           │                           │
│  │   • greeting   → skip RAG            │                           │
│  │   • small_talk → skip RAG            │                           │
│  │   • knowledge_query → trigger RAG    │                           │
│  └──────────────────────────────────────┘                           │
│           │                   │                                     │
│     (greeting /         (knowledge_query)                           │
│     small_talk)               │                                     │
│           │                   ▼                                     │
│           │    ┌──────────────────────────────────┐                 │
│           │    │   Layer 2: RAG Trigger            │                 │
│           │    │   get_retriever().invoke()        │                 │
│           │    │   Fresh Chroma connection         │                 │
│           │    │   Top-k=3 similarity search       │                 │
│           │    └──────────────────────────────────┘                 │
│           │                   │                                     │
│           │                   ▼                                     │
│           │    ┌──────────────────────────────────┐                 │
│           │    │   Layer 3: Response Generator    │                 │
│           │    │   SYSTEM_PROMPT + context +      │                 │
│           │    │   history → ChatOpenAI stream    │                 │
│           │    │   (gpt-4o-mini)                  │                 │
│           │    └──────────────────────────────────┘                 │
│           │                   │                                     │
│           └──────────┬────────┘                                     │
│                      ▼                                              │
│           ┌──────────────────────────────────┐                      │
│           │   Layer 4: Citation Formatter    │                      │
│           │   Deduplicated sources           │                      │
│           │   Max 2 citations per response   │                      │
│           │   Suppressed for off-topic Qs    │                      │
│           └──────────────────────────────────┘                      │
│                      │                                              │
│                      ▼                                              │
│                 Streamed Response + Citations                       │
└─────────────────────────────────────────────────────────────────────┘
```

### Four-Layer Design

| Layer | Component | Responsibility |
|---|---|---|
| **1. Query Classifier** | `classify_query()` in `answer.py` | Detects greetings and small talk to avoid unnecessary API calls |
| **2. RAG Trigger** | `get_retriever()` in `answer.py` | Creates a fresh Chroma connection per query; top-3 cosine similarity search |
| **3. Response Generator** | `ChatOpenAI` + prompt templates | Streams grounded answers using retrieved context and conversation history |
| **4. Citation Formatter** | `format_sources()` in `ui.py` | Cleans, deduplicates, and renders source citations; suppressed for off-topic answers |

---

## Project Structure

```
ai-research-assistant/
├── main.py              # Entry point — validates env, checks DB, launches UI
├── ingest.py            # PDF ingestion: load → chunk → embed → store in ChromaDB
├── answer.py            # Core RAG pipeline: classifier → retriever → LLM stream
├── ui.py                # Gradio Blocks UI: upload handler, chat handler, pre-load logic
├── metrics.py           # Query metrics: log to JSON, print summary report
├── test_api.py          # Utility: verify OpenAI API key and connectivity
├── test_classifier.py   # Utility: test query classifier logic
├── pyproject.toml       # Project dependencies (managed by uv)
├── .env                 # API keys (git-ignored)
└── data/
    ├── metrics.json     # Query log (auto-created on first query)
    └── vector_db/       # ChromaDB persistent store
        ├── chroma.sqlite3
        └── <uuid>/      # Binary vector index files
```

---

## Tech Stack

| Category | Technology |
|---|---|
| **Language** | Python 3.12 |
| **Package Manager** | [uv](https://docs.astral.sh/uv/) |
| **LLM** | `gpt-4o-mini` via `langchain-openai` |
| **Embeddings** | `text-embedding-3-small` (1536 dimensions) |
| **Vector Store** | [ChromaDB](https://www.trychroma.com/) with SQLite backend |
| **PDF Loading** | `PyPDFLoader` (`langchain-community`) + `pypdf` fallback |
| **Text Splitting** | `RecursiveCharacterTextSplitter` (chunk: 1500, overlap: 300) |
| **UI** | [Gradio](https://www.gradio.app/) 6.x — `gr.Blocks()` |
| **Env Management** | `python-dotenv` |

---

## Setup

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) installed
- An OpenAI API key

### 1. Clone the repository

```bash
git clone https://github.com/faniyi-akinbobola/Ai-research-assistant.git
cd ai-research-assistant
```

### 2. Install dependencies

```bash
uv sync
```

### 3. Configure environment variables

Create a `.env` file in the project root:

```env
OPENAI_API_KEY=your-openai-api-key-here
```

### 4. Launch the assistant

```bash
uv run python main.py
```

This validates your environment, confirms the vector database directory exists, and opens the Gradio UI in your browser at `http://127.0.0.1:7860`.

### 5. Upload a paper and start asking

Use the file upload area in the UI to upload one or more PDF research papers. Once ingested, the chat input unlocks and you can ask questions about the papers. Papers are persisted in the vector database — they will be available automatically the next time you start the server.

---

## Usage

**Example questions for any uploaded paper:**

- *"What is the main contribution of this paper?"*
- *"Explain the proposed architecture"*
- *"What datasets were used for evaluation?"*
- *"What were the key results and metrics?"*
- *"How does this compare to prior work?"*

**Example response:**

```
The Transformer relies entirely on attention mechanisms, dispensing with
recurrence and convolutions. The encoder maps the input sequence to
continuous representations, and the decoder generates the output
sequence auto-regressively (Page 2).

---

**Sources:**

- Attention Is All You Need — Page 2
- Attention Is All You Need — Page 3
```

Off-topic questions (not covered by the uploaded papers) receive:

```
This information is not covered in the uploaded papers.
```

No sources are shown in that case.

---

## Configuration

Key settings in `answer.py`:

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `gpt-4o-mini` | OpenAI chat model |
| `search_kwargs["k"]` | `3` | Number of chunks retrieved per query |
| `temperature` | `0.2` | LLM temperature (lower = more factual) |
| `max_retries` | `3` | Retry attempts on API failures |
| `request_timeout` | `30` | API call timeout in seconds |

Key settings in `ingest.py`:

| Variable | Default | Description |
|---|---|---|
| `chunk_size` | `1500` | Characters per text chunk |
| `chunk_overlap` | `300` | Overlap between consecutive chunks |

---

## Metrics

Every query is logged to `data/metrics.json`. To view a summary:

```bash
uv run python metrics.py
```

Output includes total queries, breakdown by query type (greeting / small_talk / knowledge_query), average/min/max response times, and error count.

---

## Testing Utilities

**Verify your API key and connectivity:**

```bash
uv run python test_api.py
```

**Test the query classifier:**

```bash
uv run python test_classifier.py
```

---

## How RAG Works

1. **Ingestion** — The PDF is split into 1,500-character chunks with 300-character overlap to preserve context at boundaries. Each chunk is embedded using `text-embedding-3-small` and stored in ChromaDB. A Python-side duplicate check prevents re-ingesting the same file.

2. **Query classification** — Before hitting the vector store, each message is classified. Greetings and small talk are handled with a lightweight conversational prompt — no retrieval, no embeddings cost.

3. **Retrieval** — For knowledge queries, a fresh Chroma connection is created (ensuring newly uploaded papers are always visible), and the question is compared against all stored chunks via cosine similarity. The top 3 chunks are returned.

4. **Augmentation** — The retrieved chunks are assembled into a context block and injected into the system prompt alongside conversation history.

5. **Generation** — The LLM streams a grounded response constrained to the provided context. If the answer cannot be found, it responds with exactly: *"This information is not covered in the uploaded papers."*

6. **Citation** — Source metadata (filename + page number) is deduplicated and formatted into a citation block appended to the response. Citations are suppressed when the answer is out-of-scope.

---

## License

MIT
