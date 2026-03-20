# 🔬 AI Research Assistant

An AI-powered research assistant built with **Retrieval-Augmented Generation (RAG)** that lets you have intelligent conversations about academic papers. Ask questions, get cited answers, and explore research — all through a clean chat interface.

> Currently configured to answer questions about the ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) paper by Vaswani et al. (2017).

---

## ✨ Features

- 📄 **PDF Ingestion** — Loads and chunks research papers into a persistent vector database
- 🧠 **RAG Pipeline** — Retrieves the most relevant context before generating an answer
- 🏷️ **Query Classifier** — Detects greetings and small talk to skip unnecessary retrieval
- 📎 **Source Citations** — Every answer includes deduplicated, page-level citations
- 💬 **Conversational Memory** — Maintains chat history across turns
- 🖥️ **Gradio UI** — Clean, browser-based chat interface

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          INGESTION PIPELINE                         │
│                                                                     │
│  PDF File ──► PyPDFLoader ──► RecursiveCharacterTextSplitter        │
│                                        │                            │
│                                        ▼                            │
│                            OpenAIEmbeddings                         │
│                         (text-embedding-3-small)                    │
│                                        │                            │
│                                        ▼                            │
│                              ChromaDB (SQLite)                      │
│                           ./data/vector_db/                         │
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
│     (greeting /          (knowledge_query)                          │
│     small_talk)               │                                     │
│           │                   ▼                                     │
│           │    ┌──────────────────────────────────┐                 │
│           │    │   Layer 2: RAG Trigger            │                 │
│           │    │   retriever.invoke(question)      │                 │
│           │    │   Top-k=3 similarity search       │                 │
│           │    │   in ChromaDB                     │                 │
│           │    └──────────────────────────────────┘                 │
│           │                   │                                     │
│           │                   ▼                                     │
│           │    ┌──────────────────────────────────┐                 │
│           │    │   Layer 3: Response Generator    │                 │
│           │    │   SYSTEM_PROMPT_TEMPLATE +       │                 │
│           │    │   retrieved context + history    │                 │
│           │    │   → ChatOpenAI (GPT)             │                 │
│           │    └──────────────────────────────────┘                 │
│           │                   │                                     │
│           └──────────┬────────┘                                     │
│                      ▼                                              │
│           ┌──────────────────────────────────┐                      │
│           │   Layer 4: Citation Formatter    │                      │
│           │   Deduplicated sources           │                      │
│           │   Max 2 citations per response   │                      │
│           │   Clean filenames, 1-indexed     │                      │
│           │   page numbers                   │                      │
│           └──────────────────────────────────┘                      │
│                      │                                              │
│                      ▼                                              │
│                 Final Response                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Four-Layer Design

| Layer                     | Component                           | Responsibility                                                                      |
| ------------------------- | ----------------------------------- | ----------------------------------------------------------------------------------- |
| **1. Query Classifier**   | `classify_query()` in `answer.py`   | Detects greetings and small talk to avoid unnecessary API calls                     |
| **2. RAG Trigger**        | `retriever.invoke()` in `answer.py` | Performs top-3 cosine similarity search against ChromaDB for knowledge queries only |
| **3. Response Generator** | `ChatOpenAI` + prompt templates     | Generates grounded answers using retrieved context and conversation history         |
| **4. Citation Formatter** | `format_response()` in `ui.py`      | Cleans, deduplicates, and renders source citations with page numbers                |

---

## 🗂️ Project Structure

```
ai-research-assistant/
├── main.py                 # Entry point — validates setup, launches UI
├── ingest.py               # PDF ingestion pipeline → ChromaDB
├── answer.py               # Core RAG pipeline (classifier + retriever + LLM)
├── ui.py                   # Gradio chat interface
├── test_api.py             # Utility: verify OpenAI API key & connectivity
├── test_classifier.py      # Utility: test query classifier logic
├── pyproject.toml          # Project dependencies (managed by uv)
├── .env                    # API keys (git-ignored)
└── data/
    ├── raw/
    │   └── attention-is-all-you-need.pdf   # Source research paper
    └── vector_db/
        ├── chroma.sqlite3                  # ChromaDB metadata store
        └── <uuid>/                         # Binary vector files
```

---

## 🛠️ Tech Stack

| Category            | Technology                                                   |
| ------------------- | ------------------------------------------------------------ |
| **Language**        | Python 3.12                                                  |
| **Package Manager** | [uv](https://docs.astral.sh/uv/)                             |
| **LLM**             | OpenAI GPT (via `langchain-openai`)                          |
| **Embeddings**      | `text-embedding-3-small` (1536 dimensions)                   |
| **Vector Store**    | [ChromaDB](https://www.trychroma.com/) with SQLite backend   |
| **PDF Loading**     | `PyPDFLoader` (`langchain-community`)                        |
| **Text Splitting**  | `RecursiveCharacterTextSplitter` (chunk: 1000, overlap: 200) |
| **UI**              | [Gradio](https://www.gradio.app/) 6.x                        |
| **Env Management**  | `python-dotenv`                                              |

---

## ⚙️ Setup

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

### 4. Add your research paper

Place the PDF you want to query in `data/raw/`. The default path expected is:

```
data/raw/attention-is-all-you-need.pdf
```

To use a different paper, update the `PDF_PATH` variable at the top of `ingest.py`.

### 5. Ingest the PDF

This step processes the PDF, creates embeddings, and populates the vector database. Run it once — or re-run whenever you change the source PDF:

```bash
uv run python ingest.py
```

Expected output:

```
🗑️  Clearing existing vector database...
Loaded 15 pages
Created 43 chunks
Creating embeddings...
Creating vector store (this may take a few minutes)...
Successfully processed 43 chunks
Vector store created with 43 documents
```

### 6. Launch the assistant

```bash
uv run python main.py
```

This validates your environment, confirms the vector database exists, and opens the Gradio UI in your browser.

---

## 💬 Usage

Once the app is running, open the browser tab Gradio launches (typically `http://127.0.0.1:7860`).

**Example questions:**

- _"What is the Transformer architecture?"_
- _"Explain the self-attention mechanism"_
- _"How does multi-head attention work?"_
- _"What were the key results of this paper?"_
- _"Why did the authors move away from recurrent networks?"_

**Example response:**

```
The Transformer uses stacked self-attention and point-wise, fully connected layers
for both the encoder and decoder...

---

**Sources:**

- 📄 Attention Is All You Need — Page 3
- 📄 Attention Is All You Need — Page 5
```

---

## 🔧 Configuration

Key settings in `answer.py`:

| Variable             | Default       | Description                            |
| -------------------- | ------------- | -------------------------------------- |
| `MODEL`              | `gpt-4o-mini` | OpenAI chat model                      |
| `search_kwargs["k"]` | `3`           | Number of chunks retrieved per query   |
| `temperature`        | `0.2`         | LLM temperature (lower = more factual) |
| `max_retries`        | `3`           | Retry attempts on API failures         |
| `request_timeout`    | `30`          | API call timeout in seconds            |

Key settings in `ingest.py`:

| Variable        | Default                                  | Description                        |
| --------------- | ---------------------------------------- | ---------------------------------- |
| `chunk_size`    | `1000`                                   | Characters per text chunk          |
| `chunk_overlap` | `200`                                    | Overlap between consecutive chunks |
| `PDF_PATH`      | `data/raw/attention-is-all-you-need.pdf` | Path to source PDF                 |

---

## 🧪 Testing Utilities

**Verify your API key and connectivity:**

```bash
uv run python test_api.py
```

**Test the query classifier:**

```bash
uv run python test_classifier.py
```

---

## 📖 How RAG Works Here

1. **Ingestion** _(one-time)_: The PDF is split into 1,000-character chunks with 200-character overlap to preserve context at boundaries. Each chunk is embedded using `text-embedding-3-small` and stored in ChromaDB.

2. **Query classification**: Before hitting the vector store, each user message is classified. Greetings and small talk are handled with a lightweight conversational prompt — no retrieval needed.

3. **Retrieval**: For knowledge queries, the user's question is embedded and compared against all stored chunks via cosine similarity. The top 3 most relevant chunks are returned.

4. **Augmentation**: The retrieved chunks are assembled into a context block and injected into the system prompt alongside conversation history.

5. **Generation**: The LLM generates a grounded response constrained to the provided context. It is instructed to cite sources and acknowledge when information is not available in the paper.

6. **Citation**: Source metadata (filename + page number) from the retrieved chunks is deduplicated and formatted into a clean citation block appended to each response.

---

## 📄 License

MIT
