# Local Python RAG — from scratch

A minimal, real, **local** retrieval-augmented generation project. It implements document ingestion, overlapping chunks, embeddings, persistent local vectors, cosine-similarity search, context assembly, and LLM answering without LangChain, LlamaIndex, FAISS, Chroma, or API keys.

## Requirements

- Python 3.11 or newer
- Ollama installed and running locally: https://ollama.com/download
- Enough free RAM/storage to run two local models

Download the models (one time):

```bash
ollama pull embeddinggemma
ollama pull llama3.2
```

If Ollama is not already running, start it in another terminal with `ollama serve` (some desktop setups start the service automatically).

## Quick start (macOS / Linux)

```bash
git clone https://github.com/sinariazi/local_rag.git
cd local_rag
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python rag.py index --docs docs
python rag.py search "Who owns Project Atlas?"
python rag.py ask "Who owns Project Atlas?"
```

Windows PowerShell activation: `.venv\Scripts\Activate.ps1` (or use `.venv\Scripts\python.exe` to avoid activation).

## Add your own documents

Copy `.txt`, `.md`, or searchable (text-based) `.pdf` files into `docs/`, then rerun:

```bash
python rag.py index --docs docs
```

This rebuilds the index from all supported documents in the directory. The data is stored in `.rag_index/` as `metadata.json` (text, source, page) and `embeddings.npy` (numerical vectors).

Index a different directory:

```bash
python rag.py index --docs "/path/to/my/files" --index "./my_index"
python rag.py ask "Summarize the requirements" --index "./my_index" --top-k 5
```

Tune the word-based chunking:

```bash
python rag.py index --docs docs --chunk-size 180 --overlap 30
```

Use an alternative local Ollama chat model:

```bash
ollama pull gemma3:4b
python rag.py ask "Explain Atlas" --chat-model gemma3:4b
```

Changing the **embedding** model requires rebuilding the index; the query always uses the embedding model recorded in index metadata.

## Architecture

```text
.txt / .md / .pdf
    -> read_document_sections()
    -> split_into_chunks()
    -> embed_texts()                 POST http://localhost:11434/api/embed
    -> embeddings.npy + metadata.json

question
    -> embed_texts()                 POST http://localhost:11434/api/embed
    -> rank_documents()             normalized dot product = cosine similarity
    -> top-k passages + source labels
    -> generate_answer()            POST http://localhost:11434/api/chat
    -> answer + retrieved sources
```

## Testing

```bash
python -m unittest discover -s tests -v
```

These tests verify chunking, cosine ranking, index persistence, and source labeling without requiring Ollama. A real end-to-end test of model output requires locally running Ollama with the models downloaded.

## Important limits

- This is a transparent learning implementation, not a production-ready RAG service.
- Whole-index similarity search is linear in the number of chunks; for millions of chunks, consider an ANN index such as FAISS/HNSW.
- Chunking is **word-based**, not token-aware or semantic, so some facts can be split across chunks.
- Image-only/scanned PDFs require OCR; `pypdf` cannot recognize text in images.
- Retrieval similarity scores are **not** calibrated probabilities that an answer is true.
- Cited answers can still be inaccurate. Review supporting chunks for important decisions.
- Text inside documents can attempt prompt injection. The prompt asks the model to treat it as untrusted, but that is not a full security boundary.
- Index contents are stored in **unencrypted local files**. Keep sensitive documents and `.rag_index/` private, and don't expose your Ollama port to the public Internet. For sensitive documents, use a directory outside the Git repository to avoid accidentally committing them.
- All API calls in this project go to `http://localhost:11434` by default; `OLLAMA_BASE_URL` can override that if intentionally configured. Model downloads do require internet access initially.

## Learning extensions

1. Build BM25 + vector **hybrid retrieval** for terms, dates, and exact identifiers.
2. Add an ANN index and incremental indexing keyed by file hashes.
3. Add a reranking stage and evaluate answer quality on a small dataset of known questions.
4. Replace basic word chunks with heading-aware or token-aware chunks.
5. Expose `/index`, `/search`, and `/ask` with FastAPI, then optionally add a frontend.

## API references

- https://docs.ollama.com/capabilities/embeddings
- https://docs.ollama.com/api/embed
- https://docs.ollama.com/api/chat
