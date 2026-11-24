# Project Atlas — Example Knowledge Base

Project Atlas is a fictional internal documentation assistant used for testing a local RAG system.
It supports documents in Markdown, plain text, and text-based PDF format.
Its primary goal is to answer questions from locally indexed files and show the passages used as evidence.

## Operations

The team reviews Atlas search quality every Friday. The project owner is the Documentation Team.
Requests to add new document formats are tracked in a development backlog.

## Privacy and limits

Atlas uses locally installed Ollama models for both embedding and answer generation.
The demo does not require an API key or any cloud account.
Scanned image-only PDFs are not searchable unless OCR is added in a future version.
TXT, MD, and PDF files are reindexed using the `index` command after they change.
