# PDF RAG Chatbot

A Streamlit chatbot that answers plain-language questions about insurance policy PDFs using retrieval-augmented generation (RAG). It retrieves relevant passages from your documents and uses OpenAI to generate an answer grounded in those passages.

The chatbot explains document text; it does not provide legal or financial advice.

## How it works

1. `ingest.py` loads PDF files from `docs/`, splits their text into overlapping chunks, creates embeddings with OpenAI's `text-embedding-3-small` model, and saves the vectors in a local Chroma database at `chroma_db/`.
2. `app.py` serves a Streamlit chat interface.
3. For each question, `rag_chain.py` retrieves the four most relevant chunks from Chroma and sends them with the question to `gpt-4o-mini`. The response is displayed in the chat.

## Requirements

- Python 3
- An OpenAI API key with access to the embedding and chat models

## Setup

From the project directory, create and activate a virtual environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Create a `.env` file in the project root and add your OpenAI API key:

```dotenv
OPENAI_API_KEY=your_openai_api_key
```

Keep your API key private. The `.env` file is excluded from version control.

## Add documents and build the index

Place the PDFs you want to query in the `docs/` directory, then run:

```bash
python ingest.py
```

This creates or updates the local vector database in `chroma_db/`. The database is excluded from version control, so build the index after cloning the repository or adding documents. Running ingestion again adds the documents to the existing index; it does not clear old entries first.

## Run the chatbot

With your virtual environment activated and the index built, start the app:

```bash
streamlit run app.py
```

Open the local URL printed by Streamlit, usually <http://localhost:8501>.

## Project files

- `app.py` — Streamlit chat interface.
- `rag_chain.py` — Document retrieval and answer-generation pipeline.
- `ingest.py` — PDF loading, chunking, embedding, and indexing.
- `docs/` — PDF documents to index.
- `chroma_db/` — Local vector database created by ingestion; not tracked by Git.
- `requirements.txt` — Python dependencies.
