# AlphaGraph

AlphaGraph is a financial research and signal extraction system that uses LangGraph to build a multi-stage RAG (Retrieval-Augmented Generation) pipeline for analyzing financial documents and extracting trading signals.

## Features

- **Document Ingestion**: Load and chunk documents from multiple formats (TXT, MD, HTML, PDF)
- **Hybrid Search**: Combine FAISS vector search with BM25 lexical search
- **Multi-Stage Pipeline**: 
  - Query planning
  - Document retrieval
  - Context synthesis
  - Named Entity Recognition (NER) for financial entities
  - Sentiment-based signal extraction
- **Configurable**: Easy YAML-based configuration
- **Rich CLI**: Beautiful command-line interface with Rich library

## Installation

1. Clone the repository
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Set up your OpenAI API key (if using OpenAI models):

```bash
export OPENAI_API_KEY="your-api-key-here"
```

## Usage

### Indexing Documents

First, index your financial documents:

```bash
python -m alphagraph.main --mode index --data_dir ./data --index_dir ./index
```

This will:
- Load all documents from `./data`
- Chunk them with configurable size and overlap
- Build FAISS and BM25 indices
- Save indices to `./index`

### Querying

Query the indexed documents:

```bash
python -m alphagraph.main "What are the latest developments in AAPL?" --mode query --index_dir ./index
```

This will:
- Generate a search plan
- Retrieve relevant documents
- Synthesize a summary
- Extract financial entities (tickers, companies, metrics)
- Generate trading signals with sentiment scores

## Configuration

Edit `config.yaml` to customize:

```yaml
embedding_model: sentence-transformers/all-MiniLM-L6-v2
summary_model: openai # or "local" for extractive summary
openai_model: gpt-4o-mini
chunk:
  size: 1200
  overlap: 200
retrieve:
  top_k: 8
  bm25_boost: 0.2
alpha:
  min_sentiment_strength: 0.3
  ticker_whitelist: [] # Optional: filter to specific tickers
```

## Architecture

The pipeline consists of five nodes in a LangGraph:

1. **Planner**: Generates a search plan from the query
2. **Retriever**: Performs hybrid search (FAISS + BM25)
3. **Synthesizer**: Summarizes retrieved context using LLM or extractive method
4. **FinancialNER**: Extracts financial entities (tickers, metrics, dates) using OpenAI function calling
5. **SignalExtractor**: Generates trading signals with sentiment analysis

## Project Structure

```
AlphaGraph-refined/
├── alphagraph/
│   ├── __init__.py          # Package exports
│   ├── config.py            # Configuration management
│   ├── graph.py             # LangGraph pipeline definition
│   ├── ingest.py            # Document loading and chunking
│   ├── main.py              # CLI entry point
│   ├── node.py              # Pipeline node implementations
│   └── store.py             # Vector store with hybrid search
├── config.yaml              # Configuration file
└── requirements.txt         # Python dependencies
```

## Dependencies

- `langgraph`: For building the multi-stage pipeline
- `sentence-transformers`: For embeddings
- `faiss-cpu`: For vector search
- `rank-bm25`: For lexical search
- `transformers`: For sentiment analysis
- `openai`: For LLM summarization and NER
- `unstructured`: For document parsing
- `rich`: For beautiful CLI output

## License

MIT License

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.
