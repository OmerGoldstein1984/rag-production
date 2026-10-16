# Telecom RAG System

A production-grade Retrieval-Augmented Generation (RAG) system for telecom documentation (eSIM, roaming specs, etc.) built with Python.

## Features

- **Document Text Extraction**: Using `pypdf` for PDFs and standard I/O for TXT files, with position tracking for citations
- **Smart Chunking**: Overlapping text chunks with configurable size and overlap
- **Vector Storage**: Local embeddings with Sentence Transformers and persistent ChromaDB
- **Semantic Retrieval**: Top-k similarity search with distance thresholding
- **Grounded Generation**: LLM-powered answers with citation formatting and "I don't know" fallback
- **REST API**: FastAPI endpoint with file upload and question answering
- **Evaluation Suite**: Golden dataset, Recall@k, MRR, and LLM-as-a-judge faithfulness metrics
- **CI/CD**: GitHub Actions workflow for automated evaluation
- **Enhanced Knowledge Base**: Four detailed telecom documents added:
  - *eSIM_document.txt*: Comprehensive eSIM technology, benefits, GSMA specifications
  - *Roaming_document.txt*: Roaming agreements, procedures, steered vs non-steered roaming
  - *LTE_5G_document.txt*: LTE/5G network architecture, specifications, and evolution
  - *SIM_Security_document.txt*: SIM authentication, AKA protocol, and security mechanisms

## Project Structure

```
Telecom RAG System/
├── src/
│   ├── load.py         # PDF loading with pypdf
│   ├── chunk.py        # Text chunking with overlap
│   ├── index.py        # Embeddings and ChromaDB storage
│   ├── retrieve.py     # Similarity search and retrieval
│   ├── answer.py       # Grounded generation with LLMs
│   └── app.py          # FastAPI web interface
├── evals/
│   ├── golden_dataset.py # Question-answer pairs for evaluation
│   ├── evaluate.py       # Evaluation runner with metrics
│   └── evaluation_results.json # Latest evaluation results
├── data/
│   └── uploads/        # Uploaded PDF and TXT documents
├── docs/
│   ├── RAG_Situations_Explained.md   # Educational document on RAG system behaviors
│   └── System_Explanations_For_Kids.md   # Simple explanations for interview preparation
├── static/             # Static web assets
├── templates/          # HTML templates
├── requirements.txt    # Python dependencies
└── README.md           # This file
```

## Installation

1. Clone the repository
2. Install Python dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Set up API keys for LLM providers (Anthropic or OpenAI):
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   # or
   export OPENAI_API_KEY="your-key-here"
   ```

## Usage

### Start the API Server

```bash
python src/app.py
```

The server will be available at `http://localhost:8000`.

### API Endpoints

- **GET `/`**: Web interface for uploading PDF/TXT files and asking questions
- **POST `/upload/`**: Upload and process PDF or TXT files
- **POST `/ask/`**: Ask a question and get an answer with citations
- **GET `/documents/`**: List uploaded documents
- **GET `/status/`**: Get system status

### Evaluation

Run the evaluation suite:

```bash
python evals/evaluate.py
```

Results will be saved to `evals/evaluation_results.json`.

## Evaluation Metrics

The system evaluates performance using:

- **Recall@k**: Proportion of expected sources found in top k retrieved results
- **MRR (Mean Reciprocal Rank)**: Average of reciprocal ranks of first relevant result
- **Faithfulness**: LLM-as-a-judge score measuring whether answers are grounded in retrieved context

## Telecommunications Focus

This RAG system is specifically designed for telecom documentation including:
- eSIM specifications (GSMA Remote SIM Provisioning)
- Roaming agreements and procedures
- LTE/5G network specifications
- SIM security and authentication protocols
- Carrier configuration requirements

## Documentation

In addition to the codebase, this system includes educational materials:
- **RAG_Situations_Explained.md**: Explains common RAG system behaviors including successful retrieval, failed retrieval, hallucination, citation issues, edge cases, and best practices
- **System_Explanations_For_Kids.md**: Simple explanations of each system component using analogies suitable for interview preparation (written at a 10-year-old's level)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a pull request

## License

MIT License - see LICENSE file for details.