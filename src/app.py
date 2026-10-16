from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import uvicorn
import os
import shutil
from pathlib import Path
import logging
from typing import List, Optional

# Import our RAG modules
from .load import load_pdf_document, load_txt_document
from .chunk import chunk_text
from .index import index_chunks
from .retrieve import retrieve_relevant_chunks
from .answer import generate_answer, format_answer_with_citations

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Telecom RAG API", version="1.0.0")

# Setup directories
UPLOAD_DIR = Path("./data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

# Mount static files and templates
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_root():
    """Serve the main upload page."""
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Telecom RAG System</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <title>Telecom RAG System</title>
        <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
        <style>
            body { padding-top: 4rem; }
            .upload-area { border: 2px dashed #ccc; border-radius: 5px; padding: 20px; text-align: center; }
            .upload-area.dragover { background-color: #f8f9fa; border-color: #007bff; }
            .result-box { margin-top: 20px; padding: 15px; border-radius: 5px; }
            .success { background-color: #d4edda; border: 1px solid #c3e6cb; }
            .error { background-color: #f8d7da; border: 1px solid #f5c6cb; }
            .info { background-color: #d1ecf1; border: 1px solid #bee5eb; }
            .citation { font-size: 0.9em; color: #666; margin-top: 10px; padding-left: 20px; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1 class="mb-4">Telecom RAG System</h1>
            <div class="row">
                <div class="col-md-8">
                    <div class="card">
                        <div class="card-header">
                            <h5>Upload Telecom Documents (PDF)</h5>
                        </div>
                        <div class="card-body">
                            <div id="upload-area" class="upload-area">
                                <p>Drag & drop PDF or TXT files here or click to select</p>
                                <input type="file" id="file-input" multiple accept=".pdf,.txt" style="display: none;">
                                <button type="button" class="btn btn-outline-primary mt-3" id="browse-btn">
                                    Select Files
                                </button>
                            </div>
                            <div id="upload-status" class="mt-3"></div>
                            <button type="button" class="btn btn-success mt-3" id="process-btn" disabled>
                                Process Documents
                            </button>
                        </div>
                    </div>

                    <div class="card mt-4">
                        <div class="card-header">
                            <h5>Ask Questions</h5>
                        </div>
                        <div class="card-body">
                            <div id="qa-section">
                                <div class="mb-3">
                                    <label for="question-input" class="form-label">Question:</label>
                                    <input type="text" class="form-control" id="question-input" placeholder="Ask about eSIM, roaming, etc.">
                                </div>
                                <button type="button" class="btn btn-primary" id="ask-btn">
                                    Ask Question
                                </button>
                            </div>
                            <div id="answer-section" class="mt-4"></div>
                        </div>
                    </div>
                </div>

                <div class="col-md-4">
                    <div class="card">
                        <div class="card-header">
                            <h5>System Status</h5>
                        </div>
                        <div class="card-body" id="system-status">
                            <p>Ready to process documents</p>
                        </div>
                    </div>

                    <div class="card mt-4">
                        <div class="card-header">
                            <h5>Uploaded Documents</h5>
                        </div>
                        <div class="card-body" id="document-list">
                            <p>No documents uploaded yet</p>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <script>
            // JavaScript for handling file upload and UI interactions
            const uploadArea = document.getElementById('upload-area');
            const fileInput = document.getElementById('file-input');
            const browseBtn = document.getElementById('browse-btn');
            const uploadStatus = document.getElementById('upload-status');
            const processBtn = document.getElementById('process-btn');
            const askBtn = document.getElementById('ask-btn');
            const questionInput = document.getElementById('question-input');
            const answerSection = document.getElementById('answer-section');
            const systemStatus = document.getElementById('system-status');
            const documentList = document.getElementById('document-list');

            let uploadedFiles = [];

            // Handle drag & drop
            uploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadArea.classList.add('dragover');
            });

            uploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
            });

            uploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadArea.classList.remove('dragover');
                const files = e.dataTransfer.files;
                handleFiles(files);
            });

            // Handle click to browse
            browseBtn.addEventListener('click', () => {
                fileInput.click();
            });

            fileInput.addEventListener('change', (e) => {
                handleFiles(fileInput.files);
            });

            function handleFiles(files) {
                uploadedFiles = Array.from(files);
                if (uploadedFiles.length > 0) {
                    uploadStatus.innerHTML = `
                        <div class="alert alert-info">
                            <strong>${uploadedFiles.length}</strong> file(s) selected:
                            <ul class="mb-0">
                                ${uploadedFiles.map(f => `<li>${f.name} (${Math.round(f.size / 1024)} KB)</li>`).join('')}
                            </ul>
                        </div>
                    `;
                    processBtn.disabled = false;
                } else {
                    uploadStatus.innerHTML = '';
                    processBtn.disabled = true;
                }
            }

            // Process documents
            processBtn.addEventListener('click', async () => {
                if (uploadedFiles.length === 0) return;

                processBtn.disabled = true;
                uploadStatus.innerHTML = '<div class="alert alert-warning">Processing documents...</div>';

                const formData = new FormData();
                uploadedFiles.forEach(file => {
                    formData.append('files', file);
                });

                try {
                    const response = await fetch('/upload/', {
                        method: 'POST',
                        body: formData
                    });

                    if (response.ok) {
                        const result = await response.json();
                        uploadStatus.innerHTML = `
                            <div class="alert alert-success">
                                Successfully processed ${result.processed_files} files!
                                <br>
                                Created ${result.total_chunks} text chunks.
                            </div>
                        `;
                        // Update document list and system status
                        await loadDocumentList();
                        await updateSystemStatus();
                    } else {
                        const error = await response.json();
                        uploadStatus.innerHTML = `
                            <div class="alert alert-danger">
                                Error: ${error.detail || 'Unknown error'}
                            </div>
                        `;
                    }
                } catch (error) {
                    uploadStatus.innerHTML = `
                        <div class="alert alert-danger">
                            Error: ${error.message || 'Network error'}
                        </div>
                    `;
                } finally {
                    processBtn.disabled = false;
                }
            });

            // Ask question
            askBtn.addEventListener('click', async () => {
                const question = questionInput.value.trim();
                if (!question) return;

                askBtn.disabled = true;
                answerSection.innerHTML = '<div class="alert alert-info">Generating answer...</div>';

                try {
                    const response = await fetch('/ask/', {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json'
                        },
                        body: JSON.stringify({ question: question })
                    });

                    if (response.ok) {
                        const result = await response.json();
                        answerSection.innerHTML = `
                            <div class="result-box success">
                                <h6>Answer:</h6>
                                <p>${result.answer}</p>
                                ${result.citations && result.citations.length > 0 ? `
                                    <h6>Citations:</h6>
                                    <div class="citations">
                                        ${result.citations.map(citation => `
                                            <div class="citation">
                                                <strong>${citation.source}</strong>, page ${citation.page}:
                                                "${citation.text_snippet}"
                                            </div>
                                        `).join('')}
                                    </div>
                                ` : ''}
                            </div>
                        `;
                    } else {
                        const error = await response.json();
                        answerSection.innerHTML = `
                            <div class="result-box error">
                                <h6>Error:</h6>
                                <p>${error.detail || 'Unknown error'}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    answerSection.innerHTML = `
                        <div class="result-box error">
                            <h6>Error:</h6>
                            <p>${error.message || 'Network error'}</p>
                        </div>
                    `;
                } finally {
                    askBtn.disabled = false;
                }
            });

            // Load document list on page load
            window.addEventListener('load', async () => {
                await loadDocumentList();
                await updateSystemStatus();
            });

            async function loadDocumentList() {
                try {
                    const response = await fetch('/documents/');
                    if (response.ok) {
                        const documents = await response.json();
                        if (documents.length === 0) {
                            documentList.innerHTML = '<p>No documents uploaded yet</p>';
                        } else {
                            documentList.innerHTML = `
                                <ul class="list-group mb-0">
                                    ${documents.map(doc => `
                                        <li class="list-group-item">
                                            <strong>${doc.name}</strong>
                                            <small class="text-muted">(${doc.pages} pages, ${doc.chunks} chunks)</small>
                                        </li>
                                    `).join('')}
                                </ul>
                            `;
                        }
                    }
                } catch (error) {
                    console.error('Failed to load document list:', error);
                    documentList.innerHTML = '<p>Error loading document list</p>';
                }
            }

            async function updateSystemStatus() {
                try {
                    const response = await fetch('/status/');
                    if (response.ok) {
                        const status = await response.json();
                        systemStatus.innerHTML = `
                            <p><strong>Documents:</strong> ${status.document_count}</p>
                            <p><strong>Text Chunks:</strong> ${status.chunk_count}</p>
                            <p><strong>Last Updated:</strong> ${new Date(status.last_updated).toLocaleTimeString()}</p>
                        `;
                    }
                } catch (error) {
                    console.error('Failed to update system status:', error);
                    systemStatus.innerHTML = '<p>Error loading system status</p>';
                }
            }
        </script>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

@app.post("/upload/")
async def upload_files(files: List[UploadFile] = File(...)):
    """Handle PDF and TXT file uploads and processing."""
    processed_files = 0
    total_chunks = 0

    for file in files:
        # Check file extension
        filename_lower = file.filename.lower()
        if not (filename_lower.endswith('.pdf') or filename_lower.endswith('.txt')):
            continue

        # Save uploaded file
        file_path = UPLOAD_DIR / file.filename
        try:
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            logger.error(f"Failed to save file {file.filename}: {e}")
            continue
        finally:
            file.file.close()

        # Process the file based on extension
        try:
            # 1. Load document
            if filename_lower.endswith('.pdf'):
                pages = load_pdf_document(file_path)
            else:  # .txt file
                pages = load_txt_document(file_path)

            if not pages:
                logger.warning(f"No content extracted from {file.filename}")
                continue

            # 2. Chunk text
            chunks = chunk_text(pages, chunk_size=500, chunk_overlap=50)
            if not chunks:
                logger.warning(f"No chunks created from {file.filename}")
                continue

            # 3. Index chunks
            index_chunks(chunks)

            processed_files += 1
            total_chunks += len(chunks)
            logger.info(f"Processed {file.filename}: {len(pages)} pages, {len(chunks)} chunks")

        except Exception as e:
            logger.error(f"Error processing {file.filename}: {e}")

    return {
        "processed_files": processed_files,
        "total_chunks": total_chunks
    }

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    """Handle question answering."""
    if not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        # 1. Retrieve relevant chunks
        relevant_chunks = retrieve_relevant_chunks(
            query=question,
            n_results=5,
            distance_threshold=0.7
        )

        # 2. Generate answer
        answer_result = generate_answer(
            query=question,
            context_chunks=relevant_chunks,
            model_provider="anthropic"  # or make configurable
        )

        # 3. Format with citations
        formatted_result = format_answer_with_citations(answer_result, relevant_chunks)

        return JSONResponse(content=formatted_result)

    except Exception as e:
        logger.error(f"Error answering question: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents/")
async def list_documents():
    """List uploaded documents with metadata."""
    documents = []
    try:
        # List both PDF and TXT files
        for file_path in UPLOAD_DIR.glob("*"):
            if file_path.is_file() and (file_path.suffix.lower() == '.pdf' or file_path.suffix.lower() == '.txt'):
                # Get basic info (in a real system, you'd store this in a database)
                stat = file_path.stat()
                documents.append({
                    "name": file_path.name,
                    "size": stat.st_size,
                    "modified": stat.st_mtime
                })
        # Sort by modification time descending
        documents.sort(key=lambda x: x["modified"], reverse=True)
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
    return documents

@app.get("/status/")
async def get_system_status():
    """Get overall system status."""
    # In a real system, you'd query ChromaDB for counts
    # For now, we'll approximate
    try:
        # Count PDF and TXT files
        pdf_count = len(list(UPLOAD_DIR.glob("*.pdf")))
        txt_count = len(list(UPLOAD_DIR.glob("*.txt")))
        document_count = pdf_count + txt_count
        # For chunk count, we'd need to query ChromaDB - simplified here
        chunk_count = document_count * 10  # placeholder
        # Get latest modification time among all documents
        all_files = list(UPLOAD_DIR.glob("*.pdf")) + list(UPLOAD_DIR.glob("*.txt"))
        last_updated = max([f.stat().st_mtime for f in all_files], default=0)
        return {
            "document_count": document_count,
            "chunk_count": chunk_count,
            "last_updated": last_updated
        }
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return {
            "document_count": 0,
            "chunk_count": 0,
            "last_updated": 0
        }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)