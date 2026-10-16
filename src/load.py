from pathlib import Path
from typing import List, Dict, Any
import pypdf
import logging

logger = logging.getLogger(__name__)

def load_pdf_document(pdf_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a PDF file and extract text page-by-page with metadata.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        List of dictionaries, each representing one page:
        [
            {
                "text": str,                   # Raw text content of the page
                "metadata": {
                    "page": int,               # 0-indexed page number
                    "source": str,             # Filename (e.g., "esim-spec.pdf")
                    "total_pages": int,        # Total pages in PDF
                    "file_path": str,          # Absolute path to PDF
                }
            },
            ...
        ]
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.is_file():
        logger.error(f"PDF file not found: {pdf_path}")
        return []

    try:
        reader = pypdf.PdfReader(str(pdf_path))
    except pypdf.errors.PdfReadError as e:
        logger.error(f"Failed to read PDF (possibly encrypted or corrupt): {pdf_path} - {e}")
        return []
    except Exception as e:
        logger.error(f"Unexpected error opening PDF {pdf_path}: {e}")
        return []

    total_pages = len(reader.pages)
    pages_data: List[Dict[str, Any]] = []

    for page_num, page in enumerate(reader.pages):
        try:
            text = page.extract_text() or ""
        except Exception as e:
            logger.warning(f"Failed to extract text from page {page_num} in {pdf_path}: {e}")
            text = ""

        if not text.strip():
            logger.debug(f"Page {page_num} in {pdf_path} extracted empty text.")

        pages_data.append({
            "text": text,
            "metadata": {
                "page": page_num,          # zero-based
                "source": pdf_path.name,
                "total_pages": total_pages,
                "file_path": str(pdf_path.resolve()),
            }
        })

    logger.info(f"Loaded PDF '{pdf_path.name}' with {total_pages} pages.")
    return pages_data


def load_txt_document(txt_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load a TXT file and treat it as a single "page" for consistency with PDF loading.

    Args:
        txt_path: Path to the TXT file.

    Returns:
        List with one dictionary representing the file content:
        [{
            "text": str,                   # Full text content
            "metadata": {
                "page": int,               # Always 0 for TXT files
                "source": str,             # Filename
                "total_pages": int,        # Always 1 for TXT files
                "file_path": str,          # Absolute path
            }
        }]
    """
    txt_path = Path(txt_path)
    if not txt_path.is_file():
        logger.error(f"TXT file not found: {txt_path}")
        return []

    try:
        with open(txt_path, 'r', encoding='utf-8') as f:
            text = f.read()
    except Exception as e:
        logger.error(f"Failed to read TXT file {txt_path}: {e}")
        return []

    # Treat TXT file as a single page
    pages_data = [{
        "text": text,
        "metadata": {
            "page": 0,                   # zero-based, only one "page"
            "source": txt_path.name,
            "total_pages": 1,            # TXT files have one logical page
            "file_path": str(txt_path.resolve()),
        }
    }]

    logger.info(f"Loaded TXT file '{txt_path.name}'.")
    return pages_data
