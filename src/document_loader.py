from pathlib import Path
from pymupdf4llm import to_markdown
from typing import List, Dict

def load_pdf_document(pdf_path: str, output_path: str) -> List[Dict]:
    """
    Load and extract text from PDF documents.
    """
    pages = to_markdown(
        pdf_path,
        page_chunks=True,     # keep page boundaries
        write_images=False    # ignore images for RAG
    )

    documents = []

    for page in pages:
        text = page.get("text", "").strip()

        # Skip empty / junk pages
        if not text:
            continue

        documents.append({
            "text": text,
            "metadata": {
                "source": pdf_path,
                "page_number": None,
                "chunk_index": None,     # filled in chunking.py
                "section_title": None    # can be inferred later
            }
        })

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:  
        for i, doc in enumerate(documents, start=1):
            f.write(f"## Page {i}\n\n")
            f.write(doc["text"])
            f.write("\n\n")

    return documents