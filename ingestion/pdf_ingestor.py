"""
PDF ingestor using PyMuPDF.
Returns a dict: {title, text, metadata, warnings}
"""
import re
from pathlib import Path
import fitz  # PyMuPDF


def ingest_pdf(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Not found: {path}")

    warnings, pages_text = [], []

    doc = fitz.open(str(path))
    if doc.is_encrypted:
        raise PermissionError(f"PDF is password-protected: {path.name}")

    for i, page in enumerate(doc):
        blocks = page.get_text("blocks", sort=True)
        text = "\n".join(b[4].strip() for b in blocks if b[4].strip())
        if text:
            pages_text.append(text)
        else:
            warnings.append(f"Page {i+1} has no text (image-only?).")

    if not pages_text:
        warnings.append("No text extracted — PDF may need OCR.")

    full_text = _clean("\n\n".join(pages_text))
    title = doc.metadata.get("title", "").strip() or path.stem

    return dict(
        title=title,
        text=full_text,
        source_ref=str(path),
        metadata={"pages": doc.page_count, "file": path.name},
        warnings=warnings,
    )


def _clean(text: str) -> str:
    for lig, rep in {"ﬁ":"fi","ﬂ":"fl","ﬀ":"ff"}.items():
        text = text.replace(lig, rep)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.replace("\u00ad", "").strip()