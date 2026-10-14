from pathlib import Path

import pdfplumber


def read_pdf(path: Path) -> str:
    """Preserve pages; never silently skip a blank or scanned page."""
    pages = []
    try:
        with pdfplumber.open(path) as pdf:
            if len(pdf.pages) > 20:
                raise ValueError("PDF exceeds the introductory 20-page limit")
            for index, page in enumerate(pdf.pages, 1):
                text = (page.extract_text() or "").strip()
                if not text:
                    raise ValueError(f"Page {index} has no selectable text; OCR is outside v1")
                pages.append(f"[Page {index}]\n{text}")
    except ValueError:
        raise
    except Exception as exc:
        raise ValueError(f"Cannot read PDF {path.name}: {exc}") from exc
    if not pages:
        raise ValueError("PDF has no selectable text")
    return "\n\n".join(pages)
