"""
Document Extractor — extracts text and images from PDF inspection/thermal reports.
Uses PyMuPDF (fitz) for fast, reliable extraction.
"""
from __future__ import annotations
import base64
import io
import re
from pathlib import Path
from typing import List, Tuple, Optional

import fitz  # PyMuPDF

from pipeline.models import DocumentExtraction, ExtractedImage


# ─────────────────────────────────────────────────────────────────────────────
def extract_document(pdf_path: str, doc_type: str = "inspection") -> DocumentExtraction:
    """
    Extract all text and embedded images from a PDF.

    Args:
        pdf_path: Absolute path to the PDF file.
        doc_type: "inspection" or "thermal"

    Returns:
        DocumentExtraction with full_text, images list, and metadata.
    """
    doc = fitz.open(pdf_path)
    full_text_parts: List[str] = []
    images: List[ExtractedImage] = []
    global_img_index = 0

    for page_num, page in enumerate(doc):
        # ── text ─────────────────────────────────────────────────────────────
        page_text = page.get_text("text")
        full_text_parts.append(page_text)

        # ── images ───────────────────────────────────────────────────────────
        img_list = page.get_images(full=True)
        for img_idx, img_info in enumerate(img_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
                img_bytes = base_image["image"]
                img_ext = base_image["ext"]  # jpeg, png, etc.
                width = base_image["width"]
                height = base_image["height"]

                # Skip tiny icons / artifacts (< 50px)
                if width < 50 or height < 50:
                    continue

                b64 = base64.b64encode(img_bytes).decode("utf-8")
                images.append(
                    ExtractedImage(
                        page_number=page_num + 1,
                        image_index_on_page=img_idx,
                        description=f"Photo {global_img_index + 1}",
                        base64_data=b64,
                        width=width,
                        height=height,
                        format=img_ext,
                    )
                )
                global_img_index += 1
            except Exception:
                pass  # Skip unreadable images silently

    full_text = "\n".join(full_text_parts)
    metadata = _extract_metadata(full_text, doc_type)

    doc.close()
    return DocumentExtraction(
        document_type=doc_type,
        total_pages=len(full_text_parts),
        full_text=full_text,
        images=images,
        metadata=metadata,
    )


# ─────────────────────────────────────────────────────────────────────────────
def _extract_metadata(text: str, doc_type: str) -> dict:
    """Pull key metadata fields from text via regex patterns."""
    meta: dict = {"doc_type": doc_type}

    patterns = {
        "inspection_date": r"Inspection Date[^:]*:\s*([^\n\r]+)",
        "inspected_by": r"Inspected By[^:]*:\s*([^\n\r]+)",
        "property_type": r"Property Type[^:]*:\s*([^\n\r]+)",
        "floors": r"Floors[^:]*:\s*([^\n\r]+)",
        "score": r"Score\s*\n?\s*([0-9.]+%)",
        "flagged_items": r"Flagged items\s*\n?\s*(\d+)",
        "customer_name": r"Customer Name\s*\n?\s*([^\n\r]+)",
        "address": r"Address[^:]*:\s*([^\n\r]+)",
    }

    for key, pattern in patterns.items():
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            meta[key] = m.group(1).strip()

    return meta


# ─────────────────────────────────────────────────────────────────────────────
def get_image_for_photo_ref(
    images: List[ExtractedImage], photo_number: int
) -> Optional[ExtractedImage]:
    """
    Map a 'Photo N' reference in the text to an actual extracted image.
    Uses 1-based indexing matching the document's photo numbering.
    """
    idx = photo_number - 1
    if 0 <= idx < len(images):
        return images[idx]
    return None


# ─────────────────────────────────────────────────────────────────────────────
def map_photos_to_sections(
    text: str, images: List[ExtractedImage]
) -> dict[str, List[ExtractedImage]]:
    """
    Parse inspection text for 'Photo N' references and group images
    by the section they appear under (Impacted Area 1–N, Checklist, etc.).
    Returns: {section_name: [ExtractedImage, ...]}
    """
    section_image_map: dict[str, List[ExtractedImage]] = {}
    current_section = "General"

    for line in text.split("\n"):
        stripped = line.strip()
        if re.match(r"Impacted Area \d+", stripped, re.IGNORECASE):
            current_section = stripped
        elif re.match(r"(Inspection Checklists?|Summary|Appendix)", stripped, re.IGNORECASE):
            current_section = stripped

        photo_refs = re.findall(r"Photo\s+(\d+)", stripped, re.IGNORECASE)
        for ref in photo_refs:
            img = get_image_for_photo_ref(images, int(ref))
            if img:
                section_image_map.setdefault(current_section, []).append(img)

    return section_image_map
