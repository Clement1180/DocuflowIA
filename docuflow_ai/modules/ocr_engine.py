"""Module OCR basé sur Tesseract pour extraction de texte depuis images et PDFs."""

import time
from pathlib import Path
from typing import Optional

import pytesseract
from PIL import Image
from pdf2image import convert_from_path

from docuflow_ai.config import TESSERACT_CMD, TESSERACT_LANG, POPPLER_PATH


pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD


def ocr_image(image_path: str, lang: str = TESSERACT_LANG) -> dict:
    """Extrait le texte d'une image via Tesseract."""
    start = time.time()
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    data = pytesseract.image_to_data(img, lang=lang, output_type=pytesseract.Output.DICT)
    elapsed = time.time() - start

    confidences = [int(c) for c in data["conf"] if int(c) > 0]
    avg_conf = sum(confidences) / len(confidences) if confidences else 0.0

    return {
        "text": text.strip(),
        "confidence": round(avg_conf, 2),
        "word_count": len([w for w in data["text"] if w.strip()]),
        "processing_time": round(elapsed, 3),
        "source": str(image_path),
    }


def ocr_pdf(pdf_path: str, lang: str = TESSERACT_LANG, dpi: int = 300) -> dict:
    """Extrait le texte d'un PDF en le convertissant en images page par page."""
    start = time.time()
    pdf_path_norm = str(Path(pdf_path).resolve())
    poppler_norm = str(Path(POPPLER_PATH).resolve())
    pages = convert_from_path(pdf_path_norm, dpi=dpi, poppler_path=poppler_norm)
    all_text = []
    total_conf = []
    total_words = 0

    for i, page_img in enumerate(pages):
        text = pytesseract.image_to_string(page_img, lang=lang)
        data = pytesseract.image_to_data(page_img, lang=lang, output_type=pytesseract.Output.DICT)
        all_text.append(text.strip())

        confidences = [int(c) for c in data["conf"] if int(c) > 0]
        total_conf.extend(confidences)
        total_words += len([w for w in data["text"] if w.strip()])

    elapsed = time.time() - start
    avg_conf = sum(total_conf) / len(total_conf) if total_conf else 0.0

    return {
        "text": "\n\n Page -\n\n".join(all_text),
        "confidence": round(avg_conf, 2),
        "word_count": total_words,
        "page_count": len(pages),
        "processing_time": round(elapsed, 3),
        "source": str(pdf_path),
    }


def process_document(file_path: str, lang: str = TESSERACT_LANG) -> dict:
    """Point d'entrée unifié : détecte le type et applique l'OCR approprié."""
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        return ocr_pdf(file_path, lang=lang)
    elif suffix in (".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"):
        return ocr_image(file_path, lang=lang)
    else:
        raise ValueError(f"Format non supporté : {suffix}")
