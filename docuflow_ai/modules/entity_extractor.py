"""Extraction d'entités structurées depuis le texte OCR : dates, montants, emails, etc."""

import re
from typing import Any


PATTERNS = {
    "dates": [
        r"\b(\d{1,2}[/\-.]\d{1,2}[/\-.]\d{2,4})\b",
        r"\b(\d{1,2}\s+(?:janvier|février|mars|avril|mai|juin|juillet|août|septembre|octobre|novembre|décembre)\s+\d{4})\b",
        r"\b(\d{1,2}\s+(?:jan|fév|mar|avr|mai|jun|jul|aoû|sep|oct|nov|déc)\.?\s+\d{4})\b",
        r"\b(\d{4}[/\-.]\d{1,2}[/\-.]\d{1,2})\b",
    ],
    "montants": [
        r"(\d[\d\s]*[.,]\d{2})\s*(?:€|EUR|euros?)",
        r"(?:€|EUR)\s*(\d[\d\s]*[.,]\d{2})",
        r"(\d[\d\s]*[.,]\d{2})\s*(?:\$|USD|dollars?)",
        r"(?:total|montant|somme|net|ttc|ht)\s*:?\s*(\d[\d\s]*[.,]\d{2})",
    ],
    "emails": [
        r"\b([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})\b",
    ],
    "telephones": [
        r"\b((?:\+33|0)\s*[1-9](?:[\s.\-]?\d{2}){4})\b",
        r"\b(\d{2}[\s.\-]\d{2}[\s.\-]\d{2}[\s.\-]\d{2}[\s.\-]\d{2})\b",
    ],
    "siret": [
        r"\b(\d{3}\s?\d{3}\s?\d{3}\s?\d{5})\b",
    ],
    "tva_intra": [
        r"\b(FR\s?\d{2}\s?\d{3}\s?\d{3}\s?\d{3})\b",
    ],
    "iban": [
        r"\b(FR\d{2}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{4}\s?\d{3})\b",
    ],
    "numero_facture": [
        r"(?:facture|invoice|fact|fac)\s*(?:n[°o]?|#|:)\s*([A-Z0-9\-/]+)",
        r"(?:n[°o]?\s*(?:de\s+)?facture)\s*:?\s*([A-Z0-9\-/]+)",
    ],
}


def extract_entities(text: str) -> dict[str, list[str]]:
    """Extrait toutes les entités reconnues du texte."""
    results: dict[str, list[str]] = {}
    text_lower = text.lower()

    for entity_type, patterns in PATTERNS.items():
        matches = []
        for pattern in patterns:
            flags = re.IGNORECASE if entity_type != "emails" else 0
            found = re.findall(pattern, text if entity_type == "emails" else text, flags=flags)
            matches.extend(found)
        results[entity_type] = list(dict.fromkeys(matches))

    return results


def entities_to_table(entities: dict[str, list[str]]) -> list[dict[str, Any]]:
    """Convertit les entités extraites en format tabulaire."""
    rows = []
    labels = {
        "dates": "Date",
        "montants": "Montant",
        "emails": "Email",
        "telephones": "Téléphone",
        "siret": "SIRET",
        "tva_intra": "TVA Intracommunautaire",
        "iban": "IBAN",
        "numero_facture": "N° Facture",
    }
    for entity_type, values in entities.items():
        for val in values:
            rows.append({
                "Type": labels.get(entity_type, entity_type),
                "Valeur": val.strip(),
            })
    return rows


def compute_extraction_score(extracted: dict, ground_truth: dict) -> dict:
    """Calcule l'exact match et le F1 entre les entités extraites et la vérité terrain."""
    all_types = set(list(extracted.keys()) + list(ground_truth.keys()))
    tp = fp = fn = 0
    exact_matches = 0
    total_fields = 0

    for etype in all_types:
        ext_set = set(extracted.get(etype, []))
        gt_set = set(ground_truth.get(etype, []))
        tp += len(ext_set & gt_set)
        fp += len(ext_set - gt_set)
        fn += len(gt_set - ext_set)

        for val in gt_set:
            total_fields += 1
            if val in ext_set:
                exact_matches += 1

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    em = exact_matches / total_fields if total_fields > 0 else 0

    return {
        "exact_match": round(em, 4),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1_score": round(f1, 4),
    }
