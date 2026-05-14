"""Classification automatique de documents en catégories (factures, reçus, etc.)."""

import re
import shutil
from pathlib import Path
from typing import Optional

from docuflow_ai.config import DOSSIERS_DIR, DOSSIER_CATEGORIES


CATEGORY_KEYWORDS = {
    "factures": [
        "facture", "invoice", "fact.", "n° facture", "numéro de facture",
        "date de facture", "échéance", "conditions de paiement",
        "sous-total", "tva", "total ttc", "total ht", "montant dû",
        "bon de commande", "référence client",
    ],
    "recus": [
        "reçu", "receipt", "ticket", "caisse", "cb", "carte bancaire",
        "espèces", "rendu monnaie", "merci de votre visite",
        "ticket de caisse", "reçu de paiement", "acquitté",
    ],
    "formulaires": [
        "formulaire", "form", "cerfa", "déclaration", "demande de",
        "à remplir", "signature", "cocher", "case à cocher",
        "nom :", "prénom :", "adresse :", "date de naissance",
    ],
    "contrats": [
        "contrat", "contract", "convention", "accord", "avenant",
        "parties", "objet du contrat", "durée", "résiliation",
        "clause", "article", "signataire", "engagement",
    ],
}


def classify_document(text: str) -> dict:
    """Classe un document selon son contenu textuel."""
    text_lower = text.lower()
    scores = {}

    for category, keywords in CATEGORY_KEYWORDS.items():
        score = 0
        matched_keywords = []
        for kw in keywords:
            count = len(re.findall(re.escape(kw), text_lower))
            if count > 0:
                score += count
                matched_keywords.append(kw)
        scores[category] = {"score": score, "matched": matched_keywords}

    best_category = max(scores, key=lambda k: scores[k]["score"])
    best_score = scores[best_category]["score"]

    if best_score == 0:
        best_category = "autres"

    confidence = min(best_score / 5.0, 1.0)

    return {
        "category": best_category,
        "confidence": round(confidence, 2),
        "scores": {k: v["score"] for k, v in scores.items()},
        "matched_keywords": scores.get(best_category, {}).get("matched", []),
    }


def move_to_dossier(file_path: str, category: str) -> str:
    """Déplace ou copie un fichier dans le dossier de sa catégorie."""
    if category not in DOSSIER_CATEGORIES:
        category = "autres"

    dest_dir = DOSSIERS_DIR / category
    dest_dir.mkdir(parents=True, exist_ok=True)

    src = Path(file_path)
    dest = dest_dir / src.name

    counter = 1
    while dest.exists():
        dest = dest_dir / f"{src.stem}_{counter}{src.suffix}"
        counter += 1

    shutil.copy2(str(src), str(dest))
    return str(dest)


def classify_and_sort(file_path: str, text: str) -> dict:
    """Pipeline complet : classifie puis trie le document."""
    classification = classify_document(text)
    dest = move_to_dossier(file_path, classification["category"])
    classification["destination"] = dest
    classification["original"] = file_path
    return classification
