"""Persistance des documents traités et métriques entre sessions."""

import json
from pathlib import Path
from typing import Any

from docuflow_ai.config import DATA_DIR


SESSION_FILE = DATA_DIR / "session_state.json"


def save_session(processed_docs: list[dict], tracker_records: list[dict], chat_history: list[dict]):
    """Sauvegarde l'état complet de la session sur disque."""
    clean_docs = []
    for doc in processed_docs:
        d = dict(doc)
        d.pop("tmp_path", None)
        clean_docs.append(d)

    state = {
        "processed_docs": clean_docs,
        "tracker_records": tracker_records,
        "chat_history": chat_history,
    }
    with open(SESSION_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)


def load_session() -> dict[str, Any]:
    """Charge l'état de la session précédente."""
    if not SESSION_FILE.exists():
        return {"processed_docs": [], "tracker_records": [], "chat_history": []}
    try:
        with open(SESSION_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, KeyError):
        return {"processed_docs": [], "tracker_records": [], "chat_history": []}


def clear_session():
    """Supprime le fichier de session."""
    if SESSION_FILE.exists():
        SESSION_FILE.unlink()
