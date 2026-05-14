"""RAG (Retrieval-Augmented Generation) en ligne de commande.

Utilise un index fichier local (JSON sur disque) au lieu d'une base vectorielle,
et Ollama pour la génération locale (aucune API cloud).
"""

import json
import hashlib
import urllib.request
import urllib.error
from pathlib import Path
from typing import Optional

from docuflow_ai.config import INDEX_DIR, OLLAMA_BASE_URL, OLLAMA_MODEL


def _get_doc_id(text: str, source: str) -> str:
    return hashlib.md5(f"{source}:{text[:200]}".encode()).hexdigest()


class DocumentStore:
    """Store léger de documents indexés, basé sur des fichiers JSON."""

    def __init__(self, store_dir: Optional[Path] = None):
        self.store_dir = store_dir or INDEX_DIR
        self.store_dir.mkdir(parents=True, exist_ok=True)
        self.index_file = self.store_dir / "document_index.json"
        self._load_index()

    def _load_index(self):
        if self.index_file.exists():
            with open(self.index_file, "r", encoding="utf-8") as f:
                self.index = json.load(f)
        else:
            self.index = {"documents": []}

    def _save_index(self):
        with open(self.index_file, "w", encoding="utf-8") as f:
            json.dump(self.index, f, ensure_ascii=False, indent=2)

    def add_document(self, text: str, source: str, metadata: Optional[dict] = None):
        doc_id = _get_doc_id(text, source)
        for doc in self.index["documents"]:
            if doc["id"] == doc_id:
                return doc_id

        chunks = self._chunk_text(text)
        doc_entry = {
            "id": doc_id,
            "source": source,
            "metadata": metadata or {},
            "chunks": chunks,
            "text_length": len(text),
        }
        self.index["documents"].append(doc_entry)
        self._save_index()

        chunk_file = self.store_dir / f"{doc_id}_chunks.json"
        with open(chunk_file, "w", encoding="utf-8") as f:
            json.dump({"doc_id": doc_id, "chunks": chunks}, f, ensure_ascii=False, indent=2)

        return doc_id

    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> list[dict]:
        words = text.split()
        chunks = []
        i = 0
        chunk_idx = 0
        while i < len(words):
            chunk_words = words[i:i + chunk_size]
            chunk_text = " ".join(chunk_words)
            chunks.append({"index": chunk_idx, "text": chunk_text})
            i += chunk_size - overlap
            chunk_idx += 1
        return chunks

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """Recherche par pertinence TF-IDF simplifiée (sans base vectorielle)."""
        query_terms = set(query.lower().split())
        results = []

        for doc in self.index["documents"]:
            for chunk in doc["chunks"]:
                chunk_lower = chunk["text"].lower()
                score = sum(1 for term in query_terms if term in chunk_lower)
                term_density = score / max(len(query_terms), 1)
                if score > 0:
                    results.append({
                        "doc_id": doc["id"],
                        "source": doc["source"],
                        "chunk_index": chunk["index"],
                        "text": chunk["text"],
                        "score": round(term_density, 4),
                        "metadata": doc.get("metadata", {}),
                    })

        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def get_all_sources(self) -> list[str]:
        return [doc["source"] for doc in self.index["documents"]]

    def get_document_count(self) -> int:
        return len(self.index["documents"])

    def clear(self):
        self.index = {"documents": []}
        self._save_index()
        for f in self.store_dir.glob("*_chunks.json"):
            f.unlink()


def _ollama_available() -> bool:
    """Vérifie si Ollama est en cours d'exécution."""
    try:
        req = urllib.request.Request(f"{OLLAMA_BASE_URL}/api/tags", method="GET")
        with urllib.request.urlopen(req, timeout=3) as resp:
            return resp.status == 200
    except (urllib.error.URLError, OSError):
        return False


def _ollama_generate(prompt: str, system: str = "") -> str:
    """Appelle Ollama en local via son API REST (aucune dépendance externe)."""
    payload = json.dumps({
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "system": system,
        "stream": False,
        "options": {"temperature": 0.1, "num_predict": 1000},
    }).encode("utf-8")

    req = urllib.request.Request(
        f"{OLLAMA_BASE_URL}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        data = json.loads(resp.read().decode("utf-8"))
        return data.get("response", "")


class RAGEngine:
    """Moteur RAG qui combine la recherche documentaire et la génération locale."""

    def __init__(self, store: Optional[DocumentStore] = None):
        self.store = store or DocumentStore()
        self._llm_available = _ollama_available()

    def check_llm(self) -> bool:
        self._llm_available = _ollama_available()
        return self._llm_available

    def index_document(self, text: str, source: str, metadata: Optional[dict] = None) -> str:
        return self.store.add_document(text, source, metadata)

    def query(self, question: str, top_k: int = 5) -> dict:
        """Interroge les documents indexés et génère une réponse."""
        relevant_chunks = self.store.search(question, top_k=top_k)

        if not relevant_chunks:
            return {
                "answer": "Aucun document pertinent trouvé pour cette question.",
                "sources": [],
                "chunks_used": 0,
            }

        context = "\n\n---\n\n".join(
            f"[Source: {c['source']}]\n{c['text']}" for c in relevant_chunks
        )

        self._llm_available = _ollama_available()

        if self._llm_available:
            answer = self._generate_with_ollama(question, context)
        else:
            answer = self._generate_extractive(question, relevant_chunks)

        sources = list(dict.fromkeys(c["source"] for c in relevant_chunks))

        return {
            "answer": answer,
            "sources": sources,
            "chunks_used": len(relevant_chunks),
            "context_preview": context[:500] + "..." if len(context) > 500 else context,
        }

    def _generate_with_ollama(self, question: str, context: str) -> str:
        system = (
            "Tu es un assistant spécialisé dans l'analyse de documents administratifs. "
            "Réponds aux questions en te basant uniquement sur le contexte fourni. "
            "Si l'information n'est pas dans le contexte, dis-le clairement. "
            "Cite les sources quand c'est pertinent. Réponds en français."
        )
        prompt = f"Contexte des documents :\n{context}\n\nQuestion : {question}"

        try:
            return _ollama_generate(prompt, system)
        except Exception as e:
            return f"Erreur Ollama : {e}\n\n(Réponse extractive ci-dessous)\n\n" + self._generate_extractive(
                question, self.store.search(question)
            )

    def _generate_extractive(self, question: str, chunks: list[dict]) -> str:
        """Réponse extractive quand Ollama n'est pas disponible."""
        best = chunks[0]
        sentences = best["text"].replace("\n", " ").split(".")
        query_terms = set(question.lower().split())

        scored = []
        for s in sentences:
            s = s.strip()
            if len(s) < 10:
                continue
            score = sum(1 for t in query_terms if t in s.lower())
            scored.append((score, s))

        scored.sort(key=lambda x: x[0], reverse=True)
        top_sentences = [s for sc, s in scored[:3] if sc > 0]

        if top_sentences:
            return (
                f"D'après le document '{best['source']}' :\n\n"
                + ". ".join(top_sentences) + "."
                + f"\n\n(Confiance de pertinence : {best['score']:.0%})"
                + "\n\n*Mode extractif — lancez Ollama pour des réponses génératives.*"
            )
        return (
            f"Le passage le plus pertinent trouvé dans '{best['source']}' :\n\n"
            f'"{best["text"][:300]}..."'
            f"\n\n(Confiance de pertinence : {best['score']:.0%})"
            "\n\n*Mode extractif — lancez Ollama pour des réponses génératives.*"
        )
