"""Module de métriques pour évaluer les performances de DocuFlow AI."""

import time
from typing import Any

from docuflow_ai.modules.entity_extractor import compute_extraction_score


def compute_cer(predicted: str, reference: str) -> float:
    """Character Error Rate (CER) via distance de Levenshtein."""
    if not reference:
        return 0.0 if not predicted else 1.0

    n = len(reference)
    m = len(predicted)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if reference[i - 1] == predicted[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return round(dp[n][m] / n, 4)


def compute_wer(predicted: str, reference: str) -> float:
    """Word Error Rate (WER)."""
    ref_words = reference.split()
    pred_words = predicted.split()

    if not ref_words:
        return 0.0 if not pred_words else 1.0

    n = len(ref_words)
    m = len(pred_words)

    dp = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        dp[i][0] = i
    for j in range(m + 1):
        dp[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if ref_words[i - 1] == pred_words[j - 1] else 1
            dp[i][j] = min(dp[i - 1][j] + 1, dp[i][j - 1] + 1, dp[i - 1][j - 1] + cost)

    return round(dp[n][m] / n, 4)


def compute_qa_accuracy(predictions: list[dict], ground_truth: list[dict]) -> dict:
    """Calcule l'accuracy des réponses Q&A.

    Chaque entrée : {"question": str, "answer": str}
    """
    if not ground_truth:
        return {"accuracy": 0.0, "total": 0, "correct": 0}

    correct = 0
    for pred, gt in zip(predictions, ground_truth):
        pred_answer = pred.get("answer", "").strip().lower()
        gt_answer = gt.get("answer", "").strip().lower()
        if pred_answer == gt_answer or gt_answer in pred_answer:
            correct += 1

    return {
        "accuracy": round(correct / len(ground_truth), 4),
        "total": len(ground_truth),
        "correct": correct,
    }


class PerformanceTracker:
    """Suit les métriques de performance au fil du temps."""

    def __init__(self):
        self.records: list[dict[str, Any]] = []

    def record(self, document: str, ocr_time: float, ocr_confidence: float,
               entities_extracted: int, category: str, classification_confidence: float):
        self.records.append({
            "document": document,
            "ocr_time_s": round(ocr_time, 3),
            "ocr_confidence": round(ocr_confidence, 2),
            "entities_extracted": entities_extracted,
            "category": category,
            "classification_confidence": round(classification_confidence, 2),
        })

    def summary(self) -> dict:
        if not self.records:
            return {"count": 0}

        times = [r["ocr_time_s"] for r in self.records]
        confs = [r["ocr_confidence"] for r in self.records]
        entities = [r["entities_extracted"] for r in self.records]

        return {
            "count": len(self.records),
            "avg_ocr_time": round(sum(times) / len(times), 3),
            "avg_ocr_confidence": round(sum(confs) / len(confs), 2),
            "total_entities": sum(entities),
            "avg_entities_per_doc": round(sum(entities) / len(entities), 1),
        }

    def to_table(self) -> list[dict]:
        return self.records
