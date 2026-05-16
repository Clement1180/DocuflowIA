"""
Rendu Jalon 2 — Préparation du dataset SROIE (Scanned Receipts OCR and IE).

Ce script télécharge un échantillon du dataset SROIE depuis le dépôt GitHub,
le restructure en splits train/validation/test, et prépare la vérité terrain
dans un format exploitable par notre pipeline.

Dataset : SROIE (ICDAR 2019 Competition)
- Images de reçus scannés (.jpg)
- Annotations clés : company, date, address, total (.json)
- Texte OCR de référence avec bounding boxes (.csv)

Source : https://github.com/zzzDavid/ICDAR-2019-SROIE
"""

import json
import os
import random
import shutil
import urllib.request
import urllib.error
import time
from pathlib import Path

DATASET_DIR = Path(__file__).parent / "dataset"
RAW_DIR = DATASET_DIR / "raw"
SPLITS = {"train": 0.6, "validation": 0.2, "test": 0.2}

BASE_URL = "https://raw.githubusercontent.com/zzzDavid/ICDAR-2019-SROIE/master/data"
SAMPLE_SIZE = 50
SEED = 42


def download_file(url: str, dest: Path, retries: int = 3) -> bool:
    """Télécharge un fichier avec retry."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True
    for attempt in range(retries):
        try:
            urllib.request.urlretrieve(url, str(dest))
            return True
        except urllib.error.HTTPError as e:
            if e.code == 404:
                return False
            if attempt < retries - 1:
                time.sleep(1)
        except Exception:
            if attempt < retries - 1:
                time.sleep(1)
    return False


def download_sroie_sample():
    """Télécharge un échantillon de SROIE (images + annotations + OCR GT)."""
    img_dir = RAW_DIR / "img"
    key_dir = RAW_DIR / "key"
    box_dir = RAW_DIR / "box"
    img_dir.mkdir(parents=True, exist_ok=True)
    key_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)

    downloaded = []
    skipped = 0
    max_id = 200  # scanner les IDs 000-199 pour trouver assez de fichiers

    print(f"  Téléchargement de {SAMPLE_SIZE} documents depuis GitHub...")
    print(f"  (scan des IDs 000 à {max_id - 1})")

    for i in range(max_id):
        if len(downloaded) >= SAMPLE_SIZE:
            break

        doc_id = f"{i:03d}"
        img_url = f"{BASE_URL}/img/{doc_id}.jpg"
        key_url = f"{BASE_URL}/key/{doc_id}.json"
        box_url = f"{BASE_URL}/box/{doc_id}.csv"

        img_path = img_dir / f"{doc_id}.jpg"
        key_path = key_dir / f"{doc_id}.json"
        box_path = box_dir / f"{doc_id}.csv"

        # Télécharger les 3 fichiers (image + annotation + OCR GT)
        img_ok = download_file(img_url, img_path)
        key_ok = download_file(key_url, key_path)
        box_ok = download_file(box_url, box_path)

        if img_ok and key_ok:
            downloaded.append(doc_id)
            if len(downloaded) % 10 == 0:
                print(f"    [{len(downloaded)}/{SAMPLE_SIZE}] téléchargés...")
        else:
            skipped += 1
            # Nettoyer les fichiers partiels
            for p in [img_path, key_path, box_path]:
                if p.exists():
                    p.unlink()

    print(f"  Téléchargés : {len(downloaded)} documents")
    if skipped > 0:
        print(f"  Ignorés (404) : {skipped}")

    return downloaded


def load_ground_truth(key_dir: Path, doc_ids: list) -> dict:
    """Charge les annotations clé-valeur SROIE (company, date, address, total)."""
    gt = {}
    for doc_id in doc_ids:
        json_path = key_dir / f"{doc_id}.json"
        if json_path.exists():
            try:
                content = json.loads(json_path.read_text(encoding="utf-8"))
                gt[doc_id] = content
            except (json.JSONDecodeError, UnicodeDecodeError):
                pass
    return gt


def load_ocr_ground_truth(box_dir: Path, doc_ids: list) -> dict:
    """Charge le texte OCR de référence depuis les fichiers CSV (bounding boxes)."""
    ocr_gt = {}
    for doc_id in doc_ids:
        csv_path = box_dir / f"{doc_id}.csv"
        if csv_path.exists():
            try:
                content = csv_path.read_text(encoding="utf-8", errors="ignore").strip()
                lines = []
                for line in content.split("\n"):
                    # Format: x1,y1,x2,y2,x3,y3,x4,y4,TEXT
                    parts = line.split(",", 8)
                    if len(parts) >= 9:
                        lines.append(parts[8].strip())
                ocr_gt[doc_id] = "\n".join(lines)
            except Exception:
                pass
    return ocr_gt


def create_splits(doc_ids: list, seed: int = SEED) -> dict:
    """Sépare les IDs en train/validation/test."""
    random.seed(seed)
    ids = list(doc_ids)
    random.shuffle(ids)

    n = len(ids)
    n_train = int(n * SPLITS["train"])
    n_val = int(n * SPLITS["validation"])

    return {
        "train": ids[:n_train],
        "validation": ids[n_train:n_train + n_val],
        "test": ids[n_train + n_val:],
    }


def build_structured_dataset(doc_ids: list):
    """Construit le dataset structuré avec splits."""
    print("\n=== Construction du dataset structuré ===")

    img_dir = RAW_DIR / "img"
    key_dir = RAW_DIR / "key"
    box_dir = RAW_DIR / "box"

    gt = load_ground_truth(key_dir, doc_ids)
    ocr_gt = load_ocr_ground_truth(box_dir, doc_ids)

    print(f"  Documents avec annotations : {len(gt)}")
    print(f"  Documents avec OCR GT      : {len(ocr_gt)}")

    splits = create_splits(doc_ids)

    for split_name, split_ids in splits.items():
        split_dir = DATASET_DIR / split_name
        split_dir.mkdir(parents=True, exist_ok=True)
        (split_dir / "images").mkdir(exist_ok=True)

        split_manifest = []
        for doc_id in split_ids:
            img_src = img_dir / f"{doc_id}.jpg"
            if not img_src.exists():
                continue
            img_dest = split_dir / "images" / f"{doc_id}.jpg"
            shutil.copy2(img_src, img_dest)

            entry = {
                "id": doc_id,
                "image": f"images/{doc_id}.jpg",
                "ground_truth": gt.get(doc_id, {}),
                "ocr_reference": ocr_gt.get(doc_id, ""),
            }
            split_manifest.append(entry)

        manifest_path = split_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(split_manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(f"  {split_name}: {len(split_ids)} documents -> {split_dir.name}/")

    summary = {
        "dataset": "SROIE (ICDAR 2019)",
        "description": "Scanned Receipts OCR and Information Extraction",
        "source": "https://github.com/zzzDavid/ICDAR-2019-SROIE",
        "total_documents": len(doc_ids),
        "splits": {k: len(v) for k, v in splits.items()},
        "entity_fields": ["company", "date", "address", "total"],
        "format": {
            "images": "JPG (reçus scannés)",
            "annotations": "JSON (company, date, address, total)",
            "ocr_reference": "CSV (x1,y1,...,x4,y4,text)"
        },
    }
    summary_path = DATASET_DIR / "dataset_info.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"\n  Resume sauvegarde dans {summary_path.name}")


def main():
    print("=" * 60)
    print("  DocuFlow AI - Preparation dataset SROIE (Jalon 2)")
    print("=" * 60)

    print("\n[1/2] Telechargement du dataset SROIE...")
    doc_ids = download_sroie_sample()

    if not doc_ids:
        print("\n[ERREUR] Aucun document telecharge. Verifiez votre connexion.")
        return

    print(f"\n[2/2] Structuration en splits train/val/test...")
    build_structured_dataset(doc_ids)

    print("\n" + "=" * 60)
    print(f"  Dataset pret dans : {DATASET_DIR.resolve()}")
    print("  Executez le notebook 01_EDA_et_Pipeline.ipynb pour la suite.")
    print("=" * 60)


if __name__ == "__main__":
    main()
