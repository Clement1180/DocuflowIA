import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
UPLOAD_DIR = DATA_DIR / "uploads"
INDEX_DIR = DATA_DIR / "indexes"
DOSSIERS_DIR = DATA_DIR / "dossiers"

DOSSIER_CATEGORIES = ["factures", "recus", "formulaires", "contrats", "autres"]

for d in [UPLOAD_DIR, INDEX_DIR] + [DOSSIERS_DIR / c for c in DOSSIER_CATEGORIES]:
    d.mkdir(parents=True, exist_ok=True)

TESSERACT_CMD = os.environ.get("TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe")
TESSERACT_LANG = os.environ.get("TESSERACT_LANG", "fra+eng")

POPPLER_PATH = os.environ.get(
    "POPPLER_PATH",
    str(Path.home() / "AppData/Local/Microsoft/WinGet/Packages"
        / "oschwartz10612.Poppler_Microsoft.Winget.Source_8wekyb3d8bbwe"
        / "poppler-25.07.0/Library/bin"),
)

OLLAMA_BASE_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434")
OLLAMA_MODEL = os.environ.get("DOCUFLOW_LLM", "batiai/gemma4-e2b:q4")
