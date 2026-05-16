# Rendu Jalon 2 — DocuFlow AI

## Contenu du dossier

| Fichier | Description |
|---------|-------------|
| `prepare_dataset.py` | Télécharge et structure le dataset SROIE (train/val/test) |
| `01_EDA_et_Pipeline.ipynb` | Notebook : EDA + évaluation baseline + métriques |
| `requirements_jalon2.txt` | Dépendances Python supplémentaires pour ce jalon |
| `baseline_metrics.json` | Généré après exécution du notebook |

## Exécution

```bat
:: 1. Installer les dépendances
"C:\Program Files\Python311\python.exe" -m pip install -r Rendu_Jalon2\requirements_jalon2.txt

:: 2. Télécharger et préparer le dataset
"C:\Program Files\Python311\python.exe" Rendu_Jalon2\prepare_dataset.py

:: 3. Lancer le notebook
"C:\Program Files\Python311\python.exe" -m jupyter notebook Rendu_Jalon2\01_EDA_et_Pipeline.ipynb
```

## Dataset utilisé

**SROIE** (ICDAR 2019 — Scanned Receipts OCR and Information Extraction)
- 60 reçus scannés (échantillon)
- Splits : 60% train / 20% validation / 20% test
- Vérité terrain : company, date, address, total + texte OCR de référence

## Métriques évaluées

| Métrique | Cible |
|----------|-------|
| CER | < 5% |
| WER | < 10% |
| Exact Match | > 80% |
| F1-Score entités | > 85% |
| Temps / document | < 10s |
