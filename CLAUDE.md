# DocuFlow AI — Documentation du Projet

## Vue d'ensemble

DocuFlow AI est un assistant intelligent local pour l'automatisation de la lecture, du tri et de l'extraction d'informations de documents administratifs (factures, reçus, formulaires, contrats). Il s'adresse aux PME souhaitant réduire le temps de traitement documentaire et les erreurs d'extraction.

L'application tourne **entièrement en local** — aucune donnée ne quitte la machine, aucune clé API cloud n'est requise.

---

## Stack technique

| Composant | Outil | Version |
|---|---|---|
| Interface | Streamlit | 1.47+ |
| OCR | Tesseract | 5.4.0 |
| Conversion PDF→image | Poppler | 25.07.0 |
| Extraction d'entités | Regex (Python natif) | — |
| LLM local | Ollama + Gemma 4 E2B Q4 | 0.23.1 |
| Persistance index RAG | JSON sur disque | — |
| Python cible | Python 3.11 | 3.11.x |

> **Important** : Streamlit et toutes les dépendances doivent être installées dans **Python 3.11** (`C:\Program Files\Python311\`), pas dans Python 3.14 ou autre.

---

## Structure du projet

```
Triage_pc_OCR/
├── CLAUDE.md                        # Ce fichier
├── requirements.txt                 # Dépendances Python
├── run.bat                          # Lanceur Windows
└── docuflow_ai/
    ├── config.py                    # Chemins, modèles, constantes
    ├── app.py                       # Interface Streamlit principale
    ├── rag_cli.py                   # CLI RAG (index / query / status / clear)
    ├── __main__.py                  # Point d'entrée : python -m docuflow_ai
    ├── modules/
    │   ├── ocr_engine.py            # OCR Tesseract (images + PDF via Poppler)
    │   ├── entity_extractor.py      # Extraction regex (dates, montants, SIRET…)
    │   ├── classifier.py            # Classification + copie/déplacement fichiers
    │   ├── rag_engine.py            # DocumentStore JSON + appel Ollama local
    │   ├── metrics.py               # CER, WER, F1, Exact Match, Accuracy Q&A
    │   └── persistence.py           # Sauvegarde/chargement session JSON
    └── data/
        ├── uploads/                 # Dossier d'upload temporaire
        ├── indexes/                 # Index RAG (document_index.json + chunks)
        ├── session_state.json       # Persistance entre sessions
        └── dossiers/
            ├── factures/
            ├── recus/
            ├── formulaires/
            ├── contrats/
            └── autres/
```

---

## Lancer l'application

```bat
# Option 1 — double-cliquer sur run.bat
run.bat

# Option 2 — ligne de commande
"C:\Program Files\Python311\python.exe" -m streamlit run docuflow_ai/app.py --server.port 8501
```

L'interface est accessible sur **http://localhost:8501**.

---

## Fonctionnalités disponibles

### 1. Import & Analyse
Deux modes d'import dans l'onglet "📤 Import & Analyse" :

- **Dossier local** : coller le chemin d'un dossier (`C:\...\MesDocuments`), tous les PDF/images sont scannés récursivement. Les fichiers déjà traités sont ignorés automatiquement.
- **Fichiers individuels** : glisser-déposer ou sélectionner des fichiers via le navigateur.

Pour chaque document, l'app effectue :
1. OCR via Tesseract (support FR+EN)
2. Extraction d'entités par regex
3. Classification automatique
4. Indexation dans le RAG

### 2. Tri en Dossiers
Onglet "📂 Tri en Dossiers" :
- Affiche la classification de tous les documents traités
- Permet de choisir le dossier de destination
- Deux actions : **Copier** (garde les originaux) ou **Déplacer** (déplace les fichiers sources)
- Les fichiers uploadés via navigateur (pas de chemin local) sont signalés comme non triables

### 3. Chat Q&A
Onglet "💬 Chat Q&A" :
- Questions en langage naturel sur tous les documents indexés
- Réponse générée par Gemma 4 E2B Q4 via Ollama (local)
- Si Ollama n'est pas actif : bascule automatiquement en mode extractif (TF-IDF)
- Les sources et passages utilisés sont affichés

### 4. Métriques
Onglet "📊 Métriques" :
- Temps OCR moyen, confiance OCR, entités extraites
- Graphiques par document
- Tableau de référence des métriques cibles (EM, F1, CER, Accuracy Q&A)

### 5. Persistance inter-sessions
- L'état (documents traités, index RAG, historique chat, métriques) est sauvegardé dans `data/session_state.json`
- Rechargé automatiquement au redémarrage de l'app
- Bouton "Réinitialiser tout" dans la sidebar pour repartir de zéro

---

## CLI RAG

Le RAG est également accessible en ligne de commande :

```bat
# Indexer un dossier entier
"C:\Program Files\Python311\python.exe" -m docuflow_ai index C:\chemin\vers\dossier

# Indexer un fichier unique
"C:\Program Files\Python311\python.exe" -m docuflow_ai index C:\chemin\vers\fichier.pdf

# Poser une question
"C:\Program Files\Python311\python.exe" -m docuflow_ai query "Quel est le montant total de la facture ?"

# Voir l'état de l'index
"C:\Program Files\Python311\python.exe" -m docuflow_ai status

# Vider l'index
"C:\Program Files\Python311\python.exe" -m docuflow_ai clear
```

---

## Entités extraites automatiquement

| Type | Exemples |
|---|---|
| Dates | `15/03/2024`, `15 mars 2024` |
| Montants | `1 234,56 €`, `500,00 EUR` |
| Emails | `contact@exemple.fr` |
| Téléphones | `06 12 34 56 78`, `+33 1 23 45 67 89` |
| SIRET | `123 456 789 00012` |
| TVA intracommunautaire | `FR12 345 678 901` |
| IBAN | `FR76 1234 …` |
| N° facture | `F-2024-001`, `FAC-0042` |

---

## Catégories de tri

| Catégorie | Mots-clés détecteurs |
|---|---|
| `factures` | facture, invoice, TVA, total TTC, montant dû… |
| `recus` | reçu, ticket, caisse, CB, carte bancaire… |
| `formulaires` | formulaire, CERFA, déclaration, à remplir… |
| `contrats` | contrat, convention, clause, résiliation… |
| `autres` | (aucun mot-clé dominant) |

---

## Modèle IA local

**Modèle actif** : `batiai/gemma4-e2b:q4`
- Taille : 3.4 GB sur disque
- VRAM requise : ~3-4 Go
- Quantisation Q4 (bon compromis qualité/vitesse)
- Contexte : 128 K tokens
- Multimodal (texte + images)

**Historique des modèles utilisés** :
- `tinyllama` — retiré (qualité insuffisante en français)
- `mistral:latest` (4.1 GB) — retiré (VRAM insuffisante sur cette machine)
- `gemma4:e2b` officiel (7.2 GB) — retiré (RAM système insuffisante : 5.4 Go requis vs 4.9 Go dispo)
- `batiai/gemma4-e2b:q4` (3.4 GB) — **modèle actuel** ✅

Pour changer de modèle :
```python
# docuflow_ai/config.py
OLLAMA_MODEL = "nom-du-modele"
# ou variable d'environnement :
# set DOCUFLOW_LLM=nom-du-modele
```

---

## Variables d'environnement

| Variable | Défaut | Description |
|---|---|---|
| `TESSERACT_CMD` | `C:\Program Files\Tesseract-OCR\tesseract.exe` | Chemin vers l'exécutable Tesseract |
| `TESSERACT_LANG` | `fra+eng` | Langues OCR (Tesseract) |
| `POPPLER_PATH` | `~\AppData\Local\Microsoft\WinGet\...\poppler-25.07.0\Library\bin` | Binaires Poppler pour pdf2image |
| `OLLAMA_URL` | `http://localhost:11434` | URL de l'API Ollama |
| `DOCUFLOW_LLM` | `batiai/gemma4-e2b:q4` | Modèle Ollama à utiliser |

---

## Problèmes résolus et correctifs appliqués

### ModuleNotFoundError: No module named 'pytesseract'
**Cause** : Streamlit utilisait Python 3.11 mais pytesseract était installé dans Python 3.14.
**Correction** : Installation explicite dans Python 3.11 :
```bat
"C:\Program Files\Python311\python.exe" -m pip install pytesseract Pillow pdf2image pandas streamlit
```

### Unable to get page count. Is poppler installed and in PATH ?
**Cause** : Poppler installé via winget dans un chemin non standard, absent du PATH.
**Correction** : Chemin Poppler déclaré explicitement dans `config.py` et transmis à `convert_from_path()` via le paramètre `poppler_path`. Les chemins sont normalisés avec `Path.resolve()` pour éviter les conflits de slashes Windows.

### Modèle Ollama : out of memory / model requires more system memory
**Cause** : La version officielle `gemma4:e2b` (7.2 GB) requiert 5.4 Go de RAM système, indisponibles.
**Correction** : Passage à `batiai/gemma4-e2b:q4` (3.4 GB, Q4), qui fonctionne sur cette configuration.

---

## Métriques cibles

| Métrique | Description | Cible |
|---|---|---|
| **Exact Match (EM)** | Champs extraits identiques à la vérité terrain | > 80% |
| **F1-Score entités** | Précision/Rappel des entités nommées | > 85% |
| **Accuracy Q&A** | Taux de bonnes réponses sur DocVQA | > 75% |
| **CER** | Character Error Rate de l'OCR | < 5% |
| **WER** | Word Error Rate de l'OCR | < 10% |
| **Temps / document** | Temps de traitement bout-en-bout | < 10s |

---

## Dépendances Python (requirements.txt)

```
streamlit>=1.30.0
pytesseract>=0.3.10
Pillow>=10.0.0
pdf2image>=1.16.3
pandas>=2.0.0
```

> **Note** : `openai` a été retiré volontairement — tout le LLM passe par Ollama en local.

---

## Outils système requis

- [Tesseract OCR 5.4](https://github.com/UB-Mannheim/tesseract/wiki) — avec langues `eng` et `fra`
- [Poppler 25.07](https://github.com/oschwartz10612/poppler-windows) — pour la conversion PDF
- [Ollama 0.23+](https://ollama.com) — runtime LLM local
- Python 3.11 (Streamlit ne supporte pas encore Python 3.14)
