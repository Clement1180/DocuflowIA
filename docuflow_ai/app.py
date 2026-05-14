"""DocuFlow AI — Interface Streamlit principale."""

import os
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from docuflow_ai.modules.ocr_engine import process_document
from docuflow_ai.modules.entity_extractor import extract_entities, entities_to_table
from docuflow_ai.modules.classifier import classify_document, classify_and_sort
from docuflow_ai.modules.rag_engine import RAGEngine
from docuflow_ai.modules.metrics import PerformanceTracker
from docuflow_ai.modules.persistence import save_session, load_session, clear_session

SUPPORTED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp", ".webp"}


def _scan_folder(folder_path: str) -> list[Path]:
    """Scanne un dossier et retourne tous les fichiers supportés."""
    folder = Path(folder_path)
    if not folder.is_dir():
        return []
    files = []
    for f in sorted(folder.rglob("*")):
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            files.append(f)
    return files


def _process_single_file(file_path: Path, file_name: str) -> dict | None:
    """Traite un fichier : OCR + entités + classification + indexation."""
    try:
        ocr_result = process_document(str(file_path))
        entities = extract_entities(ocr_result["text"])
        classification = classify_document(ocr_result["text"])

        st.session_state.rag_engine.index_document(
            text=ocr_result["text"],
            source=file_name,
            metadata={
                "confidence": ocr_result["confidence"],
                "category": classification["category"],
            },
        )

        st.session_state.tracker.record(
            document=file_name,
            ocr_time=ocr_result["processing_time"],
            ocr_confidence=ocr_result["confidence"],
            entities_extracted=sum(len(v) for v in entities.values()),
            category=classification["category"],
            classification_confidence=classification["confidence"],
        )

        return {
            "name": file_name,
            "original_path": str(file_path),
            "ocr": ocr_result,
            "entities": entities,
            "classification": classification,
        }
    except Exception as e:
        st.error(f"Erreur sur {file_name} : {e}")
        return None


def _persist():
    """Sauvegarde l'état courant sur disque."""
    save_session(
        st.session_state.processed_docs,
        st.session_state.tracker.to_table(),
        st.session_state.chat_history,
    )


#  charger la session précédente 
if "initialized" not in st.session_state:
    st.session_state.initialized = True
    st.session_state.rag_engine = RAGEngine()
    saved = load_session()
    st.session_state.processed_docs = saved.get("processed_docs", [])
    st.session_state.chat_history = saved.get("chat_history", [])
    st.session_state.tracker = PerformanceTracker()
    for rec in saved.get("tracker_records", []):
        st.session_state.tracker.records.append(rec)


#  Page config 
st.set_page_config(
    page_title="DocuFlow AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

#  Custom CSS 
st.markdown("""
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.2rem;
        border-radius: 12px;
        color: white;
        text-align: center;
    }
    .metric-card h3 { margin: 0; font-size: 0.85rem; opacity: 0.9; }
    .metric-card p { margin: 0; font-size: 1.8rem; font-weight: 700; }
    .cat-factures { background: #3b82f6; }
    .cat-recus { background: #10b981; }
    .cat-formulaires { background: #f59e0b; }
    .cat-contrats { background: #8b5cf6; }
    .cat-autres { background: #6b7280; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)


#  Sidebar 
with st.sidebar:
    st.markdown("##  DocuFlow AI")
    st.markdown("---")
    page = st.radio(
        "Navigation",
        [" Accueil", " Import & Analyse", " Tri en Dossiers", " Chat Q&A", " Métriques"],
        label_visibility="collapsed",
    )
    st.markdown("---")
    st.markdown(f"**Documents indexés** : {st.session_state.rag_engine.store.get_document_count()}")
    st.markdown(f"**Documents traités** : {len(st.session_state.processed_docs)}")

    ollama_ok = st.session_state.rag_engine.check_llm()
    if ollama_ok:
        st.success("Ollama connecté", icon="🟢")
    else:
        st.warning("Ollama non détecté — mode extractif", icon="🟡")
        st.caption("Lancez `ollama serve` puis `ollama pull tinyllama`")

    if st.button("🗑️ Réinitialiser tout"):
        st.session_state.rag_engine.store.clear()
        st.session_state.processed_docs = []
        st.session_state.chat_history = []
        st.session_state.tracker = PerformanceTracker()
        clear_session()
        st.rerun()


# Pages 

if page == "Accueil":
    st.markdown('<p class="main-header">DocuFlow AI</p>', unsafe_allow_html=True)
    st.markdown(
        '<p class="sub-header">Assistant intelligent pour la lecture, le tri et '
        "l'extraction d'informations de vos documents administratifs</p>",
        unsafe_allow_html=True,
    )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>Import & OCR</h3>
            <p></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Importez des fichiers ou un dossier complet. L'OCR extrait le texte automatiquement.")

    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>Tri Intelligent</h3>
            <p></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Classification et copie automatique dans les bons sous-dossiers.")

    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>Chat Q&A</h3>
            <p></p>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("Posez des questions en langage naturel sur l'ensemble de vos documents.")

    st.markdown("---")
    st.markdown("### Comment utiliser DocuFlow AI")
    st.markdown("""
    1. **Importez** vos documents via l'onglet Import & Analyse (fichiers individuels ou dossier entier)
    2. **Visualisez** les données extraites automatiquement (dates, montants, emails...)
    3. **Triez** vos documents dans les bons dossiers via Tri en Dossiers
    4. **Interrogez** vos documents en langage naturel via le Chat Q&A
    5. **Suivez** les performances dans l'onglet Métriques

    Les données sont **persistées entre les sessions** : fermez et rouvrez l'app, tout est conservé.
    """)


elif page == " Import & Analyse":
    st.markdown("##  Import & Analyse de Documents")

    import_mode = st.radio(
        "Mode d'import",
        [" Dossier local", " Fichiers individuels"],
        horizontal=True,
    )

    # ── Import par dossier ──
    if import_mode == " Dossier local":
        folder_path = st.text_input(
            "Chemin du dossier à scanner",
            placeholder=r"C:\Users\...\MesDocuments",
        )

        if folder_path:
            files = _scan_folder(folder_path)
            if not files:
                st.warning("Aucun fichier supporté trouvé (PDF, PNG, JPG, TIFF, BMP).")
            else:
                already = {d["name"] for d in st.session_state.processed_docs}
                new_files = [f for f in files if f.name not in already]
                skipped = len(files) - len(new_files)

                st.info(
                    f"**{len(files)}** fichier(s) trouvé(s) dans `{folder_path}`"
                    + (f" ({skipped} déjà traité(s))" if skipped else "")
                )

                if new_files:
                    with st.expander("Fichiers à traiter", expanded=False):
                        for f in new_files:
                            st.markdown(f"- `{f.name}` ({f.suffix.upper()}, {f.stat().st_size // 1024} KB)")

                    if st.button(f" Analyser les {len(new_files)} fichier(s)", type="primary"):
                        progress = st.progress(0, text="Analyse en cours...")

                        for i, file_path in enumerate(new_files):
                            progress.progress(i / len(new_files), text=f"OCR de {file_path.name}...")
                            doc = _process_single_file(file_path, file_path.name)
                            if doc:
                                st.session_state.processed_docs.append(doc)

                        progress.progress(1.0, text="Analyse terminée !")
                        _persist()
                        st.rerun()
                elif skipped:
                    st.success("Tous les fichiers de ce dossier ont déjà été traités.")

    #  Import par fichiers 
    else:
        uploaded_files = st.file_uploader(
            "Déposez vos documents ici",
            type=["pdf", "png", "jpg", "jpeg", "tiff", "bmp"],
            accept_multiple_files=True,
        )

        if uploaded_files and st.button(" Analyser les documents", type="primary"):
            progress = st.progress(0, text="Analyse en cours...")

            for i, uploaded_file in enumerate(uploaded_files):
                progress.progress(i / len(uploaded_files), text=f"Traitement de {uploaded_file.name}...")

                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=Path(uploaded_file.name).suffix
                ) as tmp:
                    tmp.write(uploaded_file.read())
                    tmp_path = tmp.name

                doc = _process_single_file(Path(tmp_path), uploaded_file.name)
                if doc:
                    st.session_state.processed_docs.append(doc)

            progress.progress(1.0, text="Analyse terminée !")
            _persist()
            st.rerun()

    #  Résultats 
    if st.session_state.processed_docs:
        st.markdown("---")
        st.markdown(f"### Résultats ({len(st.session_state.processed_docs)} documents)")

        for doc in st.session_state.processed_docs:
            cat = doc["classification"]["category"]

            with st.expander(f" {doc['name']} — {cat.upper()}", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                col1.metric("Confiance OCR", f"{doc['ocr']['confidence']}%")
                col2.metric("Mots détectés", doc["ocr"]["word_count"])
                col3.metric("Catégorie", cat.capitalize())
                col4.metric("Temps OCR", f"{doc['ocr']['processing_time']}s")

                tab1, tab2, tab3 = st.tabs(["Entités extraites", "Texte OCR", "Détails classification"])

                with tab1:
                    table_data = entities_to_table(doc["entities"])
                    if table_data:
                        st.dataframe(pd.DataFrame(table_data), use_container_width=True)
                    else:
                        st.info("Aucune entité détectée.")

                with tab2:
                    st.text_area("Texte extrait", doc["ocr"]["text"], height=300, key=f"txt_{doc['name']}")

                with tab3:
                    st.json(doc["classification"])


elif page == " Tri en Dossiers":
    st.markdown("##  Tri Automatique en Dossiers")

    if not st.session_state.processed_docs:
        st.info("Importez d'abord des documents via l'onglet Import & Analyse.")
    else:
        #  Dossier de destination 
        from docuflow_ai.config import DOSSIERS_DIR
        default_dest = str(DOSSIERS_DIR)
        dest_folder = st.text_input(
            "Dossier de destination pour le tri",
            value=default_dest,
            help="Les sous-dossiers (factures/, recus/, formulaires/, contrats/, autres/) seront créés ici.",
        )

        st.markdown("### Aperçu de la classification")

        df_data = []
        for doc in st.session_state.processed_docs:
            has_original = "original_path" in doc and Path(doc["original_path"]).exists()
            df_data.append({
                "Document": doc["name"],
                "Catégorie": doc["classification"]["category"].capitalize(),
                "Confiance": f"{doc['classification']['confidence']:.0%}",
                "Mots-clés": ", ".join(doc["classification"]["matched_keywords"][:5]),
                "Fichier source": "Disponible" if has_original else "Upload uniquement",
            })
        st.dataframe(pd.DataFrame(df_data), use_container_width=True)

        #  Visualisation par catégorie 
        st.markdown("### Répartition par dossier")
        categories: dict[str, list[dict]] = {}
        for doc in st.session_state.processed_docs:
            cat = doc["classification"]["category"]
            categories.setdefault(cat, []).append(doc)

        cols = st.columns(max(len(categories), 1))
        for idx, (cat, docs) in enumerate(categories.items()):
            with cols[idx % len(cols)]:
                st.markdown(f"####  {cat.capitalize()} ({len(docs)})")
                for d in docs:
                    st.markdown(f"- {d['name']}")

        #  Tri effectif 
        col_copy, col_move = st.columns(2)
        with col_copy:
            do_copy = st.button(" Copier dans les dossiers", type="primary")
        with col_move:
            do_move = st.button(" Déplacer dans les dossiers")

        if do_copy or do_move:
            dest_base = Path(dest_folder)
            sorted_count = 0
            skipped_count = 0
            results_log = []

            for doc in st.session_state.processed_docs:
                original = doc.get("original_path", "")
                src = Path(original)

                if not src.exists():
                    skipped_count += 1
                    results_log.append(f" **{doc['name']}** — fichier source introuvable")
                    continue

                cat = doc["classification"]["category"]
                cat_dir = dest_base / cat
                cat_dir.mkdir(parents=True, exist_ok=True)

                dest = cat_dir / src.name
                counter = 1
                while dest.exists():
                    dest = cat_dir / f"{src.stem}_{counter}{src.suffix}"
                    counter += 1

                import shutil
                if do_move:
                    shutil.move(str(src), str(dest))
                    doc["original_path"] = str(dest)
                else:
                    shutil.copy2(str(src), str(dest))

                sorted_count += 1
                action = "Déplacé" if do_move else "Copié"
                results_log.append(f" **{doc['name']}** → `{cat}/{dest.name}` ({action})")

            _persist()

            if sorted_count > 0:
                st.success(f"{sorted_count} document(s) triés avec succès !")
            if skipped_count > 0:
                st.warning(
                    f"{skipped_count} fichier(s) ignorés (uploadés via le navigateur, "
                    "pas de fichier source local). Utilisez l'import par dossier pour trier les originaux."
                )

            for line in results_log:
                st.markdown(line)

            st.markdown(f"\n Destination : `{dest_base}`")


elif page == " Chat Q&A":
    st.markdown("##  Chat Q&A sur vos Documents")

    if st.session_state.rag_engine.store.get_document_count() == 0:
        st.info("Importez et analysez des documents d'abord pour pouvoir les interroger.")
    else:
        st.markdown(
            f"*{st.session_state.rag_engine.store.get_document_count()} document(s) "
            "disponible(s) pour la recherche*"
        )

    for msg in st.session_state.chat_history:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if "sources" in msg and msg["sources"]:
                with st.expander("📎 Sources"):
                    st.markdown(msg["sources"])

    if prompt := st.chat_input("Posez votre question sur les documents..."):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            with st.spinner("Recherche dans les documents..."):
                result = st.session_state.rag_engine.query(prompt)

            st.markdown(result["answer"])

            if result["sources"]:
                sources_text = "\n".join(f"- {s}" for s in result["sources"])
                with st.expander("📎 Sources utilisées"):
                    st.markdown(sources_text)
                    if "context_preview" in result:
                        st.markdown("**Contexte :**")
                        st.text(result["context_preview"])

        st.session_state.chat_history.append({
            "role": "assistant",
            "content": result["answer"],
            "sources": "\n".join(f"- {s}" for s in result["sources"]),
        })
        _persist()


elif page == " Métriques":
    st.markdown("##  Métriques de Performance")

    summary = st.session_state.tracker.summary()

    if summary.get("count", 0) == 0:
        st.info("Analysez des documents pour voir les métriques de performance.")
    else:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Documents traités", summary["count"])
        col2.metric("Temps OCR moyen", f"{summary['avg_ocr_time']}s")
        col3.metric("Confiance OCR moy.", f"{summary['avg_ocr_confidence']}%")
        col4.metric("Entités / document", summary["avg_entities_per_doc"])

        st.markdown("### Détail par document")
        records = st.session_state.tracker.to_table()
        if records:
            df = pd.DataFrame(records)
            st.dataframe(df, use_container_width=True)

            st.markdown("### Temps de traitement OCR")
            chart_df = df[["document", "ocr_time_s"]].set_index("document")
            st.bar_chart(chart_df)

            st.markdown("### Confiance OCR par document")
            conf_df = df[["document", "ocr_confidence"]].set_index("document")
            st.bar_chart(conf_df)

        st.markdown("### Évaluation des métriques")
        st.markdown("""
        | Métrique | Description | Cible |
        |----------|-------------|-------|
        | **Exact Match** | Champs extraits identiques à la vérité terrain | > 80% |
        | **F1-Score** | Précision/Rappel des entités | > 85% |
        | **Accuracy Q&A** | Taux de bonnes réponses | > 75% |
        | **CER** | Character Error Rate de l'OCR | < 5% |
        | **Temps** | Temps de traitement par document | < 10s |
        """)
