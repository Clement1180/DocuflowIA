@echo off
echo ============================================
echo    DocuFlow AI - Lancement
echo ============================================
echo.
echo Installation des dependances...
pip install -r requirements.txt -q
echo.
echo Lancement de l'interface...
streamlit run docuflow_ai/app.py --server.port 8501
pause
