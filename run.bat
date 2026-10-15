@echo off
setlocal
cd /d "%~dp0"

echo ============================================
echo   tok_eval - starting FastAPI server
echo   http://127.0.0.1:8000
echo ============================================

python -m uvicorn app.main:app --host 127.0.0.1 --port 8000

endlocal
