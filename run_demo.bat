@echo off
REM Demo Finanzas Inteligentes (Windows)
REM   run_demo.bat              instala, prepara los datos y abre la aplicacion
REM   run_demo.bat --reiniciar  borra los datos generados y vuelve a ejecutar el pipeline
cd /d %~dp0
where python >nul 2>nul || (echo ERROR: no se encontro Python. Instalalo desde https://www.python.org/downloads/ marcando "Add python.exe to PATH". & pause & exit /b 1)
python -c "import sys; sys.exit(0 if (3,10) <= sys.version_info[:2] <= (3,13) else 1)" || (echo ERROR: se necesita Python 3.10 a 3.13. & pause & exit /b 1)
if not exist .venv (
  echo Creando entorno virtual...
  python -m venv .venv
)
call .venv\Scripts\activate.bat
if not exist .venv\instalado.txt (
  echo Instalando dependencias, la primera vez tarda unos minutos...
  python -m pip install -q --upgrade pip
  python -m pip install -q -r requirements.txt
  echo ok> .venv\instalado.txt
)
set PYTHONPATH=%cd%\src
if "%1"=="--reiniciar" (
  rmdir /s /q data\raw data\processed data\results data\audit 2>nul
)
if not exist data\results\personas.json (
  echo Ejecutando el pipeline por lotes, unos 2 minutos...
  python -m fi.pipeline
)
echo Iniciando la API en http://localhost:8000 ...
start "API Finanzas Inteligentes" /min cmd /c "python -m uvicorn fi.api:app --port 8000"
timeout /t 5 >nul
echo Abre http://localhost:8501 en tu navegador. Para detener: Ctrl + C y cierra la ventana "API Finanzas Inteligentes".
start "" http://localhost:8501
streamlit run app\streamlit_app.py --server.port 8501 --server.headless true
