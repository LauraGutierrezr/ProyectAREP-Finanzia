#!/usr/bin/env bash
# Demo Finanzas Inteligentes (macOS / Linux)
#   ./run_demo.sh              instala (solo la primera vez), prepara los datos y abre la aplicación
#   ./run_demo.sh --reiniciar  borra los datos generados y vuelve a ejecutar el pipeline
set -e
cd "$(dirname "$0")"

# 1. Buscar Python 3.10 o superior
PY=${PYTHON:-}
if [ -z "$PY" ]; then
  for c in python3.12 python3.11 python3.13 python3.10 python3; do
    if command -v "$c" >/dev/null 2>&1; then PY=$c; break; fi
  done
fi
if [ -z "$PY" ]; then
  echo "ERROR: no se encontró Python. Instálalo desde https://www.python.org/downloads/ (3.10 a 3.13)."; exit 1
fi
if ! $PY -c 'import sys; sys.exit(0 if (3, 10) <= sys.version_info[:2] <= (3, 13) else 1)'; then
  echo "ERROR: se necesita Python 3.10 a 3.13 y se encontró $($PY --version)."; exit 1
fi
echo "Usando $($PY --version)"

# 2. Entorno virtual y dependencias (solo la primera vez)
if [ ! -d .venv ]; then
  echo "Creando entorno virtual..."
  $PY -m venv .venv
fi
source .venv/bin/activate
if [ ! -f .venv/.instalado ] || [ requirements.txt -nt .venv/.instalado ]; then
  echo "Instalando dependencias (la primera vez tarda unos minutos)..."
  python -m pip install -q --upgrade pip
  python -m pip install -q -r requirements.txt
  touch .venv/.instalado
fi
export PYTHONPATH="$PWD/src"

# 3. Pipeline por lotes (datos, modelos y evaluación)
if [ "$1" = "--reiniciar" ]; then
  echo "Borrando datos generados..."
  rm -rf data/raw data/processed data/results data/audit
fi
if [ ! -f data/results/personas.json ]; then
  echo "Ejecutando el pipeline por lotes (~2 minutos)..."
  python -m fi.pipeline
fi

# 4. API + interfaz
for p in 8000 8501; do
  if lsof -i :$p >/dev/null 2>&1; then
    echo "AVISO: el puerto $p ya está en uso. Cierra la otra demo o el programa que lo use (lsof -i :$p)."
  fi
done
echo "Iniciando la API en http://localhost:8000 ..."
python -m uvicorn fi.api:app --port 8000 > data/api.log 2>&1 &
API_PID=$!
trap "kill $API_PID 2>/dev/null" EXIT
sleep 4
echo ""
echo "  Abre http://localhost:8501 en tu navegador (se abre solo en unos segundos)."
echo "  Para detener la demo: Ctrl + C"
echo ""
( sleep 4; (command -v open >/dev/null && open http://localhost:8501) || (command -v xdg-open >/dev/null && xdg-open http://localhost:8501) ) >/dev/null 2>&1 &
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
