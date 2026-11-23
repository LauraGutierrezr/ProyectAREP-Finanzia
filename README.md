# Finanzas Inteligentes · Demo del prototipo

Demo funcional del proyecto **«Sistema inteligente para el análisis y apoyo a la toma de decisiones en finanzas personales mediante inteligencia artificial»** 
AREP - Laura Valentina Gutiérrez Rico - Prof. Luis Daniel Benavides Navarro

La demo implementa la arquitectura del prototipo del documento (secciones 11.3 y 12): un **sistema de IA compuesto** donde cada tarea la resuelve el modelo adecuado y el modelo de lenguaje solo explica resultados ya calculados y verificados.

| Módulo | Qué hace | Tecnología |
|---|---|---|
| M1 · Ingesta | Lleva el dataset al esquema canónico (protocolo P1–P11) | pandas · Parquet · DuckDB |
| M4 · Motor predictivo | Modelo **global** de pronóstico del gasto por categoría | LightGBM |
| M5 · Detector de anomalías | Gastos raros **para cada persona** | Isolation Forest |
| M6 · Motor de recomendación | Reglas R1–R5 para cerrar la brecha de la meta | Reglas en YAML |
| M7 · Asistente fundamentado | RAG sobre datos estructurados + verificador de cifras + auditoría | Bedrock / Anthropic / OpenAI / Ollama o plantilla |
| M8 · Aplicación | Tablero y panel de evaluación | Streamlit + FastAPI |

---

## 1. Antes de empezar: instala Python

Necesitas **Python 3.10, 3.11, 3.12 o 3.13**. Para saber si ya lo tienes, abre una terminal y escribe:

```bash
python3 --version        # macOS / Linux
python --version         # Windows
```

Si no aparece una versión entre 3.10 y 3.13, descárgalo de <https://www.python.org/downloads/> (recomendado: 3.12).
En **Windows**, durante la instalación marca la casilla **«Add python.exe to PATH»**.


---

## 2. Correr la demo en macOS (o Linux)

1. **Descomprime** `finanzas-demo.zip` (doble clic). Queda una carpeta `finanzas-demo`, por ejemplo en *Descargas*.
2. **Abre la Terminal** (Cmd + Espacio → escribe «Terminal» → Enter).
3. **Entra a la carpeta**. Escribe `cd ` (con un espacio al final), **arrastra la carpeta `finanzas-demo` a la Terminal** y presiona Enter. Debería quedar algo como:
   ```bash
   cd ~/Downloads/finanzas-demo
   ```
4. **Ejecuta la demo:**
   ```bash
   ./run_demo.sh
   ```
   Si sale `permission denied`, ejecuta una vez `chmod +x run_demo.sh` y vuelve a intentarlo (o usa `bash run_demo.sh`).
5. **Espera.** La primera vez el script:
   - crea un entorno virtual (`.venv`) e instala las librerías (3–5 minutos),
   - genera los datos, entrena y evalúa los modelos (~2 minutos),
   - inicia la API y abre el navegador en **<http://localhost:8501>**.

   Las siguientes veces abre en unos segundos porque ya está todo instalado y calculado.
6. **Para detener la demo:** vuelve a la Terminal y presiona **Ctrl + C**.

---

## 3. Correr la demo en Windows

1. Descomprime `finanzas-demo.zip`.
2. Entra a la carpeta `finanzas-demo` y haz **doble clic en `run_demo.bat`**
   (o abre «Símbolo del sistema», entra con `cd ruta\a\finanzas-demo` y escribe `run_demo.bat`).
3. Espera a que se abra el navegador en <http://localhost:8501>.
4. Para detenerla: **Ctrl + C** en la ventana de la demo y cierra la ventana minimizada «API Finanzas Inteligentes».

---

## 4. Alternativa: paso a paso manual (dos terminales)

Útil si quieres ver qué hace cada parte o si el script falla.

```bash
cd finanzas-demo
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt
export PYTHONPATH=$PWD/src             # Windows: set PYTHONPATH=%cd%\src

python -m fi.pipeline                  # 1) datos → modelos → evaluación (una sola vez)
uvicorn fi.api:app --port 8000         # 2) API (déjala corriendo)
```

En **otra terminal** (activa de nuevo `.venv` y `PYTHONPATH`):

```bash
streamlit run app/streamlit_app.py     # 3) interfaz en http://localhost:8501
```

Otras direcciones útiles mientras corre:
- Documentación interactiva de la API: <http://localhost:8000/docs>
- Estado de la API: <http://localhost:8000/health>

---

## 5. Alternativa: Docker

Si tienes Docker Desktop instalado:

```bash
cd finanzas-demo
docker compose up --build
```

Levanta tres servicios: `pipeline` (lotes), `api` (puerto 8000) y `ui` (puerto 8501). Es la misma composición que se desplegaría en una instancia **Amazon EC2**.

---

## 6. (Opcional) Usar un modelo de lenguaje real

Sin configurar nada, el asistente responde con una **plantilla determinista** (condición C0 del documento), que no inventa cifras. Para que redacte un LLM:

1. Copia el archivo de ejemplo:
   ```bash
   cp .env.example .env          # Windows: copy .env.example .env
   ```
2. Abre `.env` con cualquier editor y quita el `#` de **un** proveedor:

| Proveedor | Qué necesitas |
|---|---|
| **Ollama** (gratis, local) | Instala <https://ollama.com>, ejecuta `ollama pull llama3.1` y deja Ollama abierto. Se detecta solo. |
| **Anthropic** | `ANTHROPIC_API_KEY=...` |
| **OpenAI** | `OPENAI_API_KEY=...` |
| **Amazon Bedrock** | `pip install boto3`, credenciales de AWS configuradas y `FI_BEDROCK_MODEL=...` |

3. Reinicia la demo y elige el proveedor en la barra lateral («Modelo de lenguaje»).

Si el LLM escribe una cifra que no está en los datos, el verificador la detecta, le pide corregirla y, si vuelve a fallar, responde con la plantilla. Todas las respuestas quedan en `data/audit/respuestas.jsonl`.

---

## 7. Qué mostrar en la presentación (≈ 5 minutos)

1. **Producto y arquitectura:** flujo del sistema compuesto, los 8 módulos con su estado y el diagrama de lo que está corriendo.
2. **Mi tablero**, cambiando de perfil en la barra lateral:
   - **Ana** (usuario estable): todo dentro de lo normal.
   - **Bruno** (gasto en aumento): una categoría sube y aparece una brecha en su meta.
   - **Camila** (gasto inusual): el detector marca un cargo nocturno muy por encima de lo habitual.
   - **David** (usuario nuevo): el modelo solo conoce 3 meses de historia.
3. **Mi meta** (con Bruno): cambia la meta o el ingreso y mira cómo cambian la brecha y las recomendaciones.
4. **Asistente** (con Camila): pregunta «¿Hay algún cobro raro?» y abre «Ver los datos que recibió el modelo». Luego pregunta «¿En qué fondo invierto?» → debe negarse (escenario E).
5. **Evaluación:** cada capacidad contra su línea base.

Resultados esperados (datos sintéticos, son siempre los mismos porque la semilla es fija):

| Capacidad | Resultado | Línea base |
|---|---|---|
| Pronóstico (MASE, usuarios vistos) | 0,836 | 0,974 (media de 3 meses) → mejora 14,1 % |
| Anomalías (PR-AUC) | 0,54 | 0,22 (z robusto) |
| Recomendaciones que aciertan | 62 % | 42 % («reduce tu mayor gasto») |
| Cifras del asistente respaldadas | 100 % | — |

---

## 8. Problemas frecuentes

| Problema | Solución |
|---|---|
| `python3: command not found` / `python no se reconoce` | Instala Python (sección 1). En Windows, reinstala marcando «Add python.exe to PATH». |
| `permission denied: ./run_demo.sh` | `chmod +x run_demo.sh` o `bash run_demo.sh`. |
| Error de LightGBM con `libomp` (macOS) | `brew install libomp`. Si no lo instalas, la demo usa el gradient boosting de scikit-learn y sigue funcionando. |
| «No se pudo conectar con la API» en el navegador | La API no arrancó. Revisa `data/api.log`; puede que el puerto 8000 esté ocupado. |
| `Address already in use` (puerto 8000 u 8501 ocupado) | Cierra la otra demo. En macOS: `lsof -i :8000` y luego `kill <PID>`. |
| Quiero recalcular todo desde cero | `./run_demo.sh --reiniciar` (Windows: `run_demo.bat --reiniciar`). |
| Se instaló con la versión de Python equivocada | Borra la carpeta `.venv` y vuelve a ejecutar el script. |

---

## 9. Usar el dataset real de Kaggle

El dataset principal del documento es *Credit Card Transactions Fraud Detection* (simulador Sparkov, Kaggle), que requiere cuenta. Por eso la demo genera transacciones **sintéticas con el mismo esquema y las mismas 14 categorías** (en pesos colombianos). Para usar el real:

1. Descarga `fraudTrain.csv` y `fraudTest.csv` de Kaggle.
2. Ponlos en `data/raw/`.
3. Ejecuta `./run_demo.sh --reiniciar`.

---

## 10. Comandos útiles

```bash
source .venv/bin/activate && export PYTHONPATH=$PWD/src
python -m fi.pipeline            # recalcula datos, modelos y métricas
python -m fi.pipeline --rapido   # igual, sin la evaluación del asistente
python -m fi.evaluation          # solo el arnés del asistente (G1, G3, G5)
pytest -q                        # 8 pruebas automáticas
```

## 11. Estructura del proyecto

```
finanzas-demo/
├── run_demo.sh / run_demo.bat   inicio con un solo comando
├── docker-compose.yml           pipeline + api + ui
├── requirements.txt             dependencias
├── .env.example                 configuración del LLM
├── config/reglas.yaml           reglas R1–R5 del recomendador (editables)
├── app/streamlit_app.py         interfaz (M8)
├── src/fi/
│   ├── data/generate.py         dataset sintético con el esquema de Sparkov
│   ├── data/prepare.py          protocolo P1–P11 → Parquet + DuckDB (M1)
│   ├── models/forecast.py       modelo global LightGBM + líneas base (M4)
│   ├── models/anomaly.py        Isolation Forest + anomalías inyectadas (M5)
│   ├── recommender.py           reglas R1–R5 + evaluación U1–U4 (M6)
│   ├── assistant/               router · context (RAG) · llm · verifier · templates (M7)
│   ├── evaluation.py            arnés del asistente
│   ├── pipeline.py              pipeline por lotes completo
│   └── api.py                   API FastAPI
├── tests/test_demo.py           pruebas con pytest
└── data/                        se llena al ejecutar (datos, resultados y auditoría)
```
