"""Configuración central de la demo Finanzas Inteligentes."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

# carga sencilla de variables desde .env (sin dependencias extra)
if (ROOT / ".env").exists():
    for _l in (ROOT / ".env").read_text(encoding="utf-8").splitlines():
        if "=" in _l and not _l.strip().startswith("#"):
            _k, _v = _l.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip().strip('"'))
DATA = Path(os.getenv("FI_DATA_DIR", ROOT / "data"))
RAW = DATA / "raw"
PROC = DATA / "processed"
RESULTS = DATA / "results"
AUDIT = DATA / "audit"
CONFIG = ROOT / "config"

for _p in (RAW, PROC, RESULTS, AUDIT):
    _p.mkdir(parents=True, exist_ok=True)

SEED = int(os.getenv("FI_SEED", "42"))
N_USERS = int(os.getenv("FI_N_USERS", "300"))
MONEDA = os.getenv("FI_MONEDA", "COP")

# Partición temporal (Tabla de partición del documento)
TRAIN_END = "2020-06"
VAL_MONTHS = ["2020-07", "2020-08", "2020-09"]
TEST_MONTHS = ["2020-10", "2020-11", "2020-12"]
MES_DEMO = "2020-12"          # mes que la demo trata como "este mes"
HOLDOUT_FRAC = 0.20           # usuarios no vistos (cold start)

# Las 14 categorías de consumo del conjunto Sparkov
CATEGORIAS = {
    "grocery_pos": "Mercado (tienda)",
    "grocery_net": "Mercado (en línea)",
    "gas_transport": "Transporte y gasolina",
    "food_dining": "Restaurantes",
    "entertainment": "Entretenimiento",
    "shopping_pos": "Compras (tienda)",
    "shopping_net": "Compras en línea",
    "health_fitness": "Salud y deporte",
    "personal_care": "Cuidado personal",
    "kids_pets": "Niños y mascotas",
    "home": "Hogar",
    "travel": "Viajes",
    "misc_pos": "Otros (tienda)",
    "misc_net": "Otros (en línea)",
}
ESENCIALES = {"grocery_pos", "grocery_net", "gas_transport", "health_fitness", "kids_pets", "home"}
DISCRECIONALES = set(CATEGORIAS) - ESENCIALES

API_URL = os.getenv("FI_API_URL", "http://localhost:8000")


def etiqueta(cat: str) -> str:
    return CATEGORIAS.get(cat, cat)


def pesos(v: float) -> str:
    """Formato colombiano: $1.234.567"""
    s = f"{abs(round(v)):,.0f}".replace(",", ".")
    return ("-$" if v < 0 else "$") + s
