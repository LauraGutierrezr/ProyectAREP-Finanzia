"""Acceso a los resultados precalculados (Parquet/DuckDB) que consume la API."""
from __future__ import annotations

import json
from functools import lru_cache

import pandas as pd

from fi import config


@lru_cache(maxsize=1)
def datos() -> dict:
    d = {
        "mensual": pd.read_parquet(config.PROC / "mensual.parquet"),
        "usuarios": pd.read_parquet(config.PROC / "usuarios.parquet").set_index("user_id"),
        "tx": pd.read_parquet(config.PROC / "transacciones.parquet"),
        "pron": pd.read_parquet(config.RESULTS / "pronosticos.parquet"),
        "anom": pd.read_parquet(config.RESULTS / "anomalias_demo.parquet"),
    }
    d["metricas"] = {}
    for f in config.RESULTS.glob("*.json"):
        d["metricas"][f.stem] = json.loads(f.read_text())
    return d


def refrescar() -> None:
    datos.cache_clear()


def estadisticas_categoria(user_id: str, mes: str) -> pd.DataFrame:
    """Mediana de 6 meses y percentil 25 de 12 meses anteriores a `mes`, por categoría."""
    m = datos()["mensual"]
    h = m[(m.user_id == user_id) & (m.month < mes)].sort_values("month")
    ult6 = h.groupby("category").tail(6).groupby("category")["spend"].median()
    ult12 = h.groupby("category").tail(12).groupby("category")["spend"]
    return pd.DataFrame({"mediana_6m": ult6, "p25_12m": ult12.quantile(0.25), "media_12m": ult12.mean()})


def pronostico_usuario(user_id: str, mes: str) -> pd.DataFrame:
    p = datos()["pron"]
    p = p[(p.user_id == user_id) & (p.month == mes)].set_index("category")
    return p


def factor_total() -> float:
    return float(datos()["metricas"].get("metricas_pronostico", {}).get("factor_calibracion_total", 1.0))
