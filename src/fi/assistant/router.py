"""Paso 2 del asistente: clasificar la pregunta en una de las 6 intenciones o marcarla fuera de alcance."""
from __future__ import annotations

import re
import unicodedata

INTENCIONES = {
    "mayor_gasto": "¿En qué categorías estoy gastando más?",
    "aumentos": "¿Qué gastos están aumentando respecto a mis hábitos?",
    "pronostico": "¿Cuánto podría gastar el próximo mes?",
    "inusual": "¿Hay algún movimiento diferente a lo habitual?",
    "meta": "¿Cómo afecta mi comportamiento a mi meta de ahorro?",
    "accion": "¿Qué acción podría realizar para mejorar?",
}

CLAVES = {
    "mayor_gasto": ["gastando mas", "gasto mas", "mayor gasto", "en que gasto", "en que estoy gastando", "donde se va",
                    "en que se me va", "mas gasto", "categorias", "en que gaste"],
    "aumentos": ["aument", "subi", "creci", "increment", "mas que antes", "mas de lo normal", "mas de lo esperado", "disparo"],
    "pronostico": ["proximo mes", "siguiente mes", "voy a gastar", "podria gastar", "pronost", "previst", "estim",
                   "cuanto gastare", "este mes voy", "fin de mes"],
    "inusual": ["raro", "inusual", "extrano", "anomal", "sospech", "diferente", "normal", "cobro", "cargo"],
    "meta": ["meta", "ahorr", "llego", "lograr", "objetivo", "alcanzar", "cumplir"],
    "accion": ["que hago", "que puedo hacer", "recomiend", "consejo", "mejorar", "reducir", "recortar", "accion", "sugier"],
}
FUERA_ALCANCE = ["invert", "invier", "inversion", "cripto", "bitcoin", "cdt", "fondo de", "fondos", "bolsa", "acciones de",
                 "prestamo", "credito", "hipoteca", "dolar", "comprar casa", "pension", "seguro de", "impuesto"]


def normalizar(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    return re.sub(r"\s+", " ", "".join(c for c in t if not unicodedata.combining(c)))


def clasificar(pregunta: str) -> str:
    q = normalizar(pregunta)
    if any(k in q for k in FUERA_ALCANCE):
        return "fuera_alcance"
    puntaje = {i: sum(k in q for k in ks) for i, ks in CLAVES.items()}
    mejor = max(puntaje, key=lambda i: puntaje[i])
    if puntaje[mejor] == 0:
        return "desconocida"
    # "¿en qué gasto más de lo esperado?" → aumentos
    if puntaje["aumentos"] and mejor == "mayor_gasto":
        return "aumentos"
    # "¿qué hago para llegar a mi meta?" → acción
    if puntaje["accion"] and puntaje["meta"] and puntaje["accion"] >= puntaje["meta"]:
        return "accion"
    return mejor
