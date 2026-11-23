"""Paso 5 del asistente: verificador de cifras.

Extrae cada número de la respuesta y comprueba que exista en el contexto entregado.
También revisa que las citas [fN] correspondan a hechos reales del contexto.
"""
from __future__ import annotations

import re

NUM = re.compile(r"(\$\s?)?(\d{1,3}(?:\.\d{3})+|\d+(?:,\d+)?)(\s?(?:millones|millón|millon|mil)\b)?(\s?%)?", re.I)
CITA = re.compile(r"\[(f\d+)\]")
FECHA = re.compile(r"\b\d{4}-\d{2}-\d{2}\b|\b(19|20)\d{2}\b")


def _valor(entero: str, escala: str | None) -> float:
    if "." in entero and "," not in entero and re.fullmatch(r"\d{1,3}(\.\d{3})+", entero):
        v = float(entero.replace(".", ""))
    else:
        v = float(entero.replace(".", "").replace(",", "."))
    if escala:
        e = escala.strip().lower()
        v *= 1_000_000 if e.startswith("mill") else 1_000
    return v


def numeros(texto: str) -> list[tuple[str, float, str]]:
    """Devuelve (texto, valor, unidad) de cada cifra; unidad ∈ {'mill', 'mil', '', '%'}."""
    limpio = FECHA.sub(" ", CITA.sub(" ", texto))
    out = []
    for m in NUM.finditer(limpio):
        crudo = m.group(0).strip()
        if not crudo or crudo in ("$",):
            continue
        esc = (m.group(3) or "").strip().lower()
        unidad = "%" if m.group(4) else ("mill" if esc.startswith("mill") else ("mil" if esc else ""))
        out.append((crudo, _valor(m.group(2), m.group(3)), unidad))
    return out


def _permitidos(contexto: list[dict]) -> list[float]:
    vals = []
    for h in contexto:
        for k, v in h.items():
            if k == "id":
                continue
            vals.extend(x[1] for x in numeros(str(v)))
    return vals


def verificar(respuesta: str, contexto: list[dict]) -> dict:
    permitidos = _permitidos(contexto)
    ids = {h["id"] for h in contexto}
    cifras = numeros(respuesta)
    no_resp = []
    for crudo, v, u in cifras:
        tol = 50_000 if u == "mill" else (500 if u == "mil" else max(1.0, abs(v) * 0.01))
        if not any(abs(v - a) <= tol for a in permitidos):
            no_resp.append(crudo)
    citas = CITA.findall(respuesta)
    invalidas = sorted({c for c in citas if c not in ids})
    return {"cifras": len(cifras), "respaldadas": len(cifras) - len(no_resp), "no_respaldadas": no_resp,
            "citas": len(citas), "citas_invalidas": invalidas, "ok": not no_resp and not invalidas}
