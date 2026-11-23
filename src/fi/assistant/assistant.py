"""M7 · Asistente fundamentado: el LLM explica, no calcula.

Flujo: 1) pregunta → 2) clasificar intención → 3) reunir contexto (RAG estructurado)
→ 4) redactar (LLM o plantilla) → 5) verificar cifras → registro de auditoría.
"""
from __future__ import annotations

import json
import time
from datetime import datetime

from fi import config
from fi.assistant import context, llm, router, templates, verifier

ABSTENCION = ("Esa pregunta está fuera de lo que puedo responder: no doy asesoría sobre inversiones, créditos ni otros "
              "productos financieros. Sí puedo explicarte tus gastos, tu pronóstico, movimientos inusuales y tu meta de ahorro.")
AYUDA = "No entendí la pregunta. Puedo responder cosas como: " + " · ".join(router.INTENCIONES.values())


def responder(user_id: str, pregunta: str, proveedor: str = "plantilla") -> dict:
    t0 = time.time()
    intencion = router.clasificar(pregunta)
    out = {"user_id": user_id, "pregunta": pregunta, "intencion": intencion, "proveedor": proveedor,
           "contexto": [], "intentos": 0, "respaldo_plantilla": False, "error": None}
    if intencion in ("fuera_alcance", "desconocida"):
        out["respuesta"] = ABSTENCION if intencion == "fuera_alcance" else AYUDA
        out["verificacion"] = verifier.verificar(out["respuesta"], [])
    else:
        ctx = context.construir(user_id, intencion).json()
        out["contexto"] = ctx
        if proveedor == "plantilla":
            texto = templates.redactar(intencion, ctx)
            out["intentos"] = 1
        else:
            texto, correccion = None, None
            for intento in (1, 2):
                out["intentos"] = intento
                try:
                    texto = llm.generar(proveedor, pregunta, ctx, correccion)
                except Exception as e:  # sin red, sin credenciales, etc.
                    out["error"] = f"{type(e).__name__}: {e}"[:300]
                    texto = None
                    break
                v = verifier.verificar(texto, ctx)
                if v["ok"]:
                    break
                correccion = ", ".join(v["no_respaldadas"] + v["citas_invalidas"])
                texto = None
            if texto is None:  # dos fallos → respuesta con plantilla (sección 5.6)
                texto = templates.redactar(intencion, ctx)
                out["respaldo_plantilla"] = True
        out["respuesta"] = texto
        out["verificacion"] = verifier.verificar(texto, ctx)
    out["latencia_s"] = round(time.time() - t0, 3)
    with open(config.AUDIT / "respuestas.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps({"ts": datetime.now().isoformat(timespec="seconds"), **out}, ensure_ascii=False) + "\n")
    return out
