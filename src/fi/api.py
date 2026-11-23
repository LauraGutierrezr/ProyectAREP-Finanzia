"""API FastAPI: el «cerebro» de la aplicación (sección 11.3).

Endpoints: /summary /forecast /anomalies /recommendations /ask + /metrics /arquitectura /personas
"""
from __future__ import annotations

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from fi import config, recommender, store
from fi.assistant import assistant, llm

app = FastAPI(title="Finanzas Inteligentes · API del prototipo", version="1.0")

MODULOS = [
    {"id": "M1", "nombre": "Ingesta y normalización", "estado": "simplificado", "detalle": "Carga del dataset (Sparkov real o sintético con el mismo esquema) al esquema canónico", "tecnologia": "pandas · Parquet · DuckDB"},
    {"id": "M2", "nombre": "Categorización", "estado": "diferido", "detalle": "Se usa la categoría del dataset; el clasificador NLP queda para trabajo futuro", "tecnologia": "—"},
    {"id": "M3", "nombre": "Perfil y metas", "estado": "simplificado", "detalle": "Ingreso simulado y meta de ahorro editable desde la interfaz", "tecnologia": "pandas"},
    {"id": "M4", "nombre": "Motor predictivo", "estado": "implementado", "detalle": "Modelo global de pronóstico con gradient boosting", "tecnologia": "LightGBM"},
    {"id": "M5", "nombre": "Detector de anomalías", "estado": "implementado", "detalle": "Detección no supervisada con variables relativas al usuario", "tecnologia": "Isolation Forest (scikit-learn)"},
    {"id": "M6", "nombre": "Motor de recomendación", "estado": "implementado", "detalle": "Reglas R1–R5 + pronóstico para cerrar la brecha de la meta", "tecnologia": "Reglas YAML"},
    {"id": "M7", "nombre": "Asistente fundamentado", "estado": "implementado", "detalle": "RAG sobre datos estructurados + verificador de cifras + auditoría", "tecnologia": "Bedrock / Anthropic / OpenAI / Ollama o plantilla"},
    {"id": "M8", "nombre": "Aplicación y administración", "estado": "simplificado", "detalle": "Tablero web y panel de evaluación", "tecnologia": "Streamlit"},
]


def _usuario(uid: str):
    us = store.datos()["usuarios"]
    if uid not in us.index:
        raise HTTPException(404, f"Usuario {uid} no existe")
    return us.loc[uid]


@app.get("/health")
def health():
    return {"ok": True, "proveedores_llm": llm.proveedores_disponibles()}


@app.get("/personas")
def personas():
    return store.datos()["metricas"].get("personas", {})


@app.get("/users")
def usuarios():
    us = store.datos()["usuarios"]
    return [{"user_id": u, "edad": int(r.edad), "ciudad": r.ciudad, "no_visto": bool(r.no_visto)} for u, r in us.iterrows()]


@app.get("/users/{uid}/summary")
def resumen(uid: str):
    u = _usuario(uid)
    m = store.datos()["mensual"]
    h = m[m.user_id == uid]
    if u.no_visto:  # usuario nuevo: solo se muestran sus últimos 3 meses
        h = h[h.month >= str(pd.Period(config.MES_DEMO, "M") - 3)]
    serie = h[h.month < config.MES_DEMO].groupby("month")["spend"].sum()   # diciembre es el mes "en curso"
    cat = h[h.month == str(pd.Period(config.MES_DEMO, "M") - 1)].groupby("category")["spend"].sum().sort_values(ascending=False)
    return {"user_id": uid, "edad": int(u.edad), "ciudad": u.ciudad, "ocupacion": u.ocupacion, "ingreso": float(u.ingreso),
            "no_visto": bool(u.no_visto), "meta_monto": float(u.meta_monto), "meta_meses": int(u.meta_meses),
            "gasto_mensual": {k: float(v) for k, v in serie.items()},
            "ultimo_mes_por_categoria": [{"categoria": c, "etiqueta": config.etiqueta(c), "monto": float(v)} for c, v in cat.items() if v > 0]}


@app.get("/users/{uid}/forecast")
def pronostico(uid: str, mes: str = config.MES_DEMO):
    _usuario(uid)
    p = store.pronostico_usuario(uid, mes)
    est = store.estadisticas_categoria(uid, mes)
    t = p.join(est)
    filas = [{"categoria": c, "etiqueta": config.etiqueta(c), "previsto": float(r.pred), "real": float(r.spend),
              "mediana_6m": float(r.mediana_6m) if pd.notna(r.mediana_6m) else None, "linea_base_media3": float(r.media3)}
             for c, r in t.sort_values("pred", ascending=False).iterrows()]
    return {"mes": mes, "total_previsto": float(p.pred.sum() * store.factor_total()), "total_real": float(p.spend.sum()),
            "categorias": filas}


@app.get("/users/{uid}/anomalies")
def anomalias(uid: str):
    _usuario(uid)
    a = store.datos()["anom"]
    a = a[(a.user_id == uid)].sort_values("score_if", ascending=False)
    return {"mes": config.MES_DEMO, "alertas": [
        {"fecha": str(r.ts), "comercio": r.merchant, "categoria": config.etiqueta(r.category), "monto": float(r.amount),
         "tipico": float(r.med_ref), "veces": float(r.ratio_med), "puntaje": float(r.score_if), "motivo": r.motivo, "origen": r.inyectada or "real"}
        for r in a[a.alerta].itertuples()], "revisadas": int(len(a))}


@app.get("/users/{uid}/recommendations")
def recomendaciones(uid: str, ingreso: float | None = None, meta_monto: float | None = None, meta_meses: int | None = None):
    _usuario(uid)
    return recommender.recomendar(uid, config.MES_DEMO, ingreso, meta_monto, meta_meses)


class Pregunta(BaseModel):
    pregunta: str
    proveedor: str = "plantilla"


@app.post("/users/{uid}/ask")
def preguntar(uid: str, body: Pregunta):
    _usuario(uid)
    return assistant.responder(uid, body.pregunta, body.proveedor)


@app.get("/metrics")
def metricas():
    return store.datos()["metricas"]


@app.get("/arquitectura")
def arquitectura():
    return {"modulos": MODULOS, "motor_pronostico": store.datos()["metricas"].get("metricas_pronostico", {}).get("motor")}
