"""Paso 4 del asistente: adaptador al modelo de lenguaje (patrón puerto-adaptador).

Proveedores: plantilla (sin LLM), bedrock (Amazon Bedrock), anthropic, openai, ollama (local).
Cambiar de proveedor no afecta al resto del sistema (decisión D3 del documento).
"""
from __future__ import annotations

import json
import os

import requests

SISTEMA = """Eres el asistente de Finanzas Inteligentes. Explicas a una persona sus propios datos financieros.
Reglas obligatorias:
1. Usa SOLO los hechos del CONTEXTO. No inventes cifras, comercios, fechas ni categorías.
2. Copia las cifras exactamente como aparecen en el contexto (por ejemplo "$1.234.567" o "12 %"). No hagas cálculos ni redondeos.
3. Después de cada cifra escribe la cita del hecho de donde sale, entre corchetes, por ejemplo [f2].
4. Si el contexto no tiene la información, dilo con claridad.
5. Nunca recomiendes productos de inversión, créditos ni fondos.
6. Responde en español, tuteando, con un tono cercano y en máximo 90 palabras."""


def proveedores_disponibles() -> list[str]:
    disp = ["plantilla"]
    if os.getenv("ANTHROPIC_API_KEY"):
        disp.append("anthropic")
    if os.getenv("OPENAI_API_KEY"):
        disp.append("openai")
    if os.getenv("FI_BEDROCK_MODEL"):
        try:
            import boto3  # noqa: F401
            disp.append("bedrock")
        except ImportError:
            pass
    try:
        if requests.get(os.getenv("OLLAMA_URL", "http://localhost:11434") + "/api/tags", timeout=0.5).ok:
            disp.append("ollama")
    except Exception:
        pass
    return disp


def _mensaje(pregunta: str, contexto: list[dict], correccion: str | None) -> str:
    txt = f"CONTEXTO:\n{json.dumps(contexto, ensure_ascii=False, indent=1)}\n\nPREGUNTA DEL USUARIO: {pregunta}"
    if correccion:
        txt += f"\n\nTu respuesta anterior tenía cifras que no están en el contexto: {correccion}. Corrígela usando solo cifras del contexto."
    return txt


def generar(proveedor: str, pregunta: str, contexto: list[dict], correccion: str | None = None) -> str:
    msg = _mensaje(pregunta, contexto, correccion)
    if proveedor == "anthropic":
        r = requests.post("https://api.anthropic.com/v1/messages", timeout=60, headers={
            "x-api-key": os.environ["ANTHROPIC_API_KEY"], "anthropic-version": "2023-06-01", "content-type": "application/json"},
            json={"model": os.getenv("FI_ANTHROPIC_MODEL", "claude-haiku-4-5"), "max_tokens": 400, "temperature": 0,
                  "system": SISTEMA, "messages": [{"role": "user", "content": msg}]})
        r.raise_for_status()
        return r.json()["content"][0]["text"].strip()
    if proveedor in ("openai", "ollama"):
        base = os.getenv("OLLAMA_URL", "http://localhost:11434") + "/v1" if proveedor == "ollama" else os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        modelo = os.getenv("FI_OLLAMA_MODEL", "llama3.1") if proveedor == "ollama" else os.getenv("FI_OPENAI_MODEL", "gpt-4o-mini")
        headers = {"Authorization": f"Bearer {os.getenv('OPENAI_API_KEY', 'ollama')}"}
        r = requests.post(base + "/chat/completions", headers=headers, timeout=120, json={
            "model": modelo, "temperature": 0, "messages": [{"role": "system", "content": SISTEMA}, {"role": "user", "content": msg}]})
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"].strip()
    if proveedor == "bedrock":
        import boto3  # opcional: pip install boto3
        cli = boto3.client("bedrock-runtime", region_name=os.getenv("AWS_REGION", "us-east-1"))
        r = cli.converse(modelId=os.getenv("FI_BEDROCK_MODEL", "anthropic.claude-3-haiku-20240307-v1:0"),
                         system=[{"text": SISTEMA}], messages=[{"role": "user", "content": [{"text": msg}]}],
                         inferenceConfig={"maxTokens": 400, "temperature": 0})
        return r["output"]["message"]["content"][0]["text"].strip()
    raise ValueError(f"Proveedor no soportado: {proveedor}")
