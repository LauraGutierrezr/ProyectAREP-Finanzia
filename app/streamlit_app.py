"""Interfaz Streamlit de la demo Finanzas Inteligentes (M8 simplificado).

La interfaz NO calcula nada: todo lo pide a la API FastAPI, igual que en la arquitectura del prototipo.
"""
from __future__ import annotations

import json
import os

import altair as alt
import pandas as pd
import requests
import streamlit as st

API = os.getenv("FI_API_URL", "http://localhost:8000")
VERDE, VERDE_OSC, VERDE_CLARO, GRIS, AMBAR = "#1E7B45", "#123D25", "#EEF5F0", "#8A9590", "#C98A12"

st.set_page_config(page_title="Finanzas Inteligentes · Demo", page_icon=":material/savings:", layout="wide")

st.markdown(f"""
<style>
  .block-container {{padding-top: 1.2rem; max-width: 1280px;}}
  h1, h2, h3 {{color: {VERDE_OSC}; letter-spacing: -0.01em;}}
  .banda {{background: {VERDE_OSC}; color: #fff; padding: 22px 28px; border-radius: 10px; margin-bottom: 14px;}}
  .banda h1 {{color: #fff; margin: 0; font-size: 2.3rem; font-weight: 800;}}
  .banda p {{margin: 4px 0 0 0; color: #CFE6D6; font-size: 1.05rem;}}
  .tarjeta {{border: 1px solid #D5E3D9; border-radius: 10px; padding: 14px 16px; background: #fff; min-height: 215px;}}
  .tarjeta h4 {{margin: 0 0 4px 0; color: {VERDE_OSC}; font-size: 1.02rem;}}
  .tarjeta p {{margin: 0; color: #3B4A41; font-size: 0.92rem;}}
  .chip {{display: inline-block; padding: 2px 9px; border-radius: 99px; font-size: 0.75rem; font-weight: 700; margin-bottom: 6px;}}
  .implementado {{background: #D8EFDF; color: #135A2F;}}
  .simplificado {{background: #FBEFD2; color: #7A5205;}}
  .diferido {{background: #ECEEED; color: #5B635F;}}
  .kpi {{background: {VERDE_CLARO}; border-radius: 10px; padding: 12px 16px;}}
  .kpi .v {{font-size: 1.6rem; font-weight: 800; color: {VERDE_OSC};}}
  .kpi .l {{font-size: 0.85rem; color: #4A5A50;}}
  .burbuja {{background: {VERDE_CLARO}; border-left: 4px solid {VERDE}; border-radius: 8px; padding: 14px 16px; font-size: 1.02rem;}}
  .alerta {{border: 1px solid #F0D9A8; background: #FFFAEE; border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;}}
  .rec {{border: 1px solid #CFE3D5; border-radius: 10px; padding: 12px 14px; margin-bottom: 8px; background: #fff;}}
  .paso {{background: {VERDE_CLARO}; border-radius: 8px; padding: 10px; text-align: center; font-weight: 700; color: {VERDE_OSC}; font-size: 0.9rem;}}
  .flecha {{text-align: center; color: {VERDE}; font-size: 1.4rem; padding-top: 6px;}}
  .stTabs [data-baseweb="tab"] {{font-weight: 700; font-size: 1rem;}}
</style>""", unsafe_allow_html=True)


def pesos(v) -> str:
    if v is None:
        return "—"
    return ("-$" if v < 0 else "$") + f"{abs(round(v)):,.0f}".replace(",", ".")


@st.cache_data(ttl=60, show_spinner=False)
def get(path: str, **params):
    r = requests.get(API + path, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def post(path: str, body: dict):
    r = requests.post(API + path, json=body, timeout=120)
    r.raise_for_status()
    return r.json()


def kpi(col, valor, etiqueta):
    col.markdown(f'<div class="kpi"><div class="v">{valor}</div><div class="l">{etiqueta}</div></div>', unsafe_allow_html=True)


# ------------------------------------------------------------------ conexión con la API
try:
    salud = get("/health")
except Exception:
    st.error(f"No se pudo conectar con la API en {API}. Inicia primero el backend (ver README).")
    st.stop()

personas = get("/personas")
usuarios = get("/users")

st.markdown('<div class="banda"><h1>Finanzas Inteligentes</h1>'
            '<p>De tus movimientos a decisiones: predicción, alertas, recomendaciones y un asistente que no inventa cifras.</p></div>',
            unsafe_allow_html=True)

# ------------------------------------------------------------------ barra lateral
with st.sidebar:
    st.markdown("### Perfil de prueba")
    opciones = {f"{p['alias']} · {p['escenario']}": p["user_id"] for p in personas.values()}
    opciones["Otro usuario del dataset"] = None
    eleccion = st.radio("Escenario", list(opciones), label_visibility="collapsed")
    uid = opciones[eleccion]
    if uid is None:
        uid = st.selectbox("Usuario (seudónimo)", [u["user_id"] for u in usuarios])
    persona = next((p for p in personas.values() if p["user_id"] == uid), None)
    if persona:
        st.caption(persona["descripcion"])
    st.markdown("### Modelo de lenguaje")
    proveedor = st.selectbox("Proveedor", salud["proveedores_llm"],
                             help="«plantilla» responde sin LLM (condición C0). Configura una API key para usar un LLM real.")
    st.markdown("---")
    st.caption(f"API: {API}  \nMes de la demo: diciembre de 2020  \nDatos sintéticos con el esquema de Sparkov")

resumen = get(f"/users/{uid}/summary")
tab_inicio, tab_tablero, tab_meta, tab_asistente, tab_eval = st.tabs(
    ["Producto y arquitectura", "Mi tablero", "Mi meta", "Asistente", "Evaluación"])

# ------------------------------------------------------------------ 1. producto y arquitectura
with tab_inicio:
    st.markdown("## Así funciona por dentro")
    st.write("Un **sistema de IA compuesto**: cada tarea la resuelve el modelo adecuado y el modelo de lenguaje "
             "solo explica resultados que ya fueron calculados y verificados.")
    pasos = ["Movimientos", "Pronóstico global\n(LightGBM)", "Anomalías\n(Isolation Forest)", "Recomendador\n(reglas R1–R5)", "Asistente\n(RAG + verificador)"]
    cols = st.columns([3, 1, 3, 1, 3, 1, 3, 1, 3])
    for i, p in enumerate(pasos):
        cols[i * 2].markdown(f'<div class="paso">{p.replace(chr(10), "<br>")}</div>', unsafe_allow_html=True)
        if i < len(pasos) - 1:
            cols[i * 2 + 1].markdown('<div class="flecha">→</div>', unsafe_allow_html=True)

    st.markdown("### Arquitectura funcional: ocho módulos")
    arq = get("/arquitectura")
    for fila in range(2):
        cols = st.columns(4)
        for j, m in enumerate(arq["modulos"][fila * 4:(fila + 1) * 4]):
            cols[j].markdown(f'<div class="tarjeta"><span class="chip {m["estado"]}">{m["estado"].upper()}</span>'
                             f'<h4>{m["id"]} · {m["nombre"]}</h4><p>{m["detalle"]}</p>'
                             f'<p style="margin-top:6px;color:{GRIS};font-size:0.82rem">{m["tecnologia"]}</p></div>',
                             unsafe_allow_html=True)
        st.write("")

    st.markdown("### Arquitectura del prototipo que está corriendo")
    st.graphviz_chart(f"""
    digraph G {{
      rankdir=TB; bgcolor="transparent"; nodesep=0.35; ranksep=0.35; node [shape=box, style="rounded,filled", fontname="Helvetica", fontsize=12, width=2.6, color="#9DBBA6", fillcolor="#FFFFFF"];
      edge [color="#5F7A68", fontsize=9, fontname="Helvetica"];
      subgraph cluster_lotes {{ label="Pipeline por lotes (python -m fi.pipeline)"; style="rounded"; color="#9DBBA6"; fontname="Helvetica";
        ds [label="Dataset Sparkov\\n(real o sintético)", fillcolor="#FFF7D6"]; prep [label="Preparación P1–P11\\npandas"];
        pq [label="Parquet + DuckDB", fillcolor="#FFF7D6"]; tr [label="Entrenar y evaluar\\n{arq['motor_pronostico']} · Isolation Forest", fillcolor="#E2F3E7"];
        res [label="Resultados precalculados\\n+ métricas", fillcolor="#FFF7D6"]; ds -> prep -> pq -> tr -> res; }}
      subgraph cluster_app {{ label="Aplicación (local o Docker Compose)"; style="rounded"; color="#9DBBA6"; fontname="Helvetica";
        ui [label="Interfaz Streamlit\\n(esta pantalla)"]; api [label="API FastAPI\\n/summary /forecast /anomalies\\n/recommendations /ask"];
        rec [label="Recomendador\\nreglas YAML", fillcolor="#E2F3E7"]; asi [label="Asistente\\nclasifica → contexto → LLM → verifica", fillcolor="#E2F3E7"];
        aud [label="Auditoría JSONL"]; ui -> api [dir=both]; api -> rec; api -> asi; rec -> asi; asi -> aud; }}
      llm [label="LLM: {proveedor}\\n(Bedrock · Anthropic · OpenAI · Ollama)", style="rounded,dashed"];
      res -> api [label="lee"]; asi -> llm [dir=both];
    }}""", use_container_width=True)

# ------------------------------------------------------------------ 2. tablero
with tab_tablero:
    titulo = f"{persona['alias']}, " if persona else ""
    st.markdown(f"## {titulo}esto pasó con tu dinero")
    c = st.columns(4)
    c[0].markdown(f"**Edad:** {resumen['edad']} años")
    c[1].markdown(f"**Ciudad:** {resumen['ciudad']}")
    c[2].markdown(f"**Ocupación:** {resumen['ocupacion']}")
    c[3].markdown(f"**Ingreso mensual:** {pesos(resumen['ingreso'])}")
    if resumen["no_visto"]:
        st.info("Usuario nuevo: el modelo nunca vio a esta persona y solo conoce sus últimos 3 meses (arranque en frío).")

    pr = get(f"/users/{uid}/forecast")
    an = get(f"/users/{uid}/anomalies")
    serie = pd.Series(resumen["gasto_mensual"])
    k = st.columns(4)
    kpi(k[0], pesos(serie.get("2020-11")), "Gasto de noviembre")
    kpi(k[1], pesos(pr["total_previsto"]), "Gasto previsto de diciembre")
    kpi(k[2], pesos(serie[serie.index < "2020-12"].tail(3).mean()), "Promedio últimos 3 meses")
    kpi(k[3], str(len(an["alertas"])), "Movimientos inusuales en diciembre")
    st.write("")

    izq, der = st.columns([3, 2])
    with izq:
        st.markdown("### Gasto mes a mes")
        df = serie.rename("Gasto").reset_index().rename(columns={"index": "Mes"})
        df["Tipo"] = "Real"
        df = pd.concat([df, pd.DataFrame([{"Mes": "2020-12", "Gasto": pr["total_previsto"], "Tipo": "Pronóstico"}])])
        base = alt.Chart(df).encode(x=alt.X("Mes:N", title=None, axis=alt.Axis(labelAngle=-45)),
                                    y=alt.Y("Gasto:Q", title="COP", axis=alt.Axis(format="~s")))
        linea = base.transform_filter(alt.datum.Tipo == "Real").mark_line(color=VERDE, point=alt.OverlayMarkDef(color=VERDE, size=40), strokeWidth=2.5)
        punto = base.transform_filter(alt.datum.Tipo == "Pronóstico").mark_point(color=AMBAR, size=160, filled=True, shape="diamond")
        st.altair_chart((linea + punto).properties(height=300), use_container_width=True)
        st.caption("Rombo: gasto total que el modelo estima para diciembre de 2020.")
    with der:
        st.markdown("### Diciembre por categoría")
        cats = pd.DataFrame(pr["categorias"]).head(8)
        largo = cats.melt(id_vars="etiqueta", value_vars=["previsto", "mediana_6m"], var_name="Serie", value_name="COP")
        largo["Serie"] = largo["Serie"].map({"previsto": "Previsto", "mediana_6m": "Tu mediana (6 meses)"})
        st.altair_chart(alt.Chart(largo).mark_bar().encode(
            y=alt.Y("etiqueta:N", title=None, sort="-x"), x=alt.X("COP:Q", axis=alt.Axis(format="~s")),
            color=alt.Color("Serie:N", scale=alt.Scale(range=[VERDE, "#B9D7C3"]), legend=alt.Legend(orient="bottom", title=None)),
            yOffset="Serie:N").properties(height=300), use_container_width=True)

    st.markdown("### Movimientos que vale la pena revisar")
    if not an["alertas"]:
        st.success(f"El detector revisó {an['revisadas']} movimientos de diciembre y ninguno se sale de lo habitual.")
    for a in an["alertas"][:5]:
        st.markdown(f'<div class="alerta"><b>{pesos(a["monto"])}</b> en {a["comercio"]} · {a["categoria"]} · '
                    f'{a["fecha"][:16]}<br><span style="color:#6B5A2E">{a["motivo"]} (lo típico: {pesos(a["tipico"])})</span></div>',
                    unsafe_allow_html=True)

    with st.expander("Ver pronóstico por categoría frente a la línea base"):
        t = pd.DataFrame(pr["categorias"])[["etiqueta", "previsto", "linea_base_media3", "real", "mediana_6m"]]
        t.columns = ["Categoría", "Previsto (LightGBM)", "Línea base (media 3 meses)", "Real (diciembre)", "Mediana 6 meses"]
        st.dataframe(t.style.format({c: pesos for c in t.columns[1:]}), use_container_width=True, hide_index=True)

# ------------------------------------------------------------------ 3. meta
with tab_meta:
    st.markdown("## ¿Cuánto te falta para tu meta?")
    st.write("La recomendación parte de una cuenta simple: **Brecha = Aporte necesario − (Ingreso − Gasto previsto)**.")
    c = st.columns(3)
    meta_monto = c[0].number_input("Meta de ahorro (COP)", 100_000, 50_000_000, int(resumen["meta_monto"]), step=100_000)
    meta_meses = c[1].number_input("Plazo (meses)", 1, 36, int(resumen["meta_meses"]))
    ingreso = c[2].number_input("Ingreso mensual (COP)", 500_000, 50_000_000, int(resumen["ingreso"]), step=50_000)
    r = get(f"/users/{uid}/recommendations", ingreso=ingreso, meta_monto=meta_monto, meta_meses=meta_meses)
    mt = r["meta"]
    k = st.columns(4)
    kpi(k[0], pesos(mt["aporte_necesario"]), "Debes ahorrar al mes")
    kpi(k[1], pesos(mt["gasto_previsto"]), "Gasto previsto (diciembre)")
    kpi(k[2], pesos(mt["ahorro_proyectado"]), "Ahorro proyectado")
    kpi(k[3], pesos(max(mt["brecha"], 0)), "Te falta al mes" if mt["brecha"] > 0 else "Sin brecha")
    st.write("")
    sin = f"{mt['meses_sin_cambios']:.0f} meses" if mt["meses_sin_cambios"] and mt["meses_sin_cambios"] < 120 else "no se alcanza"
    con = f"{mt['meses_con_acciones']:.0f} meses" if mt["meses_con_acciones"] and mt["meses_con_acciones"] < 120 else "no se alcanza"
    st.markdown(f"**Sin cambios:** {sin} · **Aplicando las recomendaciones:** {con}")
    st.markdown("### Recomendaciones del mes")
    for rec in r["recomendaciones"]:
        imp = f'<br><span style="color:{VERDE};font-weight:700">Impacto estimado: {pesos(rec["impacto"])} al mes</span>' \
            if rec["regla"] in ("R1", "R3", "R4") else ""
        st.markdown(f'<div class="rec"><span class="chip implementado">{rec["regla"]} · {rec["nombre"]}</span><br>{rec["texto"]}{imp}</div>',
                    unsafe_allow_html=True)
    st.caption("Límites: nunca toca gastos esenciales · no pide bajar de un nivel que ya lograste (percentil 25) · máximo 3 al mes.")

# ------------------------------------------------------------------ 4. asistente
with tab_asistente:
    st.markdown("## Pregúntale a tu asistente")
    st.write("El asistente sigue cinco pasos: clasifica la pregunta, reúne los datos ya calculados, redacta, "
             "**revisa cada cifra** y guarda la respuesta en la auditoría.")
    sugeridas = ["¿En qué estoy gastando más?", "¿Qué gastos están aumentando?", "¿Cuánto voy a gastar este mes?",
                 "¿Hay algún cobro raro?", "¿Por qué no llego a mi meta?", "¿Qué puedo hacer para mejorar?", "¿En qué fondo invierto?"]
    cols = st.columns(4)
    for i, q in enumerate(sugeridas):
        if cols[i % 4].button(q, use_container_width=True, key=f"q{i}"):
            st.session_state["pregunta"] = q
    pregunta = st.text_input("Escribe tu pregunta", value=st.session_state.get("pregunta", ""))
    if pregunta:
        with st.spinner("Calculando y verificando…"):
            resp = post(f"/users/{uid}/ask", {"pregunta": pregunta, "proveedor": proveedor})
        st.markdown(f'<div class="burbuja">{resp["respuesta"]}</div>', unsafe_allow_html=True)
        v = resp["verificacion"]
        c = st.columns(4)
        c[0].metric("Intención", resp["intencion"])
        c[1].metric("Cifras respaldadas", f'{v["respaldadas"]} de {v["cifras"]}')
        c[2].metric("Proveedor", resp["proveedor"] + (" → plantilla" if resp["respaldo_plantilla"] else ""))
        c[3].metric("Latencia", f'{resp["latencia_s"]:.2f} s')
        if v["ok"]:
            st.success("Verificación superada: cada cifra de la respuesta existe en los datos calculados.")
        else:
            st.error(f"Cifras no respaldadas: {', '.join(v['no_respaldadas'])}")
        if resp.get("error"):
            st.warning(f"El LLM no respondió ({resp['error']}); se usó la plantilla.")
        with st.expander("Ver los datos que recibió el modelo (contexto con identificadores)"):
            st.code(json.dumps(resp["contexto"], ensure_ascii=False, indent=1), language="json")

# ------------------------------------------------------------------ 5. evaluación
with tab_eval:
    st.markdown("## ¿Funciona mejor que lo simple?")
    st.write("Cada capacidad se compara contra una línea base, sobre los meses de prueba (octubre a diciembre de 2020).")
    met = get("/metrics")
    fp, fa = met.get("metricas_pronostico", {}), met.get("metricas_anomalias", {})
    fr, fs = met.get("metricas_recomendacion", {}), met.get("metricas_asistente", {})
    fd = met.get("ficha_datos", {})
    c = st.columns(4)
    if fp:
        v = fp["usuarios_vistos"]
        kpi(c[0], f"{v['mejora_MASE']:.0%}", f"Mejora del pronóstico en MASE vs {v['mejor_linea_base']}")
    if fa:
        pm = fa["promedio_5_semillas"]
        kpi(c[1], f"{pm['PR_AUC_isolation_forest']:.2f} vs {pm['PR_AUC_z_robusto']:.2f}", "PR-AUC anomalías: Isolation Forest vs z robusto")
    if fr:
        kpi(c[2], f"{fr['U1_precision_sistema']:.0%} vs {fr['U1_precision_B1']:.0%}", "Recomendaciones que aciertan (sistema vs «reduce tu mayor gasto»)")
    if fs:
        g = fs["proveedores"].get("plantilla", {})
        kpi(c[3], f"{g.get('G1_cifras_respaldadas', 0):.0%}", "Cifras del asistente respaldadas por los datos")
    st.write("")

    if fp:
        st.markdown("### Pronóstico (M4)")
        filas = []
        for grupo, nombre in (("usuarios_vistos", "Usuarios vistos"), ("usuarios_no_vistos", "Usuarios nuevos (cold start)")):
            for metodo, et in (("naive", "Último mes"), ("estacional", "Mismo mes año anterior"), ("media3", "Media 3 meses"), ("pred", fp["motor"])):
                mm = fp[grupo]["metricas"][metodo]
                filas.append({"Grupo": nombre, "Método": et, "MAE": pesos(mm["MAE"]), "WAPE": f"{mm['WAPE']:.1%}", "MASE": round(mm["MASE"], 3)})
        st.dataframe(pd.DataFrame(filas), use_container_width=True, hide_index=True)
        v, n = fp["usuarios_vistos"], fp["usuarios_no_vistos"]
        st.caption(f"Usuarios vistos: mejora {v['mejora_MASE']:.1%} (p = {v['p_wilcoxon']:.1e}) → criterio de éxito ≥ 10 %: "
                   f"{'cumple' if v['exito'] else 'no cumple'}. Usuarios nuevos: mejora {n['mejora_MASE']:.1%} → "
                   f"{'cumple' if n['exito'] else 'no cumple'}.")
    if fa:
        st.markdown("### Anomalías (M5)")
        pm = fa["promedio_5_semillas"]
        t = pd.DataFrame([
            {"Tipo inyectado": "A1 · monto 3–10 veces lo habitual", "Isolation Forest": pm["recall_A1"], "z robusto ≥ 3,5": pm["recall_z_A1"]},
            {"Tipo inyectado": "A2 · categoría nunca usada", "Isolation Forest": pm["recall_A2"], "z robusto ≥ 3,5": pm["recall_z_A2"]},
            {"Tipo inyectado": "A3 · hora nunca usada", "Isolation Forest": pm["recall_A3"], "z robusto ≥ 3,5": pm["recall_z_A3"]},
            {"Tipo inyectado": "A4 · ráfaga en el mismo comercio", "Isolation Forest": pm["recall_A4"], "z robusto ≥ 3,5": pm["recall_z_A4"]},
        ])
        st.dataframe(t.style.format({"Isolation Forest": "{:.0%}", "z robusto ≥ 3,5": "{:.0%}"}), use_container_width=True, hide_index=True)
        st.caption(f"Recall por tipo (promedio de 5 semillas). Alertas por usuario al mes: {pm['alertas_usuario_mes']:.2f} (límite 2). "
                   f"PR-AUC con la etiqueta de fraude: {fa['fraude_secundario']['PR_AUC_isolation_forest']:.2f}.")
    if fr:
        st.markdown("### Recomendaciones (M6)")
        t = pd.DataFrame([{"Alfa (adherencia)": f"{float(a):.0%}", "Sistema": f"{x['sistema']:.0%}", "Línea base B1": f"{x['B1']:.0%}"}
                          for a, x in fr["U2_en_camino"].items()])
        a, b = st.columns(2)
        a.markdown(f"- **U1** Precisión@3: {fr['U1_precision_sistema']:.0%} vs {fr['U1_precision_B1']:.0%} (B1)\n"
                   f"- **U3** Factibles: {fr['U3_factibles']:.0%} · en gastos esenciales: {fr['U3_en_esenciales']}\n"
                   f"- **U4** Recomendaciones por usuario al mes: {fr['U4_recomendaciones_por_usuario_mes']:.1f}\n"
                   f"- **U5** Estudio con personas: {fr['U5_estudio_humano']}")
        b.markdown("**U2** Usuarios en camino a su meta (simulación)")
        b.dataframe(t, use_container_width=True, hide_index=True)
    if fs:
        st.markdown("### Asistente (M7)")
        nombres = {"G1_cifras_respaldadas": "G1 · cifras respaldadas", "G3_respuestas_con_cifra_no_respaldada": "G3 · respuestas con cifra no respaldada",
                   "G5_abstencion_correcta": "G5 · abstención correcta", "rechazos_indebidos": "Rechazos indebidos",
                   "exactitud_enrutador": "Exactitud del enrutador", "uso_respaldo_plantilla": "Uso de plantilla de respaldo",
                   "latencia_p95_s": "Latencia p95 (s)", "C1_movimientos_crudos_respuestas_ok": "C1 · respuestas OK con movimientos crudos"}
        t = pd.DataFrame(fs["proveedores"]).rename(index=nombres)
        t = t.apply(lambda col: [f"{v:.1%}" if (isinstance(v, float) and "Latencia" not in i) else (f"{v:.2f}" if isinstance(v, float) else v)
                                 for i, v in col.items()])
        st.dataframe(t.reset_index().rename(columns={"index": "Métrica"}), use_container_width=True, hide_index=True)
        st.caption(f"{fs['n_preguntas']} preguntas: 240 normales (40 perfiles × 6 intenciones) y 60 trampa.")
    if fd:
        with st.expander("Ficha de datos (P11)"):
            st.json(fd)
