"""M6 · Motor de recomendación basado en conocimiento: reglas R1–R5 + pronóstico.

Brecha = Aporte necesario − (Ingreso − Gasto previsto)
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd
import yaml

from fi import config, store
from fi.config import etiqueta, pesos

REGLAS = yaml.safe_load((config.CONFIG / "reglas.yaml").read_text(encoding="utf-8"))
P = REGLAS["parametros"]


@dataclass
class Recomendacion:
    regla: str
    nombre: str
    texto: str
    categoria: str | None = None
    previsto: float | None = None
    objetivo: float | None = None
    impacto: float = 0.0
    evidencia: dict = field(default_factory=dict)


def _txt(regla: str, **kw) -> str:
    return REGLAS["reglas"][regla]["texto"].format(**kw)


def situacion_meta(user_id: str, mes: str, ingreso=None, meta_monto=None, meta_meses=None) -> dict:
    u = store.datos()["usuarios"].loc[user_id]
    ingreso = float(ingreso if ingreso is not None else u.ingreso)
    meta_monto = float(meta_monto if meta_monto is not None else u.meta_monto)
    meta_meses = int(meta_meses if meta_meses is not None else u.meta_meses)
    pron = store.pronostico_usuario(user_id, mes)
    gasto_previsto = float(pron["pred"].sum() * store.factor_total())
    aporte = meta_monto / meta_meses
    ahorro = ingreso - gasto_previsto
    return {"ingreso": ingreso, "meta_monto": meta_monto, "meta_meses": meta_meses, "aporte_necesario": aporte,
            "gasto_previsto": gasto_previsto, "ahorro_proyectado": ahorro, "brecha": aporte - ahorro}


def recomendar(user_id: str, mes: str = config.MES_DEMO, ingreso=None, meta_monto=None, meta_meses=None,
               usar_anomalias: bool = True) -> dict:
    meta = situacion_meta(user_id, mes, ingreso, meta_monto, meta_meses)
    pron = store.pronostico_usuario(user_id, mes)
    est = store.estadisticas_categoria(user_id, mes)
    tabla = est.join(pron[["pred"]], how="inner")
    # piso: nunca por debajo del percentil 25 del propio usuario (ni de la mitad de su mediana)
    tabla["piso"] = np.maximum(tabla["p25_12m"], 0.5 * tabla["mediana_6m"])
    cand: dict[str, Recomendacion] = {}

    # R1: categoría discrecional en aumento
    for c, r in tabla.iterrows():
        if c in config.DISCRECIONALES and r.mediana_6m > 0 and r.pred > r.mediana_6m * (1 + P["delta_aumento"]):
            obj = max(r.mediana_6m, r.piso)
            imp = r.pred - obj
            cand[c] = Recomendacion("R1", REGLAS["reglas"]["R1"]["nombre"],
                                    _txt("R1", categoria=etiqueta(c).lower(), previsto=pesos(r.pred), mediana=pesos(r.mediana_6m),
                                         objetivo=pesos(obj), impacto=pesos(imp)),
                                    c, float(r.pred), float(obj), float(imp),
                                    {"previsto": r.pred, "mediana_6m": r.mediana_6m, "p25_12m": r.piso})

    # R3: cerrar la brecha con las categorías discrecionales de mayor margen (nunca bajo el piso)
    restante = meta["brecha"]
    if restante > 0:
        margen = (tabla.loc[tabla.index.isin(config.DISCRECIONALES), "pred"] - tabla["piso"]).dropna().sort_values(ascending=False)
        for c, mg in margen.items():
            if restante <= 0 or mg <= 1000:
                break
            r = tabla.loc[c]
            imp = float(min(mg, restante))
            obj = float(r.pred - imp)
            cand[c] = Recomendacion("R3", REGLAS["reglas"]["R3"]["nombre"],
                                    _txt("R3", aporte=pesos(meta["aporte_necesario"]), ahorro=pesos(meta["ahorro_proyectado"]),
                                         categoria=etiqueta(c).lower(), previsto=pesos(r.pred), objetivo=pesos(obj), impacto=pesos(imp)),
                                    c, float(r.pred), obj, imp,
                                    {"previsto": r.pred, "p25_12m": r.piso, "brecha": meta["brecha"]})
            restante -= imp

    lista = sorted(cand.values(), key=lambda x: -x.impacto)

    # R2: gasto inusual del mes (detector de anomalías)
    if usar_anomalias and mes == config.MES_DEMO:
        an = store.datos()["anom"]
        an = an[(an.user_id == user_id) & an.alerta].sort_values("score_if", ascending=False)
        if len(an):
            a = an.iloc[0]
            lista.insert(0, Recomendacion("R2", REGLAS["reglas"]["R2"]["nombre"],
                                          _txt("R2", monto=pesos(a.amount), comercio=a.merchant, veces=f"{a.ratio_med:.1f}".replace(".", ","),
                                               categoria=etiqueta(a.category).lower()),
                                          a.category, None, None, float(a.amount - a.med_ref),
                                          {"monto": a.amount, "tipico": a.med_ref, "fecha": str(a.ts.date())}))

    # R4: gastos hormiga en el último mes cerrado
    tx = store.datos()["tx"]
    mes_ant = str(pd.Period(mes, "M") - 1)
    h = tx[(tx.user_id == user_id) & (tx.month == mes_ant) & tx.discretionary & (tx.amount <= P["hormiga_monto_max"])]
    tot_mes = tx[(tx.user_id == user_id) & (tx.month == mes_ant)]["amount"].sum()
    if len(h):
        g = h.groupby("category")["amount"].agg(["size", "sum"]).sort_values("sum", ascending=False).iloc[0]
        if g["size"] >= P["hormiga_min_compras"] and g["sum"] >= P["hormiga_min_participacion"] * tot_mes:
            c = h.groupby("category")["amount"].sum().idxmax()
            lista.append(Recomendacion("R4", REGLAS["reglas"]["R4"]["nombre"],
                                       _txt("R4", n=int(g["size"]), categoria=etiqueta(c).lower(), total=pesos(g["sum"]), impacto=pesos(g["sum"] / 2)),
                                       c, None, None, float(g["sum"] / 2), {"compras": int(g["size"]), "total": g["sum"]}))

    lista = lista[: P["max_recomendaciones"]]
    if meta["brecha"] <= 0 and len(lista) < P["max_recomendaciones"]:
        lista.append(Recomendacion("R5", REGLAS["reglas"]["R5"]["nombre"],
                                   _txt("R5", ahorro=pesos(meta["ahorro_proyectado"]), aporte=pesos(meta["aporte_necesario"]))))

    ahorro_con = meta["ahorro_proyectado"] + sum(r.impacto for r in lista if r.regla in ("R1", "R3", "R4"))
    meta["meses_sin_cambios"] = float(meta["meta_monto"] / meta["ahorro_proyectado"]) if meta["ahorro_proyectado"] > 0 else None
    meta["meses_con_acciones"] = float(meta["meta_monto"] / ahorro_con) if ahorro_con > 0 else None
    meta["ahorro_con_acciones"] = float(ahorro_con)
    return {"user_id": user_id, "mes": mes, "meta": meta,
            "recomendaciones": [{k: (float(v) if isinstance(v, (np.floating,)) else v) for k, v in asdict(r).items()} for r in lista]}


# ----------------------------------------------------------------- evaluación U1–U4 (sección 9)
def evaluar(verbose: bool = True) -> dict:
    d = store.datos()
    m = d["mensual"]
    filas, b1, sim = [], [], []
    for uid in d["usuarios"].index:
        for mes in config.TEST_MONTHS:
            res = recomendar(uid, mes, usar_anomalias=False)
            real = m[(m.user_id == uid) & (m.month == mes)].set_index("category")["spend"]
            est = store.estadisticas_categoria(uid, mes)
            recs = [r for r in res["recomendaciones"] if r["categoria"]]
            for r in recs:
                c = r["categoria"]
                filas.append({"user_id": uid, "mes": mes, "regla": r["regla"], "categoria": c,
                              "pertinente": bool(real.get(c, 0) > est.loc[c, "mediana_6m"]),
                              "factible": r["objetivo"] is None or r["objetivo"] >= est.loc[c, "p25_12m"] - 1,
                              "esencial": c in config.ESENCIALES})
            # B1: «reduce 10 % tu categoría de mayor gasto»
            top = est["mediana_6m"].idxmax()
            b1.append({"pertinente": bool(real.get(top, 0) > est.loc[top, "mediana_6m"]), "esencial": top in config.ESENCIALES})
            # U2: simulación de adherencia
            meta = res["meta"]
            ahorro_real = meta["ingreso"] - real.sum()
            fila = {"user_id": uid, "mes": mes, "aporte": meta["aporte_necesario"]}
            for a in (0, 0.25, 0.5, 1.0):
                red = sum(a * min(r["impacto"], max(real.get(r["categoria"], 0) - (r["objetivo"] or 0), 0))
                          for r in recs if r["regla"] in ("R1", "R3"))
                red += sum(a * r["impacto"] for r in recs if r["regla"] == "R4")
                fila[f"sis_{a}"] = ahorro_real + red >= meta["aporte_necesario"]
                fila[f"b1_{a}"] = ahorro_real + a * 0.10 * real.get(top, 0) >= meta["aporte_necesario"]
            sim.append(fila)
    f, b, s = pd.DataFrame(filas), pd.DataFrame(b1), pd.DataFrame(sim)
    n_um = len(d["usuarios"]) * len(config.TEST_MONTHS)
    resultados = {
        "U1_precision_sistema": float(f.pertinente.mean()) if len(f) else None,
        "U1_precision_B1": float(b.pertinente.mean()),
        "U1_por_regla": f.groupby("regla")["pertinente"].mean().round(3).to_dict() if len(f) else {},
        "U2_en_camino": {str(a): {"sistema": float(s[f"sis_{a}"].mean()), "B1": float(s[f"b1_{a}"].mean())}
                         for a in (0, 0.25, 0.5, 1.0)},
        "U3_factibles": float(f.factible.mean()) if len(f) else None,
        "U3_en_esenciales": int(f.esencial.sum()) if len(f) else 0,
        "U3_B1_en_esenciales": float(b.esencial.mean()),
        "U4_recomendaciones_por_usuario_mes": float(len(f) / n_um),
        "U5_estudio_humano": "pendiente (requiere participantes; protocolo en la sección 9)",
    }
    resultados["exito_U1"] = bool(resultados["U1_precision_sistema"] - resultados["U1_precision_B1"] >= 0.15)
    (config.RESULTS / "metricas_recomendacion.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    if verbose:
        print(f"Recomendación: precisión@3 sistema={resultados['U1_precision_sistema']:.3f} vs B1={resultados['U1_precision_B1']:.3f}; "
              f"en camino (50 %): {resultados['U2_en_camino']['0.5']}")
    return resultados


if __name__ == "__main__":
    print(json.dumps(recomendar(store.datos()["usuarios"].index[0]), indent=2, ensure_ascii=False, default=str))
    evaluar()
