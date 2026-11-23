"""Paso 3 del asistente: reunir los hechos YA CALCULADOS por los modelos (RAG sobre datos estructurados).

Cada hecho lleva un identificador [fN]. Las cifras se entregan ya formateadas para
que el modelo de lenguaje no tenga que calcular ni redondear nada.
"""
from __future__ import annotations

import pandas as pd

from fi import config, recommender, store
from fi.config import etiqueta, pesos

MESES = {1: "enero", 2: "febrero", 3: "marzo", 4: "abril", 5: "mayo", 6: "junio", 7: "julio", 8: "agosto",
         9: "septiembre", 10: "octubre", 11: "noviembre", 12: "diciembre"}


def _nombre_mes(m: str) -> str:
    p = pd.Period(m, "M")
    return f"{MESES[p.month]} de {p.year}"


class Contexto:
    def __init__(self):
        self.hechos: list[dict] = []

    def add(self, descripcion: str, **valores) -> str:
        fid = f"f{len(self.hechos) + 1}"
        self.hechos.append({"id": fid, "descripcion": descripcion, **valores})
        return fid

    def json(self) -> list[dict]:
        return self.hechos


def construir(user_id: str, intencion: str, mes: str = config.MES_DEMO) -> Contexto:
    d = store.datos()
    ctx = Contexto()
    m = d["mensual"]
    mes_ant = str(pd.Period(mes, "M") - 1)
    ctx.add("Mes de análisis", mes_actual=_nombre_mes(mes), ultimo_mes_cerrado=_nombre_mes(mes_ant))

    if intencion == "mayor_gasto":
        sub = m[(m.user_id == user_id) & (m.month == mes_ant)].set_index("category")["spend"].sort_values(ascending=False)
        total = sub.sum()
        ctx.add("Gasto total del último mes cerrado", monto=pesos(total))
        for c, v in sub.head(3).items():
            ctx.add("Categoría con mayor gasto", categoria=etiqueta(c), monto=pesos(v), porcentaje=f"{round(100 * v / total)} %")

    elif intencion == "aumentos":
        est = store.estadisticas_categoria(user_id, mes_ant)
        ult = m[(m.user_id == user_id) & (m.month == mes_ant)].set_index("category")["spend"]
        t = est.join(ult.rename("ultimo"))
        t = t[(t.mediana_6m > 20_000) & (t.ultimo > t.mediana_6m * 1.15)]
        t["aumento"] = (t.ultimo / t.mediana_6m - 1)
        if t.empty:
            ctx.add("No hay categorías por encima de lo habitual", detalle="ninguna categoría supera su mediana de 6 meses en más de 15 %")
        for c, r in t.sort_values("aumento", ascending=False).head(3).iterrows():
            ctx.add("Categoría que aumentó frente a la mediana de los 6 meses anteriores", categoria=etiqueta(c),
                    gasto_ultimo_mes=pesos(r.ultimo), mediana_6_meses=pesos(r.mediana_6m), aumento=f"{round(100 * r.aumento)} %")

    elif intencion == "pronostico":
        pr = store.pronostico_usuario(user_id, mes)
        tot = pr["pred"].sum() * store.factor_total()
        prom3 = m[(m.user_id == user_id) & (m.month < mes)].groupby("month")["spend"].sum().tail(3).mean()
        ctx.add("Gasto total previsto por el modelo para el mes", mes=_nombre_mes(mes), monto=pesos(tot))
        ctx.add("Promedio de gasto de los últimos 3 meses", monto=pesos(prom3))
        for c, v in pr["pred"].sort_values(ascending=False).head(3).items():
            ctx.add("Categoría con mayor gasto previsto", categoria=etiqueta(c), monto=pesos(v))

    elif intencion == "inusual":
        an = d["anom"]
        an = an[(an.user_id == user_id) & an.alerta].sort_values("score_if", ascending=False)
        if an.empty:
            ctx.add("Sin movimientos inusuales en el mes", detalle="el detector no marcó ningún movimiento")
        else:
            ctx.add("Cantidad de movimientos marcados como inusuales", cantidad=str(min(len(an), 3)))
        for a in an.head(3).itertuples():
            ctx.add("Movimiento marcado como inusual", fecha=f"{a.ts.day} de {MESES[a.ts.month]}", comercio=a.merchant,
                    categoria=etiqueta(a.category), monto=pesos(a.amount), gasto_tipico=pesos(a.med_ref),
                    veces_lo_habitual=f"{a.ratio_med:.1f}".replace(".", ","), motivo=a.motivo)

    elif intencion in ("meta", "accion"):
        r = recommender.recomendar(user_id, mes)
        mt = r["meta"]
        ctx.add("Meta de ahorro", monto=pesos(mt["meta_monto"]), meses=str(mt["meta_meses"]), aporte_mensual_necesario=pesos(mt["aporte_necesario"]))
        ctx.add("Ahorro proyectado del mes (ingreso menos gasto previsto)", monto=pesos(mt["ahorro_proyectado"]))
        if mt["brecha"] > 0:
            ctx.add("Brecha mensual (lo que falta para la meta)", monto=pesos(mt["brecha"]))
        else:
            ctx.add("Sin brecha: el ahorro proyectado cubre la meta", excedente=pesos(-mt["brecha"]))
        if mt["meses_sin_cambios"] and mt["meses_sin_cambios"] <= 36:
            ctx.add("Meses para cumplir la meta sin cambios", meses=str(round(mt["meses_sin_cambios"])))
        if intencion == "meta":
            est = store.estadisticas_categoria(user_id, mes)
            pr = store.pronostico_usuario(user_id, mes)
            t = est.join(pr[["pred"]])
            t = t[t.index.isin(config.DISCRECIONALES)]
            c = (t.pred - t.mediana_6m).idxmax()
            ctx.add("Categoría discrecional con mayor diferencia frente a lo habitual", categoria=etiqueta(c),
                    previsto=pesos(t.loc[c, "pred"]), mediana_6_meses=pesos(t.loc[c, "mediana_6m"]))
        else:
            for rec in r["recomendaciones"]:
                ctx.add(f"Recomendación ({rec['nombre']})", texto=rec["texto"],
                        **({"ahorro_estimado": pesos(rec["impacto"])} if rec["regla"] in ("R1", "R3", "R4") else {}))
            if mt["meses_con_acciones"] and mt["meses_con_acciones"] <= 36:
                ctx.add("Meses para cumplir la meta aplicando las recomendaciones", meses=str(round(mt["meses_con_acciones"])))
    return ctx
