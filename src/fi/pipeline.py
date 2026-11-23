"""Pipeline por lotes completo: preparación → modelos → recomendaciones → evaluación → personas de la demo.

Uso:  python -m fi.pipeline            (todo)
      python -m fi.pipeline --rapido   (omite la evaluación del asistente)
"""
from __future__ import annotations

import json
import sys
import time

import numpy as np
import pandas as pd

from fi import config, store


def elegir_personas() -> dict:
    """Escenarios A–E del documento, con un usuario sintético representativo para cada uno."""
    from fi import recommender
    d = store.datos()
    m, us, an = d["mensual"], d["usuarios"], d["anom"]
    vistos = us[~us.no_visto].index
    tot = m[m.month <= config.TRAIN_END].groupby(["user_id", "month"])["spend"].sum()
    cv = (tot.groupby("user_id").std() / tot.groupby("user_id").mean()).reindex(vistos)
    usados = set()

    def libre(serie):
        for u in serie.index:
            if u not in usados:
                usados.add(u)
                return u

    # C: el gasto inusual más claro del mes
    alertas = an[an.alerta & an.user_id.isin(vistos)].sort_values("ratio_med", ascending=False)
    c = libre(alertas.drop_duplicates("user_id").set_index("user_id")["ratio_med"])
    # B: mayor aumento previsto en una categoría discrecional
    pr = d["pron"][(d["pron"].month == config.MES_DEMO) & d["pron"].visto & d["pron"].category.isin(config.DISCRECIONALES)]
    pr = pr[pr.rm6 > 50_000].assign(r=lambda x: x.pred / x.rm6)
    crec = pr.groupby("user_id")["r"].max().sort_values(ascending=False)
    b = next((u for u in crec.index[:25] if u not in usados and recommender.situacion_meta(u, config.MES_DEMO)["brecha"] > 0), None)
    b = b or libre(crec)
    usados.add(b)
    # A: el más estable que además cumple su meta
    candidatos = cv.sort_values()
    a = None
    for u in candidatos.index[:40]:
        if u in usados:
            continue
        if recommender.situacion_meta(u, config.MES_DEMO)["brecha"] <= 0:
            a = u; break
    a = a or libre(candidatos)
    usados.add(a)
    # D: usuario no visto (cold start) con gasto típico
    nov = us[us.no_visto]
    dd = libre((nov.gasto_mediano - nov.gasto_mediano.median()).abs().sort_values())
    personas = {
        "A": {"user_id": a, "alias": "Ana", "escenario": "Usuario estable", "descripcion": "Gasta parecido todos los meses y va bien con su meta."},
        "B": {"user_id": b, "alias": "Bruno", "escenario": "Gasto en aumento", "descripcion": "Una categoría discrecional viene subiendo y atrasa su meta."},
        "C": {"user_id": c, "alias": "Camila", "escenario": "Gasto inusual", "descripcion": "Este mes aparece un cargo muy por encima de lo habitual."},
        "D": {"user_id": dd, "alias": "David", "escenario": "Usuario nuevo", "descripcion": "Solo tiene 3 meses de historia (arranque en frío)."},
    }
    (config.RESULTS / "personas.json").write_text(json.dumps(personas, indent=2, ensure_ascii=False))
    return personas


def main(rapido: bool = False) -> None:
    t0 = time.time()
    from fi.data.prepare import preparar
    from fi.models import anomaly, forecast
    print("== 1/5 Preparación de datos (P1–P11)"); preparar()
    print("== 2/5 Motor predictivo (modelo global)"); forecast.entrenar_y_evaluar()
    print("== 3/5 Detector de anomalías (Isolation Forest)"); anomaly.entrenar_y_evaluar()
    store.refrescar()
    from fi import recommender
    print("== 4/5 Motor de recomendación (evaluación U1–U4)"); recommender.evaluar()
    personas = elegir_personas()
    print("   Personas de la demo:", {k: v["user_id"] for k, v in personas.items()})
    if not rapido:
        from fi import evaluation
        print("== 5/5 Asistente fundamentado (G1, G5)"); evaluation.evaluar()
    print(f"Listo en {time.time() - t0:.0f} s. Resultados en {config.RESULTS}")


if __name__ == "__main__":
    main(rapido="--rapido" in sys.argv)
