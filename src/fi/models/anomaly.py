"""M5 · Detector de anomalías: Isolation Forest con variables relativas al usuario.

Una anomalía es un gasto que se aleja del historial del PROPIO usuario (no
necesariamente fraude). Como no existen etiquetas de "gasto raro", se inyectan
anomalías controladas (P10) y se mide si el detector las encuentra.
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import average_precision_score

from fi import config

FEATS = ["log_amt", "ratio_med", "z_rob", "cat_nueva", "hora_rara", "hora_nueva", "rafaga", "cat_code"]
CAT_CODE = {c: i for i, c in enumerate(config.CATEGORIAS)}
TIPOS = {"A1": "monto muy superior a lo habitual", "A2": "categoría que nunca usa",
         "A3": "hora poco habitual", "A4": "varias compras seguidas en el mismo comercio"}


# ----------------------------------------------------------------- inyección (P10)
def inyectar(periodo: pd.DataFrame, hist: pd.DataFrame, seed: int, tasa: float = 0.005) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    p = periodo.copy()
    p["inyectada"] = ""
    n = max(4, int(len(p) * tasa))
    por_tipo = n // 4
    cats_usadas = hist.groupby("user_id")["category"].agg(set)
    med_user = hist.groupby("user_id")["amount"].median()
    horas_user = hist.assign(h=hist.ts.dt.hour).groupby("user_id")["h"].agg(set)
    nuevas = []
    # A1: monto 3x-10x
    idx = rng.choice(p.index[p.inyectada == ""], por_tipo, replace=False)
    p.loc[idx, "amount"] = (p.loc[idx, "amount"] * rng.uniform(3, 10, len(idx))).round(-2)
    p.loc[idx, "inyectada"] = "A1"
    # A2: gasto alto en categoría nunca usada
    for i in rng.choice(p.index, por_tipo, replace=False):
        r = p.loc[i].copy()
        libres = sorted(set(config.CATEGORIAS) - cats_usadas.get(r.user_id, set())) or sorted(config.DISCRECIONALES)
        r["category"] = rng.choice(libres)
        r["amount"] = round(float(med_user.get(r.user_id, 80_000) * rng.uniform(4, 9)), -2)
        r["merchant"] = "Comercio nuevo"
        r["ts"] = r["ts"] + pd.Timedelta(minutes=int(rng.integers(5, 300)))
        r["inyectada"] = "A2"
        nuevas.append(r)
    # A3: hora nunca observada para ese usuario
    for i in rng.choice(p.index, por_tipo, replace=False):
        r = p.loc[i].copy()
        vistas = horas_user.get(r.user_id, set())
        libres = [h for h in (1, 2, 3, 4, 5, 0, 23) if h not in vistas] or [3]
        r["ts"] = r["ts"].normalize() + pd.Timedelta(hours=int(rng.choice(libres)), minutes=int(rng.integers(0, 59)))
        r["inyectada"] = "A3"
        nuevas.append(r)
    # A4: ráfaga de 4 compras en el mismo comercio en menos de una hora
    for i in rng.choice(p.index, max(1, por_tipo // 4), replace=False):
        for k in range(4):
            r = p.loc[i].copy()
            r["ts"] = r["ts"] + pd.Timedelta(minutes=int(8 * (k + 1)))
            r["inyectada"] = "A4"
            nuevas.append(r)
    p = pd.concat([p, pd.DataFrame(nuevas)], ignore_index=True)
    p["mes_cat"] = p["ts"].dt.to_period("M").astype(str)
    return p


# ----------------------------------------------------------------- variables
def variables(periodo: pd.DataFrame, hist: pd.DataFrame) -> pd.DataFrame:
    p = periodo.copy()
    uc = hist.groupby(["user_id", "category"])["amount"].agg(med="median",
                                                               mad=lambda s: (s - s.median()).abs().median())
    u = hist.groupby("user_id")["amount"].agg(med_u="median", mad_u=lambda s: (s - s.median()).abs().median())
    p = p.join(uc, on=["user_id", "category"]).join(u, on="user_id")
    p["cat_nueva"] = p["med"].isna().astype(int)
    med = p["med"].fillna(p["med_u"])
    mad = p["mad"].fillna(p["mad_u"]).clip(lower=1_000)
    p["med_ref"] = med
    p["log_amt"] = np.log1p(p["amount"])
    p["ratio_med"] = p["amount"] / med
    p["z_rob"] = (p["amount"] - med) / (1.4826 * mad)
    hh = hist.assign(h=hist.ts.dt.hour).groupby(["user_id", "h"]).size()
    tot = hist.groupby("user_id").size()
    hora = p.ts.dt.hour
    frec = sum(hh.reindex(list(zip(p.user_id, (hora + k) % 24))).fillna(0).values for k in (-1, 0, 1))
    p["hora_rara"] = 1 - frec / tot.reindex(p.user_id).values
    p["hora_nueva"] = (hh.reindex(list(zip(p.user_id, hora))).fillna(0).values == 0).astype(int)
    p = p.sort_values("ts")
    p["rafaga"] = 0
    g = p.groupby(["user_id", "merchant"])["ts"]
    p["rafaga"] = (g.diff().dt.total_seconds().lt(3600).astype(int)
                   + g.diff(-1).dt.total_seconds().abs().lt(3600).astype(int))
    p["cat_code"] = p["category"].map(CAT_CODE)
    return p


def _motivo(r) -> str:
    if r.cat_nueva:
        return f"Primera compra en {config.etiqueta(r.category)}"
    if r.rafaga >= 1 and r.ratio_med < 3:
        return f"Varias compras seguidas en {r.merchant}"
    if r.hora_nueva and r.ratio_med < 3:
        return f"Compra a una hora poco habitual ({r.ts:%H:%M})"
    veces = f"{r.ratio_med:.0f}" if r.ratio_med >= 10 else f"{r.ratio_med:.1f}".replace(".", ",")
    return f"{veces} veces tu gasto típico en {config.etiqueta(r.category).lower()}"


# ----------------------------------------------------------------- entrenamiento y evaluación
def entrenar_y_evaluar(verbose: bool = True) -> dict:
    tx = pd.read_parquet(config.PROC / "transacciones.parquet")
    fraude = pd.read_parquet(config.PROC / "fraude.parquet")
    hist_val = tx[tx.month <= config.TRAIN_END]
    hist_test = tx[tx.month <= config.VAL_MONTHS[-1]]

    # Isolation Forest entrenado con comportamiento normal (historia hasta junio 2020)
    # se entrena con meses "nuevos" frente a su propia historia (ene-jun 2020 vs 2019) para que las
    # variables de novedad (categoría u hora nunca vistas) tengan variación natural
    base = variables(hist_val[hist_val.month >= "2020-01"], tx[tx.month <= "2019-12"])
    iforest = IsolationForest(n_estimators=300, max_samples=4096, contamination="auto", random_state=config.SEED)
    iforest.fit(base[FEATS].fillna(0))

    def puntuar(p):
        p = p.copy()
        p["score_if"] = -iforest.score_samples(p[FEATS].fillna(0))
        p["score_z"] = p["z_rob"].abs()
        return p

    # umbral elegido en validación (otra semilla): máximo F1 con <= 2 alertas por usuario al mes
    v = puntuar(variables(inyectar(tx[tx.month.isin(config.VAL_MONTHS)], hist_val, seed=100), hist_val))
    y = (v.inyectada != "").astype(int)
    mejor, umbral = -1, None
    for q in np.linspace(0.95, 0.999, 60):
        t = v.score_if.quantile(q)
        pred = v.score_if >= t
        alertas = pred.groupby([v.user_id, v.mes_cat]).sum().mean()
        tp = (pred & (y == 1)).sum()
        f1 = 2 * tp / (pred.sum() + y.sum())
        if alertas <= 2 and f1 > mejor:
            mejor, umbral = f1, float(t)

    filas, demo = [], None
    periodo_test = tx[tx.month.isin(config.TEST_MONTHS)]
    for seed in range(1, 6):
        p = puntuar(variables(inyectar(periodo_test, hist_test, seed=seed), hist_test))
        y = (p.inyectada != "").astype(int)
        pred = p.score_if >= umbral
        fila = {"semilla": seed,
                "PR_AUC_isolation_forest": average_precision_score(y, p.score_if),
                "PR_AUC_z_robusto": average_precision_score(y, p.score_z),
                "PR_AUC_percentil99": average_precision_score(y, p.ratio_med),
                "alertas_usuario_mes": float(pred.groupby([p.user_id, p.mes_cat]).sum().mean()),
                "precision": float((pred & (y == 1)).sum() / max(pred.sum(), 1))}
        for t in TIPOS:
            m = p.inyectada == t
            fila[f"recall_{t}"] = float(pred[m].mean()) if m.any() else None
            fila[f"recall_z_{t}"] = float((p.score_z[m] >= 3.5).mean()) if m.any() else None
        filas.append(fila)
        if seed == 1:
            demo = p
    df = pd.DataFrame(filas)
    # evaluación secundaria con la etiqueta de fraude (sin inyección)
    f_test = fraude[fraude.month.isin(config.TEST_MONTHS)]
    pf = puntuar(variables(pd.concat([periodo_test, f_test]), hist_test))
    resultados = {
        "umbral_isolation_forest": umbral,
        "promedio_5_semillas": {k: float(df[k].mean()) for k in df.columns if k != "semilla" and df[k].notna().all()},
        "desviacion_5_semillas": {k: float(df[k].std()) for k in ("PR_AUC_isolation_forest", "PR_AUC_z_robusto")},
        "por_semilla": filas,
        "fraude_secundario": {"PR_AUC_isolation_forest": average_precision_score(pf.label_fraud, pf.score_if),
                              "PR_AUC_z_robusto": average_precision_score(pf.label_fraud, pf.score_z),
                              "proporcion_fraude": float(pf.label_fraud.mean())},
    }
    resultados["exito"] = bool((df.PR_AUC_isolation_forest > df.PR_AUC_z_robusto).all()
                               and resultados["promedio_5_semillas"]["alertas_usuario_mes"] <= 2)

    # datos para la demo: mes de diciembre con las anomalías inyectadas de la semilla 1 y el fraude simulado
    f_dic = puntuar(variables(f_test[f_test.month == config.MES_DEMO].assign(inyectada="fraude"), hist_test))
    demo = pd.concat([demo[demo.mes_cat == config.MES_DEMO], f_dic], ignore_index=True)
    demo["alerta"] = demo.score_if >= umbral
    demo["motivo"] = [_motivo(r) if a else "" for r, a in zip(demo.itertuples(), demo.alerta)]
    cols = ["user_id", "ts", "merchant", "category", "amount", "med_ref", "ratio_med", "score_if", "score_z",
            "alerta", "motivo", "inyectada"]
    demo[cols].to_parquet(config.RESULTS / "anomalias_demo.parquet", index=False)
    (config.RESULTS / "metricas_anomalias.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    if verbose:
        pm = resultados["promedio_5_semillas"]
        print(f"Anomalías: PR-AUC IF={pm['PR_AUC_isolation_forest']:.3f} vs z={pm['PR_AUC_z_robusto']:.3f} "
              f"alertas/usuario/mes={pm['alertas_usuario_mes']:.2f}")
    return resultados


if __name__ == "__main__":
    entrenar_y_evaluar()
