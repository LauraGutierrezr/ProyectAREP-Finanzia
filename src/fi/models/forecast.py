"""M4 · Motor predictivo: modelo GLOBAL de pronóstico con gradient boosting (LightGBM).

Un solo modelo aprende de las series usuario x categoría de todos los usuarios
(Montero-Manso y Hyndman, 2021). Se compara contra tres líneas base ingenuas.
"""
from __future__ import annotations

import json
import warnings

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from fi import config

warnings.filterwarnings("ignore")

try:
    import lightgbm as lgb
    MOTOR = "LightGBM"
except Exception:  # p. ej. macOS sin libomp: mismo tipo de modelo (gradient boosting) de scikit-learn
    lgb = None
    from sklearn.ensemble import HistGradientBoostingRegressor
    MOTOR = "HistGradientBoosting (scikit-learn)"

FEATS = ["lag1", "lag2", "lag3", "lag12", "rm3", "rm6", "rs6", "m12", "ratio12", "tot_lag1", "mes", "cat_code"]
CAT_CODE = {c: i for i, c in enumerate(config.CATEGORIAS)}


def construir_variables(mensual: pd.DataFrame) -> pd.DataFrame:
    """P8: todas las variables usan solo meses anteriores al mes objetivo (sin fuga)."""
    d = mensual.sort_values(["user_id", "category", "month"]).copy()
    g = d.groupby(["user_id", "category"])["spend"]
    for k in (1, 2, 3, 12):
        d[f"lag{k}"] = g.shift(k)
    prev = g.shift(1)
    d["rm3"] = prev.groupby([d.user_id, d.category]).transform(lambda s: s.rolling(3, min_periods=1).mean())
    d["rm6"] = prev.groupby([d.user_id, d.category]).transform(lambda s: s.rolling(6, min_periods=1).mean())
    d["rs6"] = prev.groupby([d.user_id, d.category]).transform(lambda s: s.rolling(6, min_periods=2).std())
    d["m12"] = prev.groupby([d.user_id, d.category]).transform(lambda s: s.rolling(12, min_periods=1).mean())
    tot = d.groupby(["user_id", "month"])["spend"].sum().rename("tot").reset_index()
    tot["tot_lag1"] = tot.groupby("user_id")["tot"].shift(1)
    d = d.merge(tot[["user_id", "month", "tot_lag1"]], on=["user_id", "month"], how="left")
    d["ratio12"] = d["lag12"] / (d["m12"] + 1)   # efecto estacional del mismo mes del año anterior
    d["mes"] = pd.PeriodIndex(d["month"], freq="M").month
    d["cat_code"] = d["category"].map(CAT_CODE)
    return d


def _truncar_cold_start(d: pd.DataFrame) -> pd.DataFrame:
    """Usuarios no vistos: solo se conocen los últimos 3 meses antes de cada origen."""
    d = d.copy()
    d["lag12"] = np.nan
    d["rm6"] = d["rm3"]
    d["m12"] = d["rm3"]
    d["rs6"] = d[["lag1", "lag2", "lag3"]].std(axis=1)
    d["ratio12"] = np.nan
    return d


def _modelo():
    # Objetivo L1 (mediana): MAE y MASE premian la mediana, y el gasto mensual es asimétrico.
    if lgb is not None:
        return lgb.LGBMRegressor(objective="l1", n_estimators=600,
                                 learning_rate=0.03, num_leaves=31, min_child_samples=40,
                                 subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                                 random_state=config.SEED, verbose=-1)
    return HistGradientBoostingRegressor(loss="absolute_error", max_iter=400, learning_rate=0.05, random_state=config.SEED)


def _metricas(df: pd.DataFrame, col: str, escala: pd.Series) -> dict:
    e = (df["spend"] - df[col]).abs()
    por_serie = e.groupby([df.user_id, df.category]).mean()
    esc = escala.reindex(por_serie.index)
    ok = esc > 0
    return {
        "MAE": float(e.mean()),
        "RMSE": float(np.sqrt(((df["spend"] - df[col]) ** 2).mean())),
        "WAPE": float(e.sum() / df["spend"].sum()),
        "MASE": float((por_serie[ok] / esc[ok]).mean()),
    }


def entrenar_y_evaluar(verbose: bool = True) -> dict:
    mensual = pd.read_parquet(config.PROC / "mensual.parquet")
    usuarios = pd.read_parquet(config.PROC / "usuarios.parquet")
    no_vistos = set(usuarios.loc[usuarios.no_visto, "user_id"])
    d = construir_variables(mensual)
    d = d[d.month >= "2019-04"]
    d["visto"] = ~d.user_id.isin(no_vistos)

    train = d[(d.month <= config.TRAIN_END) & d.visto]
    val = d[d.month.isin(config.VAL_MONTHS) & d.visto]
    m = _modelo()
    if lgb is not None:
        m.fit(train[FEATS], train["spend"], eval_set=[(val[FEATS], val["spend"])],
              callbacks=[lgb.early_stopping(50, verbose=False)])
        mejor = m.best_iteration_ or 600
    else:
        m.fit(train[FEATS], train["spend"]); mejor = None

    # factor para calibrar el TOTAL mensual: la suma de medianas por categoría subestima el total
    pv = np.clip(m.predict(val[FEATS]), 0, None)
    factor = float(val["spend"].sum() / pv.sum())

    # modelo final: entrenado hasta septiembre 2020 (entrenamiento + validación)
    final = _modelo()
    if lgb is not None:
        final.set_params(n_estimators=mejor)
    trval = d[(d.month <= config.VAL_MONTHS[-1]) & d.visto]
    final.fit(trval[FEATS], trval["spend"])

    test = d[d.month.isin(config.TEST_MONTHS)].copy()
    vis, nov = test[test.visto].copy(), _truncar_cold_start(test[~test.visto])
    vis["pred"] = np.clip(final.predict(vis[FEATS]), 0, None)
    nov["pred"] = np.clip(final.predict(nov[FEATS]), 0, None)
    test = pd.concat([vis, nov])
    test["naive"] = test["lag1"]
    test["estacional"] = test["lag12"].fillna(test["lag1"])
    test["media3"] = test["rm3"]
    test.loc[~test.visto, "estacional"] = test.loc[~test.visto, "lag1"]  # sin historia de 12 meses

    # escala MASE: error del ingenuo dentro del periodo de entrenamiento de cada serie
    hist = mensual[mensual.month <= config.TRAIN_END].sort_values("month")
    escala = hist.groupby(["user_id", "category"])["spend"].apply(lambda s: s.diff().abs().mean())

    resultados = {"motor": MOTOR, "mejor_iteracion": mejor, "factor_calibracion_total": factor}
    for grupo, sub in (("usuarios_vistos", test[test.visto]), ("usuarios_no_vistos", test[~test.visto])):
        res = {c: _metricas(sub, c, escala) for c in ("naive", "estacional", "media3", "pred")}
        base = min(("naive", "estacional", "media3"), key=lambda c: res[c]["MASE"])
        e_m = (sub["spend"] - sub["pred"]).abs().groupby([sub.user_id, sub.category]).mean()
        e_b = (sub["spend"] - sub[base]).abs().groupby([sub.user_id, sub.category]).mean()
        p = float(wilcoxon(e_m, e_b, alternative="less").pvalue)
        mejora = 1 - res["pred"]["MASE"] / res[base]["MASE"]
        resultados[grupo] = {"metricas": res, "mejor_linea_base": base, "mejora_MASE": float(mejora),
                             "p_wilcoxon": p, "exito": bool(mejora >= 0.10 and p < 0.05)}
    # pronóstico total por usuario-mes
    tot = test.groupby(["user_id", "month"])[["spend", "pred", "naive"]].sum()
    tot["pred"] *= factor
    resultados["total_mensual_WAPE"] = {"modelo": float((tot.spend - tot.pred).abs().sum() / tot.spend.sum()),
                                        "naive": float((tot.spend - tot.naive).abs().sum() / tot.spend.sum())}
    if lgb is not None:
        imp = pd.Series(final.feature_importances_, index=FEATS).sort_values(ascending=False)
        resultados["importancia_variables"] = {k: int(v) for k, v in imp.items()}

    cols = ["user_id", "category", "month", "spend", "pred", "naive", "estacional", "media3", "rm3", "rm6", "visto"]
    test[cols].to_parquet(config.RESULTS / "pronosticos.parquet", index=False)
    (config.RESULTS / "metricas_pronostico.json").write_text(json.dumps(resultados, indent=2, ensure_ascii=False))
    _mlflow("pronostico", resultados)
    if verbose:
        v = resultados["usuarios_vistos"]
        print(f"Pronóstico ({MOTOR}): MASE modelo={v['metricas']['pred']['MASE']:.3f} vs {v['mejor_linea_base']}="
              f"{v['metricas'][v['mejor_linea_base']]['MASE']:.3f}  mejora={v['mejora_MASE']:.1%}  p={v['p_wilcoxon']:.2g}")
    return resultados


def _mlflow(nombre: str, res: dict) -> None:
    """Registra el experimento en MLflow si está instalado (opcional)."""
    try:
        import mlflow
    except Exception:
        return
    mlflow.set_tracking_uri((config.DATA / "mlruns").as_uri())
    mlflow.set_experiment("finanzas-inteligentes")
    with mlflow.start_run(run_name=nombre):
        v = res["usuarios_vistos"]
        mlflow.log_param("motor", res["motor"])
        for k, val in v["metricas"]["pred"].items():
            mlflow.log_metric(k, val)
        mlflow.log_metric("mejora_MASE", v["mejora_MASE"])


if __name__ == "__main__":
    entrenar_y_evaluar()
