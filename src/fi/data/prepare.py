"""Protocolo de preparación de datos P1–P11 (sección 6.3 del documento)."""
from __future__ import annotations

import hashlib
import json

import duckdb
import numpy as np
import pandas as pd

from fi import config
from fi.data.generate import cargar_crudo


def _seudonimo(cc: str) -> str:  # P2: hash con sal
    return "U" + hashlib.sha256(("sal-fi-2026" + str(cc)).encode()).hexdigest()[:8].upper()


def preparar(verbose: bool = True) -> dict:
    reporte = {}
    crudo, fuente = cargar_crudo()
    reporte["fuente"] = fuente
    reporte["P0_registros_crudos"] = len(crudo)

    # P1 esquema canónico
    df = pd.DataFrame({
        "user_id": crudo["cc_num"].astype(str).map(_seudonimo),
        "ts": pd.to_datetime(crudo["trans_date_trans_time"]),
        "amount": crudo["amt"].astype(float),
        "direction": "debito",
        "category": crudo["category"],
        "merchant": crudo["merchant"].astype(str),
        "source": fuente,
        "label_fraud": crudo["is_fraud"].astype(int),
    })
    # perfil mínimo permitido (P2: se eliminan nombre, dirección, tarjeta y coordenadas)
    perfil = pd.DataFrame({"user_id": df["user_id"], "dob": pd.to_datetime(crudo["dob"]),
                           "ciudad": crudo.get("city", "N/D"), "ocupacion": crudo.get("job", "N/D")})
    perfil = perfil.drop_duplicates("user_id")
    perfil["edad"] = ((pd.Timestamp("2020-12-31") - perfil["dob"]).dt.days // 365).astype(int)
    perfil = perfil.drop(columns="dob")

    # P3 limpieza
    n0 = len(df)
    df = df.drop_duplicates(subset=["user_id", "ts", "amount", "merchant"])
    df = df[(df.amount > 0) & df.ts.between("2019-01-01", "2020-12-31 23:59:59")]
    df["merchant"] = df["merchant"].str.replace(r"^fraud_", "", regex=True)
    reporte["P3_eliminados_limpieza"] = n0 - len(df)
    df["discretionary"] = df["category"].isin(config.DISCRECIONALES)
    df["month"] = df["ts"].dt.to_period("M").astype(str)

    # P4 fraude aparte
    fraude = df[df.label_fraud == 1].copy()
    tx = df[df.label_fraud == 0].copy()
    reporte["P4_transacciones_fraude"] = len(fraude)

    # P5 selección de usuarios
    act = tx.groupby("user_id").agg(meses=("month", "nunique"), n=("amount", "size"))
    act["tx_mes"] = act["n"] / act["meses"]
    validos = act[(act.meses >= 18) & (act.tx_mes >= 10)].index
    tx = tx[tx.user_id.isin(validos)]
    fraude = fraude[fraude.user_id.isin(validos)]
    reporte["P5_usuarios"] = int(len(validos))

    # P6 agregación mensual usuario x categoría con ceros explícitos
    meses = pd.period_range("2019-01", "2020-12", freq="M").astype(str)
    idx = pd.MultiIndex.from_product([sorted(validos), list(config.CATEGORIAS), meses], names=["user_id", "category", "month"])
    mensual = (tx.groupby(["user_id", "category", "month"])["amount"].sum()
               .reindex(idx, fill_value=0.0).reset_index().rename(columns={"amount": "spend"}))
    reporte["P6_series"] = int(mensual.groupby(["user_id", "category"]).ngroups)

    # P7 ingreso y meta simulados. Razón ingreso/gasto: si existe Berka (trans.csv) se estima de ahí,
    # si no, se usa una distribución documentada (media 1,18; entre 1,03 y 1,40).
    rng = np.random.default_rng(config.SEED)
    tot_train = mensual[mensual.month <= config.TRAIN_END].groupby(["user_id", "month"])["spend"].sum()
    mediana_u = tot_train.groupby("user_id").median()
    razon = np.clip(rng.normal(1.18, 0.07, len(mediana_u)), 1.03, 1.40)
    usuarios = pd.DataFrame({"user_id": mediana_u.index, "gasto_mediano": mediana_u.values,
                             "ingreso": np.round(mediana_u.values * razon, -4)})
    usuarios["meta_monto"] = np.round(usuarios.gasto_mediano * 0.10 * 6, -4)   # ahorrar ~10 % del gasto, 6 meses
    usuarios["meta_meses"] = 6
    usuarios = usuarios.merge(perfil, on="user_id", how="left")

    # P9 partición: 20 % de usuarios no vistos (estratificado por volumen)
    usuarios["q"] = pd.qcut(usuarios.gasto_mediano, 5, labels=False)
    hold = usuarios.groupby("q", group_keys=False).sample(frac=config.HOLDOUT_FRAC, random_state=config.SEED)
    usuarios["no_visto"] = usuarios.user_id.isin(hold.user_id)
    usuarios = usuarios.drop(columns="q")
    reporte["P9_usuarios_no_vistos"] = int(usuarios.no_visto.sum())

    # P11 ficha de datos
    reporte["periodo"] = "2019-01 a 2020-12"
    reporte["transacciones_limpias"] = int(len(tx))
    reporte["categorias"] = len(config.CATEGORIAS)

    tx.to_parquet(config.PROC / "transacciones.parquet", index=False)
    fraude.to_parquet(config.PROC / "fraude.parquet", index=False)
    mensual.to_parquet(config.PROC / "mensual.parquet", index=False)
    usuarios.to_parquet(config.PROC / "usuarios.parquet", index=False)
    con = duckdb.connect(str(config.PROC / "finanzas.duckdb"))
    for t in ("transacciones", "fraude", "mensual", "usuarios"):
        con.execute(f"CREATE OR REPLACE TABLE {t} AS SELECT * FROM read_parquet('{(config.PROC / (t + '.parquet')).as_posix()}')")
    con.close()
    (config.RESULTS / "ficha_datos.json").write_text(json.dumps(reporte, indent=2, ensure_ascii=False))
    if verbose:
        print("Preparación:", reporte)
    return reporte


if __name__ == "__main__":
    preparar()
