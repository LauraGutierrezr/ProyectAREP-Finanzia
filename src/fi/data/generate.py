"""Generador sintético con el MISMO esquema del dataset Sparkov de Kaggle.

El dataset original ("Credit Card Transactions Fraud Detection", Kaggle) requiere
una cuenta para descargarse. Para que la demo corra sin pasos manuales, este módulo
genera transacciones con las mismas columnas y las mismas 14 categorías. Si el
usuario coloca fraudTrain.csv y fraudTest.csv en data/raw/, el pipeline usa los
datos reales en su lugar.

Los montos se generan en pesos colombianos (COP) para que la demo sea legible.
"""
from __future__ import annotations

import hashlib

import numpy as np
import pandas as pd

from fi import config

# participación típica del gasto y ticket promedio (COP) por categoría
PERFIL_CAT = {
    "grocery_pos": (0.19, 85_000), "grocery_net": (0.05, 120_000),
    "gas_transport": (0.12, 55_000), "food_dining": (0.12, 42_000),
    "entertainment": (0.06, 60_000), "shopping_pos": (0.09, 115_000),
    "shopping_net": (0.08, 105_000), "health_fitness": (0.05, 90_000),
    "personal_care": (0.04, 48_000), "kids_pets": (0.05, 80_000),
    "home": (0.07, 140_000), "travel": (0.04, 420_000),
    "misc_pos": (0.02, 35_000), "misc_net": (0.02, 30_000),
}
ESTACIONAL = {  # multiplicador por mes (1..12) para algunas categorías
    "shopping_pos": {11: 1.15, 12: 1.45, 1: 0.8}, "shopping_net": {11: 1.25, 12: 1.4, 1: 0.85},
    "food_dining": {12: 1.3, 1: 0.85}, "travel": {6: 1.6, 7: 1.5, 12: 1.8, 1: 1.3, 2: 0.6},
    "entertainment": {12: 1.25}, "home": {1: 1.15, 12: 1.1},
}
NOMBRES = {
    "grocery_pos": ["Supermercado El Ahorro", "Mercado Central", "Tienda La Economía", "Fruver Los Andes", "Autoservicio San José"],
    "grocery_net": ["MercaYa en línea", "Domicilios Frescos", "SuperApp Mercado"],
    "gas_transport": ["Estación Terpel 80", "Taxi Ya", "Uber Trip", "Recarga TuLlave", "EDS La Autopista"],
    "food_dining": ["Restaurante La Esquina", "Corrientazo Doña Rosa", "Pizzería Napoli", "Café Juan Valdez", "Rappi Restaurantes", "Hamburguesas El Parque"],
    "entertainment": ["Cine Colombia", "Netflix", "Spotify", "Teatro Nacional", "Bolera Galerías"],
    "shopping_pos": ["Almacén Éxito", "Falabella", "Tienda de Ropa Moda", "Zapatería Paso Firme"],
    "shopping_net": ["Mercado Libre", "Amazon", "Linio", "Dafiti"],
    "health_fitness": ["Droguería La Rebaja", "Gimnasio BodyTech", "Consultorio Médico", "Óptica Visión"],
    "personal_care": ["Peluquería Estilo", "Spa Relax", "Barbería Clásica"],
    "kids_pets": ["Veterinaria Huellitas", "Juguetería Mundo", "Colegio Pagos"],
    "home": ["Homecenter", "Ferretería El Tornillo", "Muebles y Más", "Servicios Públicos"],
    "travel": ["Avianca", "Hotel Dann", "Despegar", "Terminal de Transporte"],
    "misc_pos": ["Papelería La 19", "Kiosco Universitario", "Tienda Don Pedro"],
    "misc_net": ["App Store", "Google Play", "Suscripción Digital"],
}
NOMBRES_PERSONA = ["Ana", "Camilo", "Daniela", "Juan", "Laura", "Mateo", "Sofía", "Andrés", "Valentina", "Santiago",
                   "Mariana", "Felipe", "Isabela", "Nicolás", "Paula", "Sebastián", "Carolina", "David", "Natalia", "Julián"]
APELLIDOS = ["Gómez", "Rodríguez", "Martínez", "López", "García", "Pérez", "Ramírez", "Torres", "Díaz", "Vargas"]
OFICIOS = ["Estudiante que trabaja", "Ingeniera de software", "Docente", "Diseñador independiente", "Enfermera",
           "Contador", "Vendedora", "Analista de datos", "Mesero", "Abogada"]
CIUDADES = ["Bogotá", "Medellín", "Cali", "Barranquilla", "Bucaramanga", "Pereira", "Manizales"]


def _hash(s: str) -> str:
    return hashlib.sha1(s.encode()).hexdigest()[:10]


def generar(n_users: int = config.N_USERS, seed: int = config.SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    meses = pd.period_range("2019-01", "2020-12", freq="M")
    cats = list(PERFIL_CAT)
    filas = []
    for u in range(n_users):
        cc = str(4000_0000_0000_0000 + u * 7919)
        nombre, apellido = rng.choice(NOMBRES_PERSONA), rng.choice(APELLIDOS)
        perfil = rng.choice(["normal", "estable", "creciente", "hormiga"], p=[0.5, 0.2, 0.18, 0.12])
        gasto_base = rng.lognormal(np.log(1_900_000), 0.35)
        share = rng.dirichlet([PERFIL_CAT[c][0] * 40 for c in cats])
        ticket = {c: PERFIL_CAT[c][1] * rng.lognormal(0, 0.2) for c in cats}
        ruido = 0.06 if perfil == "estable" else 0.17
        cat_crece = rng.choice(sorted(config.DISCRECIONALES - {"travel"})) if perfil == "creciente" else None
        hora_media = rng.uniform(10, 19)
        dob = pd.Timestamp("1975-01-01") + pd.Timedelta(days=int(rng.integers(0, 365 * 27)))  # 18 a 45 años
        oficio, ciudad = rng.choice(OFICIOS), rng.choice(CIUDADES)
        for k, mes in enumerate(meses):
            nivel = rng.lognormal(0, ruido)
            for ci, c in enumerate(cats):
                mult = ESTACIONAL.get(c, {}).get(mes.month, 1.0)
                if c == cat_crece and k >= 16:          # sube en los últimos 8 meses
                    mult *= 1 + 0.12 * (k - 15)
                esperado = gasto_base * share[ci] * mult * nivel
                lam = max(esperado / ticket[c], 0.05)
                if perfil == "hormiga" and c in ("misc_pos", "food_dining"):
                    lam *= 3.0
                n = rng.poisson(lam)
                if n == 0:
                    continue
                tk = ticket[c] / (3.0 if perfil == "hormiga" and c in ("misc_pos", "food_dining") else 1.0)
                montos = rng.lognormal(np.log(tk) - 0.08, 0.4, n)
                dias = rng.integers(0, mes.days_in_month, n)
                horas = np.clip(rng.normal(hora_media, 2.8, n), 6, 22.99)
                for m, d, h in zip(montos, dias, horas):
                    ts = mes.start_time + pd.Timedelta(days=int(d), hours=float(h))
                    filas.append((ts, cc, rng.choice(NOMBRES[c]), c, round(float(m), -2), nombre, apellido, oficio, ciudad, dob, 0))
        # fraude simulado (~0,5 %), como en Sparkov: noche, montos altos, compras en línea
        n_fraude = rng.binomial(24 * 45, 0.005)
        for _ in range(n_fraude):
            mes = meses[rng.integers(0, len(meses))]
            c = rng.choice(["shopping_net", "misc_net", "grocery_pos"])
            ts = mes.start_time + pd.Timedelta(days=int(rng.integers(0, 28)), hours=float(rng.choice([0, 1, 2, 3, 23])) + rng.uniform(0, 0.9))
            filas.append((ts, cc, "fraud_" + rng.choice(NOMBRES[c]), c, round(float(rng.lognormal(np.log(900_000), 0.5)), -2),
                          nombre, apellido, oficio, ciudad, dob, 1))
    df = pd.DataFrame(filas, columns=["trans_date_trans_time", "cc_num", "merchant", "category", "amt", "first", "last",
                                      "job", "city", "dob", "is_fraud"])
    # como en Sparkov, todos los comercios llevan el prefijo "fraud_"
    df["merchant"] = np.where(df["merchant"].str.startswith("fraud_"), df["merchant"], "fraud_" + df["merchant"])
    df = df.sort_values("trans_date_trans_time").reset_index(drop=True)
    df["trans_num"] = [_hash(f"{a}{b}{i}") for i, (a, b) in enumerate(zip(df.cc_num, df.trans_date_trans_time))]
    return df


def cargar_crudo() -> tuple[pd.DataFrame, str]:
    """Usa el Sparkov real si existe en data/raw; si no, genera el sintético."""
    reales = [config.RAW / "fraudTrain.csv", config.RAW / "fraudTest.csv"]
    if all(p.exists() for p in reales):
        df = pd.concat([pd.read_csv(p) for p in reales], ignore_index=True)
        return df, "sparkov_kaggle"
    destino = config.RAW / "transacciones_sinteticas.csv"
    if destino.exists():
        return pd.read_csv(destino, parse_dates=["trans_date_trans_time", "dob"]), "sintetico_sparkov"
    df = generar()
    df.to_csv(destino, index=False)
    return df, "sintetico_sparkov"
