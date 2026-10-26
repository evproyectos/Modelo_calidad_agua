"""
Pronóstico de lluvia (Open-Meteo) como entrada del pronóstico a 24 h.

Para entrenar se usan pronósticos PASADOS, tal como estaban disponibles en su momento: la
Previous Runs API de Open-Meteo guarda, para cada hora, la lluvia que pronosticaba cada modelo
meteorológico 24 h antes (`precipitation_previous_day1`). Así el modelo aprende a usar un
pronóstico con errores reales, no la lluvia que de verdad cayó.

Se usa el PROMEDIO de cuatro modelos globales (ECMWF IFS 0,25°, GFS, ICON y GEM): acierta mejor
la lluvia de la zona que cualquiera de ellos por separado y es estable en el tiempo. La opción
"best_match" de Open-Meteo no sirve para entrenar, porque cambia de modelo con el tiempo y su
sesgo cambia con él (notebook 06).

Archivo: clima/pronostico_lluvia_modelos.csv (o la ruta de la variable de entorno
PRONOSTICO_LLUVIA), descargado de
  https://previous-runs-api.open-meteo.com/v1/forecast?latitude=10.3606&longitude=-84.5135
    &hourly=precipitation_previous_day1&models=ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless
    &start_date=2024-08-01&end_date=2026-08-31&timezone=America%2FCosta_Rica&format=csv
(coordenadas de la estación 69-701 ITCR, San Carlos, del IMN). En operación se usa el pronóstico
actual de los mismos cuatro modelos (Forecast API de Open-Meteo, parámetro `models`).

Convención de Open-Meteo, igual que la columna lluvia_mm_h de los datos: el valor de las 14:00
es la lluvia que cae de 13:00 a 14:00.

Sin fuga de información: en la hora t, el pronóstico para cualquier hora v entre t+1 y t+24
hecho 24 h antes se emitió en v − 24 ≤ t, así que ya existía en t. Para las horas más cercanas
es un pronóstico algo más viejo que el que habría en operación: el resultado es conservador.
"""

import os
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO = RAIZ / "clima" / "pronostico_lluvia_modelos.csv"
MODELOS = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]
TRAMOS = [(0, 6), (6, 12), (12, 24)]          # horas hacia adelante desde el momento del pronóstico
COLUMNAS = [f"lluvia_pron_{a}_{b}h" for a, b in TRAMOS]


def ruta_archivo():
    return Path(os.environ.get("PRONOSTICO_LLUVIA", ARCHIVO))


def leer(ruta=None):
    """Tabla del archivo de Open-Meteo: una columna por modelo, indexada por hora válida."""
    ruta = Path(ruta) if ruta else ruta_archivo()
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró el pronóstico de lluvia en {ruta}")
    t = pd.read_csv(ruta, skiprows=3)
    t.index = pd.to_datetime(t.pop("time"))
    t.columns = [c.split(" (")[0].replace("precipitation_previous_day1_", "") for c in t.columns]
    return t.clip(lower=0)


def cargar_pronostico(ruta=None, modelos=MODELOS):
    """Lluvia pronosticada 24 h antes (mm en la hora), promedio de los modelos, por hora válida."""
    t = leer(ruta)
    return t[modelos].mean(axis=1).rename("lluvia_pron").astype("float32")


def ventanas(lluvia_pron, horas):
    """
    Para cada hora t de `horas`, la lluvia pronosticada en cada tramo hacia adelante:
    lluvia_pron_0_6h = horas válidas t+1 … t+6, lluvia_pron_6_12h = t+7 … t+12, etc.
    `lluvia_pron` está indexada por hora válida (en operación, el pronóstico más reciente).
    """
    horas = pd.DatetimeIndex(horas)
    s = lluvia_pron.astype("float64").reindex(
        pd.date_range(horas.min(), horas.max() + pd.Timedelta(hours=TRAMOS[-1][1]), freq="1h"))
    def suma(a, b):          # lluvia de las horas válidas t+a+1 … t+b (NaN si falta alguna)
        return s.rolling(b - a, min_periods=b - a).sum().reindex(horas + pd.Timedelta(hours=b)).values
    return pd.DataFrame({c: suma(a, b) for c, (a, b) in zip(COLUMNAS, TRAMOS)}, index=horas).astype("float32")
