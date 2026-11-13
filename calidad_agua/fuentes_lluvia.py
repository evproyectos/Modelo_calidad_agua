"""
Fuente de la lluvia observada con que se entrenan los modelos (RNF-009): capa de ingesta.

Los sensores de la red no miden lluvia. La columna lluvia_mm_h del dataset es la de la estación
69-701 del IMN (ITCR, San Carlos), que el simulador mete en las lecturas; en operación el servicio
de monitoreo la saca de una fuente que se pueda consultar en vivo (Monitoreo_calidad_agua,
backend/monitoreo/fuentes_lluvia.py). Los modelos tienen que entrenarse con la misma fuente con que
van a operar (docs/modelos/fuente-lluvia.md), así que carga.cargar() cambia esa columna por la de la
fuente elegida al leer los datos. Ningún módulo de modelado se entera.

  lecturas    la del dataset (la estación del IMN)
  open-meteo  la de Open-Meteo en el mismo punto: clima/lluvia_open_meteo_historica.csv (Historical
              Forecast API, la misma serie que da en vivo la Forecast API), promedio de los modelos
              meteorológicos elegidos

Se elige en hiperparametros.yaml (sección datos) y se puede cambiar con las variables de entorno
FUENTE_LLUVIA y LLUVIA_MODELOS (las mismas del servicio de monitoreo). Convención: el valor
horario de las 14:00 es la lluvia de 13:00 a 14:00, y va en las lecturas de las 14:00 a las 14:45.
"""

import os
import warnings
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from . import configuracion

RAIZ = Path(__file__).resolve().parents[1]
ARCHIVO = RAIZ / "clima" / "lluvia_open_meteo_historica.csv"
COLUMNA = "lluvia_mm_h"
FUENTES = ("lecturas", "open-meteo")
CUATRO = ["ecmwf_ifs025", "gfs_seamless", "icon_seamless", "gem_seamless"]


def _config():
    return configuracion.cargar().get("datos", {})


def elegida():
    fuente = (os.environ.get("FUENTE_LLUVIA") or _config().get("fuente_lluvia") or "lecturas").strip()
    if fuente not in FUENTES:
        raise ValueError(f"FUENTE_LLUVIA debe ser una de: {', '.join(FUENTES)} (no {fuente!r})")
    return fuente


def modelos():
    valor = os.environ.get("LLUVIA_MODELOS") or _config().get("lluvia_modelos") or "ecmwf_ifs025"
    return [m.strip() for m in valor.split(",") if m.strip()] if isinstance(valor, str) else list(valor)


def descripcion():
    """Lo que queda en modelos/fuente_lluvia.json al entrenar (el servicio lo lee)."""
    fuente = elegida()
    return {"fuente": fuente, "modelos": modelos() if fuente == "open-meteo" else []}


def leer_open_meteo(ruta=None):
    """CSV de Open-Meteo: una columna por modelo meteorológico (mm en la hora), más el promedio de
    los cuatro globales. Acepta varios CSV unidos, cada uno con su encabezado."""
    t = pd.read_csv(Path(ruta) if ruta else ARCHIVO, skiprows=3)
    t = t[t["time"] != "time"]
    t.index = pd.to_datetime(t.pop("time"))
    t = t.apply(pd.to_numeric, errors="coerce")
    t.columns = [c.split(" (")[0].replace("precipitation_", "").replace("precipitation", "best_match")
                 for c in t.columns]
    t = t[~t.index.duplicated()].sort_index().clip(lower=0)
    if all(m in t for m in CUATRO):
        t["promedio"] = t[CUATRO].mean(axis=1)
    return t


@lru_cache(maxsize=4)
def _serie(modelos_, ruta):
    t = leer_open_meteo(ruta)
    faltan = [m for m in modelos_ if m not in t]
    if faltan:
        raise ValueError(f"{ruta or ARCHIVO} no tiene la lluvia de {faltan}")
    return t[list(modelos_)].mean(axis=1).astype("float32")


def serie_open_meteo(modelos_=None, ruta=None):
    """Lluvia horaria de Open-Meteo (mm, índice = fin de la hora), promedio de `modelos_`."""
    return _serie(tuple(modelos_ or modelos()), str(ruta) if ruta else None)


def aplicar(t, fuente=None):
    """La tabla de lecturas con lluvia_mm_h de la fuente elegida (sin cambios con 'lecturas')."""
    fuente = fuente or elegida()
    if fuente == "lecturas" or COLUMNA not in t:
        return t
    nueva = serie_open_meteo().reindex(t.index.floor("h")).to_numpy(dtype="float32")
    faltan = int(np.isnan(nueva).sum())
    if faltan:
        warnings.warn(f"La lluvia de Open-Meteo no cubre {faltan} lecturas; quedan sin lluvia", stacklevel=2)
    t = t.copy()
    t[COLUMNA] = nueva
    return t
