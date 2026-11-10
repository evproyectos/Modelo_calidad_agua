"""
Limpieza de lecturas según lo encontrado en el notebook 01 (calidad de datos).

Una lectura de calidad (CL2, TEMP, TURB, PH) se considera NO válida si:
  - tiene una falla de sensor identificada (columna falla_<PARAM>_<nodo> distinta de 0), o
  - la presión del nodo es menor a PRESION_MINIMA (tubería sin agua: el sensor no mide agua), o
  - el valor está fuera del rango físico posible en agua potable (RANGO_FISICO).

lecturas_operativas() aplica después el preprocesamiento (preprocesamiento.py, RF-001.3): quita
los picos sueltos por IQR, interpola los huecos de hasta 30 min y agrega flag_imputacion.
"""

import numpy as np

from . import carga, preprocesamiento

PRESION_MINIMA = 1.0          # m
CALIDAD = ["CL2", "TEMP", "TURB", "PH"]
RANGO_FISICO = {"CL2": (0, 5), "TEMP": (10, 40), "TURB": (0, 100), "PH": (4, 10)}

# Decreto 38924-S, Reglamento para la Calidad del Agua Potable (Costa Rica)
NORMA = {
    "CL2": {"minimo": 0.3, "maximo": 1.0},                      # cloro residual libre, mg/L
    "TURB": {"recomendado": 1.0, "maximo": 5.0},                # UNT
    "PH": {"minimo": 6.5, "maximo": 8.5},
}


# Las lecturas se guardan en float32 (así se entrenaron los modelos). Los datos que llegan en
# vivo por la API vienen en JSON y Python los lee como float64: si no se convierten, algunas
# variables que comparan contra un umbral cambian en el último decimal y los modelos dan otro
# resultado. Por eso la limpieza fija siempre el tipo.
TIPO_LECTURAS = "float32"


def _como_float32(t):
    t = t.copy()
    flotantes = t.select_dtypes(include="floating").columns
    t[flotantes] = t[flotantes].astype(TIPO_LECTURAS)
    return t


def mascara_invalida(t, p, n):
    """True en las lecturas de `p` en el nodo `n` que no se deben usar."""
    x = t[f"{p}_{n}"]
    lo, hi = RANGO_FISICO[p]
    invalida = (t[f"P_{n}"] < PRESION_MINIMA) | (x < lo) | (x > hi)
    if f"falla_{p}_{n}" in t:
        invalida |= t[f"falla_{p}_{n}"] != 0
    return invalida.values


def lecturas_validas(t):
    """Copia de la tabla con las lecturas no válidas puestas en NaN."""
    t = _como_float32(t)
    for n in carga.sensores(t):
        for p in CALIDAD:
            col = f"{p}_{n}"
            t.loc[mascara_invalida(t, p, n), col] = np.nan
    return t


# ---------------------------------------------------------------------------
# Reglas que se pueden aplicar en operación (sin etiquetas de falla)
# ---------------------------------------------------------------------------
LECTURAS_PEGADO = 16          # 4 h sin ningún cambio = sensor pegado (notebook 01)


def pegado(x, lecturas=LECTURAS_PEGADO):
    """True donde la serie lleva `lecturas` valores seguidos idénticos (sin contar ceros)."""
    sin_cambio = (x.diff() == 0).astype(int).rolling(lecturas - 1).sum() == lecturas - 1
    return (sin_cambio & (x > 0)).values


def lecturas_operativas(t, preprocesar=True):
    """
    Copia de la tabla con las lecturas no válidas en NaN, usando SOLO reglas que funcionan en
    operación: presión mínima, rango físico y sensor pegado. No usa las columnas falla_*, que en
    operación no existen. Con `preprocesar` (por defecto), después quita los picos (IQR) e
    interpola los huecos cortos (preprocesamiento.py); la deriva y el ruido quedan en los datos.
    """
    t = _como_float32(t)
    for n in carga.sensores(t):
        for p in CALIDAD:
            col = f"{p}_{n}"
            x = t[col]
            lo, hi = RANGO_FISICO[p]
            malo = ((t[f"P_{n}"] < PRESION_MINIMA) | (x < lo) | (x > hi)).values | pegado(x)
            t.loc[malo, col] = np.nan
    return preprocesamiento.preprocesar(t) if preprocesar else t
