"""
Limpieza de lecturas según lo encontrado en el notebook 01 (calidad de datos).

Una lectura de calidad (CL2, TEMP, TURB, PH) se considera NO válida si:
  - tiene una falla de sensor identificada (columna falla_<PARAM>_<nodo> distinta de 0), o
  - la presión del nodo es menor a PRESION_MINIMA (tubería sin agua: el sensor no mide agua), o
  - el valor está fuera del rango físico posible en agua potable (RANGO_FISICO).
"""

import numpy as np

from src import carga

PRESION_MINIMA = 1.0          # m
CALIDAD = ["CL2", "TEMP", "TURB", "PH"]
RANGO_FISICO = {"CL2": (0, 5), "TEMP": (10, 40), "TURB": (0, 100), "PH": (4, 10)}

# Decreto 38924-S, Reglamento para la Calidad del Agua Potable (Costa Rica)
NORMA = {
    "CL2": {"minimo": 0.3, "maximo": 1.0},                      # cloro residual libre, mg/L
    "TURB": {"recomendado": 1.0, "maximo": 5.0},                # UNT
    "PH": {"minimo": 6.5, "maximo": 8.5},
}


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
    t = t.copy()
    for n in carga.sensores(t):
        for p in CALIDAD:
            col = f"{p}_{n}"
            t.loc[mascara_invalida(t, p, n), col] = np.nan
    return t
