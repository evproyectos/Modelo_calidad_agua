"""
Modelo de pronóstico a 24 h (scikit-learn, HistGradientBoostingRegressor).

Un modelo por parámetro, compartido por los 21 sensores (el sensor es una variable más).
El modelo no pronostica el valor directamente, sino el CAMBIO respecto al valor actual,
que es el pronóstico "igual que ayer a la misma hora" para dentro de 24 h. Así solo tiene
que aprender lo que la línea base no explica.

  - Se minimiza el error absoluto: es más robusto a los eventos (cambios grandes y raros)
    y se alinea con la meta de "acertar dentro de un margen".
  - La turbidez se modela en escala logarítmica, porque su tolerancia es relativa (± 20 %).
"""

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

from . import configuracion

LOG = {"TURB"}
PISO_LOG = 0.05            # UNT, para no tomar logaritmo de cero

HIPERPARAMETROS = configuracion.seccion("pronostico_arboles")      # hiperparametros.yaml


def _a_escala(p, x):
    return np.log(np.clip(x, PISO_LOG, None)) if p in LOG else x


def _desde_escala(p, x):
    return np.exp(x) if p in LOG else x


def objetivo(tabla, p):
    """Cambio entre el valor actual y el de dentro de 24 h (en la escala del modelo)."""
    return _a_escala(p, tabla[f"objetivo_{p}"]) - _a_escala(p, tabla[p])


def entrenar(tabla, p, columnas, filas):
    y = objetivo(tabla, p)
    filas = filas & y.notna() & tabla[p].notna()
    mod = HistGradientBoostingRegressor(categorical_features=[columnas.index("sensor")], **HIPERPARAMETROS)
    mod.fit(tabla.loc[filas, columnas], y[filas])
    return mod


def pronosticar(mod, tabla, p, columnas):
    """Pronóstico del valor dentro de 24 h para cada fila de la tabla."""
    cambio = mod.predict(tabla[columnas])
    return _desde_escala(p, _a_escala(p, tabla[p].values) + cambio)
