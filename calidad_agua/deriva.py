"""
Deriva de los datos (RF-004.2): ¿los datos que llegan en operación se parecen a los del
entrenamiento?

Los modelos aprendieron con una distribución de cloro, turbidez, pH, temperatura, presión y
caudal. Si con el tiempo los datos cambian (un sensor que se descalibra, una época más lluviosa que
cualquiera del entrenamiento, cambios en la red o en la planta), los modelos pierden exactitud
aunque nadie lo note. Se mide con el índice de estabilidad de la población (PSI) de cada variable:
se parte el rango del entrenamiento en 10 tramos con la misma cantidad de datos y se compara qué
fracción de los datos recientes cae en cada tramo.

  PSI < 0,10        estable
  0,10 a 0,25       vigilar
  > 0,25            deriva: conviene revisar los sensores y volver a entrenar

La referencia se calcula con el entrenamiento (herramientas/referencia_deriva.py la guarda en
modelos/referencia_deriva.joblib) y el servicio la compara cada día con los últimos 7 días.
"""

import numpy as np
import pandas as pd

TRAMOS = 10
VIGILAR, DERIVA = 0.10, 0.25
PARAMETROS = ("CL2", "TURB", "PH", "TEMP", "P", "Q")
MINIMO_DATOS = 48                      # horas con dato para que la comparación tenga sentido


def columnas(t):
    """Columnas de lecturas que se vigilan (cada parámetro de cada sensor y el caudal)."""
    return [c for c in t.columns if c.split("_")[0] in PARAMETROS and not c.startswith(("evento", "falla"))]


def referencia(h, cols=None):
    """Bordes de los tramos (cuantiles) y fracción de datos en cada uno, por columna. `h`: lecturas
    horarias del entrenamiento."""
    ref = {}
    for c in cols or columnas(h):
        x = h[c].dropna().values
        if len(x) < MINIMO_DATOS:
            continue
        bordes = np.unique(np.quantile(x, np.linspace(0, 1, TRAMOS + 1)[1:-1]))
        ref[c] = {"bordes": bordes, "fracciones": _fracciones(x, bordes)}
    return ref


def _fracciones(x, bordes):
    cuenta = np.bincount(np.searchsorted(bordes, x, side="right"), minlength=len(bordes) + 1)
    return cuenta / max(cuenta.sum(), 1)


def psi(esperadas, observadas, piso=1e-4):
    e = np.clip(np.asarray(esperadas, float), piso, None)
    o = np.clip(np.asarray(observadas, float), piso, None)
    return float(np.sum((o - e) * np.log(o / e)))


def estado(valor):
    if valor != valor:
        return "sin datos"
    return "deriva" if valor > DERIVA else ("vigilar" if valor > VIGILAR else "estable")


def revisar(ref, h):
    """PSI de cada columna de la referencia con los datos horarios `h` (p. ej. los últimos 7 días)."""
    filas = []
    for c, r in ref.items():
        x = h[c].dropna().values if c in h else np.array([])
        v = psi(r["fracciones"], _fracciones(x, r["bordes"])) if len(x) >= MINIMO_DATOS else np.nan
        filas.append({"columna": c, "psi": v, "estado": estado(v), "horas": len(x)})
    return pd.DataFrame(filas, columns=["columna", "psi", "estado", "horas"])
