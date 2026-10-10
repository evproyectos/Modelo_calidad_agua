"""
Definición de la meta del proyecto: cuándo un pronóstico a 24 h se considera acertado.

Meta:
    Al menos el 80 % de los pronósticos a 24 h dentro de la tolerancia en cloro, turbidez y pH,
    en operación normal, y mejor que el pronóstico ingenuo "igual que ayer a la misma hora" en
    cada parámetro. El acierto durante eventos se reporta por separado.

Tolerancias (criterio técnico, ver notebook 03):
    CL2   ± 0,1 mg/L            ~2 veces el ruido del sensor; 1/7 del rango de la norma (0,3–1,0)
    TURB  ± 20 % del valor,     se adapta al nivel: ± 0,2 UNT en 0,5 UNT y ± 2 UNT en 10 UNT;
          mínimo ± 0,2 UNT      es la forma usual de expresar la precisión de un turbidímetro
    PH    ± 0,1                 ~3 veces el ruido del sensor
    TEMP  ± 0,5 °C              solo se reporta: no tiene límite en la norma, se usa como entrada
"""

import numpy as np

HORIZONTE_H = 24
META_PCT = 80.0
PARAMETROS_META = ["CL2", "TURB", "PH"]

TOLERANCIA = {
    "CL2": {"absoluta": 0.1},
    "TURB": {"relativa": 0.20, "minima": 0.2},
    "PH": {"absoluta": 0.1},
    "TEMP": {"absoluta": 0.5},
}


def tolerancia(parametro, observado):
    """Tolerancia aplicable a cada valor observado (array o Series)."""
    t = TOLERANCIA[parametro]
    obs = np.asarray(observado, dtype=float)
    if "relativa" in t:
        return np.maximum(t["minima"], t["relativa"] * np.abs(obs))
    return np.full(obs.shape, t["absoluta"])


def acierto(parametro, observado, pronosticado):
    """True donde el pronóstico queda dentro de la tolerancia; NaN si falta alguno de los dos."""
    obs = np.asarray(observado, dtype=float)
    pron = np.asarray(pronosticado, dtype=float)
    ok = np.abs(obs - pron) <= tolerancia(parametro, obs)
    return np.where(np.isnan(obs) | np.isnan(pron), np.nan, ok)


def texto_tolerancia(parametro):
    t = TOLERANCIA[parametro]
    if "relativa" in t:
        return f"± {t['relativa']:.0%} (mín. ± {t['minima']:g})"
    return f"± {t['absoluta']:g}"
