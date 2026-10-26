"""
Construcción de la tabla de entrenamiento para el pronóstico a 24 h.

Una fila = un sensor en una hora t. Las variables usan solo información disponible
hasta la hora t (nada del futuro). El objetivo es el valor horario en t + 24 h.

Variables (por lo encontrado en el EDA):
  - del propio sensor: valor actual de cada parámetro, cambios en las últimas horas,
    media y desviación de las últimas 6 y 24 h, valor de hace 24 h (ciclo diario);
  - de la fuente (J280) y de la red: valores actuales y de hace unas horas, porque los
    cambios de la fuente llegan a los sensores con 1 a 16 h de retraso;
  - del sensor como lugar: identificador y valor típico (cloro muy distinto entre sensores);
  - de calendario: hora del día (ciclo diario). La época del año NO se usa: con 16 meses de
    entrenamiento el modelo ve cada época una sola vez y aprende cómo fue ese año en particular,
    que no se repite el siguiente (notebook 06);
  - de clima y operación: lluvia acumulada reciente (6, 24 y 72 h), lluvia pronosticada para las
    próximas 24 h por tramos (calidad_agua.clima, si se pasa `pronostico`) y caudal de la fuente;
  - de calidad del dato: cuántas lecturas válidas tuvo la hora.
"""

import numpy as np
import pandas as pd

from . import carga, calidad

FUENTE = "J280"
PARAMS = ["CL2", "TURB", "PH", "TEMP"]


def horario(t15):
    """Promedios horarios de las lecturas válidas, más eventos (máximo) y partición de cada hora."""
    num = t15.drop(columns=[c for c in t15.columns if c == "particion"])
    h = num.resample("1h").mean()
    S = carga.sensores(t15)
    for n in S:
        # en operación no hay etiquetas: si la columna no existe, queda vacía
        h[f"evento_{n}"] = t15[f"evento_{n}"].resample("1h").max() if f"evento_{n}" in t15 else np.nan
        h[f"lecturas_{n}"] = t15[f"CL2_{n}"].notna().resample("1h").sum()      # lecturas válidas en la hora
    h["lluvia_mm"] = t15["lluvia_mm_h"].resample("1h").mean()
    h["particion"] = t15["particion"].resample("1h").first() if "particion" in t15 else None
    return h


def tabla(h, horizonte=24, tipicos=None, h_objetivo=None, pronostico=None):
    """
    Tabla larga (sensor × hora) con variables y objetivos.
    `h`: tabla horaria de las ENTRADAS. Debe limpiarse igual que en operación
    (calidad.lecturas_operativas), porque en vivo no hay etiquetas de falla.
    `h_objetivo`: tabla horaria de la que salen los OBJETIVOS (el valor real que se quiere
    pronosticar, limpio de fallas del sensor: calidad.lecturas_validas). Si es None se usa `h`.
    `tipicos`: valor típico de cada sensor y parámetro (si es None se calcula con el
    periodo de entrenamiento y se devuelve para reutilizarlo). En operación se pasan los
    guardados al entrenar.
    `pronostico`: lluvia pronosticada por tramos para cada hora (calidad_agua.clima.ventanas),
    indexada como `h`. Si es None, esas columnas no se agregan.
    """
    S = [c[4:] for c in h.columns if c.startswith("CL2_")]
    entr = h["particion"] == "entrenamiento"
    if tipicos is None:
        tipicos = {(p, n): float(h.loc[entr & (h[f"evento_{n}"] == 0), f"{p}_{n}"].median())
                   for n in S for p in PARAMS}

    # variables comunes a todos los sensores (fuente, red, clima, calendario)
    comun = pd.DataFrame(index=h.index)
    for p in PARAMS:
        f = h[f"{p}_{FUENTE}"]
        comun[f"fuente_{p}"] = f
        for k in [3, 6, 12, 24]:
            comun[f"fuente_{p}_hace{k}h"] = f.shift(k)
        red = h[[f"{p}_{n}" for n in S]]
        comun[f"red_{p}_mediana"] = red.median(axis=1)
    comun["caudal"] = h["Q_R1"]
    comun["caudal_media24h"] = h["Q_R1"].rolling(24, min_periods=12).mean()
    for k in [6, 24, 72]:
        comun[f"lluvia_{k}h"] = h["lluvia_mm"].rolling(k, min_periods=1).sum()
    if pronostico is not None:
        comun = comun.join(pronostico.reindex(h.index))
    hora_obj = (h.index + pd.Timedelta(hours=horizonte)).hour
    comun["hora_sin"] = np.sin(2 * np.pi * hora_obj / 24)
    comun["hora_cos"] = np.cos(2 * np.pi * hora_obj / 24)

    partes = []
    for i, n in enumerate(S):
        d = comun.copy()
        d["sensor"] = i
        for p in PARAMS:
            x = h[f"{p}_{n}"]
            d[f"{p}"] = x
            d[f"{p}_tipico"] = tipicos[(p, n)]
            d[f"{p}_hace24h"] = x.shift(24)
            d[f"{p}_hace23h"] = x.shift(23)
            for k in [1, 3, 6]:
                d[f"{p}_cambio{k}h"] = x - x.shift(k)
            for k in [6, 24]:
                d[f"{p}_media{k}h"] = x.rolling(k, min_periods=k // 2).mean()
                d[f"{p}_desv{k}h"] = x.rolling(k, min_periods=k // 2).std()
            d[f"{p}_vs_fuente"] = x - h[f"{p}_{FUENTE}"]
            # objetivo: valor en t + horizonte
            d[f"objetivo_{p}"] = (h if h_objetivo is None else h_objetivo)[f"{p}_{n}"].shift(-horizonte)
        d["P"] = h[f"P_{n}"]
        d["lecturas_hora"] = h[f"lecturas_{n}"]
        d["evento_objetivo"] = h[f"evento_{n}"].shift(-horizonte)
        d["particion"] = h["particion"]
        d["nodo"] = n
        partes.append(d)
    return pd.concat(partes).rename_axis("fecha").reset_index(), tipicos


def columnas_entrada(tabla_):
    fuera = {"fecha", "particion", "nodo", "evento_objetivo"}
    return [c for c in tabla_.columns if c not in fuera and not c.startswith("objetivo_")]
