"""
Preprocesamiento de las lecturas cada 15 min (RF-001.3), después de quitar las lecturas no válidas
(calidad.lecturas_operativas) y antes de armar las variables de los modelos.

Pasos, en este orden:
  1. Valores atípicos por el método IQR (k = 1,5, configurable). Se aplica a la diferencia entre
     cada lectura y la mediana de las lecturas vecinas (las 2 de antes y las 2 de después), no al
     valor mismo: así se detectan los picos sueltos de un sensor (1 o 2 lecturas raras), pero no
     el cambio sostenido de un evento real (el cloro que baja durante horas en una contaminación),
     que es justo lo que los modelos tienen que ver. Una lectura es atípica si esa diferencia cae
     fuera de [Q1 - k·IQR, Q3 + k·IQR] y además se aparta del vecindario más que `minimo` (en
     las unidades del parámetro), para no tocar el ruido normal del sensor. Las atípicas se
     quitan y se rellenan como un faltante más. Solo se revisan cloro, turbidez, pH y
     temperatura: en la presión y el caudal los cambios bruscos son reales (las bombas), igual
     que un aguacero en la lluvia.
  2. Faltantes: interpolación lineal en los huecos de hasta 30 min (2 lecturas seguidas); los
     huecos más largos no se inventan: quedan vacíos y se marcan.
  3. flag_imputacion por fila: 'original' (todas las lecturas medidas), 'interpolado' (alguna
     rellenada) o 'marcado' (alguna vacía por un hueco largo). preprocesar() también devuelve el
     detalle por columna.

La normalización (StandardScaler) y las ventanas deslizantes (24 a 72 h) van dentro de cada
modelo que las necesita (red_neuronal.py, anticipacion.py y fugas.py), porque dependen del
algoritmo.

Registro: cada vez que se aplica, escribe una línea JSON con la fecha, los parámetros y cuántas
lecturas tocó cada paso en el logger "calidad_agua.preprocesamiento". registrar_en(ruta) lo
manda a un archivo (los notebooks lo dejan en registros/preprocesamiento.log).

En operación (el servicio de monitoreo) es lo mismo, con una diferencia que no se puede evitar:
la última lectura recibida todavía no tiene vecinas de después, así que no se le revisa si es
atípica ni se interpola un hueco que todavía no terminó.
"""

import json
import logging
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from . import carga, configuracion as configuracion_

PARAMETROS = ["CL2", "TEMP", "TURB", "PH", "P"]
OTRAS = ["Q_R1", "lluvia_mm_h"]
ORIGINAL, INTERPOLADO, MARCADO = "original", "interpolado", "marcado"
VECINAS = 2                      # lecturas de cada lado para la mediana del vecindario

registro = logging.getLogger("calidad_agua.preprocesamiento")
registro.addHandler(logging.NullHandler())


def configuracion(cfg=None):
    c = dict(configuracion_.cargar().get("preprocesamiento", {}))
    c.update(cfg or {})
    return c


def registrar_en(ruta):
    """Escribe el registro del preprocesamiento en `ruta` (una línea JSON por aplicación)."""
    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    if not any(isinstance(h, logging.FileHandler) and Path(h.baseFilename) == ruta.resolve()
               for h in registro.handlers):
        manejador = logging.FileHandler(ruta, encoding="utf-8")
        manejador.setFormatter(logging.Formatter("%(message)s"))
        registro.addHandler(manejador)
    registro.setLevel(logging.INFO)
    return ruta


def columnas_lectura(t):
    """Columnas que se preprocesan: los cinco parámetros de cada sensor, el caudal y la lluvia."""
    cols = [f"{p}_{n}" for n in carga.sensores(t) for p in PARAMETROS if f"{p}_{n}" in t]
    return cols + [c for c in OTRAS if c in t]


def atipicos(x, k=1.5, minimo=0.0):
    """
    Máscara de lecturas atípicas de una serie (cada 15 min) por IQR sobre la diferencia con la
    mediana de sus vecinas. Solo se revisan las lecturas con el vecindario completo.
    """
    v = x.astype("float64")
    mediana = v.rolling(2 * VECINAS + 1, center=True, min_periods=2 * VECINAS + 1).median()
    r = v - mediana
    q1, q3 = np.nanpercentile(r, [25, 75]) if r.notna().any() else (np.nan, np.nan)
    if not np.isfinite(q1):
        return pd.Series(False, index=x.index)
    rango = q3 - q1
    fuera = (r < q1 - k * rango) | (r > q3 + k * rango)
    return (fuera & (r.abs() > minimo)).fillna(False)


def interpolar(x, max_lecturas=2):
    """
    Interpola linealmente los huecos de hasta `max_lecturas` lecturas que tienen dato antes y
    después. Devuelve (serie, máscara de interpoladas, máscara de marcadas).
    """
    falta = x.isna()
    grupo = (falta != falta.shift()).cumsum()
    largo = falta.groupby(grupo).transform("sum")
    dentro = x.ffill().notna() & x.bfill().notna()                  # con dato a los dos lados
    corto = falta & (largo <= max_lecturas) & dentro
    lleno = x.interpolate(method="linear", limit_area="inside")
    y = x.where(~corto, lleno)
    return y, corto, falta & ~corto


def preprocesar(t, cfg=None, detalle=False):
    """
    Aplica los pasos a una tabla de lecturas cada 15 min (índice de fechas). Devuelve una copia
    con los valores tratados y la columna flag_imputacion; con `detalle=True`, también la tabla
    de estado por lectura ('original', 'atipico', 'interpolado' o 'marcado').
    """
    c = configuracion(cfg)
    paso = int(c.get("paso_min", 15))
    max_lect = int(c.get("max_hueco_min", 30)) // paso
    k = float(c.get("iqr_k", 1.5))
    minimos = c.get("iqr_minimo", {})
    con_iqr = set(c.get("iqr_parametros", ["CL2", "TURB", "PH", "TEMP"]))
    t = t.copy()
    estado = pd.DataFrame(ORIGINAL, index=t.index, columns=columnas_lectura(t))
    cuenta = {"atipicos": {}, "interpoladas": {}, "marcadas": {}}
    for col in estado.columns:
        x = t[col]
        revisar = c.get("iqr", True) and _parametro(col) in con_iqr
        malas = atipicos(x, k, minimos.get(_parametro(col), 0.0)) if revisar else pd.Series(False, index=x.index)
        if malas.any():
            x = x.mask(malas)
            estado.loc[malas.values, col] = "atipico"
            cuenta["atipicos"][col] = int(malas.sum())
        y, interp, marc = interpolar(x, max_lect)
        t[col] = y.astype(t[col].dtype)
        estado.loc[interp.values & (estado[col] == ORIGINAL).values, col] = INTERPOLADO
        estado.loc[marc.values, col] = MARCADO
        for clave, m in (("interpoladas", interp), ("marcadas", marc)):
            if m.any():
                cuenta[clave][col] = int(m.sum())
    t = t.copy().assign(flag_imputacion=_flag_fila(estado))   # copy: la tabla queda en un solo bloque
    _anotar(t, c, cuenta)
    return (t, estado) if detalle else t


def _parametro(col):
    return "Q" if col.startswith("Q_") else col.split("_")[0]


def _flag_fila(estado):
    marcada = (estado == MARCADO).any(axis=1)
    tocada = estado.isin([INTERPOLADO, "atipico"]).any(axis=1)
    return pd.Series(np.where(marcada, MARCADO, np.where(tocada, INTERPOLADO, ORIGINAL)), index=estado.index)


def _anotar(t, c, cuenta):
    if not registro.isEnabledFor(logging.INFO):
        return
    registro.info(json.dumps({
        "fecha": datetime.now().isoformat(timespec="seconds"),
        "datos": [str(t.index.min()), str(t.index.max())] if len(t) else None,
        "lecturas": int(len(t)),
        "parametros": {"iqr": c.get("iqr", True), "iqr_k": c.get("iqr_k", 1.5), "iqr_minimo": c.get("iqr_minimo", {}),
                       "iqr_parametros": c.get("iqr_parametros"),
                       "vecinas": VECINAS, "max_hueco_min": c.get("max_hueco_min", 30)},
        "atipicos": sum(cuenta["atipicos"].values()), "interpoladas": sum(cuenta["interpoladas"].values()),
        "marcadas": sum(cuenta["marcadas"].values()), "por_columna": cuenta,
    }, ensure_ascii=False))


def resumen(estado):
    """Porcentaje de lecturas en cada estado, por columna (para el notebook 01)."""
    return (estado.apply(lambda s: s.value_counts(normalize=True)).T.fillna(0) * 100).round(2)
