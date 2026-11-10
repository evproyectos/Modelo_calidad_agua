"""
Vector de etiquetas supervisadas (RF-001.6): a cada lectura de 15 min le asigna 0 (operación
normal) o 1 (evento crítico: contaminación o fuga) y de dónde salió esa etiqueta.

Fuentes, de la más confiable a la menos:
  laboratorio      resultados del Ministerio de Salud: un muestreo que incumple el Decreto
                   38924-S marca las lecturas de ese sensor en la hora del muestreo (y uno que
                   cumple las marca como normales, aunque la regla del umbral diga otra cosa).
                   Tabla con fecha, nodo, parametro, valor.
  registro_asada   fugas reportadas por la ASADA, con inicio y fin: marca esas lecturas.
                   Tabla con inicio, fin y, si se sabe, nodo (si no, toda la red).
  umbral_iot       si no hay laboratorio para ese momento: 3 o más lecturas seguidas de un sensor
                   fuera de la norma (cloro < 0,3 mg/L, turbidez > 5 UNT o pH fuera de 6,5–8,5).
                   Si un sensor pasa más de la mitad del tiempo fuera de la norma en un parámetro
                   (incumplimiento crónico, como el cloro bajo al final de la red), ese parámetro
                   no cuenta para la regla en ese sensor: ahí la norma no distingue un evento de lo
                   de siempre. Esos casos quedan en etq.attrs["cronicos"] para reportarlos.

Con los datos del simulador no hay laboratorio ni registros de la ASADA: los modelos se entrenan
con la verdad del simulador (columnas evento_*), y este módulo sirve para comparar la regla del
umbral contra esa verdad (notebook 05) y para armar las etiquetas cuando lleguen los datos reales.

Balanceo: los eventos son menos del 5 % de las lecturas. En vez de crear lecturas sintéticas
(SMOTE), los modelos pesan cada clase según su frecuencia (class_weight: balanced en
hiperparametros.yaml), que equivale a sobremuestrear los eventos sin inventar datos en una serie de
tiempo.
"""

import numpy as np
import pandas as pd

from . import carga
from .calidad import NORMA

LECTURAS_SEGUIDAS = 3
NORMAL, EVENTO = 0, 1
FUENTES = ("laboratorio", "registro_asada", "umbral_iot")


CRONICO = 0.5                   # fuera de norma más de la mitad del tiempo: incumplimiento crónico


def fuera_por_parametro(t, n):
    """{parámetro: lecturas de ese sensor fuera de la norma} para cloro, turbidez y pH (NaN no cuenta)."""
    m = {}
    if f"CL2_{n}" in t:
        m["CL2"] = t[f"CL2_{n}"] < NORMA["CL2"]["minimo"]
    if f"TURB_{n}" in t:
        m["TURB"] = t[f"TURB_{n}"] > NORMA["TURB"]["maximo"]
    if f"PH_{n}" in t:
        m["PH"] = (t[f"PH_{n}"] < NORMA["PH"]["minimo"]) | (t[f"PH_{n}"] > NORMA["PH"]["maximo"])
    return {p: x.fillna(False) for p, x in m.items()}


def fuera_de_norma(t, n, ignorar=()):
    """Lecturas de un sensor fuera de la norma en algún parámetro (salvo los de `ignorar`)."""
    m = pd.Series(False, index=t.index)
    for p, x in fuera_por_parametro(t, n).items():
        if p not in ignorar:
            m |= x
    return m


def sostenido(m, lecturas=LECTURAS_SEGUIDAS):
    """Marca los tramos de al menos `lecturas` True seguidos (todo el tramo, no solo desde la tercera)."""
    grupo = (m != m.shift()).cumsum()
    largo = m.groupby(grupo).transform("size")
    return m & (largo >= lecturas)


def incumple(parametro, valor):
    n = NORMA.get(parametro, {})
    return (("minimo" in n and valor < n["minimo"]) or ("maximo" in n and valor > n["maximo"]))


def cronicos(t, limite=CRONICO):
    """{sensor: [parámetros]} fuera de la norma más que `limite` de las lecturas con dato."""
    out = {}
    for n in carga.sensores(t):
        for p, x in fuera_por_parametro(t, n).items():
            con_dato = t[f"{p}_{n}"].notna()
            if con_dato.any() and x[con_dato].mean() > limite:
                out.setdefault(n, []).append(p)
    return out


def construir(t, laboratorio=None, registros=None, lecturas=LECTURAS_SEGUIDAS, cronico=CRONICO):
    """
    Etiquetas por sensor y por fila para la tabla de lecturas `t` (cada 15 min).
    Devuelve (por_fila, por_sensor): por_fila tiene `etiqueta` (0/1) y `fuente_etiqueta`;
    por_sensor tiene una columna 0/1 por sensor (y en attrs["cronicos"], los sensores y parámetros
    con incumplimiento crónico que la regla no usa).
    """
    S = carga.sensores(t)
    cron = cronicos(t, cronico) if cronico is not None else {}
    etq = pd.DataFrame({n: sostenido(fuera_de_norma(t, n, cron.get(n, ())), lecturas).astype(int) for n in S},
                       index=t.index)
    fuente = pd.DataFrame("umbral_iot", index=t.index, columns=S)
    if registros is not None and len(registros):
        for r in registros.itertuples():
            nodos = [r.nodo] if getattr(r, "nodo", None) in S else S
            sel = (t.index >= pd.Timestamp(r.inicio)) & (t.index <= pd.Timestamp(r.fin))
            etq.loc[sel, nodos] = EVENTO
            fuente.loc[sel, nodos] = "registro_asada"
    if laboratorio is not None and len(laboratorio):
        for r in laboratorio.itertuples():
            if r.nodo not in S:
                continue
            hora = pd.Timestamp(r.fecha).floor("h")
            sel = (t.index >= hora) & (t.index < hora + pd.Timedelta(hours=1))
            if incumple(r.parametro, r.valor):
                etq.loc[sel, r.nodo] = EVENTO
            elif fuente.loc[sel, r.nodo].ne("registro_asada").all():
                etq.loc[sel, r.nodo] = NORMAL           # el laboratorio dice que cumple
            fuente.loc[sel, r.nodo] = "laboratorio"
    fila = etq.max(axis=1)
    # la fuente de la fila es la más confiable entre los sensores que la deciden
    rango = {f: i for i, f in enumerate(FUENTES)}
    decide = fuente.where(etq.eq(fila, axis=0))
    fuente_fila = decide.apply(lambda x: min(x.dropna(), key=rango.get) if x.notna().any() else "umbral_iot", axis=1)
    por_fila = pd.DataFrame({"etiqueta": fila.astype(int), "fuente_etiqueta": fuente_fila}, index=t.index)
    etq.attrs["cronicos"] = cron
    return por_fila, etq


def comparar_con_verdad(etq, t, tipos=(1, 5)):
    """
    Precisión y sensibilidad de las etiquetas contra la verdad del simulador (evento_<nodo> con
    contaminación = 1 o fuga = 5), lectura por lectura.
    """
    S = [n for n in etq.columns if f"evento_{n}" in t]
    y = np.concatenate([t[f"evento_{n}"].isin(tipos).values for n in S])
    p = np.concatenate([etq[n].values == EVENTO for n in S])
    vp = int((y & p).sum())
    return {"lecturas_evento": int(y.sum()), "lecturas_marcadas": int(p.sum()), "aciertos": vp,
            "precision": vp / p.sum() if p.sum() else np.nan, "sensibilidad": vp / y.sum() if y.sum() else np.nan,
            "proporcion_evento": float(y.mean())}
