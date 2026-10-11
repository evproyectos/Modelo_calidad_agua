"""
Pronóstico a 24 h con una red neuronal recurrente (TensorFlow/Keras, LSTM).

Mismo problema que src/modelo.py (cambio de CL2, TURB y PH dentro de 24 h, turbidez en escala
logarítmica), pero en vez de resúmenes calculados a mano la red recibe la SECUENCIA de las
últimas VENTANA_H horas y aprende sola qué patrones usar.

Entrada de cada ejemplo (un sensor en la hora t):
  - secuencia (VENTANA_H × variables): cloro, turbidez, pH, temperatura y presión del sensor,
    los mismos de la fuente (J280), la mediana de la red, lluvia, caudal, hora del día y época
    del año, más una marca de "dato faltante" por parámetro del sensor;
  - estáticas: el sensor (embedding) y su valor típico de cloro, turbidez y pH.
Salida: los tres cambios a 24 h a la vez (un solo modelo para los tres parámetros).

Los datos faltantes se ponen en 0 después de normalizar y se marcan; la pérdida (error absoluto)
ignora los objetivos faltantes.
"""

import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
import keras
from keras import layers
import tensorflow as tf

from src.carga import sensores as _sensores

FUENTE = "J280"
VENTANA_H = 48
HORIZONTE_H = 24
SALIDAS = ["CL2", "TURB", "PH"]
PISO_LOG = 0.05


def _log(x):
    return np.log(np.clip(x, PISO_LOG, None))


def preparar(h, tipicos):
    """
    Arreglos por sensor listos para la red. `h` es la tabla horaria de src.variables.horario y
    `tipicos` el diccionario de src.variables.tabla. Devuelve un diccionario con todo lo necesario.
    """
    S = [c[4:] for c in h.columns if c.startswith("CL2_")]
    entr = (h["particion"] == "entrenamiento").values
    hora = h.index.hour
    dia = h.index.dayofyear
    comunes = {
        "hora_sin": np.sin(2 * np.pi * hora / 24), "hora_cos": np.cos(2 * np.pi * hora / 24),
        "anio_sin": np.sin(2 * np.pi * dia / 365.25), "anio_cos": np.cos(2 * np.pi * dia / 365.25),
        "lluvia": np.log1p(h["lluvia_mm"].fillna(0).values), "caudal": h["Q_R1"].values,
    }
    for p in ["CL2", "TURB", "PH", "TEMP"]:
        f = h[f"{p}_{FUENTE}"].values
        comunes[f"fuente_{p}"] = _log(f) if p == "TURB" else f
        red = h[[f"{p}_{n}" for n in S]].median(axis=1).values
        comunes[f"red_{p}"] = _log(red) if p == "TURB" else red

    secuencias, objetivos, base = [], [], []
    for n in S:
        propio = {}
        for p in ["CL2", "TURB", "PH", "TEMP", "P"]:
            x = h[f"{p}_{n}"].values
            propio[p] = _log(x) if p == "TURB" else x
        X = np.column_stack([propio[p] for p in propio] + [comunes[k] for k in comunes]).astype("float32")
        secuencias.append(X)
        actual = np.column_stack([_log(h[f"{p}_{n}"].values) if p == "TURB" else h[f"{p}_{n}"].values
                                  for p in SALIDAS])
        futuro = np.vstack([actual[HORIZONTE_H:], np.full((HORIZONTE_H, len(SALIDAS)), np.nan)])
        objetivos.append((futuro - actual).astype("float32"))
        base.append(actual.astype("float32"))
    nombres = ["CL2", "TURB", "PH", "TEMP", "P"] + list(comunes)

    # normalización con el periodo de entrenamiento
    todo = np.concatenate([s[entr] for s in secuencias])
    media, desv = np.nanmean(todo, axis=0), np.nanstd(todo, axis=0) + 1e-6
    obj_todo = np.concatenate([o[entr] for o in objetivos])
    media_y, desv_y = np.zeros(len(SALIDAS), "float32"), (np.nanstd(obj_todo, axis=0) + 1e-6).astype("float32")
    for i in range(len(S)):
        s = (secuencias[i] - media) / desv
        faltante = np.isnan(s[:, :4]).astype("float32")              # marcas para CL2, TURB, PH, TEMP propios
        secuencias[i] = np.column_stack([np.nan_to_num(s, nan=0.0), faltante]).astype("float32")
        objetivos[i] = objetivos[i] / desv_y
    estaticas = np.array([[tipicos[(p, n)] for p in SALIDAS] for n in S], dtype="float32")
    estaticas[:, 1] = _log(estaticas[:, 1])
    estaticas = (estaticas - estaticas.mean(axis=0)) / (estaticas.std(axis=0) + 1e-6)
    return {"sensores": S, "fechas": h.index, "particion": h["particion"].values,
            "evento": np.column_stack([h[f"evento_{n}"].values for n in S]),
            "secuencias": secuencias, "objetivos": objetivos, "base": base, "estaticas": estaticas,
            "variables": nombres + ["falta_CL2", "falta_TURB", "falta_PH", "falta_TEMP"],
            "media": media, "desv": desv, "desv_y": desv_y}


def indices(datos, particion, cada_h=1, solo_normal=False):
    """
    Pares (sensor, hora) con ventana completa y al menos un objetivo, en la partición dada.
    Con solo_normal=True se excluyen los ejemplos cuya hora pronosticada tiene evento en el sensor.
    """
    filas = []
    horas = np.where(datos["particion"] == particion)[0]
    horas = horas[(horas >= VENTANA_H - 1) & (horas < len(datos["fechas"]) - HORIZONTE_H)][::cada_h]
    for i in range(len(datos["sensores"])):
        ok = ~np.isnan(datos["objetivos"][i][horas]).all(axis=1)
        if solo_normal:
            ok &= np.nan_to_num(datos["evento"][horas + HORIZONTE_H, i]) == 0
        filas.append(np.column_stack([np.full(ok.sum(), i), horas[ok]]))
    return np.concatenate(filas)


class Lotes(keras.utils.PyDataset):
    """Arma los lotes sobre la marcha (no guarda todas las ventanas en memoria)."""

    def __init__(self, datos, idx, tamano=512, barajar=False, semilla=0, **kw):
        super().__init__(**kw)
        self.d, self.idx, self.tamano, self.barajar = datos, idx.copy(), tamano, barajar
        self.rng = np.random.default_rng(semilla)
        if barajar:
            self.rng.shuffle(self.idx)

    def __len__(self):
        return int(np.ceil(len(self.idx) / self.tamano))

    def __getitem__(self, k):
        lote = self.idx[k * self.tamano:(k + 1) * self.tamano]
        seq = np.stack([self.d["secuencias"][i][t - VENTANA_H + 1:t + 1] for i, t in lote])
        sens = lote[:, 0:1].astype("int32")
        est = self.d["estaticas"][lote[:, 0]]
        y = np.stack([self.d["objetivos"][i][t] for i, t in lote])
        return (seq, sens, est), y

    def on_epoch_end(self):
        if self.barajar:
            self.rng.shuffle(self.idx)


def mae_con_faltantes(y, pred):
    """Error absoluto medio ignorando los objetivos faltantes (NaN)."""
    valido = tf.math.logical_not(tf.math.is_nan(y))
    y0 = tf.where(valido, y, tf.zeros_like(y))
    err = tf.abs(y0 - pred) * tf.cast(valido, pred.dtype)
    return tf.reduce_sum(err) / tf.maximum(tf.reduce_sum(tf.cast(valido, pred.dtype)), 1.0)


def construir(n_variables, n_sensores, unidades=48, abandono=0.1, l2=0.0):
    seq = keras.Input((VENTANA_H, n_variables), name="secuencia")
    sens = keras.Input((1,), dtype="int32", name="sensor")
    est = keras.Input((len(SALIDAS),), name="tipicos")
    reg = keras.regularizers.l2(l2) if l2 else None
    x = layers.LSTM(unidades, kernel_regularizer=reg, name="lstm")(seq)
    e = layers.Flatten()(layers.Embedding(n_sensores, 4, name="embedding_sensor")(sens))
    z = layers.Concatenate()([x, e, est])
    z = layers.Dropout(abandono)(z)
    z = layers.Dense(64, activation="relu", kernel_regularizer=reg)(z)
    z = layers.Dropout(abandono)(z)
    salida = layers.Dense(len(SALIDAS), name="cambio_24h")(z)
    mod = keras.Model([seq, sens, est], salida)
    mod.compile(optimizer=keras.optimizers.Adam(1e-3), loss=mae_con_faltantes)
    return mod


def pronosticar(mod, datos, idx, tamano=2048):
    """Pronóstico (en unidades originales) de los tres parámetros para cada par (sensor, hora)."""
    cambio = mod.predict(Lotes(datos, idx, tamano), verbose=0) * datos["desv_y"]
    base = np.stack([datos["base"][i][t] for i, t in idx])
    pron = base + cambio
    pron[:, SALIDAS.index("TURB")] = np.exp(pron[:, SALIDAS.index("TURB")])
    return pron
