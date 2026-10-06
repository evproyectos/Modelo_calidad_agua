"""
Carga de los datos de los sensores.

Los datos se leen de la carpeta indicada en la variable de entorno
DATOS_SENSORES. Si no está definida, se usan los de la carpeta datos/ de este
proyecto (no se versiona en git).

Archivos que se leen:
  entrenamiento / validacion / prueba .parquet
      Lecturas de los sensores, cada 15 min, más las etiquetas:
        CL2_<nodo>, TEMP_<nodo>, TURB_<nodo>, PH_<nodo>, P_<nodo>   lecturas
        Q_R1, lluvia_mm_h                                           caudal de la fuente y lluvia
        falla_<PARAM>_<nodo>   0 = sin falla; si no, código de falla del sensor
        evento_<nodo>          tipo de evento visible en ese sensor (0 = normal)
        evento_visible         evento visible en algún sensor (0-4)
        evento_red             evento ocurriendo en la red, aunque no se vea
  verdad_<particion>.parquet   valor de referencia de cada parámetro, sin ruido ni fallas
  sensores.csv, fallas_sensor_catalogo.csv, metadatos.json
"""

import json
import os
from pathlib import Path

import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
PARTICIONES = ["entrenamiento", "validacion", "prueba"]
PARAMETROS = ["CL2", "TEMP", "TURB", "PH", "P"]
UNIDADES = {"CL2": "mg/L", "TEMP": "°C", "TURB": "UNT", "PH": "", "P": "m", "Q": "L/s", "lluvia": "mm/h"}
NOMBRES_PARAMETRO = {"CL2": "Cloro residual libre", "TEMP": "Temperatura", "TURB": "Turbidez",
                     "PH": "pH", "P": "Presión"}
TIPOS_EVENTO = {0: "normal", 1: "contaminacion", 2: "falla_cloro", 3: "turbidez_fuente", 4: "falla_ph"}


def carpeta_datos():
    ruta = Path(os.environ.get("DATOS_SENSORES", RAIZ / "datos"))
    if not (ruta / "entrenamiento.parquet").exists():
        raise FileNotFoundError(
            f"No se encontraron los datos en {ruta}.\n"
            "Copia los archivos de datos en la carpeta datos/ del proyecto "
            "o define la variable de entorno DATOS_SENSORES con su ubicación.")
    return ruta


def cargar(particion="entrenamiento", verdad=False):
    """Una partición. Con verdad=True devuelve los valores reales sin ruido ni fallas."""
    nombre = f"verdad_{particion}" if verdad else particion
    return pd.read_parquet(carpeta_datos() / f"{nombre}.parquet")


def cargar_todo(verdad=False):
    """Las tres particiones unidas, con una columna 'particion'."""
    partes = []
    for p in PARTICIONES:
        t = cargar(p, verdad)
        t["particion"] = p
        partes.append(t)
    return pd.concat(partes).sort_index()


def sensores(t):
    """Nodos con sensor, en el orden del dataset."""
    return [c[4:] for c in t.columns if c.startswith("CL2_")]


def columnas(t, parametro):
    """Columnas de lectura de un parámetro (p. ej. 'CL2' -> ['CL2_J280', ...])."""
    return [c for c in t.columns if c.startswith(parametro + "_")]


def larga(t, parametros=("CL2", "TEMP", "TURB", "PH", "P")):
    """Formato largo: una fila por fecha y sensor, una columna por parámetro (útil para seaborn)."""
    filas = []
    for n in sensores(t):
        cols = {f"{p}_{n}": p for p in parametros if f"{p}_{n}" in t}
        d = t[list(cols)].rename(columns=cols)
        d["sensor"] = n
        if f"evento_{n}" in t:
            d["evento"] = t[f"evento_{n}"].map(TIPOS_EVENTO)
        filas.append(d)
    return pd.concat(filas).reset_index()


def tabla_sensores():
    return pd.read_csv(carpeta_datos() / "sensores.csv")


def catalogo_fallas():
    return pd.read_csv(carpeta_datos() / "fallas_sensor_catalogo.csv", parse_dates=["inicio", "fin"])


def metadatos():
    texto = (carpeta_datos() / "metadatos.json").read_bytes()
    try:
        return json.loads(texto.decode("utf-8"))
    except UnicodeDecodeError:          # archivos viejos escritos en Windows con cp1252
        return json.loads(texto.decode("cp1252"))
