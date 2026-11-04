"""
Carga de los datos de los sensores.

Los datos se leen de la carpeta indicada en la variable de entorno
DATOS_SENSORES. Si no está definida, se usan los de la carpeta datos/ de este
proyecto (no se versiona en git).

Sensores del sistema: el simulador exporta muchos (21), pero el sistema usa solo los de
hiperparametros.yaml, sección red (la salida de la fuente, con su caudalímetro, y dos
puntos de la red de distribución). Al cargar se quitan los demás y evento_visible se
recalcula con los que quedan. La variable de entorno SENSORES (p. ej. "J280,J69,J296")
cambia la lista sin tocar el archivo, y SENSORES=todos usa todos (sirve para elegirlos).

Archivos que se leen:
  entrenamiento / validacion / prueba .parquet
      Lecturas de los sensores, cada 15 min, más las etiquetas:
        CL2_<nodo>, TEMP_<nodo>, TURB_<nodo>, PH_<nodo>, P_<nodo>   lecturas
        Q_R1, lluvia_mm_h                                           caudal de la fuente y lluvia
        falla_<PARAM>_<nodo>   0 = sin falla; si no, código de falla del sensor
        evento_<nodo>          tipo de evento visible en ese sensor (0 = normal)
        evento_visible         evento visible en algún sensor del sistema (0-5)
        evento_red             evento ocurriendo en la red, aunque no se vea
  verdad_<particion>.parquet   valor de referencia de cada parámetro, sin ruido ni fallas
  sensores.csv, fallas_sensor_catalogo.csv, eventos_catalogo.csv, metadatos.json
"""

import json
import os
from pathlib import Path

import pandas as pd

from . import configuracion

RAIZ = Path(__file__).resolve().parents[1]
PARTICIONES = ["entrenamiento", "validacion", "prueba"]
PARAMETROS = ["CL2", "TEMP", "TURB", "PH", "P"]
UNIDADES = {"CL2": "mg/L", "TEMP": "°C", "TURB": "UNT", "PH": "", "P": "m", "Q": "L/s", "lluvia": "mm/h"}
NOMBRES_PARAMETRO = {"CL2": "Cloro residual libre", "TEMP": "Temperatura", "TURB": "Turbidez",
                     "PH": "pH", "P": "Presión"}
TIPOS_EVENTO = {0: "normal", 1: "contaminacion", 2: "falla_cloro", 3: "turbidez_fuente", 4: "falla_ph",
                5: "fuga"}
PREFIJOS_SENSOR = ("CL2", "TEMP", "TURB", "PH", "P", "evento", "falla")


def sensores_sistema():
    """Sensores del sistema (lista) o None si se usan todos los del dataset."""
    entorno = os.environ.get("SENSORES", "").strip()
    if entorno:
        return None if entorno.lower() == "todos" else [s.strip() for s in entorno.split(",") if s.strip()]
    return list(configuracion.seccion("red")["sensores"]) or None


def filtrar_sensores(t, nodos):
    """Deja solo las columnas de esos sensores (más caudal, lluvia y etiquetas de la red) y
    recalcula evento_visible con ellos."""
    if nodos is None:
        return t
    faltan = [n for n in nodos if f"CL2_{n}" not in t and f"P_{n}" not in t]
    if faltan:
        raise ValueError(f"Sensores {faltan} no están en el dataset")
    quedan = [c for c in t.columns if c.split("_")[0] not in PREFIJOS_SENSOR or c in ("evento_visible", "evento_red")
              or c.rsplit("_", 1)[-1] in nodos]
    t = t[quedan].copy()
    if "evento_visible" in t:
        t["evento_visible"] = t[[f"evento_{n}" for n in nodos]].max(axis=1).astype(t["evento_visible"].dtype)
    return t


def carpeta_datos():
    ruta = Path(os.environ.get("DATOS_SENSORES", RAIZ / "datos"))
    if not (ruta / "entrenamiento.parquet").exists():
        raise FileNotFoundError(
            f"No se encontraron los datos en {ruta}.\n"
            "Copia los archivos de datos en la carpeta datos/ del proyecto "
            "o define la variable de entorno DATOS_SENSORES con su ubicación.")
    return ruta


def cargar(particion="entrenamiento", verdad=False, todos=False):
    """Una partición. Con verdad=True devuelve los valores reales sin ruido ni fallas. Con
    todos=True, con todos los sensores del dataset (para elegir dónde van los del sistema)."""
    nombre = f"verdad_{particion}" if verdad else particion
    t = pd.read_parquet(carpeta_datos() / f"{nombre}.parquet")
    return t if todos else filtrar_sensores(t, sensores_sistema())


def cargar_todo(verdad=False, todos=False):
    """Las tres particiones unidas, con una columna 'particion'."""
    partes = []
    for p in PARTICIONES:
        t = cargar(p, verdad, todos)
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


MESES = ["ene", "feb", "mar", "abr", "may", "jun", "jul", "ago", "sep", "oct", "nov", "dic"]


def periodos():
    """Inicio y fin [inicio, fin) de cada partición, tal como vienen en metadatos.json.

    Las fechas no se escriben a mano en los notebooks: así, si el dataset cambia de periodo,
    basta con volver a correrlos."""
    m = metadatos()["particiones"]
    return {p: (pd.Timestamp(m[p][0]), pd.Timestamp(m[p][1])) for p in PARTICIONES}


def texto_periodo(particion):
    """'sep 2024 – dic 2025', o 'may – ago 2026' si empieza y termina el mismo año."""
    ini, fin = periodos()[particion]
    ult = fin - pd.Timedelta(days=1)
    if ini.year == ult.year:
        return f"{MESES[ini.month - 1]} – {MESES[ult.month - 1]} {ult.year}"
    return f"{MESES[ini.month - 1]} {ini.year} – {MESES[ult.month - 1]} {ult.year}"


def tabla_sensores():
    t = pd.read_csv(carpeta_datos() / "sensores.csv")
    nodos = sensores_sistema()
    return t if nodos is None else t[t["nodo"].isin(nodos)].reset_index(drop=True)


def distancias_sensores():
    """Distancia por tubería (m) de cada nodo de consumo a cada sensor (simulador/exportar.py)."""
    d = pd.read_csv(carpeta_datos() / "distancias_sensores.csv", index_col="nodo")
    nodos = sensores_sistema()
    return d if nodos is None else d[nodos]


def catalogo_eventos():
    """Eventos del simulador: tipo, código, inicio, fin y, en los datos nuevos, causa y
    deterioro previo. Sirve para medir cuánto se anticipa cada evento (notebook 09)."""
    cat = pd.read_csv(carpeta_datos() / "eventos_catalogo.csv")
    for c in ("inicio", "fin", "inicio_precursor"):
        if c in cat:
            cat[c] = pd.to_datetime(cat[c])
    if "causa" not in cat:
        cat["causa"] = "aleatoria"
    return cat


def catalogo_fallas():
    return pd.read_csv(carpeta_datos() / "fallas_sensor_catalogo.csv", parse_dates=["inicio", "fin"])


def metadatos():
    texto = (carpeta_datos() / "metadatos.json").read_bytes()
    try:
        return json.loads(texto.decode("utf-8"))
    except UnicodeDecodeError:          # archivos viejos escritos en Windows con cp1252
        return json.loads(texto.decode("cp1252"))
