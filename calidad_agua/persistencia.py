"""
Guardar los modelos con joblib sin que dependan de dill.

TensorFlow importa dill si está instalado, y dill, al importarse, agrega sus funciones a la tabla con
que el pickle de Python guarda cada tipo. joblib copia esa tabla cuando se importa: si se importó
después que TensorFlow (como en el notebook 08), algunos objetos (por ejemplo el functools.partial
que lleva dentro HistGradientBoosting) quedan guardados con tipos de dill, y el archivo ya no se abre
donde dill no está instalado (como la imagen del servicio de monitoreo). guardar() usa la tabla
original mientras guarda y después comprueba, abriendo el archivo en otro proceso sin dill, que se
pueda usar en cualquier lado.
"""

import subprocess
import sys
from pathlib import Path

import joblib
from joblib import numpy_pickle

PRUEBA = ("import sys; sys.modules['dill'] = None; import joblib; joblib.load(sys.argv[1])")


def se_abre_sin_dill(ruta):
    r = subprocess.run([sys.executable, "-c", PRUEBA, str(ruta)], capture_output=True, text=True)
    return r.returncode == 0


def guardar(objeto, ruta, compress=3, comprobar=True):
    """joblib.dump sin tipos de dill; error si el archivo igual quedara dependiendo de dill."""
    dill = sys.modules.get("dill")
    original = getattr(getattr(dill, "_dill", None), "pickle_dispatch_copy", None)   # la tabla sin dill
    anterior = numpy_pickle.NumpyPickler.dispatch
    if original is not None:
        numpy_pickle.NumpyPickler.dispatch = dict(original)
    try:
        joblib.dump(objeto, ruta, compress=compress)
    finally:
        numpy_pickle.NumpyPickler.dispatch = anterior
    if comprobar and dill is not None and not se_abre_sin_dill(ruta):
        raise RuntimeError(f"{Path(ruta).name} necesita dill para abrirse; el servicio no lo podría cargar")
    return ruta
