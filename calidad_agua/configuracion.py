"""
Hiperparámetros de los modelos (RF-002.4), leídos de hiperparametros.yaml, que viaja con el
paquete. Cada sección se devuelve como copia: modificarla no cambia la de los demás.
"""

import copy
from pathlib import Path

import yaml

ARCHIVO = Path(__file__).with_name("hiperparametros.yaml")
_leidos = None


def cargar(ruta=None):
    """Todo el archivo como diccionario (el del paquete, o `ruta` si se da otro)."""
    global _leidos
    if ruta is not None:
        return yaml.safe_load(Path(ruta).read_text(encoding="utf-8"))
    if _leidos is None:
        _leidos = yaml.safe_load(ARCHIVO.read_text(encoding="utf-8"))
    return copy.deepcopy(_leidos)


def seccion(nombre):
    return cargar()[nombre]
