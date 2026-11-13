"""
Corre los notebooks en orden y los guarda con sus resultados (como "Run All" + guardar).

Uso, desde la raíz del proyecto y con el entorno activado:

    python herramientas/correr_notebooks.py              # todos (01 a 10)
    python herramientas/correr_notebooks.py 06 07 09     # solo esos

Si un notebook falla, se detiene y muestra el error; los anteriores quedan guardados.

Lluvia (RNF-009): el EDA (01 a 05) describe el dataset tal como viene, con la lluvia de la estación
del IMN; los modelos (06 a 10) se entrenan con la de la fuente de hiperparametros.yaml, sección datos
(o FUENTE_LLUVIA), y al terminarlos se escribe modelos/fuente_lluvia.json para que el servicio de
monitoreo use esa misma fuente.
"""

import json
import os
import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "notebooks"
EDA = {"01", "02", "03", "04", "05"}
MODELOS = {"06", "07", "08", "09", "10"}
sys.path.insert(0, str(RAIZ))


def fuente_lluvia(numero, original):
    """El entorno del kernel: el EDA con la lluvia del dataset; los modelos, con la configurada."""
    if numero in EDA:
        os.environ["FUENTE_LLUVIA"] = "lecturas"
    elif original is None:
        os.environ.pop("FUENTE_LLUVIA", None)
    else:
        os.environ["FUENTE_LLUVIA"] = original


def anotar_fuente_lluvia():
    from calidad_agua import fuentes_lluvia
    destino = RAIZ / "modelos" / "fuente_lluvia.json"
    destino.parent.mkdir(exist_ok=True)
    destino.write_text(json.dumps(fuentes_lluvia.descripcion(), ensure_ascii=False) + "\n", encoding="utf-8")
    return destino


def correr(ruta):
    nb = nbformat.read(ruta, as_version=4)
    ep = ExecutePreprocessor(timeout=None, kernel_name="python3")
    t0 = time.time()
    try:
        ep.preprocess(nb, {"metadata": {"path": str(CARPETA)}})
    finally:
        nbformat.write(nb, ruta)             # se guarda aunque falle, para ver hasta dónde llegó
    return time.time() - t0


if __name__ == "__main__":
    pedidos = sys.argv[1:]
    notebooks = sorted(p for p in CARPETA.glob("[0-9][0-9]_*.ipynb")
                       if not pedidos or p.name[:2] in pedidos)
    total = time.time()
    original = os.environ.get("FUENTE_LLUVIA")
    for ruta in notebooks:
        fuente_lluvia(ruta.name[:2], original)
        print(f"{ruta.name} ...", end=" ", flush=True)
        try:
            print(f"listo en {correr(ruta) / 60:.1f} min", flush=True)
        except CellExecutionError as e:
            print("FALLÓ\n")
            print(str(e)[-3000:])
            sys.exit(1)
    fuente_lluvia("", original)
    if any(r.name[:2] in MODELOS for r in notebooks):
        print(f"Lluvia de los modelos: {anotar_fuente_lluvia().read_text(encoding='utf-8').strip()}")
    print(f"\nTodo listo en {(time.time() - total) / 60:.1f} min")
