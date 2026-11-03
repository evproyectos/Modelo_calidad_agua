"""
Corre los notebooks en orden y los guarda con sus resultados (como "Run All" + guardar).

Uso, desde la raíz del proyecto y con el entorno activado:

    python herramientas/correr_notebooks.py              # todos (01 a 09)
    python herramientas/correr_notebooks.py 06 07 09     # solo esos

Si un notebook falla, se detiene y muestra el error; los anteriores quedan guardados.
"""

import sys
import time
from pathlib import Path

import nbformat
from nbconvert.preprocessors import CellExecutionError, ExecutePreprocessor

CARPETA = Path(__file__).resolve().parents[1] / "notebooks"


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
    for ruta in notebooks:
        print(f"{ruta.name} ...", end=" ", flush=True)
        try:
            print(f"listo en {correr(ruta) / 60:.1f} min", flush=True)
        except CellExecutionError as e:
            print("FALLÓ\n")
            print(str(e)[-3000:])
            sys.exit(1)
    print(f"\nTodo listo en {(time.time() - total) / 60:.1f} min")
