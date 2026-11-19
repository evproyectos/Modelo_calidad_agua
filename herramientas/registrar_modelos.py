"""
Versiones de los modelos en el Model Registry de MLflow (RF-004.2): registra los modelos de
modelos/ que todavía no tengan versión, lista las versiones o vuelve a una anterior. Ver
calidad_agua/registro_modelos.py. reentrenar.py lo llama al final, así que normalmente no hace
falta correrlo a mano.

Uso, desde la raíz del proyecto y con el entorno activado:

    python herramientas/registrar_modelos.py                      # registra lo que hay en modelos/
    python herramientas/registrar_modelos.py --lista              # versiones de cada modelo
    python herramientas/registrar_modelos.py --restaurar fugas 2  # vuelve a la versión 2 de fugas

Las versiones se ven también en MLflow (http://127.0.0.1:5000), pestaña Models.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calidad_agua import registro_modelos as R  # noqa: E402


def mostrar(resumen):
    for nombre, r in resumen.items():
        estado = "versión nueva" if r["nueva"] else "ya estaba registrada"
        print(f"{nombre:22} versión {r['version']:<3} {estado} (corrida {r['run_id']})")
    if not resumen:
        print("No hay modelos en modelos/: corre primero reentrenar.py.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Versiones de los modelos en MLflow")
    ap.add_argument("--lista", action="store_true", help="listar las versiones de cada modelo")
    ap.add_argument("--restaurar", nargs=2, metavar=("MODELO", "VERSION"),
                    help=f"volver a una versión anterior ({', '.join(R.MODELOS)})")
    args = ap.parse_args()
    try:
        if args.lista:
            filas = R.lista()
            for f in filas:
                marca = "  <- en producción" if f["en_produccion"] else ""
                print(f"{f['modelo']:22} versión {f['version']:<3} {f['creada']}  {f['algoritmo']:14}{marca}")
            if not filas:
                print("Todavía no hay versiones registradas.")
        elif args.restaurar:
            nombre, version = args.restaurar
            mostrar(R.restaurar(nombre, version))
            print(f"\nmodelos/ tiene ahora la versión {version} de {nombre}. "
                  "El servicio de monitoreo la carga solo en menos de 10 minutos.")
        else:
            mostrar(R.registrar())
    except R.ErrorRegistro as e:
        sys.exit(f"No se pudo: {e}")
