"""
Reentrenamiento completo (RF-004.2): vuelve a correr los notebooks de modelos (06 a 10) con los
datos de DATOS_SENSORES, guarda los modelos nuevos en modelos/ y la referencia de deriva, y deja
todo registrado en MLflow: una corrida nueva por notebook y una versión nueva de cada modelo en el
Model Registry (herramientas/registrar_modelos.py), con el alias `produccion`. El servicio de
monitoreo revisa cada 10 minutos si cambiaron los archivos de modelos/ y los vuelve a cargar solo,
sin reiniciarlo.

Los notebooks corren en paralelo, en tres filas (ninguno usa lo que guarda otro):

    09                 anticipación: es el más largo (unos 18 min en la PC de la práctica)
    06 y luego 08      pronóstico a 24 h
    07 y luego 10      detector de eventos y fugas

Así el total queda cerca de lo que tarda el 09, en vez de la suma de todos (34,7 min en serie en
esa PC). El resultado es el mismo que en serie: solo cambia el orden. Cada notebook escribe lo
que imprime el kernel (avisos de TensorFlow, etc.) en artefactos/reentrenamiento/<notebook>.log, y si
falla se muestra el final de ese registro.

Uso, desde la raíz del proyecto y con el entorno activado:

    python herramientas/reentrenar.py              # 06 a 10, la referencia de deriva y las versiones
    python herramientas/reentrenar.py 07 10        # solo esos notebooks (y la referencia)
    python herramientas/reentrenar.py --en-serie   # uno por uno (con poca memoria, menos de 8 GB)

Solo los modelos que se volvieron a entrenar suben de versión.
"""

import argparse
import subprocess
import sys
import threading
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
HERRAMIENTAS = RAIZ / "herramientas"
REGISTROS = RAIZ / "artefactos" / "reentrenamiento"
NOTEBOOKS = ("06", "07", "08", "09", "10")
FILAS = (("09",), ("06", "08"), ("07", "10"))      # la más larga primero


def filas(pedidos, en_serie=False):
    """Los notebooks pedidos repartidos en filas que corren a la vez (una sola fila si en_serie)."""
    if en_serie:
        return [tuple(n for n in NOTEBOOKS if n in pedidos)]
    return [f for f in (tuple(n for n in fila if n in pedidos) for fila in FILAS) if f]


def correr_fila(fila, errores, imprimir):
    """Corre los notebooks de una fila, uno tras otro, en su propio proceso."""
    REGISTROS.mkdir(parents=True, exist_ok=True)
    for nb in fila:
        log = REGISTROS / f"{nb}.log"
        t0 = time.time()
        with open(log, "w", encoding="utf-8") as salida:
            r = subprocess.run([sys.executable, str(HERRAMIENTAS / "correr_notebooks.py"), "--sin-anotar", nb],
                               cwd=RAIZ, stdout=salida, stderr=subprocess.STDOUT)
        if r.returncode != 0:
            texto = log.read_text(encoding="utf-8", errors="replace")
            errores.append(f"Falló el notebook {nb} (registro completo en {log}):\n{texto[-3000:]}")
            imprimir(f"  {nb}  FALLÓ a los {(time.time() - t0) / 60:.1f} min")
            return
        imprimir(f"  {nb}  listo en {(time.time() - t0) / 60:.1f} min")


def correr_notebooks(pedidos, en_serie=False):
    candado = threading.Lock()

    def imprimir(texto):
        with candado:
            print(texto, flush=True)

    errores = []
    reparto = filas(pedidos, en_serie)
    print("Notebooks " + ("en serie" if en_serie else "en paralelo") + ": "
          + " | ".join(" → ".join(f) for f in reparto), flush=True)
    hilos = [threading.Thread(target=correr_fila, args=(f, errores, imprimir)) for f in reparto]
    for h in hilos:
        h.start()
    for h in hilos:
        h.join()
    return errores


def paso(script, *args):
    if subprocess.run([sys.executable, str(HERRAMIENTAS / script), *args], cwd=RAIZ).returncode != 0:
        sys.exit(f"Falló: {script} {' '.join(args)}".strip())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Reentrenamiento de los modelos (RF-004.2)")
    ap.add_argument("notebooks", nargs="*", help="06 a 10 (por defecto, todos)")
    ap.add_argument("--en-serie", action="store_true", help="correr los notebooks uno por uno")
    args = ap.parse_args()
    otros = [n for n in args.notebooks if n not in NOTEBOOKS]
    if otros:
        sys.exit(f"Solo se reentrenan los notebooks {', '.join(NOTEBOOKS)} (no {', '.join(otros)}).")
    t0 = time.time()
    errores = correr_notebooks(args.notebooks or list(NOTEBOOKS), args.en_serie)
    if errores:
        sys.exit("\n\n".join(errores))
    print(f"Notebooks listos en {(time.time() - t0) / 60:.1f} min", flush=True)
    paso("correr_notebooks.py", "--solo-anotar")          # modelos/fuente_lluvia.json, una vez
    paso("referencia_deriva.py")
    paso("registrar_modelos.py")
    print(f"\nReentrenamiento listo en {(time.time() - t0) / 60:.1f} min. Modelos en {RAIZ / 'modelos'}")
