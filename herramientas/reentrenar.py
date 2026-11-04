"""
Reentrenamiento completo (RF-004.2): vuelve a correr los notebooks de modelos (06 a 10) con los
datos de DATOS_SENSORES, guarda los modelos nuevos en modelos/ y la referencia de deriva, y deja
todo registrado en MLflow con una versión nueva de cada experimento. El servicio de monitoreo
revisa cada 10 minutos si cambiaron los archivos de modelos/ y los vuelve a cargar solo, sin
reiniciarlo.

Uso, desde la raíz del proyecto y con el entorno activado:

    python herramientas/reentrenar.py              # 06 a 10 y la referencia de deriva
    python herramientas/reentrenar.py 07 10        # solo esos notebooks (y la referencia)
"""

import subprocess
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]

if __name__ == "__main__":
    notebooks = sys.argv[1:] or ["06", "07", "08", "09", "10"]
    t0 = time.time()
    for paso in ([sys.executable, str(RAIZ / "herramientas" / "correr_notebooks.py"), *notebooks],
                 [sys.executable, str(RAIZ / "herramientas" / "referencia_deriva.py")]):
        if subprocess.run(paso, cwd=RAIZ).returncode != 0:
            sys.exit(f"Falló: {' '.join(paso[1:])}")
    print(f"\nReentrenamiento listo en {(time.time() - t0) / 60:.1f} min. Modelos en {RAIZ / 'modelos'}")
