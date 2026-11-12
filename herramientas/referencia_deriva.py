"""
Guarda la referencia para vigilar la deriva de los datos (calidad_agua/deriva.py): la distribución
de cada lectura del sistema en el entrenamiento. El servicio la lee de modelos/ y compara cada
día los últimos 7 días contra ella.

Uso, desde la raíz del proyecto y con el entorno activado (después de los notebooks):

    python herramientas/referencia_deriva.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from calidad_agua import calidad, carga, deriva, experimentos, persistencia  # noqa: E402

if __name__ == "__main__":
    t15 = calidad.lecturas_operativas(carga.cargar("entrenamiento"))
    h = t15[deriva.columnas(t15)].resample("1h").mean()
    ref = deriva.referencia(h)
    destino = Path(__file__).resolve().parents[1] / "modelos" / "referencia_deriva.joblib"
    destino.parent.mkdir(exist_ok=True)
    persistencia.guardar({"referencia": ref, "sensores": carga.sensores(t15),
                          "periodo": carga.texto_periodo("entrenamiento"),
                          "dataset_sha256": experimentos.huella_datos()[0]}, destino)
    print(f"Guardado: {destino} ({len(ref)} columnas: {', '.join(ref)})")
