"""
Dataset maestro (RF-001.2): un solo comando que junta las lecturas de los sensores del sistema con
la lluvia, las limpia y preprocesa (calidad.lecturas_operativas), le agrega el vector de etiquetas
(etiquetas.py) y lo exporta en Parquet y CSV. Es idempotente: con los mismos datos da los mismos
archivos (y escribe su huella SHA-256 al lado).

Uso, desde la raíz del proyecto y con el entorno activado:

    python herramientas/dataset_maestro.py                 # en datos_maestro/
    python herramientas/dataset_maestro.py otra/carpeta

Esquema: docs/datos/esquema-dataset.md.
"""

import hashlib
import json
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from calidad_agua import calidad, carga, etiquetas, preprocesamiento  # noqa: E402


def construir():
    t = carga.cargar_todo()                                   # sensores del sistema (hiperparametros.yaml)
    limpio = calidad.lecturas_operativas(t)
    fila, _ = etiquetas.construir(limpio)
    return limpio.join(fila)


def exportar(d, carpeta):
    carpeta = Path(carpeta)
    carpeta.mkdir(parents=True, exist_ok=True)
    d.to_parquet(carpeta / "dataset_maestro.parquet")
    d.to_csv(carpeta / "dataset_maestro.csv", float_format="%.4f")
    huellas = {a.name: hashlib.sha256(a.read_bytes()).hexdigest() for a in sorted(carpeta.glob("dataset_maestro.*"))}
    resumen = {"filas": len(d), "desde": str(d.index.min()), "hasta": str(d.index.max()),
               "sensores": carga.sensores(d), "flag_imputacion": d["flag_imputacion"].value_counts().to_dict(),
               "etiqueta_1_pct": round(100 * float(d["etiqueta"].mean()), 3), "sha256": huellas}
    (carpeta / "dataset_maestro.json").write_text(json.dumps(resumen, ensure_ascii=False, indent=2), encoding="utf-8")
    return resumen


if __name__ == "__main__":
    destino = Path(sys.argv[1]) if len(sys.argv) > 1 else RAIZ / "datos_maestro"
    preprocesamiento.registrar_en(RAIZ / "registros" / "preprocesamiento.log")
    r = exportar(construir(), destino)
    print(f"{r['filas']:,} filas · {len(r['sensores'])} sensores · {r['desde']} a {r['hasta']}")
    print(f"flag_imputacion: {r['flag_imputacion']} · etiqueta 1: {r['etiqueta_1_pct']} %")
    print(f"Archivos en {destino}")
