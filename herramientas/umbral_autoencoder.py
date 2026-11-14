"""
Umbral del Autoencoder de fugas (RF-002.2): el percentil 95 del error de reconstrucción en la
operación normal, con su distribución y lo que separa. Escribe docs/modelos/umbral-autoencoder.json
y figuras/umbral_autoencoder.png, que usa docs/modelos/umbral-autoencoder.md.

Usa los mismos datos, horas normales y regresión que el notebook 10, y entrena el Autoencoder dos
veces con la misma semilla para comprobar que el umbral se reproduce.

Uso (desde la raíz del proyecto):  python herramientas/umbral_autoencoder.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

from calidad_agua import carga, calidad, detector, fuentes_lluvia  # noqa: E402
from calidad_agua import fugas as F  # noqa: E402

PERCENTILES = [50, 75, 90, 95, 99, 99.9]


def datos():
    """Lo mismo que la primera celda del notebook 10."""
    crudo = carga.cargar_todo()
    h = detector.horario(calidad.lecturas_operativas(crudo))
    cat = carga.catalogo_eventos()
    cat_f = cat[cat["tipo"] == "fuga"]
    y = F.clase(h, cat).values
    cerca = np.zeros(len(h), bool)
    for _, ev in cat_f.iterrows():
        cerca |= (h.index >= ev["inicio_precursor"]) & (h.index < ev["fin"] + pd.Timedelta(hours=12))
    normal = ~cerca & (y == 0)
    part = h["particion"].values
    entr = part == "entrenamiento"
    X = F.tabla(h, F.ajustar_normal(h, entr & normal))
    return X, y, normal, part


def resumen(puntaje, mascara):
    p = puntaje[mascara]
    return {"horas": int(mascara.sum()),
            "pct_sobre_umbral": round(100 * float((p >= 1).mean()), 2),
            "pct_sobre_4_umbrales": round(100 * float((p >= 4).mean()), 2),
            "mediana_en_umbrales": round(float(np.median(p)), 3)}


def main():
    X, y, normal, part = datos()
    entr = part == "entrenamiento"
    X_normal = X[entr & normal]
    ae = F.Autoencoder().fit(X_normal)
    otra = F.Autoencoder().fit(X_normal)                   # misma semilla: tiene que dar lo mismo
    error = ae.error(X_normal)
    puntaje = ae.puntaje(X)
    grupos = {}
    for p in ["entrenamiento", "validacion", "prueba"]:
        en = part == p
        grupos[p] = {"normal": resumen(puntaje, en & normal), "fuga": resumen(puntaje, en & (y == 1))}
    r = {
        "fuente_lluvia": fuentes_lluvia.descripcion(),
        "arquitectura": F.HIPERPARAMETROS["autoencoder"],
        "variables": X.shape[1],
        "horas_normales_entrenamiento": int(len(X_normal)),
        "umbral_percentil_95": ae.umbral_,
        "umbral_segunda_corrida": otra.umbral_,
        "diferencia_relativa_entre_corridas": abs(ae.umbral_ - otra.umbral_) / ae.umbral_,
        "percentiles_error_normal_entrenamiento": {str(q): float(np.percentile(error, q)) for q in PERCENTILES},
        "error_medio_normal_entrenamiento": float(np.mean(error)),
        "puntaje_por_particion": grupos,
    }
    salida = RAIZ / "docs" / "modelos" / "umbral-autoencoder.json"
    salida.write_text(json.dumps(r, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    figura(puntaje, y, normal, part)
    print(json.dumps({k: v for k, v in r.items() if k != "arquitectura"}, indent=2, ensure_ascii=False))


def figura(puntaje, y, normal, part):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calidad_agua.graficos import COLORES, GRIS, estilo, guardar
    estilo()
    val = part == "validacion"
    bordes = np.logspace(-1, np.log10(20), 50)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.hist(puntaje[val & normal], bins=bordes, density=True, color=COLORES[0], alpha=0.6, label="horas normales")
    ax.hist(puntaje[val & (y == 1)], bins=bordes, density=True, color=COLORES[7], alpha=0.6, label="horas con fuga")
    ax.axvline(1, color=GRIS, ls="--", lw=1)
    ax.axvline(4, color="#0b0b0b", ls="-", lw=1)
    ax.annotate("percentil 95\n(umbral del RF-002.2)", (1, ax.get_ylim()[1] * 0.6), xytext=(6, 0),
                textcoords="offset points", fontsize=8, color=GRIS)
    ax.annotate("4 veces el percentil 95\n(elegido por bloques)", (4, ax.get_ylim()[1] * 0.85), xytext=(6, 0),
                textcoords="offset points", fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.1, 20)
    ax.set_xlabel("error de reconstrucción, en veces el percentil 95 de la operación normal")
    ax.set_ylabel("densidad")
    ax.set_title("Autoencoder de fugas: error en la validación (ene–abr 2026)", fontsize=10)
    ax.legend(fontsize=8, loc="center right")
    guardar(fig, "umbral_autoencoder")


if __name__ == "__main__":
    main()
