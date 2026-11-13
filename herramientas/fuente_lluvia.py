"""
Segunda fuente de lluvia (RNF-009): ¿los modelos funcionan con la lluvia de Open-Meteo en lugar de
la de la estación del IMN?

En operación los sensores no miden lluvia: la del dataset es la de la estación 69-701 del IMN (ITCR,
San Carlos), que el simulador mete en las lecturas. El servicio de monitoreo la consigue de una
fuente de lluvia observada elegida por configuración (adaptadores en Monitoreo_calidad_agua); la
que se puede consultar en vivo es Open-Meteo. Este script mide qué pasaría con los modelos si se
cambia la fuente, SIN modificar ningún módulo de modelado: solo cambia la columna lluvia_mm_h de los
datos al cargarlos y reutiliza los validadores de calidad_agua/validacion.py. Trabaja siempre con la
lluvia del dataset como referencia (FUENTE_LLUVIA=lecturas), aunque los modelos se entrenen con otra
(hiperparametros.yaml, sección datos).

Archivo: clima/lluvia_open_meteo_historica.csv, de la Historical Forecast API de Open-Meteo (las
primeras horas de cada pronóstico, lo mismo que devuelve en vivo la Forecast API con `past_days`):
  https://historical-forecast-api.open-meteo.com/v1/forecast?latitude=10.3606&longitude=-84.5135
    &hourly=precipitation&models=best_match,ecmwf_ifs025,gfs_seamless,icon_seamless,gem_seamless
    &start_date=2022-01-01&end_date=2026-09-30&timezone=America%2FCosta_Rica&format=csv
(misma convención que lluvia_mm_h: el valor de las 14:00 es la lluvia de 13:00 a 14:00, y en las
lecturas va de las 14:00 a las 14:45).

Pasos:
  1. Concordancia de cada serie de Open-Meteo con la del IMN (horaria, diaria, 24 h y 30 días).
  2. Los tres modelos que dan alarmas en el periodo de prueba (may–ago 2026), entrenados con
     entrenamiento + validación, en cuatro casos:
       IMN                 como están hoy, con el punto de operación guardado (referencia)
       cambio en vivo      entrenados con el IMN, pero en la prueba la lluvia es la de Open-Meteo
                           (lo que pasa si se cambia la fuente sin reentrenar); también cuánto
                           cambian las probabilidades
       IMN, punto nuevo    con el IMN, pero eligiendo otra vez el punto de operación con la regla de
                           los notebooks en los 5 pliegues: la referencia justa del caso siguiente
       reentrenados        entrenados y probados con Open-Meteo, con el punto elegido igual
  Deja docs/modelos/fuente-lluvia.md (y .json) y figuras/fuente_lluvia.png.

Uso, desde la raíz del proyecto:  python herramientas/fuente_lluvia.py [--serie promedio]
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from calidad_agua import carga, graficos  # noqa: E402
from calidad_agua.fuentes_lluvia import ARCHIVO, CUATRO, leer_open_meteo  # noqa: E402,F401
from calidad_agua import validacion as V  # noqa: E402

DOC = RAIZ / "docs" / "modelos" / "fuente-lluvia.md"
LECTURA = RAIZ / "docs" / "modelos" / "fuente-lluvia-lectura.md"
NOMBRES = {"best_match": "Open-Meteo best_match", "promedio": "promedio de 4 modelos",
           "ecmwf_ifs025": "ECMWF IFS", "gfs_seamless": "GFS", "icon_seamless": "ICON", "gem_seamless": "GEM"}
LLUEVE_MM_H, DIA_LLUVIOSO_MM = 1.0, 10.0
MODELOS = {"detector": "Detector de eventos", "anticipacion": "Anticipación a 24 h", "fugas": "Fugas"}


# ---------------------------------------------------------------------------- series
def lluvia_imn(crudo):
    """Lluvia horaria del dataset (la del IMN), con la convención de Open-Meteo."""
    return crudo["lluvia_mm_h"].resample("1h", label="left", closed="left").mean()


def concordancia(imn, om):
    j = pd.concat([imn.rename("imn"), om.rename("om")], axis=1, sort=True).dropna()
    d = j.resample("1D").sum()
    s24, s30 = j.rolling(24).sum().dropna(), j.rolling(24 * 30).sum().dropna()
    llueve, dia = j >= LLUEVE_MM_H, d >= DIA_LLUVIOSO_MM

    def deteccion(x):
        acierta = (x.imn & x.om).sum()
        return acierta / max(x.imn.sum(), 1), 1 - acierta / max(x.om.sum(), 1)
    pod_h, far_h = deteccion(llueve)
    pod_d, far_d = deteccion(dia)
    return {"horas": int(len(j)), "total_imn_mm": float(j.imn.sum()), "total_om_mm": float(j.om.sum()),
            "razon_total": float(j.om.sum() / j.imn.sum()), "r_horaria": float(j.corr().iloc[0, 1]),
            "r_diaria": float(d.corr().iloc[0, 1]), "r_24h": float(s24.corr().iloc[0, 1]),
            "r_30d": float(s30.corr().iloc[0, 1]), "pod_hora": float(pod_h), "far_hora": float(far_h),
            "pod_dia": float(pod_d), "far_dia": float(far_d), "faltantes": int(om.reindex(imn.index).isna().sum())}


# ---------------------------------------------------------------------------- modelos
def datos_con_lluvia(serie=None, desde=None):
    """V.Datos con lluvia_mm_h cambiada por `serie` (horaria), desde `desde` o en todo el periodo."""
    original = carga.cargar_todo

    def cargar(*a, **k):
        c = original(*a, **k)
        if serie is not None:
            nueva = serie.reindex(c.index.floor("h")).to_numpy(dtype="float32")
            m = np.ones(len(c), bool) if desde is None else np.asarray(c.index >= desde)
            col = c["lluvia_mm_h"].to_numpy(dtype="float32").copy()
            col[m] = nueva[m]
            c["lluvia_mm_h"] = col
        return c
    carga.cargar_todo = cargar
    try:
        return V.Datos()
    finally:
        carga.cargar_todo = original


def pliegue_prueba():
    p = carga.periodos()
    sep = pd.Timedelta(hours=V.CFG.get("separacion_h", 24))
    return V.Pliegue(0, p["entrenamiento"][0], p["prueba"][0] - sep, p["prueba"][0], p["prueba"][1])


def evaluar(clave_modelo, datos, punto=None, elegir=False):
    val = V.VALIDADORES[clave_modelo](datos)
    if elegir:                                       # punto de operación con la regla de los notebooks
        plgs = V.pliegues(datos.horas_cv)
        punto = val.elegir(V.predecir_pliegues(val, plgs), plgs)
    punto = punto if punto is not None else val.punto_final()
    pl = pliegue_prueba()
    r = V.validar(val, [pl], punto=punto)
    prob = r.prob[val.en(pl.desde, pl.hasta)]
    prob = np.asarray(prob.values if isinstance(prob, pd.DataFrame) else prob, dtype=float)
    return {**r.pliegues.iloc[0][["eventos", "detectados", "alarmas", "falsas", "falsas_por_semana",
                                  "precision", "sensibilidad", "f1"]].to_dict(), "punto": _texto_punto(punto)}, prob


def _texto_punto(p):
    if isinstance(p, dict):
        return "; ".join(f"tipo {k}: {u:g} / {c} h" for k, (u, c) in sorted(p.items()))
    return f"{p[0]:g} / {p[1]} h"


# ---------------------------------------------------------------------------- reporte
def figura(imn, om, nombre):
    graficos.estilo()
    fig, (a, b) = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2.2, 1]})
    j = pd.concat([imn.rename("imn"), om.rename("om")], axis=1, sort=True).dropna()
    m = j.resample("MS").sum()
    x = np.arange(len(m))
    a.bar(x - 0.2, m.imn, 0.4, label="IMN (estación 69-701)", color=graficos.COLORES[0])
    a.bar(x + 0.2, m.om, 0.4, label=f"Open-Meteo ({nombre})", color=graficos.COLORES[1])
    a.set_xticks(x[::2], [f"{carga.MESES[d.month - 1]} {d:%y}" for d in m.index[::2]], rotation=45)
    a.set_ylabel("mm en el mes")
    a.set_title("Lluvia mensual")
    a.legend()
    s = j.rolling(24 * 30).sum().dropna()
    b.scatter(s.imn, s.om, s=2, alpha=0.3, color=graficos.COLORES[1])
    lim = float(max(s.max()))
    b.plot([0, lim], [0, lim], color=graficos.GRIS, lw=1)
    b.set_xlabel("IMN (mm en 30 días)")
    b.set_ylabel("Open-Meteo (mm en 30 días)")
    b.set_title("Lluvia de 30 días (la de la anticipación)")
    graficos.guardar(fig, "fuente_lluvia")
    plt.close(fig)


def _pct(x):
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{100 * x:.0f} %"


def escribir(conc, serie, res, segundos):
    L = ["# Segunda fuente de lluvia (RNF-009)", "",
         "Generado por `herramientas/fuente_lluvia.py`; no editar a mano (la lectura va en "
         "`fuente-lluvia-lectura.md`, que se agrega al final).", "",
         "## Concordancia con la estación del IMN", "",
         f"Periodo del dataset (sep 2024 – ago 2026). Hora lluviosa: ≥ {LLUEVE_MM_H:g} mm; día lluvioso: "
         f"≥ {DIA_LLUVIOSO_MM:g} mm. POD: de las horas (días) lluviosas en el IMN, cuántas también lo son en "
         "Open-Meteo; FAR: de las que Open-Meteo marca, cuántas no lo fueron en el IMN.", "",
         "| Serie | Total / IMN | r horaria | r diaria | r 24 h | r 30 días | POD hora | FAR hora | POD día "
         "| FAR día |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    for k, c in conc.items():
        L.append(f"| {NOMBRES.get(k, k)}{' **(elegida)**' if k == serie else ''} | {c['razon_total']:.2f} | "
                 f"{c['r_horaria']:.2f} | {c['r_diaria']:.2f} | {c['r_24h']:.2f} | {c['r_30d']:.2f} | "
                 f"{_pct(c['pod_hora'])} | {_pct(c['far_hora'])} | {_pct(c['pod_dia'])} | {_pct(c['far_dia'])} |")
    c = conc[serie]
    L += ["", f"Lluvia total en el periodo: IMN {c['total_imn_mm']:.0f} mm, {NOMBRES.get(serie, serie)} "
          f"{c['total_om_mm']:.0f} mm.", "", "![Lluvia mensual y de 30 días](../../figuras/fuente_lluvia.png)", "",
          "## Los modelos en el periodo de prueba (may – ago 2026)", "",
          f"Entrenados con entrenamiento + validación; lluvia de Open-Meteo: {NOMBRES.get(serie, serie)}. "
          "Punto de operación: umbral / horas de confirmación; el guardado en `modelos/`, salvo en "
          "los dos casos que lo eligen otra vez en los 5 pliegues con la regla de los notebooks "
          "(compárense entre sí).", ""]
    for m, filas in res.items():
        L += [f"### {MODELOS[m]}", "", "| Caso | Eventos | Detectados | Alarmas | Falsas | Falsas/semana | "
              "Precisión | Sensibilidad | F1 | Punto |", "|---|---|---|---|---|---|---|---|---|---|"]
        for caso, r in filas.items():
            L.append(f"| {caso} | {r['eventos']:.0f} | {r['detectados']:.0f} | {r['alarmas']:.0f} | "
                     f"{r['falsas']:.0f} | "
                     f"{r['falsas_por_semana']:.2f} | {_pct(r['precision'])} | {_pct(r['sensibilidad'])} | "
                     f"{_pct(r['f1'])} | {r['punto']} |")
        c = filas.get("cambio en vivo (entrenados con IMN)", {})
        if "cambio_prob_medio" in c:
            L += ["", f"Con la lluvia de Open-Meteo, la probabilidad del modelo entrenado con el IMN cambia en "
                  f"promedio {c['cambio_prob_medio']:.3f} (percentil 99: {c['cambio_prob_p99']:.3f})."]
        L.append("")
    L += [f"Duración: {segundos / 60:.1f} min.", ""]
    if LECTURA.exists():
        L += [LECTURA.read_text(encoding="utf-8").strip(), ""]
    DOC.write_text("\n".join(L), encoding="utf-8")
    DOC.with_suffix(".json").write_text(json.dumps({"serie": serie, "concordancia": conc, "modelos": res},
                                                   indent=1, ensure_ascii=False, default=float), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--serie", default=None, help="serie de Open-Meteo para los modelos (por defecto, "
                    "la de mejor correlación de 30 días con el IMN)")
    ap.add_argument("--modelos", nargs="*", default=list(MODELOS))
    args = ap.parse_args()
    os.environ["FUENTE_LLUVIA"] = "lecturas"         # la referencia es la lluvia del dataset (IMN)
    t0 = time.time()
    om = leer_open_meteo()
    imn = lluvia_imn(carga.cargar_todo())
    conc = {k: concordancia(imn, om[k]) for k in om.columns}
    serie = args.serie or max(conc, key=lambda k: conc[k]["r_30d"])
    for k, c in conc.items():
        print(f"{k:14s} total {c['razon_total']:.2f}  r_h {c['r_horaria']:.2f}  r_d {c['r_diaria']:.2f}  "
              f"r_24h {c['r_24h']:.2f}  r_30d {c['r_30d']:.2f}", flush=True)
    print(f"serie elegida: {serie}", flush=True)
    figura(imn, om[serie], NOMBRES.get(serie, serie))

    inicio = carga.periodos()["prueba"][0]
    imn_datos = datos_con_lluvia()
    casos = {"IMN (como hoy)": dict(datos=imn_datos),
             "cambio en vivo (entrenados con IMN)": dict(datos=datos_con_lluvia(om[serie], desde=inicio)),
             "IMN, punto elegido otra vez": dict(datos=imn_datos, elegir=True),
             "reentrenados con Open-Meteo": dict(datos=datos_con_lluvia(om[serie]), elegir=True)}
    res = {}
    for m in args.modelos:
        res[m], probs = {}, {}
        for caso, kw in casos.items():
            res[m][caso], probs[caso] = evaluar(m, **kw)
            r = res[m][caso]
            print(f"{m:12s} {caso:38s} {r['detectados']:.0f}/{r['eventos']:.0f}  falsas {r['falsas']:.0f}  "
                  f"F1 {r['f1']:.3f}  ({r['punto']})", flush=True)
        dif = np.abs(probs["cambio en vivo (entrenados con IMN)"] - probs["IMN (como hoy)"])
        res[m]["cambio en vivo (entrenados con IMN)"]["cambio_prob_medio"] = float(np.nanmean(dif))
        res[m]["cambio en vivo (entrenados con IMN)"]["cambio_prob_p99"] = float(np.nanpercentile(dif, 99))
    escribir(conc, serie, res, time.time() - t0)
    print(f"Listo: {DOC.relative_to(RAIZ)}")


if __name__ == "__main__":
    main()
