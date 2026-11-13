"""
Validación cruzada temporal y búsqueda de hiperparámetros (RF-002.3 y RF-002.4) del detector de
eventos, la anticipación a 24 h y las fugas. Un solo comando, desde la raíz del proyecto:

    python herramientas/validacion_cruzada.py                      # los tres modelos (~1,5 h)
    python herramientas/validacion_cruzada.py fugas anticipacion   # solo esos
    python herramientas/validacion_cruzada.py --reanudar           # sigue donde quedó

Para cada modelo:
  1. F1 por evento en los 5 pliegues de Time Series Split, con el modelo final tal como está en
     modelos/ (sus hiperparámetros y su umbral); media y desviación estándar.
  2. Curva de aprendizaje: error (1 − F1) según cuánto entrenamiento se usa.
  3. Sensibilidad: un hiperparámetro a la vez alrededor del modelo final.
  4. Búsqueda al azar (Random Search) en la grilla de hiperparametros.yaml (sección validacion),
     comparada con el modelo final con las mismas reglas.

Deja el reporte en docs/modelos/validacion-cruzada.md (y los números en validacion-cruzada.json),
las figuras en figuras/validacion_*.png, y una corrida por modelo en MLflow, en el experimento de
ese modelo, con una corrida hija por configuración de la búsqueda (hiperparámetros, F1 y duración).
Los resultados de cada modelo se guardan en registros/validacion/ al terminarlo (--reanudar los usa).
Pliegues, grillas y semilla: hiperparametros.yaml, sección validacion. Detalle: calidad_agua/validacion.py.
"""

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ))

import joblib  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from calidad_agua import experimentos  # noqa: E402
from calidad_agua import validacion as V  # noqa: E402

DOC = RAIZ / "docs" / "modelos" / "validacion-cruzada.md"
LECTURA = RAIZ / "docs" / "modelos" / "validacion-cruzada-lectura.md"
GUARDADO = RAIZ / "registros" / "validacion"
MEJORA_MINIMA_PP = 2.0                     # RF-002.4: mejorar el F1 en al menos 2 puntos
DESVIACION_MAXIMA = 0.10                   # RF-002.3: desviación del F1 entre pliegues


def correr_modelo(clave_modelo, datos, plgs, n_busqueda, semilla, con_mlflow):
    t0 = time.time()
    val = V.VALIDADORES[clave_modelo](datos)
    cfg = V.CFG[getattr(val, "grillas", clave_modelo)]
    print(f"\n== {val.nombre} ==", flush=True)
    reg = experimentos.iniciar(val.experimento, "RF-002.3/4 · validación cruzada y búsqueda",
                               "herramientas/validacion_cruzada.py") if con_mlflow else None
    try:
        final = V.validar(val, plgs, punto=val.punto_final())
        print(f"modelo final: F1 {final.f1_media:.3f} ± {final.f1_desv:.3f}", flush=True)
        cache = {V.clave(val.base()): V.validar(val, plgs, prob=final.prob)}
        curva = V.curva_aprendizaje(val, plgs, val.punto_final())
        print("curva de aprendizaje lista", flush=True)
        sens = V.sensibilidad(val, plgs, cfg["sensibilidad"], cache)
        print("sensibilidad lista", flush=True)

        def registrar(params, r):
            print(f"  {params or 'modelo final'}: F1 {r.f1_media:.3f} ± {r.f1_desv:.3f} ({r.segundos:.0f} s)",
                  flush=True)
            if reg is None:
                return
            nombre = "modelo final" if not params else ", ".join(f"{k}={v}" for k, v in params.items())
            with reg.hija(f"búsqueda · {nombre}") as hija:
                hija.parametros({**val.base(), **params, "punto": str(r.punto)})
                hija.metricas({"f1_media": r.f1_media, "f1_desv": r.f1_desv, "segundos": r.segundos,
                               "detectados": r.pliegues["detectados"].sum(), "eventos": r.pliegues["eventos"].sum(),
                               "falsas_por_semana": r.pliegues["falsas_por_semana"].mean()})
                hija.metricas({f"f1_pliegue_{int(p)}": f for p, f in zip(r.pliegues["pliegue"], r.pliegues["f1"])})

        bus = V.busqueda(val, plgs, cfg["espacio"], n_busqueda, semilla, cache, registrar)
        base_regla = cache[V.clave(val.base())]
        mejor = bus.iloc[0]
        mejora_pp = 100 * (mejor["f1_media"] - base_regla.f1_media)
        params_mejor = {k: mejor[k] for k in cfg["espacio"] if k in mejor and pd.notna(mejor[k])}
        res = {"nombre": val.nombre, "experimento": val.experimento, "modelo": type(val).__name__,
               "base": {k: str(v) for k, v in val.base().items()},
               "punto_final": str(val.punto_final()),
               "final": final.pliegues, "f1_media": final.f1_media, "f1_desv": final.f1_desv,
               "base_regla": {"f1_media": base_regla.f1_media, "f1_desv": base_regla.f1_desv,
                              "punto": str(base_regla.punto)},
               "curva": curva, "sensibilidad": sens, "busqueda": bus,
               "grillas": getattr(val, "grillas", clave_modelo),
               "mejor": {"configuracion": mejor["configuracion"], "params": params_mejor,
                         "f1_media": float(mejor["f1_media"]), "f1_desv": float(mejor["f1_desv"]),
                         "punto": mejor["punto"]},
               "mejora_pp": mejora_pp, "minutos": (time.time() - t0) / 60}
        if reg is not None:
            reg.parametros({"pliegues": len(plgs), "separacion_h": V.CFG.get("separacion_h"),
                            "busqueda_configuraciones": n_busqueda, "semilla": semilla,
                            "punto_final": str(val.punto_final()),
                            "mejor_configuracion": json.dumps(params_mejor, default=str)}, "validacion")
            reg.metricas({"f1_media": final.f1_media, "f1_desv": final.f1_desv,
                          "f1_media_base_regla": base_regla.f1_media, "f1_media_mejor": mejor["f1_media"],
                          "mejora_pp": mejora_pp, "minutos": res["minutos"]}, "validacion")
            reg.tabla(final.pliegues, "validacion_pliegues")
            reg.tabla(curva, "curva_aprendizaje")
            reg.tabla(sens, "sensibilidad")
            reg.tabla(bus, "busqueda")
            reg.terminar()
        return res
    except BaseException:
        if reg is not None:
            reg.terminar("FAILED")
        raise


# ---------------------------------------------------------------------------- figuras
def figuras(res):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from calidad_agua.graficos import COLORES, GRIS, estilo, guardar
    estilo()
    claves = list(res)

    fig, axes = plt.subplots(1, len(claves), figsize=(4.2 * len(claves), 3.2), squeeze=False)
    for ax, k in zip(axes[0], claves):
        p = res[k]["final"]
        ax.bar(p["pliegue"].astype(str), p["f1"], color=COLORES[0])
        ax.axhline(res[k]["f1_media"], color=COLORES[1], lw=1.2, label=f"media {res[k]['f1_media']:.2f}")
        ax.set_ylim(0, 1)
        ax.set_xlabel("pliegue")
        ax.set_ylabel("F1 por evento")
        ax.set_title(f"{res[k]['nombre']} (desv. {res[k]['f1_desv']:.2f})", fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    guardar(fig, "validacion_pliegues")
    plt.close(fig)

    fig, axes = plt.subplots(1, len(claves), figsize=(4.2 * len(claves), 3.2), squeeze=False)
    for ax, k in zip(axes[0], claves):
        c = res[k]["curva"]
        ax.plot(c["meses"], c["error_entrenamiento"], "o-", color=COLORES[0], label="entrenamiento")
        ax.plot(c["meses"], c["error_validacion"], "o-", color=COLORES[1], label="validación (pliegue 5)")
        ax.set_ylim(0, 1)
        ax.set_xlabel("meses de entrenamiento")
        ax.set_ylabel("error (1 − F1)")
        ax.set_title(res[k]["nombre"], fontsize=10)
        ax.legend(fontsize=7)
    fig.tight_layout()
    guardar(fig, "validacion_curvas_aprendizaje")
    plt.close(fig)

    filas = sum(len(res[k]["sensibilidad"]["hiperparametro"].unique()) for k in claves)
    fig, axes = plt.subplots(1, filas, figsize=(3.0 * filas, 3.0), squeeze=False)
    i = 0
    for k in claves:
        s = res[k]["sensibilidad"]
        for hp, g in s.groupby("hiperparametro", sort=False):
            ax = axes[0][i]
            i += 1
            x = np.arange(len(g))
            ax.errorbar(x, g["f1_media"], yerr=g["f1_desv"], fmt="o-", color=COLORES[0], capsize=3)
            fin = g["es_el_final"].values
            ax.scatter(x[fin], g["f1_media"].values[fin], s=90, facecolors="none", edgecolors=COLORES[7], zorder=5)
            ax.set_xticks(x, g["valor"], fontsize=7)
            ax.set_ylim(0, 1)
            ax.set_title(f"{res[k]['nombre']}\n{hp}", fontsize=8)
            ax.axhline(res[k]["base_regla"]["f1_media"], color=GRIS, ls="--", lw=0.8)
    axes[0][0].set_ylabel("F1 medio (± desv.)")
    fig.tight_layout()
    guardar(fig, "validacion_sensibilidad")
    plt.close(fig)


# ---------------------------------------------------------------------------- reporte
def _tabla_md(df, columnas, formatos=None):
    formatos = formatos or {}
    filas = ["| " + " | ".join(columnas.values()) + " |", "|" + "---|" * len(columnas)]
    for _, r in df.iterrows():
        celdas = []
        for c in columnas:
            v = r[c]
            f = formatos.get(c)
            celdas.append(f.format(v) if f and pd.notna(v) else ("—" if pd.isna(v) else str(v)))
        filas.append("| " + " | ".join(celdas) + " |")
    return "\n".join(filas)


def reporte(res, plgs, segundos):
    git = experimentos.estado_git()
    huella, _ = experimentos.huella_datos()
    pct = "{:.0%}"
    L = ["# Validación cruzada y búsqueda de hiperparámetros (RF-002.3 y RF-002.4)", "",
         "> Generado por `python herramientas/validacion_cruzada.py` el "
         f"{datetime.now():%d/%m/%Y %H:%M} ({segundos / 60:.0f} min de cálculo) · "
         f"commit `{git.get('git_commit', '?')[:8]}`"
         f"{' (con cambios sin commit)' if git.get('git_cambios_sin_guardar') == 'sí' else ''} · "
         f"datos sha256 `{huella[:12]}`. "
         "No editar a mano: se vuelve a generar.", "",
         "## Cómo se valida", "",
         "**Time Series Split con k = 5** sobre las horas de entrenamiento y validación (sep 2024 – abr 2026). "
         "Cada pliegue entrena solo con el pasado (ventana creciente) y deja "
         f"{V.CFG.get('separacion_h', 24)} h de separación antes del pliegue, para que las ventanas de las variables y "
         "la etiqueta de la anticipación (que mira 24 h adelante) no crucen. Lo que se ajusta con datos "
         "(valores típicos "
         "de cada sensor, regresión de la operación normal de las fugas) se recalcula en cada pliegue con su propio "
         "entrenamiento. El periodo de prueba (may–ago 2026) no se usa.", "",
         _tabla_md(pd.DataFrame([{"n": p.numero, "e": f"{p.entrena_desde:%d/%m/%Y} – {p.entrena_hasta:%d/%m/%Y}",
                                  "v": f"{p.desde:%d/%m/%Y} – {p.hasta:%d/%m/%Y}",
                                  "m": (p.entrena_hasta - p.entrena_desde).days / 30.4} for p in plgs]),
                   {"n": "Pliegue", "e": "Entrena", "v": "Valida", "m": "Meses de entrenamiento"}, {"m": "{:.1f}"}), "",
         "**Métrica: F1 por evento**, la de los notebooks. Precisión = alarmas (o avisos) que coinciden con "
         "un evento / "
         "alarmas; sensibilidad = eventos detectados (o anticipados) / eventos. El F1 depende del umbral y la "
         "confirmación: el del **modelo final** es el que usa el servicio; en la sensibilidad y la búsqueda cada "
         "configuración elige el suyo con la regla de los notebooks (el que más eventos detecta sin pasar del tope de "
         "falsas alarmas por semana en ningún pliegue), para comparar todas con las mismas reglas.", "",
         "## Resumen", ""]
    filas = []
    for k, r in res.items():
        filas.append({"modelo": r["nombre"], "f1": f"{r['f1_media']:.2f} ± {r['f1_desv']:.2f}",
                      "estable": "sí" if r["f1_desv"] <= DESVIACION_MAXIMA else "no",
                      "base": f"{r['base_regla']['f1_media']:.2f}", "mejor": f"{r['mejor']['f1_media']:.2f}",
                      "mejora": f"{r['mejora_pp']:+.1f}",
                      "decision": "se adopta la mejor configuración" if r["mejora_pp"] >= MEJORA_MINIMA_PP
                      else "el modelo final ya era óptimo"})
    columnas = {"modelo": "Modelo", "f1": "F1 por pliegue (media ± desv.)",
                "estable": f"Desv. ≤ {DESVIACION_MAXIMA:.2f}", "base": "F1 del modelo final (regla)",
                "mejor": "F1 de la mejor búsqueda", "mejora": "Mejora (puntos)", "decision": "Decisión (RF-002.4)"}
    L += [_tabla_md(pd.DataFrame(filas), columnas), "",
          f"Criterios: RF-002.3 pide una desviación del F1 entre pliegues ≤ {DESVIACION_MAXIMA:.2f}; RF-002.4, que la "
          f"búsqueda mejore el F1 en al menos {MEJORA_MINIMA_PP:.0f} puntos o que se documente que el modelo ya era "
          "óptimo. Los hiperparámetros del modelo final están en `calidad_agua/hiperparametros.yaml`.", "",
          "![F1 por pliegue](../../figuras/validacion_pliegues.png)", ""]
    for k, r in res.items():
        L += [f"## {r['nombre']}", "",
              f"Hiperparámetros del modelo final: `{json.dumps(r['base'], ensure_ascii=False)}`; umbral y "
              f"confirmación: `{r['punto_final']}`. Experimento de MLflow: `{r['experimento']}`.", "",
              "### F1 por pliegue (modelo final)", "",
              _tabla_md(r["final"], {"pliegue": "Pliegue", "eventos": "Eventos", "detectados": "Detectados",
                                     "alarmas": "Alarmas", "falsas": "Falsas", "falsas_por_semana": "Falsas/semana",
                                     "precision": "Precisión", "sensibilidad": "Sensibilidad", "f1": "F1"},
                        {"falsas_por_semana": "{:.2f}", "precision": pct, "sensibilidad": pct, "f1": "{:.2f}"}), "",
              f"Media {r['f1_media']:.2f}, desviación estándar {r['f1_desv']:.2f}.", "",
              "### Curva de aprendizaje", "",
              "Último pliegue; se entrena con la parte más reciente del pasado.", "",
              _tabla_md(r["curva"], {"meses": "Meses", "desde": "Desde", "eventos_entrenamiento": "Eventos (entr.)",
                                     "error_entrenamiento": "Error entrenamiento",
                                     "eventos_validacion": "Eventos (valid.)", "error_validacion": "Error validación"},
                        {"meses": "{:.1f}", "error_entrenamiento": "{:.2f}", "error_validacion": "{:.2f}"}), "",
              "### Sensibilidad a los hiperparámetros", "",
              "Un hiperparámetro a la vez; los demás, como en el modelo final.", "",
              _tabla_md(r["sensibilidad"].assign(
                  valor=lambda d: d["valor"] + np.where(d["es_el_final"], " (final)", "")),
                        {"hiperparametro": "Hiperparámetro", "valor": "Valor", "f1_media": "F1 medio",
                         "f1_desv": "Desv.", "detectados": "Detectados", "eventos": "De",
                         "falsas_por_semana": "Falsas/semana", "punto": "Umbral elegido"},
                        {"f1_media": "{:.2f}", "f1_desv": "{:.2f}", "falsas_por_semana": "{:.2f}"}), "",
              f"### Búsqueda al azar ({len(r['busqueda']) - 1} configuraciones y el modelo final)", ""]
        cols = {"configuracion": "Configuración"}
        cols.update({c: c for c in V.CFG[r.get("grillas", k)]["espacio"] if c in r["busqueda"]})
        cols.update({"f1_media": "F1 medio", "f1_desv": "Desv.", "detectados": "Detectados", "eventos": "De",
                     "falsas_por_semana": "Falsas/semana", "segundos": "Segundos"})
        L += [_tabla_md(r["busqueda"], cols, {"f1_media": "{:.2f}", "f1_desv": "{:.2f}", "falsas_por_semana": "{:.2f}",
                                              "segundos": "{:.0f}"}), "",
              f"Mejor: {r['mejor']['configuracion']} (F1 {r['mejor']['f1_media']:.2f}); "
              "modelo final con la misma regla: "
              f"{r['base_regla']['f1_media']:.2f} → diferencia {r['mejora_pp']:+.1f} puntos.", ""]
    L += ["## Curvas de aprendizaje y sensibilidad", "",
          "![Curvas de aprendizaje](../../figuras/validacion_curvas_aprendizaje.png)", "",
          "![Sensibilidad](../../figuras/validacion_sensibilidad.png)", ""]
    if LECTURA.exists():
        L += [LECTURA.read_text(encoding="utf-8").strip(), ""]
    L += ["## Cómo repetirlo", "",
          "Con el entorno de este proyecto: `python herramientas/validacion_cruzada.py`. Con Docker, "
          "desde esta carpeta: "
          "`docker compose run --rm validacion` (ver README, **Docker**). Pliegues, grillas y semilla: "
          "`hiperparametros.yaml`, sección `validacion`.", ""]
    DOC.parent.mkdir(parents=True, exist_ok=True)
    DOC.write_text("\n".join(L), encoding="utf-8")
    numeros = {k: {**{c: v for c, v in r.items() if not isinstance(v, pd.DataFrame)},
                   **{c: v.to_dict(orient="records") for c, v in r.items() if isinstance(v, pd.DataFrame)}}
               for k, r in res.items()}
    DOC.with_suffix(".json").write_text(json.dumps(numeros, ensure_ascii=False, indent=1, default=str),
                                        encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("modelos", nargs="*", help=f"de {', '.join(V.VALIDADORES)} (sin nada: los tres)")
    ap.add_argument("--busqueda", type=int, default=V.CFG.get("busqueda", 10), help="configuraciones al azar")
    ap.add_argument("--reanudar", action="store_true", help="usar los modelos ya terminados en registros/validacion")
    ap.add_argument("--sin-mlflow", action="store_true")
    a = ap.parse_args()
    modelos = a.modelos or list(V.VALIDADORES)
    if set(modelos) - set(V.VALIDADORES):
        ap.error(f"modelos posibles: {', '.join(V.VALIDADORES)}")
    t0 = time.time()
    datos = V.Datos()
    plgs = V.pliegues(datos.horas_cv)
    GUARDADO.mkdir(parents=True, exist_ok=True)
    hechos = {}
    for k in sorted(V.VALIDADORES, key=["fugas", "anticipacion", "detector"].index):   # los rápidos primero
        archivo = GUARDADO / f"{k}.joblib"
        if k in modelos and not (a.reanudar and archivo.exists()):
            try:
                hechos[k] = correr_modelo(k, datos, plgs, a.busqueda, V.CFG.get("semilla", 0), not a.sin_mlflow)
                joblib.dump(hechos[k], archivo)
                continue
            except V.SinBusqueda as e:
                print(f"\n== {k}: se omite ({e}); queda la corrida anterior, si la hay", flush=True)
        if archivo.exists():                      # de una corrida anterior, para el reporte completo
            hechos[k] = joblib.load(archivo)
    res = {k: hechos[k] for k in V.VALIDADORES if k in hechos}
    figuras(res)
    reporte(res, plgs, 60 * sum(r["minutos"] for r in res.values()))
    print(f"\nListo en {(time.time() - t0) / 60:.0f} min: {DOC.relative_to(RAIZ)}")
    subprocess.run(["git", "status", "--short", str(DOC.parent)], cwd=RAIZ)


if __name__ == "__main__":
    main()
