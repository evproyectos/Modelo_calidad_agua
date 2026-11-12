"""
Validación cruzada temporal y búsqueda de hiperparámetros (RF-002.3 y RF-002.4) de los modelos que
dan alarmas: el detector de eventos (notebook 07), la anticipación a 24 h (09) y las fugas (10).

Pliegues: Time Series Split (k = 5) sobre las horas de entrenamiento y validación (sep 2024 – abr
2026). Cada pliegue entrena SOLO con lo anterior a él (ventana creciente) y deja `separacion_h`
horas entre el fin del entrenamiento y el inicio del pliegue, para que las ventanas de las
variables y la etiqueta de la anticipación (que mira 24 h adelante) no crucen. Todo lo que se
ajusta con datos (los valores típicos de cada sensor, la regresión de la operación normal de las
fugas) se calcula en cada pliegue con su propio entrenamiento. El periodo de prueba (may–ago 2026)
no se toca.

Métrica: F1 por evento, el mismo de los notebooks. Precisión = alarmas (o avisos) que coinciden
con un evento / alarmas; sensibilidad = eventos detectados (o anticipados) / eventos.

Punto de operación (umbral y confirmación): el F1 depende de él. Para el modelo final se usa el que
tiene guardado (el que usa el servicio). Para comparar configuraciones (sensibilidad y búsqueda),
cada una elige el suyo con la regla de los notebooks sobre los 5 pliegues: el que más eventos
detecta sin pasar del tope de falsas alarmas por semana en ningún pliegue. Así cada configuración
compite con su mejor umbral y con las mismas reglas.

Uso: herramientas/validacion_cruzada.py arma el reporte (docs/modelos/validacion-cruzada.md) y lo
registra en MLflow.
"""

import itertools
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from . import anticipacion as A
from . import carga, calidad, clima, configuracion, detector
from . import fugas as F

CFG = configuracion.seccion("validacion")
RAIZ = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Pliegue:
    numero: int
    entrena_desde: pd.Timestamp
    entrena_hasta: pd.Timestamp          # sin incluir
    desde: pd.Timestamp                  # pliegue de validación [desde, hasta)
    hasta: pd.Timestamp


def pliegues(horas, k=None, separacion_h=None):
    """Time Series Split sobre `horas` (índice horario ordenado)."""
    k = k or CFG.get("pliegues", 5)
    separacion_h = CFG.get("separacion_h", 24) if separacion_h is None else separacion_h
    horas = pd.DatetimeIndex(horas)
    una = pd.Timedelta(hours=1)
    return [Pliegue(i, horas[e[0]], horas[e[-1]] + una, horas[p[0]], horas[p[-1]] + una)
            for i, (e, p) in enumerate(TimeSeriesSplit(n_splits=k, gap=separacion_h).split(horas), 1)]


def f1(precision, sensibilidad):
    if not precision or not sensibilidad or np.isnan(precision) or np.isnan(sensibilidad):
        return 0.0
    return 2 * precision * sensibilidad / (precision + sensibilidad)


def _metricas(eventos, detectados, alarmas, falsas, falsas_semana):
    precision = (alarmas - falsas) / alarmas if alarmas else np.nan
    sensibilidad = detectados / eventos if eventos else np.nan
    return {"eventos": eventos, "detectados": detectados, "alarmas": alarmas, "falsas": falsas,
            "falsas_por_semana": falsas_semana, "precision": precision, "sensibilidad": sensibilidad,
            "f1": f1(precision, sensibilidad)}


def clave(params):
    return tuple(sorted((k, str(v)) for k, v in (params or {}).items()))


# ---------------------------------------------------------------------------- datos comunes
class Datos:
    """Lecturas preprocesadas y tablas horarias, cargadas una sola vez para los tres modelos."""

    def __init__(self):
        crudo = carga.cargar_todo()
        self.t15 = calidad.lecturas_operativas(crudo)
        self.h = detector.horario(self.t15)
        self.h["evento_red"] = crudo["evento_red"].resample("1h").max()
        self.catalogo = carga.catalogo_eventos()
        p = carga.periodos()
        self.desde, self.hasta = p["entrenamiento"][0], p["validacion"][1]
        horas = self.h.index
        self.horas_cv = horas[(horas >= self.desde) & (horas < self.hasta)]


# ---------------------------------------------------------------------------- modelos
class Validador:
    """Lo que cada modelo tiene que saber hacer para validarlo y buscar sus hiperparámetros."""

    nombre = experimento = archivo = ""
    seccion = ""                        # sección de hiperparametros.yaml con los del modelo final

    def __init__(self, datos):
        self.datos = datos
        self._preparado = {}
        self.final = joblib.load(RAIZ / "modelos" / self.archivo)

    def preparado(self, pl):
        if pl.numero not in self._preparado:
            self._preparado[pl.numero] = self.preparar(pl)
        return self._preparado[pl.numero]

    def base(self):
        """Hiperparámetros del modelo final (hiperparametros.yaml)."""
        return dict(configuracion.seccion(self.seccion[0])[self.seccion[1]])


class ValidadorDetector(Validador):
    nombre, experimento, archivo = "Detector de eventos", "detector_eventos", "detector_eventos.joblib"
    seccion = ("detector", "bosque")

    def __init__(self, datos):
        super().__init__(datos)
        self.tabla, _ = detector.tabla(datos.h)
        self.columnas = detector.columnas_entrada(self.tabla)
        self.fechas = self.tabla["fecha"].values

    def preparar(self, pl):
        h = self.datos.h.copy()
        dentro = (h.index >= pl.entrena_desde) & (h.index < pl.entrena_hasta)
        h["particion"] = np.where(dentro, "entrenamiento", "fuera")       # típicos solo con el pasado
        t, _ = detector.tabla(h)
        assert (t["fecha"].values == self.fechas).all() and (t["nodo"].values == self.tabla["nodo"].values).all()
        return {"tabla": t, "entrena": (t["particion"] == "entrenamiento").values}

    def ajustar(self, prep, params=None, filas=None):
        filas = prep["entrena"] if filas is None else filas
        return detector.entrenar(prep["tabla"], self.columnas, filas, modelo="bosque", params=params)

    def vacio(self):
        return np.full(len(self.tabla), np.nan)

    def en(self, desde, hasta):
        return (self.fechas >= np.datetime64(desde)) & (self.fechas < np.datetime64(hasta))

    def probabilidad(self, prep, mod, filas):
        p, _ = detector.probabilidades(mod, prep["tabla"][filas], self.columnas)
        return p

    def punto_final(self):
        return (float(self.final["umbral"]), int(self.final["confirmar_h"]))

    def evaluar(self, prob, punto, desde, hasta):
        m = self.en(desde, hasta)
        alarma = detector.alarmas_sensor(self.tabla[m], np.nan_to_num(prob[m]), *punto)
        _, _, r = detector.evaluar(self.datos.h, alarma, desde, hasta)
        return _metricas(r["episodios visibles"], r["detectados"], r["alarmas"], r["falsas alarmas"],
                         r["falsas por semana"])

    def puntos(self):
        el = configuracion.seccion("eleccion_alarma")
        return ([(u, c) for c in el["confirmaciones_h"] for u in el["umbrales"]], el["max_falsas_por_semana"])


class ValidadorAnticipacion(Validador):
    nombre, experimento, archivo = "Anticipación a 24 h", "anticipacion_eventos", "anticipacion_24h.joblib"
    seccion = ("anticipacion", "bosque")

    def __init__(self, datos):
        super().__init__(datos)
        self.h = A.horario(datos.t15)
        self.objetivo = A.objetivo(self.h.index, datos.catalogo)
        self.y = self.objetivo["clase"].values
        self.utiles = A.filas_utiles(self.h, self.objetivo)
        self.pronostico = clima.ventanas(clima.cargar_pronostico(), self.h.index)
        self.fechas = self.h.index.values
        self.arboles = self.final.get("nombre") == "árboles"
        if self.arboles:
            self.seccion = ("anticipacion", "arboles")

    def preparar(self, pl):
        dentro = (self.h.index >= pl.entrena_desde) & (self.h.index < pl.entrena_hasta)
        entrena = dentro & self.utiles
        t, _ = A.tabla(self.h, pronostico=self.pronostico, normal=entrena & (self.y == 0))
        return {"tabla": t, "entrena": entrena}

    def ajustar(self, prep, params=None, filas=None):
        filas = prep["entrena"] if filas is None else filas & self.utiles
        X = prep["tabla"].loc[filas, A.columnas_entrada(prep["tabla"])]
        entrenar = A.entrenar_arboles if self.arboles else A.entrenar_bosque
        return entrenar(X, self.y[filas], params)

    def vacio(self):
        return pd.DataFrame(np.nan, index=self.h.index, columns=A.CODIGOS)

    def en(self, desde, hasta):
        return (self.h.index >= desde) & (self.h.index < hasta)

    def probabilidad(self, prep, mod, filas):
        t = prep["tabla"]
        return A.probabilidades(mod, t.loc[filas, A.columnas_entrada(t)])[A.CODIGOS].values

    def punto_final(self):
        return {int(k): tuple(v) for k, v in self.final["eleccion"].items()}

    def evaluar(self, prob, punto, desde, hasta):
        aviso, tipo = A.avisos_por_tipo(prob.fillna(0), punto)
        _, av, r = A.evaluar(self.h.index, aviso, self.datos.catalogo, self.utiles, desde, hasta, tipo=tipo)
        return _metricas(r["eventos"], r["anticipados"], len(av), int(av["falso"].sum()), r["falsos_por_semana"])

    def elegir(self, prob, plgs):
        el = A.HIPERPARAMETROS["eleccion_aviso"]
        eleccion, _ = A.elegir_umbrales(self.h.index, prob.fillna(0), self.datos.catalogo, self.utiles,
                                        [(p.desde, p.hasta) for p in plgs], el["umbrales"], el["confirmaciones_h"],
                                        el["max_falsos_por_semana"])
        return eleccion


class ValidadorFugas(Validador):
    nombre, experimento, archivo = "Fugas", "fugas", "fugas.joblib"

    def __init__(self, datos):
        super().__init__(datos)
        h, cat = datos.h, datos.catalogo
        self.y = F.clase(h, cat).values
        cerca = np.zeros(len(h), bool)                   # como el notebook 10: lejos de toda fuga
        for _, ev in cat[cat["tipo"] == "fuga"].iterrows():
            cerca |= (h.index >= ev["inicio_precursor"]) & (h.index < ev["fin"] + pd.Timedelta(hours=12))
        self.normal = ~cerca & (self.y == 0)
        self.fechas = h.index.values
        self.modelo = self.final.get("nombre", "SVM")
        self.seccion = ("fugas", {"SVM": "svm", "Random Forest": "bosque"}.get(self.modelo, "svm"))

    def preparar(self, pl):
        h = self.datos.h
        dentro = (h.index >= pl.entrena_desde) & (h.index < pl.entrena_hasta)
        return {"tabla": F.tabla(h, F.ajustar_normal(h, dentro & self.normal)), "entrena": dentro}

    def ajustar(self, prep, params=None, filas=None):
        filas = prep["entrena"] if filas is None else filas
        entrenar = {"SVM": F.entrenar_svm, "Random Forest": F.entrenar_bosque}.get(self.modelo)
        if entrenar is None:
            raise ValueError(f"la búsqueda no está hecha para el modelo de fugas {self.modelo!r}")
        return entrenar(prep["tabla"][filas], self.y[filas], params)

    def vacio(self):
        return np.full(len(self.datos.h), np.nan)

    def en(self, desde, hasta):
        return (self.datos.h.index >= desde) & (self.datos.h.index < hasta)

    def probabilidad(self, prep, mod, filas):
        return F.probabilidad(mod, prep["tabla"][filas])

    def punto_final(self):
        return (float(self.final["umbral"]), int(self.final["confirmar_h"]))

    def evaluar(self, prob, punto, desde, hasta):
        a = F.alarma(np.nan_to_num(prob), *punto)
        _, r = F.evaluar(self.datos.h.index, a, self.y, desde, hasta)
        return _metricas(r["fugas"], r["detectadas"], r["alarmas"], r["falsas"], r["falsas_por_semana"])

    def puntos(self):
        el = F.HIPERPARAMETROS["eleccion"]
        return ([(u, c) for c in el["confirmaciones_h"] for u in el["umbrales"]], el["max_falsas_por_semana"])


def _elegir_por_grilla(val, prob, plgs):
    """Regla de los notebooks: más eventos detectados con a lo sumo el tope de falsas por semana
    en CADA pliegue; a igualdad, menos falsas y mayor F1."""
    candidatos, tope = val.puntos()
    filas = []
    for punto in candidatos:
        rs = [val.evaluar(prob, punto, p.desde, p.hasta) for p in plgs]
        filas.append({"punto": punto, "detectados": sum(r["detectados"] for r in rs),
                      "falsas": sum(r["falsas"] for r in rs),
                      "peor": max(r["falsas_por_semana"] for r in rs),
                      "f1": np.mean([r["f1"] for r in rs])})
    g = pd.DataFrame(filas)
    ok = g[g["peor"] <= tope]
    ok = ok if len(ok) else g.nsmallest(1, "peor")
    return ok.sort_values(["detectados", "falsas", "f1"], ascending=[False, True, False]).iloc[0]["punto"]


for _cls in (ValidadorDetector, ValidadorFugas):
    _cls.elegir = _elegir_por_grilla

VALIDADORES = {"detector": ValidadorDetector, "anticipacion": ValidadorAnticipacion, "fugas": ValidadorFugas}


# ---------------------------------------------------------------------------- validación
@dataclass
class Resultado:
    params: dict
    punto: object
    pliegues: pd.DataFrame
    segundos: float
    prob: object = field(default=None, repr=False)

    @property
    def f1_media(self):
        return float(self.pliegues["f1"].mean())

    @property
    def f1_desv(self):
        return float(self.pliegues["f1"].std(ddof=0))


def predecir_pliegues(val, plgs, params=None):
    """Probabilidades fuera de muestra: cada pliegue con un modelo entrenado solo con el pasado."""
    prob = val.vacio()
    for pl in plgs:
        prep = val.preparado(pl)
        filas = val.en(pl.desde, pl.hasta)
        p = val.probabilidad(prep, val.ajustar(prep, params), filas)
        if isinstance(prob, pd.DataFrame):
            prob.loc[filas] = p
        else:
            prob[filas] = p
    return prob


def validar(val, plgs, params=None, punto=None, prob=None):
    """F1 y lo demás por pliegue. Sin `punto`, la configuración elige el suyo (regla del notebook)."""
    t0 = time.time()
    if prob is None:
        prob = predecir_pliegues(val, plgs, params)
    punto = punto if punto is not None else val.elegir(prob, plgs)
    filas = [{"pliegue": p.numero, "entrena_desde": p.entrena_desde.date(), "entrena_hasta": p.entrena_hasta.date(),
              "desde": p.desde.date(), "hasta": p.hasta.date(), **val.evaluar(prob, punto, p.desde, p.hasta)}
             for p in plgs]
    return Resultado(dict(params or {}), punto, pd.DataFrame(filas), time.time() - t0, prob)


def curva_aprendizaje(val, plgs, punto, fracciones=None, params=None):
    """
    Error (1 − F1 por evento) según cuánto entrenamiento se usa, en el último pliegue: se entrena con
    la parte más reciente del pasado (20 %, 40 %… 100 %) y se mide en el entrenamiento y en el pliegue.
    """
    fracciones = fracciones or CFG.get("fracciones", [0.2, 0.4, 0.6, 0.8, 1.0])
    pl = plgs[-1]
    prep = val.preparado(pl)
    largo = pl.entrena_hasta - pl.entrena_desde
    filas = []
    for f in fracciones:
        inicio = (pl.entrena_hasta - largo * f).ceil("h")
        entrena = prep["entrena"] & val.en(inicio, pl.entrena_hasta)
        mod = val.ajustar(prep, params, entrena)
        prob = val.vacio()
        for desde, hasta in [(inicio, pl.entrena_hasta), (pl.desde, pl.hasta)]:
            m = val.en(desde, hasta)
            if isinstance(prob, pd.DataFrame):
                prob.loc[m] = val.probabilidad(prep, mod, m)
            else:
                prob[m] = val.probabilidad(prep, mod, m)
        r_e = val.evaluar(prob, punto, inicio, pl.entrena_hasta)
        r_v = val.evaluar(prob, punto, pl.desde, pl.hasta)
        filas.append({"fraccion": f, "meses": round(largo.days * f / 30.4, 1), "desde": inicio.date(),
                      "eventos_entrenamiento": r_e["eventos"], "error_entrenamiento": 1 - r_e["f1"],
                      "eventos_validacion": r_v["eventos"], "error_validacion": 1 - r_v["f1"]})
    return pd.DataFrame(filas)


def sensibilidad(val, plgs, variaciones, cache=None):
    """Un hiperparámetro a la vez alrededor del modelo final: F1 medio y desviación entre pliegues."""
    cache = {} if cache is None else cache
    base = val.base()
    filas = []
    for nombre, valores in variaciones.items():
        for v in valores:
            params = {nombre: v}
            k = clave({**base, **params})
            if k not in cache:
                cache[k] = validar(val, plgs, params)
            r = cache[k]
            filas.append({"hiperparametro": nombre, "valor": str(v), "es_el_final": str(base.get(nombre)) == str(v),
                          "f1_media": r.f1_media, "f1_desv": r.f1_desv, "punto": str(r.punto),
                          "detectados": int(r.pliegues["detectados"].sum()),
                          "eventos": int(r.pliegues["eventos"].sum()),
                          "falsas_por_semana": float(r.pliegues["falsas_por_semana"].mean()),
                          "segundos": r.segundos})
    return pd.DataFrame(filas)


def configuraciones_al_azar(espacio, n, semilla=0, excluir=None):
    """Random Search: `n` combinaciones distintas de la grilla `espacio` (sin la de `excluir`)."""
    nombres = sorted(espacio)
    todas = [dict(zip(nombres, c)) for c in itertools.product(*(espacio[k] for k in nombres))]
    if excluir is not None:
        todas = [c for c in todas if clave(c) != clave({k: excluir.get(k) for k in nombres})]
    rng = np.random.default_rng(semilla)
    idx = rng.choice(len(todas), size=min(n, len(todas)), replace=False)
    return [todas[i] for i in sorted(idx)]


def busqueda(val, plgs, espacio, n, semilla=0, cache=None, al_terminar=None):
    """Prueba `n` configuraciones al azar y la del modelo final; devuelve la tabla ordenada por F1."""
    cache = {} if cache is None else cache
    base = val.base()
    candidatos = [{}] + configuraciones_al_azar(espacio, n, semilla, excluir=base)
    filas = []
    for i, params in enumerate(candidatos):
        k = clave({**base, **params})
        if k not in cache:
            cache[k] = validar(val, plgs, params)
        r = cache[k]
        fila = {"configuracion": "modelo final" if not params else f"#{i}",
                **{k_: str(v) for k_, v in params.items()},
                "f1_media": r.f1_media, "f1_desv": r.f1_desv, "punto": str(r.punto),
                "detectados": int(r.pliegues["detectados"].sum()), "eventos": int(r.pliegues["eventos"].sum()),
                "falsas_por_semana": float(r.pliegues["falsas_por_semana"].mean()), "segundos": r.segundos}
        filas.append(fila)
        if al_terminar:
            al_terminar(params, r)
    return pd.DataFrame(filas).sort_values("f1_media", ascending=False).reset_index(drop=True)
