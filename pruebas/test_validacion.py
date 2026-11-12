"""validacion.py: pliegues de Time Series Split, F1 por evento y la búsqueda (con un modelo de juguete)."""

import numpy as np
import pandas as pd
import pytest

from calidad_agua import fugas as F
from calidad_agua import validacion as V


def test_pliegues_entrenan_solo_con_el_pasado():
    horas = pd.date_range("2025-01-01", periods=24 * 100, freq="h")
    plgs = V.pliegues(horas, k=5, separacion_h=24)
    assert [p.numero for p in plgs] == [1, 2, 3, 4, 5]
    for p in plgs:
        assert p.entrena_desde == horas[0]                       # ventana creciente
        assert p.desde - p.entrena_hasta == pd.Timedelta(hours=24)   # separación
        assert p.entrena_hasta <= p.desde < p.hasta
    for a, b in zip(plgs, plgs[1:]):
        assert b.entrena_hasta > a.entrena_hasta and b.desde >= a.hasta    # crecen y no se pisan
    assert plgs[-1].hasta == horas[-1] + pd.Timedelta(hours=1)


def test_f1_y_metricas():
    assert V.f1(0.5, 1.0) == pytest.approx(2 / 3)
    assert V.f1(np.nan, 0.5) == 0.0 and V.f1(0.0, 0.5) == 0.0
    m = V._metricas(eventos=10, detectados=8, alarmas=10, falsas=2, falsas_semana=0.5)
    assert m["precision"] == 0.8 and m["sensibilidad"] == 0.8 and m["f1"] == pytest.approx(0.8)
    assert V._metricas(5, 0, 0, 0, 0)["f1"] == 0.0               # sin alarmas: F1 0, no un error


def test_configuraciones_al_azar_distintas_reproducibles_y_sin_la_base():
    espacio = {"a": [1, 2, 3], "b": ["x", "y"]}
    c1 = V.configuraciones_al_azar(espacio, 4, semilla=0, excluir={"a": 1, "b": "x", "otro": 9})
    c2 = V.configuraciones_al_azar(espacio, 4, semilla=0, excluir={"a": 1, "b": "x"})
    assert c1 == c2 and len(c1) == 4
    assert {"a": 1, "b": "x"} not in c1
    assert len({V.clave(c) for c in c1}) == 4
    assert len(V.configuraciones_al_azar(espacio, 50)) == 6       # no más que la grilla


class Juguete(V.Validador):
    """Serie horaria con eventos conocidos; el 'modelo' ve el evento con ruido según `ruido`."""

    nombre, experimento = "juguete", "juguete"

    def __init__(self, n=24 * 120, semilla=1):
        rng = np.random.default_rng(semilla)
        self.index = pd.date_range("2025-01-01", periods=n, freq="h")
        self.y = np.zeros(n, int)
        for ini in rng.choice(n - 10, size=30, replace=False):
            self.y[ini:ini + 6] = 1
        self.ruido = rng.random(n)
        self._preparado = {}
        self.entrenados = []

    def base(self):
        return {"ruido": 0.3, "otro": 1}

    def preparar(self, pl):
        return {"entrena": self.en(pl.entrena_desde, pl.entrena_hasta)}

    def ajustar(self, prep, params=None, filas=None):
        filas = prep["entrena"] if filas is None else filas
        self.entrenados.append(int(filas.sum()))
        return {**self.base(), **(params or {})}["ruido"]

    def vacio(self):
        return np.full(len(self.index), np.nan)

    def en(self, desde, hasta):
        return (self.index >= desde) & (self.index < hasta)

    def probabilidad(self, prep, ruido, filas):
        return (1 - ruido) * self.y[filas] + ruido * self.ruido[filas]

    def punto_final(self):
        return (0.5, 1)

    def evaluar(self, prob, punto, desde, hasta):
        _, r = F.evaluar(self.index, F.alarma(np.nan_to_num(prob), *punto), self.y, desde, hasta)
        return V._metricas(r["fugas"], r["detectadas"], r["alarmas"], r["falsas"], r["falsas_por_semana"])

    def puntos(self):
        return [(u, 1) for u in (0.3, 0.5, 0.7, 0.9)], 1.0

    elegir = V._elegir_por_grilla


def test_validar_con_punto_fijo_y_eligiendo_el_suyo():
    j = Juguete()
    plgs = V.pliegues(j.index)
    r = V.validar(j, plgs, punto=(0.5, 1))
    assert list(r.pliegues["pliegue"]) == [1, 2, 3, 4, 5]
    assert r.punto == (0.5, 1) and 0 <= r.f1_desv <= r.f1_media <= 1
    assert np.isnan(r.prob[: plgs[0].desde.hour + 1]).all()        # antes del pliegue 1 no se predice
    entrenados = len(j.entrenados)
    elegido = V.validar(j, plgs, prob=r.prob)                         # misma predicción, umbral por regla
    assert len(j.entrenados) == entrenados                            # no vuelve a entrenar
    assert elegido.punto in j.puntos()[0]
    assert (elegido.pliegues["falsas_por_semana"] <= j.puntos()[1]).all()   # respeta el tope en cada pliegue


def test_un_modelo_peor_tiene_menos_f1():
    j = Juguete()
    plgs = V.pliegues(j.index)
    bueno = V.validar(j, plgs, {"ruido": 0.1}, punto=(0.5, 1))
    malo = V.validar(j, plgs, {"ruido": 0.9}, punto=(0.5, 1))
    assert bueno.f1_media > malo.f1_media


def test_curva_de_aprendizaje_usa_partes_crecientes_del_ultimo_pliegue():
    j = Juguete()
    plgs = V.pliegues(j.index)
    c = V.curva_aprendizaje(j, plgs, (0.5, 1), fracciones=[0.25, 0.5, 1.0])
    assert list(c["fraccion"]) == [0.25, 0.5, 1.0]
    assert j.entrenados[0] < j.entrenados[1] < j.entrenados[2]
    assert j.entrenados[2] == j.preparado(plgs[-1])["entrena"].sum()     # 100 %: todo el pasado del pliegue
    assert ((c["error_validacion"] >= 0) & (c["error_validacion"] <= 1)).all()


def test_sensibilidad_y_busqueda_reusan_lo_ya_calculado():
    j = Juguete()
    plgs = V.pliegues(j.index)
    cache = {}
    s = V.sensibilidad(j, plgs, {"ruido": [0.1, 0.3, 0.9]}, cache)
    assert list(s["es_el_final"]) == [False, True, False]
    assert s.set_index("valor").loc["0.1", "f1_media"] > s.set_index("valor").loc["0.9", "f1_media"]
    n = len(j.entrenados)
    vistos = []
    b = V.busqueda(j, plgs, {"ruido": [0.1, 0.3, 0.5, 0.9]}, 2, semilla=0, cache=cache,
                   al_terminar=lambda p, r: vistos.append(p))
    assert len(b) == 3 and "modelo final" in set(b["configuracion"])
    assert b["f1_media"].is_monotonic_decreasing
    assert vistos[0] == {} and len(vistos) == 3
    nuevos = sum(1 for p in vistos[1:] if p["ruido"] not in (0.1, 0.9))
    assert len(j.entrenados) - n == 5 * nuevos                     # la base, 0,1 y 0,9 ya estaban
