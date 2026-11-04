"""Fugas (fugas.py) y ubicación de los sensores (ubicacion.py)."""

import numpy as np
import pandas as pd
import pytest

from calidad_agua import fugas as F
from calidad_agua import ubicacion as U
from conftest import red_sintetica


@pytest.fixture
def h():
    """Red sintética horaria con tres fugas: el caudal sube 2 L/s y la presión de J20 baja 3 m."""
    t = red_sintetica(dias=30)
    hh = t.resample("1h").mean(numeric_only=True)
    for c in [c for c in t.columns if c.startswith("evento_")]:
        hh[c] = 0
    for ini in ["2025-03-08 02:00", "2025-03-15 10:00", "2025-03-24 20:00"]:
        v = slice(pd.Timestamp(ini), pd.Timestamp(ini) + pd.Timedelta(hours=11))
        hh.loc[v, "Q_R1"] += 2.0
        hh.loc[v, "P_J20"] -= 3.0
        hh.loc[v, ["evento_J280", "evento_J20"]] = F.CODIGO_FUGA
    return hh


def test_tramos_con_los_dos_sensores_mas_cercanos():
    d = pd.DataFrame({"J280": [10, 500, 900], "J10": [50, 20, 800], "J20": [900, 30, 100]},
                     index=pd.Index(["A", "B", "C"], name="nodo"))
    assert F.tramos(d).tolist() == ["J280 – J10", "J10 – J20", "J10 – J20"]


def test_residuos_y_clase(h):
    y = F.clase(h)
    assert y.sum() == 36
    normal = (y == 0).values
    mods = F.ajustar_normal(h, normal)
    X = F.tabla(h, mods)
    assert abs(X.loc[normal, "caudal_residuo"].mean()) < 0.05          # la normal queda explicada
    assert X.loc[~normal, "caudal_residuo"].mean() == pytest.approx(2.0, abs=0.3)
    assert X.loc[~normal, "presion_J20_residuo"].mean() < -2.5
    assert X.loc[~normal, "presion_J10_residuo"].abs().mean() < 0.5


def test_modelos_aprenden_la_fuga_y_el_tramo(h):
    y = F.clase(h).values
    X = F.tabla(h, F.ajustar_normal(h, y == 0))
    for entrenar in (F.entrenar_bosque, F.entrenar_svm):
        p = F.probabilidad(entrenar(X, y), X)
        assert p[y == 1].mean() > 0.6 and p[y == 0].mean() < 0.2
    horas = X.index[y == 1]
    assert F.tramo_por_regla(X, ["J280", "J10", "J20"], horas).endswith("J20")
    mod = F.entrenar_tramo(X.loc[horas], np.array(["J280 – J20"] * len(horas)))
    assert F.tramo_por_evento(mod, X, horas)[0] == "J280 – J20"


def test_evaluar_y_elegir_umbral(h):
    y = F.clase(h).values
    p = y * 0.9 + 0.05
    p[100:102] = 0.8                                  # una falsa alarma de 2 h
    ev, r = F.evaluar(h.index, F.alarma(p, 0.5), y, h.index[0], h.index[-1] + pd.Timedelta(hours=1))
    assert r["fugas"] == 3 and r["detectadas"] == 3 and r["falsas"] == 1 and r["retraso_mediano_h"] == 0
    assert r["precision"] == pytest.approx(0.75) and r["sensibilidad"] == 1
    _, r2 = F.evaluar(h.index, F.alarma(p, 0.5, 3), y, h.index[0], h.index[-1] + pd.Timedelta(hours=1))
    assert r2["falsas"] == 0 and r2["retraso_mediano_h"] == 2
    (u, c), grilla = F.elegir_umbral(h.index, p, y, [(h.index[0], h.index[-1] + pd.Timedelta(hours=1))],
                                     [0.5, 0.85], [1, 3], 0.1)
    assert (u, c) == (0.85, 1) and len(grilla) == 4


def test_autoencoder_marca_lo_raro(h):
    pytest.importorskip("tensorflow")
    y = F.clase(h).values
    X = F.tabla(h, F.ajustar_normal(h, y == 0))
    cfg = dict(F.HIPERPARAMETROS["autoencoder"], epocas=30)
    ae = F.Autoencoder(cfg).fit(X[y == 0])
    s = F.probabilidad(ae, X)
    assert np.mean(s[y == 0] > 1) == pytest.approx(0.05, abs=0.03)     # percentil 95
    assert np.median(s[y == 1]) > np.median(s[y == 0])


def test_ubicacion_por_eventos_vistos():
    idx = pd.date_range("2025-03-01", periods=400, freq="15min")
    t = pd.DataFrame(0, index=idx, columns=[f"evento_{n}" for n in ["J280", "A", "B", "C"]])
    cat = pd.DataFrame({"id": [1, 2, 3], "tipo": ["contaminacion", "fuga", "falla_cloro"], "codigo": [1, 5, 2],
                        "inicio": idx[[10, 150, 300]], "fin": idx[[20, 170, 310]]})
    t.loc[idx[12]:idx[20], "evento_A"] = 1
    t.loc[idx[150]:idx[170], "evento_C"] = 5
    t.loc[idx[300]:idx[310], ["evento_J280", "evento_A", "evento_B", "evento_C"]] = 2
    V = U.visibilidad(t, cat)
    assert V.loc[1, "A"] == 0.5 and np.isnan(V.loc[1, "B"]) and V.loc[3, "J280"] == 0
    pares = U.mejores_pares(V, cat)
    assert pares.loc[0, "par"] == "A y C" and pares.loc[0, "vistos"] == 3
    r = U.evaluar_conjunto(V, cat, ["J280", "B"])
    assert r.loc["falla_cloro", "vistos"] == 1 and r.loc["contaminacion", "vistos"] == 0
