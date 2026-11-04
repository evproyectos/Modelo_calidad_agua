"""
Datos sintéticos para las pruebas unitarias: una red pequeña con la misma forma que el dataset
(lecturas cada 15 min, columnas CL2_<nodo>, TEMP_, TURB_, PH_, P_, caudal, lluvia, fallas y
eventos), para que las pruebas corran en segundos y sin los archivos de datos.
"""

import numpy as np
import pandas as pd
import pytest

SENSORES = ["J280", "J10", "J20"]          # J280 es la fuente, como en la red real


@pytest.fixture(autouse=True)
def todos_los_sensores(monkeypatch):
    """Las pruebas usan todos los sensores de la red sintética, no los de hiperparametros.yaml."""
    monkeypatch.setenv("SENSORES", "todos")


def red_sintetica(dias=8, inicio="2025-03-01", semilla=0, particion="entrenamiento"):
    rng = np.random.default_rng(semilla)
    idx = pd.date_range(inicio, periods=dias * 96, freq="15min")
    hora = idx.hour.values + idx.minute.values / 60
    ciclo = np.sin(2 * np.pi * hora / 24)
    t = pd.DataFrame(index=idx)

    def ruido(e):
        return rng.normal(0, e, len(idx))

    for k, n in enumerate(SENSORES):
        t[f"CL2_{n}"] = 0.8 - 0.15 * k + 0.05 * ciclo + ruido(0.01)
        t[f"TEMP_{n}"] = 25 + 2 * ciclo + ruido(0.1)
        t[f"TURB_{n}"] = 0.5 + 0.1 * k + ruido(0.02)
        t[f"PH_{n}"] = 7.2 + 0.05 * ciclo + ruido(0.01)
        t[f"P_{n}"] = 30 - 5 * k + ruido(0.2)
    t["Q_R1"] = 12 + ciclo + rng.normal(0, 0.1, len(idx))
    lluvia = np.zeros(len(idx))
    lluvia[200:212] = 4.0                     # un aguacero de 3 h
    t["lluvia_mm_h"] = lluvia
    for n in SENSORES:
        for p in ["CL2", "TEMP", "TURB", "PH"]:
            t[f"falla_{p}_{n}"] = 0
        t[f"evento_{n}"] = 0
    t["evento_visible"] = 0
    t["evento_red"] = 0
    t["particion"] = particion
    return t


def igual(a, b):
    """Mismos valores e índice (la frecuencia del índice no importa)."""
    pd.testing.assert_series_equal(a, b, check_names=False, check_freq=False)


@pytest.fixture
def red():
    return red_sintetica()


@pytest.fixture
def sensores():
    return list(SENSORES)
