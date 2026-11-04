"""Deriva de los datos (deriva.py)."""

import numpy as np
import pandas as pd

from calidad_agua import deriva


def test_psi_estable_y_con_deriva():
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=2000, freq="1h")
    h = pd.DataFrame({"CL2_J280": rng.normal(0.8, 0.05, 2000), "P_J10": rng.normal(30, 1, 2000),
                      "Q_R1": rng.normal(260, 10, 2000), "evento_J10": 0, "lluvia_mm_h": 0.0}, index=idx)
    assert deriva.columnas(h) == ["CL2_J280", "P_J10", "Q_R1"]
    ref = deriva.referencia(h.iloc[:1500])
    r = deriva.revisar(ref, h.iloc[1500:]).set_index("columna")
    assert (r["estado"] == "estable").all() and r.loc["Q_R1", "horas"] == 500
    reciente = h.iloc[1500:].copy()
    reciente["CL2_J280"] -= 0.1                       # el cloro bajó: deriva
    reciente["P_J10"] += 0.3                          # cambio chico: estable o vigilar
    reciente["Q_R1"] = np.nan                         # sin datos
    r = deriva.revisar(ref, reciente).set_index("columna")
    assert r.loc["CL2_J280", "estado"] == "deriva" and r.loc["CL2_J280", "psi"] > 1
    assert r.loc["P_J10", "estado"] in ("estable", "vigilar")
    assert r.loc["Q_R1", "estado"] == "sin datos"
    assert deriva.estado(0.05) == "estable" and deriva.estado(0.2) == "vigilar" and deriva.estado(0.3) == "deriva"
