"""herramientas/fuente_lluvia.py (RNF-009): lectura del CSV de Open-Meteo y concordancia con el IMN."""

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RUTA = Path(__file__).resolve().parents[1] / "herramientas" / "fuente_lluvia.py"
spec = importlib.util.spec_from_file_location("fuente_lluvia", RUTA)
F = importlib.util.module_from_spec(spec)
spec.loader.exec_module(F)

ENCABEZADO = ("latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n"
              "10.36,-84.53,166.0,-21600,America/Costa_Rica,GMT-6\n\n")


def test_leer_open_meteo_varios_csv_unidos(tmp_path):
    cols = ",".join(f"precipitation_{m} (mm)" for m in ["best_match"] + F.CUATRO)
    filas = ["2026-05-01T00:00,0.1,0.2,0.4,0.0,0.6", "2026-05-01T01:00,1.0,2.0,,1.0,3.0"]
    texto = ENCABEZADO + f"time,{cols}\n{filas[0]}\ntime,{cols}\n{filas[1]}\n{filas[1]}\n"   # dos partes y una repetida
    (tmp_path / "om.csv").write_text(texto)
    t = F.leer_open_meteo(tmp_path / "om.csv")
    assert list(t.index) == list(pd.date_range("2026-05-01", periods=2, freq="h"))
    assert list(t.columns) == ["best_match", *F.CUATRO, "promedio"]
    assert t["promedio"].iloc[0] == pytest.approx(0.3) and t["promedio"].iloc[1] == pytest.approx(2.0)   # sin GFS


def test_un_solo_modelo(tmp_path):
    (tmp_path / "om.csv").write_text(ENCABEZADO + "time,precipitation (mm)\n2026-05-01T00:00,-0.1\n")
    t = F.leer_open_meteo(tmp_path / "om.csv")
    assert list(t.columns) == ["best_match"] and t.iloc[0, 0] == 0.0          # nada negativo


def test_concordancia():
    horas = pd.date_range("2026-01-01", periods=24 * 60, freq="h")
    rng = np.random.default_rng(0)
    imn = pd.Series(rng.exponential(1, len(horas)) * (rng.random(len(horas)) < 0.2), index=horas)
    c = F.concordancia(imn, imn * 0.5)
    assert c["razon_total"] == pytest.approx(0.5) and c["r_horaria"] == pytest.approx(1.0)
    assert c["r_30d"] == pytest.approx(1.0) and c["far_hora"] == 0.0 and c["pod_hora"] < 1
    assert F.concordancia(imn, imn)["pod_dia"] == 1.0
