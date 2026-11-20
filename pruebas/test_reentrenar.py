"""Reparto de los notebooks en filas paralelas (herramientas/reentrenar.py, RF-004.2)."""

import importlib.util
from pathlib import Path

RUTA = Path(__file__).resolve().parents[1] / "herramientas" / "reentrenar.py"
spec = importlib.util.spec_from_file_location("reentrenar", RUTA)
reentrenar = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reentrenar)


def test_todos_en_tres_filas_con_el_09_solo():
    assert reentrenar.filas(list(reentrenar.NOTEBOOKS)) == [("09",), ("06", "08"), ("07", "10")]


def test_cada_notebook_una_sola_vez():
    reparto = [n for f in reentrenar.FILAS for n in f]
    assert sorted(reparto) == sorted(reentrenar.NOTEBOOKS)


def test_solo_los_pedidos_y_sin_filas_vacias():
    assert reentrenar.filas(["07", "10"]) == [("07", "10")]
    assert reentrenar.filas(["08", "09"]) == [("09",), ("08",)]


def test_en_serie_una_sola_fila_en_orden():
    assert reentrenar.filas(["10", "06", "09"], en_serie=True) == [("06", "09", "10")]
