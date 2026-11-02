"""Variables del pronóstico a 24 h (variables.py)."""

import numpy as np
import pandas as pd
import pytest

from calidad_agua import calidad, clima, variables
from conftest import igual, SENSORES


@pytest.fixture
def h(red):
    red.loc[red.index[400:440], "evento_J10"] = 1            # un evento de 10 h en J10
    red.loc[red.index[400:440], "CL2_J10"] = 0.1
    red.loc[red.index[500:], "particion"] = "validacion"
    return variables.horario(calidad.lecturas_operativas(red))


def test_horario_promedia_las_cuatro_lecturas(red):
    red.loc[red.index[1], "CL2_J20"] = np.nan                 # una lectura que no llegó
    hh = variables.horario(red)
    assert hh["CL2_J10"].iloc[0] == pytest.approx(red["CL2_J10"].iloc[:4].mean())
    assert hh["CL2_J20"].iloc[0] == pytest.approx(red["CL2_J20"].iloc[[0, 2, 3]].mean())
    assert hh["lecturas_J20"].iloc[0] == 3 and hh["lecturas_J10"].iloc[0] == 4
    assert hh["lluvia_mm"].iloc[50] == pytest.approx(4.0)       # el aguacero de la conftest
    assert hh["particion"].iloc[0] == "entrenamiento"
    assert len(hh) == len(red) // 4


def test_horario_en_operacion_sin_etiquetas(red):
    vivo = red.drop(columns=[c for c in red.columns if c.startswith(("evento_", "falla_")) or c == "particion"])
    hh = variables.horario(vivo)
    assert hh["evento_J10"].isna().all()
    assert hh["particion"].isna().all()


def test_tabla_tiene_una_fila_por_sensor_y_hora(h):
    tabla, tipicos = variables.tabla(h)
    assert len(tabla) == len(h) * len(SENSORES)
    assert list(tabla["nodo"].unique()) == SENSORES
    assert tabla.groupby("nodo")["sensor"].first().to_dict() == {n: i for i, n in enumerate(SENSORES)}
    assert set(tipicos) == {(p, n) for n in SENSORES for p in variables.PARAMS}


def test_tipicos_salen_del_entrenamiento_en_operacion_normal(h):
    _, tipicos = variables.tabla(h)
    normal = (h["particion"] == "entrenamiento") & (h["evento_J10"] == 0)
    assert tipicos[("CL2", "J10")] == pytest.approx(h.loc[normal, "CL2_J10"].median())
    # con tipicos dados (como en operación) se usan tal cual
    fijos = {k: 1.0 for k in tipicos}
    tabla, devueltos = variables.tabla(h, tipicos=fijos)
    assert devueltos is fijos and (tabla["CL2_tipico"] == 1.0).all()


def test_variables_de_un_sensor_iguales_a_calcularlas_por_separado(h):
    tabla, _ = variables.tabla(h)
    d = tabla[tabla["nodo"] == "J10"].set_index("fecha")
    x = h["CL2_J10"]
    igual(d["CL2_hace24h"], x.shift(24))
    igual(d["CL2_cambio3h"], x - x.shift(3))
    igual(d["CL2_media24h"], x.rolling(24, min_periods=12).mean())
    igual(d["TURB_desv6h"], h["TURB_J10"].rolling(6, min_periods=3).std())
    igual(d["PH_vs_fuente"], h["PH_J10"] - h["PH_J280"])
    igual(d["objetivo_CL2"], x.shift(-24))
    igual(d["evento_objetivo"], h["evento_J10"].shift(-24))


def test_variables_comunes_de_fuente_red_y_clima(h):
    tabla, _ = variables.tabla(h)
    d = tabla[tabla["nodo"] == "J20"].set_index("fecha")
    igual(d["fuente_CL2_hace6h"], h["CL2_J280"].shift(6))
    red = h[[f"TURB_{n}" for n in SENSORES]].median(axis=1)
    igual(d["red_TURB_mediana"], red)
    assert d["lluvia_72h"].max() == pytest.approx(h["lluvia_mm"].sum())
    hora_objetivo = (d.index + pd.Timedelta(hours=24)).hour
    np.testing.assert_allclose(d["hora_sin"], np.sin(2 * np.pi * hora_objetivo / 24))


def test_objetivo_de_otra_tabla(h):
    limpia = h.copy()
    limpia["CL2_J20"] = 0.42
    tabla, _ = variables.tabla(h, h_objetivo=limpia)
    obj = tabla.loc[tabla["nodo"] == "J20", "objetivo_CL2"].dropna()
    assert (obj == 0.42).all() and len(obj) == len(h) - 24


def test_pronostico_de_lluvia_como_variable(h):
    lluvia = pd.Series(0.5, index=pd.date_range(h.index[0], periods=len(h) + 30, freq="1h"))
    pron = clima.ventanas(lluvia, h.index)
    tabla, _ = variables.tabla(h, pronostico=pron)
    assert set(clima.COLUMNAS) <= set(tabla.columns)
    assert tabla["lluvia_pron_12_24h"].iloc[0] == pytest.approx(6.0)
    sin, _ = variables.tabla(h)
    assert not set(clima.COLUMNAS) & set(sin.columns)


def test_columnas_de_entrada_no_incluyen_objetivos_ni_etiquetas(h):
    tabla, _ = variables.tabla(h)
    cols = variables.columnas_entrada(tabla)
    assert "sensor" in cols and "CL2_media6h" in cols
    assert not {"fecha", "particion", "nodo", "evento_objetivo"} & set(cols)
    assert not [c for c in cols if c.startswith("objetivo_")]
