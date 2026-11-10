"""Preprocesamiento (preprocesamiento.py, RF-001.3): atípicos por IQR, interpolación y flag_imputacion."""

import json
import logging

import numpy as np
import pandas as pd
import pytest

from calidad_agua import preprocesamiento as pp


def serie(n=200, semilla=0, nivel=0.8, ruido=0.01):
    rng = np.random.default_rng(semilla)
    return pd.Series(nivel + rng.normal(0, ruido, n), index=pd.date_range("2025-01-01", periods=n, freq="15min"))


def test_pico_suelto_es_atipico_pero_un_cambio_sostenido_no():
    x = serie()
    x.iloc[50] = 0.1                                     # pico de una lectura (falla del sensor)
    x.iloc[100:140] -= 0.5                               # el cloro baja y se queda abajo: evento real
    m = pp.atipicos(x, k=1.5, minimo=0.15)
    assert m.iloc[50]
    assert not m.iloc[95:145].any()                      # ni el inicio ni el fin del evento se tocan
    assert m.sum() == 1


def test_el_ruido_normal_no_se_toca_por_el_minimo():
    x = serie(n=2000, ruido=0.02)
    assert pp.atipicos(x, k=1.5, minimo=0.0).sum() > 0   # sin mínimo, el ruido cae fuera del IQR
    assert pp.atipicos(x, k=1.5, minimo=0.15).sum() == 0


def test_k_mas_grande_es_mas_permisivo():
    x = serie(n=500, ruido=0.05)
    x.iloc[[40, 200, 300]] += [0.4, 0.25, 0.6]
    assert pp.atipicos(x, k=1.5).sum() >= pp.atipicos(x, k=3.0).sum() >= pp.atipicos(x, k=10.0).sum()


def test_bordes_sin_vecinas_no_se_revisan():
    x = serie(n=20)
    x.iloc[-1] = 5.0
    x.iloc[0] = 5.0
    assert not pp.atipicos(x, minimo=0.15).iloc[[0, -1]].any()


def test_interpolacion_solo_en_huecos_de_hasta_30_min():
    x = serie(n=30)
    x.iloc[5] = np.nan                                   # 15 min
    x.iloc[10:12] = np.nan                               # 30 min
    x.iloc[20:23] = np.nan                               # 45 min: se marca
    x.iloc[-1] = np.nan                                  # al final, sin dato después: se marca
    y, interp, marc = pp.interpolar(x, max_lecturas=2)
    assert interp.sum() == 3 and list(np.flatnonzero(interp)) == [5, 10, 11]
    assert list(np.flatnonzero(marc)) == [20, 21, 22, 29]
    assert y.iloc[20:23].isna().all() and y.iloc[[5, 10, 11]].notna().all()
    assert y.iloc[10] == pytest.approx(x.iloc[9] + (x.iloc[12] - x.iloc[9]) / 3)


def test_preprocesar_con_flag_por_fila(red):
    red.loc[red.index[10], "CL2_J10"] = np.nan                     # se interpola
    red.loc[red.index[30:34], "TURB_J20"] = np.nan                 # 1 h: se marca
    red.loc[red.index[50], "PH_J280"] = 3.5 + red["PH_J280"].iloc[50]   # pico de pH
    red.loc[red.index[200:212], "lluvia_mm_h"] = 30.0              # aguacero: no es un pico
    red.loc[red.index[60], "Q_R1"] += 40                           # arranque de una bomba: real
    t, estado = pp.preprocesar(red, detalle=True)
    f = t["flag_imputacion"]
    assert f.iloc[10] == "interpolado" and f.iloc[31] == "marcado" and f.iloc[50] == "interpolado"
    assert f.iloc[100] == "original" and set(f.unique()) <= {"original", "interpolado", "marcado"}
    assert estado.loc[red.index[50], "PH_J280"] == "atipico"
    assert abs(t["PH_J280"].iloc[50] - 7.2) < 0.2                   # el pico se reemplazó
    assert (estado["lluvia_mm_h"] != "atipico").all() and t["lluvia_mm_h"].iloc[205] == 30.0
    assert (estado["Q_R1"] != "atipico").all() and t["Q_R1"].iloc[60] > 50
    assert t["TURB_J20"].iloc[30:34].isna().all()
    assert len(t) == len(red) and "flag_imputacion" not in red     # no cambia la entrada
    r = pp.resumen(estado)
    assert r.loc["TURB_J20", "marcado"] == pytest.approx(100 * 4 / len(red), abs=0.01)


def test_iqr_se_puede_apagar_y_configurar(red):
    red.loc[red.index[50], "PH_J280"] += 3.5
    t = pp.preprocesar(red, {"iqr": False})
    assert t["PH_J280"].iloc[50] > 10
    t = pp.preprocesar(red, {"iqr_k": 1.5, "iqr_minimo": {"PH": 5.0}})           # el pico no supera el mínimo
    assert t["PH_J280"].iloc[50] > 10
    t = pp.preprocesar(red, {"max_hueco_min": 0, "iqr": False})
    assert (t["flag_imputacion"] != "interpolado").all()


def test_registro_con_fecha_y_parametros(red, tmp_path):
    ruta = pp.registrar_en(tmp_path / "pre.log")
    pp.registrar_en(tmp_path / "pre.log")                          # no duplica el archivo
    red.loc[red.index[10], "CL2_J10"] = np.nan
    pp.preprocesar(red)
    for h in list(pp.registro.handlers):
        if isinstance(h, logging.FileHandler):
            h.flush()
            pp.registro.removeHandler(h)
            h.close()
    (linea,) = ruta.read_text(encoding="utf-8").splitlines()
    r = json.loads(linea)
    assert r["fecha"] and r["parametros"]["iqr_k"] == 1.5 and r["parametros"]["max_hueco_min"] == 30
    assert r["interpoladas"] >= 1 and r["por_columna"]["interpoladas"]["CL2_J10"] == 1
    assert r["lecturas"] == len(red)


def test_configuracion_del_yaml():
    c = pp.configuracion()
    assert c["iqr_k"] == 1.5 and c["max_hueco_min"] == 30 and c["iqr_parametros"] == ["CL2", "TURB", "PH", "TEMP"]
