"""Limpieza de lecturas (calidad.py): presión mínima, rango físico, sensor pegado y fallas."""

import numpy as np

from calidad_agua import calidad


def test_presion_baja_anula_las_lecturas_de_calidad(red):
    red.loc[red.index[10:20], "P_J10"] = 0.5          # tubería sin agua
    t = calidad.lecturas_operativas(red)
    for p in calidad.CALIDAD:
        assert t[f"{p}_J10"].iloc[10:20].isna().all()
        assert t[f"{p}_J10"].iloc[20:30].notna().all()
        assert t[f"{p}_J20"].iloc[10:20].notna().all()      # los demás sensores no se tocan
    assert t["P_J10"].iloc[10:20].notna().all()             # la presión misma se conserva


def test_valores_fuera_del_rango_fisico(red):
    red.loc[red.index[5], "CL2_J20"] = 6.0
    red.loc[red.index[6], "PH_J20"] = 3.0
    red.loc[red.index[7], "TEMP_J20"] = 45.0
    red.loc[red.index[8], "TURB_J20"] = -1.0
    t = calidad.lecturas_operativas(red, preprocesar=False)
    assert np.isnan(t["CL2_J20"].iloc[5])
    assert np.isnan(t["PH_J20"].iloc[6])
    assert np.isnan(t["TEMP_J20"].iloc[7])
    assert np.isnan(t["TURB_J20"].iloc[8])
    assert t["CL2_J20"].iloc[6:9].notna().all()


def test_sensor_pegado_se_detecta_a_las_16_lecturas_iguales(red):
    red.loc[red.index[100:116], "CL2_J10"] = 0.7       # 16 lecturas idénticas = 4 h sin cambio
    red.loc[red.index[200:215], "CL2_J20"] = 0.6       # 15 lecturas: todavía no está pegado
    t = calidad.lecturas_operativas(red, preprocesar=False)
    assert np.isnan(t["CL2_J10"].iloc[115])
    assert t["CL2_J10"].iloc[100:115].notna().all()
    assert t["CL2_J20"].iloc[200:215].notna().all()


def test_ceros_repetidos_no_cuentan_como_pegado():
    import pandas as pd
    x = pd.Series([0.0] * 30)
    assert not calidad.pegado(x).any()
    y = pd.Series([0.5] * 30)
    assert calidad.pegado(y)[15:].all() and not calidad.pegado(y)[:15].any()


def test_lecturas_validas_usa_las_etiquetas_de_falla(red):
    red.loc[red.index[50:54], "falla_TURB_J20"] = 3
    validas = calidad.lecturas_validas(red)
    operativas = calidad.lecturas_operativas(red)
    assert validas["TURB_J20"].iloc[50:54].isna().all()
    assert operativas["TURB_J20"].iloc[50:54].notna().all()   # en operación no hay etiquetas


def test_mascara_invalida_sin_columna_de_falla(red):
    red = red.drop(columns=["falla_CL2_J10"])
    red.loc[red.index[3], "P_J10"] = 0.2
    m = calidad.mascara_invalida(red, "CL2", "J10")
    assert m[3] and m.sum() == 1


def test_resultado_en_float32_y_sin_modificar_la_entrada(red):
    original = red.copy()
    red.loc[red.index[0], "CL2_J10"] = 9.0
    copia = red.copy()
    t = calidad.lecturas_operativas(red)
    assert t["CL2_J10"].dtype == np.float32
    assert t["Q_R1"].dtype == np.float32
    assert red.equals(copia)                          # la entrada queda igual
    assert len(t) == len(original)


def test_norma_del_decreto():
    assert calidad.NORMA["CL2"] == {"minimo": 0.3, "maximo": 1.0}
    assert calidad.NORMA["TURB"]["maximo"] == 5.0
    assert calidad.NORMA["PH"] == {"minimo": 6.5, "maximo": 8.5}


def test_con_preprocesamiento_la_lectura_invalida_suelta_se_interpola(red):
    red.loc[red.index[5], "CL2_J20"] = 6.0                     # fuera de rango: se quita...
    t = calidad.lecturas_operativas(red)
    antes, despues = red["CL2_J20"].iloc[4], red["CL2_J20"].iloc[6]
    assert abs(t["CL2_J20"].iloc[5] - (antes + despues) / 2) < 1e-5   # ...y se rellena (hueco de 15 min)
    assert t["flag_imputacion"].iloc[5] == "interpolado" and t["flag_imputacion"].iloc[20] == "original"
