"""Vector de etiquetas (etiquetas.py, RF-001.6): umbral IoT, registros de la ASADA y laboratorio."""

import numpy as np
import pandas as pd

from calidad_agua import etiquetas as E


def test_tres_lecturas_seguidas_fuera_de_norma(red):
    red.loc[red.index[10:12], "CL2_J10"] = 0.1                 # 2 lecturas: no alcanza
    red.loc[red.index[40:44], "TURB_J20"] = 8.0                # 4 lecturas: sí
    red.loc[red.index[60:63], "PH_J280"] = 9.0                 # 3 lecturas de pH alto: sí
    fila, sensor = E.construir(red)
    assert sensor["J10"].sum() == 0
    assert sensor["J20"].iloc[40:44].eq(1).all() and sensor["J20"].sum() == 4
    assert sensor["J280"].iloc[60:63].eq(1).all()
    assert fila["etiqueta"].iloc[41] == 1 and fila["etiqueta"].iloc[0] == 0
    assert set(fila["etiqueta"].unique()) == {0, 1}
    assert (fila["fuente_etiqueta"] == "umbral_iot").all()


def test_registro_de_fuga_de_la_asada(red):
    reg = pd.DataFrame({"inicio": [red.index[100]], "fin": [red.index[110]], "nodo": [None]})
    fila, sensor = E.construir(red, registros=reg)
    assert sensor.iloc[100:111].eq(1).all().all()                    # sin nodo: toda la red
    assert (fila["fuente_etiqueta"].iloc[100:111] == "registro_asada").all()
    assert fila["fuente_etiqueta"].iloc[99] == "umbral_iot"


def test_laboratorio_manda_sobre_la_regla(red):
    red.loc[red.index[200:206], "CL2_J10"] = 0.2                    # la regla diría evento...
    lab = pd.DataFrame({"fecha": [red.index[200], red.index[300]], "nodo": ["J10", "J20"],
                        "parametro": ["CL2", "TURB"], "valor": [0.5, 7.0]})
    fila, sensor = E.construir(red, laboratorio=lab)
    hora = red.index[200].floor("h")
    en_hora = (red.index >= hora) & (red.index < hora + pd.Timedelta(hours=1))
    assert sensor.loc[en_hora, "J10"].eq(0).all()                   # ...pero el laboratorio dice que cumple
    assert (fila.loc[en_hora, "fuente_etiqueta"] == "laboratorio").all()
    hora2 = red.index[300].floor("h")
    en_hora2 = (red.index >= hora2) & (red.index < hora2 + pd.Timedelta(hours=1))
    assert sensor.loc[en_hora2, "J20"].eq(1).all() and fila.loc[en_hora2, "etiqueta"].eq(1).all()


def test_comparacion_con_la_verdad_del_simulador(red):
    red.loc[red.index[40:48], "evento_J20"] = 1                     # contaminación real en J20
    red.loc[red.index[42:48], "CL2_J20"] = 0.1                      # la regla la ve desde la tercera lectura
    _, sensor = E.construir(red)
    r = E.comparar_con_verdad(sensor, red)
    assert r["lecturas_evento"] == 8 and r["aciertos"] == 6 and r["precision"] == 1.0
    assert np.isclose(r["sensibilidad"], 6 / 8)
    assert E.incumple("CL2", 0.2) and not E.incumple("CL2", 0.5) and E.incumple("PH", 9)


def test_incumplimiento_cronico_no_cuenta_para_la_regla(red):
    red["CL2_J20"] = 0.15                                           # al final de la red, siempre bajo
    red.loc[red.index[40:44], "TURB_J20"] = 8.0                     # pero la turbidez sí cuenta
    fila, sensor = E.construir(red)
    assert sensor.attrs["cronicos"] == {"J20": ["CL2"]}
    assert sensor["J20"].sum() == 4 and fila["etiqueta"].sum() == 4
    _, sin = E.construir(red, cronico=None)
    assert sin["J20"].sum() == len(red)                             # con la regla literal, todo es evento
