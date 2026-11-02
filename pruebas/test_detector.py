"""Detector de eventos (detector.py): variables, entrenamiento, alarmas y evaluación por episodio."""

import numpy as np
import pandas as pd
import pytest

from calidad_agua import calidad, detector
from conftest import igual, SENSORES, red_sintetica


@pytest.fixture
def hd():
    t = red_sintetica(dias=10)
    # eventos de contaminación en J10 (cloro baja) y de turbidez en la fuente, que se ven en la red
    for ini in (300, 620):
        t.loc[t.index[ini:ini + 24], "evento_J10"] = 1
        t.loc[t.index[ini:ini + 24], "CL2_J10"] -= 0.4
        t.loc[t.index[ini:ini + 24], "evento_red"] = 1
    t.loc[t.index[800:830], ["evento_J280", "evento_J10", "evento_J20"]] = 3
    for n in SENSORES:
        t.loc[t.index[800:830], f"TURB_{n}"] += 2.0
    t.loc[t.index[800:830], "evento_red"] = 3
    t.loc[t.index[700:], "particion"] = "validacion"
    return detector.horario(calidad.lecturas_operativas(t))


def test_horario_toma_solo_lecturas_caudal_y_lluvia(hd):
    assert "lluvia_mm_h" in hd and "Q_R1" in hd and "P_J10" in hd
    assert not [c for c in hd.columns if c.startswith("falla_")]
    assert hd["evento_J10"].max() == 3 and hd["lecturas_J10"].max() == 4


def test_tipicos_por_hora_del_dia(hd):
    tip = detector.tipicos_por_hora(hd, SENSORES)
    assert len(tip["CL2", "J20"]) == 24
    normal = (hd["particion"] == "entrenamiento") & (hd["evento_J20"] == 0) & (hd.index.hour == 7)
    assert tip["CL2", "J20"][7] == pytest.approx(hd.loc[normal, "CL2_J20"].median())


def test_tabla_desvios_respecto_a_lo_tipico(hd):
    tabla, tip = detector.tabla(hd)
    assert len(tabla) == len(hd) * len(SENSORES)
    d = tabla[tabla["nodo"] == "J10"].set_index("fecha")
    esperado = hd["CL2_J10"] - tip["CL2", "J10"].reindex(hd.index.hour).values
    np.testing.assert_allclose(d["desvio_CL2"], esperado)
    # mismas ventanas que calcularlas sensor por sensor
    igual(d["desvio_CL2_media12h"], d["desvio_CL2"].rolling(12, min_periods=3).mean())
    igual(d["desv_TURB_6h"], hd["TURB_J10"].rolling(6, min_periods=3).std())
    desvios = pd.DataFrame({n: tabla.loc[tabla["nodo"] == n].set_index("fecha")["desvio_PH"] for n in SENSORES})
    np.testing.assert_allclose(d["desvio_PH_vs_red"], desvios["J10"] - desvios.median(axis=1))
    assert d["clase"].max() == 3 and set(tabla["clase"].unique()) == {0, 1, 3}


def test_turbidez_en_la_fuente_se_ve_en_toda_la_red(hd):
    tabla, _ = detector.tabla(hd)
    d = tabla[tabla["nodo"] == "J20"].set_index("fecha")
    hora_evento = hd.index[205]                          # dentro de las lecturas 800–830
    assert d.loc[hora_evento, "red_sensores_desviados_TURB"] == len(SENSORES)
    assert d.loc[hora_evento, "parametros_desviados"] >= 1
    assert d.loc[hd.index[20], "red_sensores_desviados_TURB"] == 0


def test_entrenar_y_probabilidades(hd):
    tabla, _ = detector.tabla(hd)
    cols = detector.columnas_entrada(tabla)
    assert not {"fecha", "particion", "nodo", "clase"} & set(cols)
    mod = detector.entrenar(tabla, cols, tabla["particion"] == "entrenamiento")
    p, tipo = detector.probabilidades(mod, tabla, cols)
    assert ((p >= 0) & (p <= 1)).all()
    assert 0 not in set(tipo)                               # el tipo sugerido nunca es "normal"
    evento = tabla["clase"].values == 1                        # el tipo que vio al entrenar
    assert p[evento].mean() > p[~evento].mean()


def test_alarma_solo_con_horas_seguidas_sobre_el_umbral():
    fechas = pd.date_range("2025-03-01", periods=6, freq="1h")
    t = pd.DataFrame({"fecha": fechas, "nodo": "J10"})
    p = np.array([0.95, 0.2, 0.95, 0.95, 0.95, 0.1])
    a = detector.alarmas_sensor(t, p, umbral=0.9, confirmar_h=2)["J10"].tolist()
    assert a == [False, False, False, True, True, False]
    assert detector.alarmas_sensor(t, p, 0.9, confirmar_h=1)["J10"].tolist() == list(p >= 0.9)


def test_episodios_de_alarma_se_unen_si_estan_cerca():
    idx = pd.date_range("2025-03-01", periods=20, freq="1h")
    red = pd.Series(False, index=idx)
    red.iloc[[2, 3, 5, 6, 15]] = True                         # hueco de 1 h y luego uno de 8 h
    eps = detector.episodios_alarma(red, unir_h=2)
    assert len(eps) == 2
    assert eps.iloc[0]["inicio"] == idx[2] and eps.iloc[0]["fin"] == idx[6] and eps.iloc[0]["horas"] == 5
    assert detector.episodios_alarma(pd.Series(False, index=idx)).empty


def test_evaluar_cuenta_detectados_retraso_y_falsas_alarmas():
    idx = pd.date_range("2025-03-01", periods=24 * 14, freq="1h")
    h = pd.DataFrame(0, index=idx, columns=[f"evento_{n}" for n in SENSORES] + ["evento_red"])
    h.iloc[10:20, [1, 3]] = 1                                  # evento visible en J10
    h.iloc[100:110, 3] = 2                                     # evento que ningún sensor ve
    alarma = pd.DataFrame(False, index=idx, columns=SENSORES)
    alarma.iloc[12:16, 1] = True                               # J10 lo detecta 2 h después
    alarma.iloc[200:203, 2] = True                             # J20 sin evento: falsa alarma
    tipos = pd.DataFrame(1, index=idx, columns=SENSORES)
    det, alarmas, r = detector.evaluar(h, alarma, idx[0], idx[-1] + pd.Timedelta(hours=1), tipo_predicho=tipos)
    assert r["episodios visibles"] == 1 and r["detectados"] == 1
    assert det.iloc[0]["retraso_h"] == 2 and det.iloc[0]["tipo_sugerido"] == "contaminacion"
    assert r["alarmas"] == 2 and r["falsas alarmas"] == 1
    assert r["falsas por semana"] == pytest.approx(1 / ((len(idx) - 10) / 168))
