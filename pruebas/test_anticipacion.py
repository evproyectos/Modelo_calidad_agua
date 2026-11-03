"""Anticipación de eventos a 24 h (anticipacion.py): objetivo, variables, avisos y evaluación."""

import numpy as np
import pandas as pd
import pytest

from calidad_agua import anticipacion as A
from calidad_agua import calidad
from conftest import red_sintetica


def catalogo(*eventos):
    """eventos: (id, código, inicio)."""
    tipos = {1: "contaminacion", 2: "falla_cloro", 3: "turbidez_fuente", 4: "falla_ph"}
    return pd.DataFrame({"id": [e[0] for e in eventos], "codigo": [e[1] for e in eventos],
                         "tipo": [tipos[e[1]] for e in eventos],
                         "inicio": pd.to_datetime([e[2] for e in eventos]),
                         "causa": "aleatoria"})


@pytest.fixture
def horas():
    return pd.date_range("2025-03-01", periods=24 * 10, freq="1h")


def test_objetivo_marca_las_24_h_antes_del_inicio(horas):
    cat = catalogo((1, 3, "2025-03-03 12:00"), (2, 2, "2025-03-07 05:30"))
    o = A.objetivo(horas, cat)
    ini = pd.Timestamp("2025-03-03 12:00")
    assert o.loc[ini - pd.Timedelta(hours=24), "clase"] == 3           # justo 24 h antes: sí
    assert o.loc[ini - pd.Timedelta(hours=25), "clase"] == 0           # 25 h antes: todavía no
    assert o.loc[ini - pd.Timedelta(hours=1), "horas_para_inicio"] == 1
    assert o.loc[ini, "clase"] == 0                                    # ya empezó: no es anticipación
    # un inicio a media hora se cuenta desde la primera hora completa (06:00)
    assert o.loc[pd.Timestamp("2025-03-07 05:00"), "horas_para_inicio"] == 1
    assert o.loc[pd.Timestamp("2025-03-07 05:00"), "id_evento"] == 2
    assert (o["clase"] > 0).sum() == 48


def test_filas_utiles_quitan_eventos_en_curso(horas):
    cat = catalogo((1, 1, "2025-03-05 00:00"))
    o = A.objetivo(horas, cat)
    h = pd.DataFrame({"evento_red": 0}, index=horas)
    h.loc["2025-03-02":"2025-03-03", "evento_red"] = 3           # un evento anterior todavía visible
    h.loc["2025-03-04 20:00":"2025-03-04 23:00", "evento_red"] = 1   # (se cruza con el que viene)
    u = A.filas_utiles(h, o)
    assert not u[h.index.get_loc(pd.Timestamp("2025-03-02 10:00"))]
    assert u[h.index.get_loc(pd.Timestamp("2025-03-04 21:00"))]     # hay uno por empezar: cuenta
    assert u[h.index.get_loc(pd.Timestamp("2025-03-08 00:00"))]


@pytest.fixture
def h():
    t = red_sintetica(dias=12)
    t.loc[t.index[600:700], "evento_red"] = 2
    return A.horario(calidad.lecturas_operativas(t))


def test_tabla_sin_informacion_del_futuro(h):
    T, tip = A.tabla(h)
    corte = h.index[150]
    T2, _ = A.tabla(h.loc[:corte], tipicos=tip)
    pd.testing.assert_frame_equal(T.loc[:corte].iloc[-50:], T2.iloc[-50:], check_freq=False)


def test_tabla_tendencia_de_la_fuente(h):
    h2 = h.copy()
    h2["CL2_J280"] = h2["CL2_J280"] - np.linspace(0, 0.3, len(h2))     # el cloro va bajando
    T, _ = A.tabla(h, tipicos=None)
    T2, _ = A.tabla(h2, tipicos=A.tipicos_por_hora(h))
    dif = (T2["fuente_CL2_tendencia"] - T["fuente_CL2_tendencia"]).dropna()
    assert dif.max() < 0 and dif.mean() == pytest.approx(-0.3 * 18 / (len(h) - 1), rel=0.1)
    assert T2["fuente_CL2_cambio24h"].iloc[-1] < -0.02


def test_lluvia_de_30_dias_desde_afuera(h):
    T, tip = A.tabla(h)
    T2, _ = A.tabla(h.iloc[-60:], tipicos=tip, lluvia_mes=A.lluvia_30d(h["lluvia_mm_h"]))
    pd.testing.assert_series_equal(T["lluvia_30d"].iloc[-30:], T2["lluvia_30d"].iloc[-30:], check_freq=False)


def test_tabla_lluvia_y_pronostico(h):
    pron = pd.DataFrame({c: 1.0 for c in ["lluvia_pron_0_6h", "lluvia_pron_6_12h", "lluvia_pron_12_24h"]},
                        index=h.index)
    T, _ = A.tabla(h, pronostico=pron)
    assert (T["lluvia_pron_24h"] == 3.0).all()
    assert T["lluvia_6h"].max() == pytest.approx(h["lluvia_mm_h"].rolling(6, min_periods=1).sum().max())
    assert T["red_sensores_presion_baja"].max() == 0          # la red sintética tiene presión de sobra


def test_avisos_con_confirmacion():
    p = np.array([0.1, 0.9, 0.2, 0.9, 0.9, 0.9, 0.1])
    assert A.avisos(p, 0.5).tolist() == [False, True, False, True, True, True, False]
    assert A.avisos(p, 0.5, 2).tolist() == [False, False, False, False, True, True, False]


def test_evaluar_anticipacion_y_falsos(horas):
    cat = catalogo((1, 3, "2025-03-04 12:00"), (2, 1, "2025-03-08 00:00"))
    aviso = np.zeros(len(horas), bool)
    aviso[horas.get_indexer(pd.date_range("2025-03-04 02:00", "2025-03-04 11:00", freq="1h"))] = True  # 10 h antes
    aviso[horas.get_indexer(pd.date_range("2025-03-06 00:00", "2025-03-06 02:00", freq="1h"))] = True  # nada después
    utiles = np.ones(len(horas), bool)
    tipo = np.where(aviso, 3, 0)
    ev, av, r = A.evaluar(horas, aviso, cat, utiles, horas[0], horas[-1] + pd.Timedelta(hours=1), tipo=tipo)
    assert ev.set_index("id").loc[1, "anticipado"] and ev.set_index("id").loc[1, "anticipacion_h"] == 10
    assert ev.set_index("id").loc[1, "tipo_sugerido"] == "turbidez_fuente"
    assert not ev.set_index("id").loc[2, "anticipado"]
    assert r["avisos"] == 2 and r["falsos"] == 1 and r["anticipados"] == 1


def test_modelos_aprenden_una_causa_simple():
    """Si la lluvia anuncia los eventos de turbidez, el modelo de ese tipo lo aprende."""
    rng = np.random.default_rng(0)
    idx = pd.date_range("2025-01-01", periods=3000, freq="1h")
    lluvia = np.zeros(len(idx))
    inicios = []
    for k in range(100, 2900, 120):
        lluvia[k:k + 3] = 20
        inicios.append(idx[k + 6])
    X = pd.DataFrame({"lluvia_6h": pd.Series(lluvia).rolling(6, min_periods=1).sum().values,
                      "ruido": rng.normal(size=len(idx))}, index=idx)
    cat = catalogo(*[(i + 1, 3, t) for i, t in enumerate(inicios)])
    y = A.objetivo(idx, cat, horizonte=12)["clase"].values
    for entrenar in (A.entrenar_arboles, A.entrenar_bosque):
        P = A.probabilidades(entrenar(X, y), X)
        assert list(P.columns) == A.CODIGOS
        r = A.metricas_hora(y == 3, P[3].values, 0.5, np.ones(len(idx), bool))
        assert r["auc_roc"] > 0.8
        assert P[1].max() < 0.5                   # tipos sin ejemplos: probabilidad baja
    assert np.all(A.probabilidad_alguno(P) >= P.max(axis=1).values - 1e-12)
    iso = A.calibrar_isolation(A.entrenar_isolation(X[y == 0]), X[y == 0])
    P_iso = A.probabilidades(iso, X)
    assert list(P_iso.columns) == [0] and 0 <= P_iso[0].min() and P_iso[0].max() <= 1


def test_avisos_por_tipo():
    idx = pd.date_range("2025-01-01", periods=5, freq="1h")
    P = pd.DataFrame({1: [0.1, 0.6, 0.6, 0.1, 0.1], 3: [0.1, 0.9, 0.2, 0.2, 0.95]}, index=idx)
    aviso, tipo = A.avisos_por_tipo(P, {1: (0.5, 1), 3: (0.8, 1)})
    assert aviso.tolist() == [False, True, True, False, True]
    assert tipo.tolist() == [0, 1, 1, 0, 3]        # en la hora 1, 0.6/0.5 > 0.9/0.8


def test_elegir_umbrales_por_tipo(horas):
    cat = catalogo((1, 3, "2025-03-04 12:00"), (2, 2, "2025-03-08 12:00"))
    P = pd.DataFrame({3: 0.2, 2: 0.2}, index=horas)
    P.loc["2025-03-04 00:00":"2025-03-04 11:00", 3] = 0.8
    P.loc["2025-03-08 06:00":"2025-03-08 11:00", 2] = 0.6
    P.loc["2025-03-02 00:00":"2025-03-02 03:00", 2] = 0.9       # falsa alarma fuerte del tipo 2
    eleccion, grilla = A.elegir_umbrales(horas, P, cat, np.ones(len(horas), bool),
                                         [(horas[0], horas[-1] + pd.Timedelta(hours=1))],
                                         [0.1, 0.5, 0.95], [1, 2], 0.6)
    assert eleccion[3] == (0.5, 1)
    assert eleccion[2] == (0.95, 1)        # con 0.5 anticipa, pero se pasa de falsos: se sacrifica
    assert len(grilla) == 12
    # evaluar un tipo: el aviso del tipo 3 antes del evento 2 no es falso si cuenta todo el catálogo
    a3 = A.avisos(P[3].values, 0.5)
    _, _, r = A.evaluar(horas, a3, cat[cat["codigo"] == 3], np.ones(len(horas), bool), horas[0],
                        horas[-1] + pd.Timedelta(hours=1), catalogo_falsos=cat)
    assert r["anticipados"] == 1 and r["falsos"] == 0


def test_secuencias():
    X = np.arange(12, dtype=float).reshape(6, 2)
    s = A.secuencias(X, 3)
    assert s.shape == (6, 3, 2)
    assert s[5].tolist() == [[6, 7], [8, 9], [10, 11]]
    assert s[0].tolist() == [[0, 0], [0, 0], [0, 1]]


def test_retrasos_propagacion():
    idx = pd.date_range("2025-03-01", periods=200, freq="1h")
    S = ["J280", "J10", "J20"]
    E = pd.DataFrame(0, index=idx, columns=[f"evento_{n}" for n in S])
    cat = catalogo((1, 3, "2025-03-02 00:00"), (2, 2, "2025-03-06 00:00"), (3, 1, "2025-03-08 00:00"))
    for ini, cod, r10 in [("2025-03-02 00:00", 3, 4), ("2025-03-06 00:00", 2, 6)]:
        t0 = pd.Timestamp(ini)
        E.loc[t0:t0 + pd.Timedelta(hours=20), "evento_J280"] = cod
        E.loc[t0 + pd.Timedelta(hours=r10):t0 + pd.Timedelta(hours=30), "evento_J10"] = cod
    E.loc["2025-03-08 02:00":"2025-03-08 05:00", "evento_J20"] = 1     # contaminación: no es de la fuente
    r = A.retrasos_propagacion(E, cat, S)
    assert r.loc["J280", "mediana"] == 0 and r.loc["J10", "mediana"] == 5 and r.loc["J10", "eventos"] == 2
    assert np.isnan(r.loc["J20", "mediana"])
    llegada = A.llegada_estimada(r, "2025-04-01 10:00")
    assert llegada["J10"] == pd.Timestamp("2025-04-01 15:00") and "J20" not in llegada
