"""Pronóstico de lluvia (clima.py): lectura del archivo de Open-Meteo y ventanas hacia adelante."""

import numpy as np
import pandas as pd
import pytest

from calidad_agua import clima


def archivo_open_meteo(ruta, horas=48):
    """Mismo formato que la descarga de Open-Meteo: 3 líneas de cabecera y una columna por modelo."""
    idx = pd.date_range("2025-03-01", periods=horas, freq="1h")
    lineas = ["latitude,longitude,elevation", "10.36,-84.51,170.0", ""]
    cab = ["time"] + [f"precipitation_previous_day1_{m} (mm)" for m in clima.MODELOS]
    lineas.append(",".join(cab))
    for k, f in enumerate(idx):
        vals = [1.0, 3.0, -0.5, 2.0] if k == 5 else [0.0, 0.0, 0.0, 0.0]   # un valor negativo
        lineas.append(",".join([f.strftime("%Y-%m-%dT%H:%M")] + [str(v) for v in vals]))
    ruta.write_text("\n".join(lineas), encoding="utf-8")
    return ruta


def test_leer_recorta_negativos_y_nombra_las_columnas_por_modelo(tmp_path):
    t = clima.leer(archivo_open_meteo(tmp_path / "pron.csv"))
    assert list(t.columns) == clima.MODELOS
    assert t.iloc[5].tolist() == [1.0, 3.0, 0.0, 2.0]
    assert isinstance(t.index, pd.DatetimeIndex)


def test_cargar_pronostico_promedia_los_modelos(tmp_path):
    s = clima.cargar_pronostico(archivo_open_meteo(tmp_path / "pron.csv"))
    assert s.dtype == np.float32
    assert s.iloc[5] == pytest.approx(1.5)                      # (1 + 3 + 0 + 2) / 4
    solo_dos = clima.cargar_pronostico(tmp_path / "pron.csv", modelos=clima.MODELOS[:2])
    assert solo_dos.iloc[5] == pytest.approx(2.0)


def test_archivo_inexistente(tmp_path, monkeypatch):
    monkeypatch.setenv("PRONOSTICO_LLUVIA", str(tmp_path / "nada.csv"))
    assert clima.ruta_archivo() == tmp_path / "nada.csv"
    with pytest.raises(FileNotFoundError):
        clima.leer()


def test_ventanas_suman_las_horas_siguientes():
    idx = pd.date_range("2025-03-01", periods=60, freq="1h")
    lluvia = pd.Series(np.arange(60, dtype=float), index=idx)      # la lluvia de la hora k vale k
    v = clima.ventanas(lluvia, idx[:10])
    assert list(v.columns) == clima.COLUMNAS
    # en t = 0: horas 1…6, 7…12 y 13…24
    assert v.iloc[0].tolist() == [sum(range(1, 7)), sum(range(7, 13)), sum(range(13, 25))]
    assert v.iloc[3]["lluvia_pron_0_6h"] == sum(range(4, 10))
    assert v.dtypes.eq(np.float32).all()


def test_ventanas_quedan_vacias_si_falta_una_hora():
    idx = pd.date_range("2025-03-01", periods=40, freq="1h")
    lluvia = pd.Series(1.0, index=idx)
    lluvia.iloc[3] = np.nan
    v = clima.ventanas(lluvia, idx[:5])
    assert np.isnan(v.iloc[0]["lluvia_pron_0_6h"])                  # incluye la hora 3
    assert v.iloc[3]["lluvia_pron_0_6h"] == 6.0                     # horas 4…9, completas
    assert v.iloc[0]["lluvia_pron_6_12h"] == 6.0
    corto = clima.ventanas(lluvia.iloc[:10], idx[:2])
    assert np.isnan(corto.iloc[0]["lluvia_pron_12_24h"])             # el pronóstico no llega tan lejos
