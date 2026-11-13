"""calidad_agua/fuentes_lluvia.py (RNF-009): la lluvia con que se entrena, según la fuente elegida."""

import pandas as pd
import pytest

from calidad_agua import fuentes_lluvia as fl

CSV = ("latitude,longitude,elevation,utc_offset_seconds,timezone,timezone_abbreviation\n"
       "10.36,-84.53,166.0,-21600,America/Costa_Rica,GMT-6\n\n"
       "time,precipitation_ecmwf_ifs025 (mm),precipitation_gfs_seamless (mm)\n"
       "2026-05-01T00:00,1.0,3.0\n2026-05-01T01:00,2.0,0.0\n")


@pytest.fixture
def archivo(tmp_path, monkeypatch):
    ruta = tmp_path / "lluvia.csv"
    ruta.write_text(CSV)
    monkeypatch.setattr(fl, "ARCHIVO", ruta)
    fl._serie.cache_clear()
    yield ruta
    fl._serie.cache_clear()


def lecturas():
    i = pd.date_range("2026-05-01 00:00", "2026-05-01 01:45", freq="15min")
    return pd.DataFrame({"CL2_J10": 0.7, "lluvia_mm_h": 9.0}, index=i)


def test_fuente_por_configuracion_y_entorno(monkeypatch):
    monkeypatch.delenv("FUENTE_LLUVIA", raising=False)
    monkeypatch.delenv("LLUVIA_MODELOS", raising=False)
    assert fl.descripcion() == {"fuente": "open-meteo", "modelos": ["ecmwf_ifs025"]}   # hiperparametros.yaml
    monkeypatch.setenv("FUENTE_LLUVIA", "lecturas")
    assert fl.descripcion() == {"fuente": "lecturas", "modelos": []}
    monkeypatch.setenv("FUENTE_LLUVIA", "imn")
    with pytest.raises(ValueError, match="lecturas, open-meteo"):
        fl.elegida()


def test_aplicar_open_meteo(archivo, monkeypatch):
    monkeypatch.setenv("FUENTE_LLUVIA", "open-meteo")
    monkeypatch.setenv("LLUVIA_MODELOS", "ecmwf_ifs025,gfs_seamless")
    original = lecturas()
    t = fl.aplicar(original)
    assert list(t["lluvia_mm_h"]) == [2.0] * 4 + [1.0] * 4      # la hora que terminó: (1+3)/2 y (2+0)/2
    assert (original["lluvia_mm_h"] == 9.0).all() and (t["CL2_J10"] == 0.7).all()


def test_lecturas_no_cambia_y_horas_sin_dato(archivo, monkeypatch):
    t = lecturas()
    assert fl.aplicar(t, "lecturas") is t
    monkeypatch.setenv("LLUVIA_MODELOS", "ecmwf_ifs025")
    tarde = t.copy()
    tarde.index = tarde.index + pd.Timedelta(days=1)          # fuera del archivo
    with pytest.warns(UserWarning, match="no cubre 8 lecturas"):
        assert fl.aplicar(tarde, "open-meteo")["lluvia_mm_h"].isna().all()


def test_modelo_que_no_esta_en_el_archivo(archivo, monkeypatch):
    monkeypatch.setenv("LLUVIA_MODELOS", "icon_seamless")
    with pytest.raises(ValueError, match="icon_seamless"):
        fl.aplicar(lecturas(), "open-meteo")
