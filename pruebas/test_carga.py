"""Lectura del dataset (carga.py) con una carpeta de datos pequeña creada en la prueba."""

import json

import pandas as pd
import pytest

from calidad_agua import carga
from conftest import SENSORES, red_sintetica

FECHAS = {"entrenamiento": ("2025-03-01", "2025-03-09"), "validacion": ("2025-03-09", "2025-03-13"),
          "prueba": ("2025-03-13", "2025-03-17")}


@pytest.fixture
def carpeta(tmp_path, monkeypatch):
    for p, (ini, fin) in FECHAS.items():
        dias = (pd.Timestamp(fin) - pd.Timestamp(ini)).days
        t = red_sintetica(dias=dias, inicio=ini).drop(columns="particion")
        t.to_parquet(tmp_path / f"{p}.parquet")
        (t[[c for c in t.columns if c.startswith(("CL2_", "TURB_"))]] * 0 + 1).to_parquet(
            tmp_path / f"verdad_{p}.parquet")
    (tmp_path / "metadatos.json").write_text(
        json.dumps({"particiones": {p: list(f) for p, f in FECHAS.items()}}), encoding="utf-8")
    pd.DataFrame({"nodo": SENSORES}).to_csv(tmp_path / "sensores.csv", index=False)
    pd.DataFrame({"inicio": ["2025-03-02"], "fin": ["2025-03-03"], "falla": ["deriva"]}).to_csv(
        tmp_path / "fallas_sensor_catalogo.csv", index=False)
    monkeypatch.setenv("DATOS_SENSORES", str(tmp_path))
    return tmp_path


def test_sin_datos_da_un_error_que_explica_que_hacer(tmp_path, monkeypatch):
    monkeypatch.setenv("DATOS_SENSORES", str(tmp_path / "no_existe"))
    with pytest.raises(FileNotFoundError, match="DATOS_SENSORES"):
        carga.carpeta_datos()


def test_cargar_particiones_y_verdad(carpeta):
    t = carga.cargar("validacion")
    assert t.index[0] == pd.Timestamp("2025-03-09")
    assert (carga.cargar("prueba", verdad=True) == 1).all().all()
    todo = carga.cargar_todo()
    assert list(todo["particion"].unique()) == ["entrenamiento", "validacion", "prueba"]
    assert todo.index.is_monotonic_increasing


def test_periodos_y_texto_desde_metadatos(carpeta):
    p = carga.periodos()
    assert p["prueba"] == (pd.Timestamp("2025-03-13"), pd.Timestamp("2025-03-17"))
    assert carga.texto_periodo("entrenamiento") == "mar – mar 2025"


def test_texto_periodo_en_dos_anios(carpeta):
    meta = {"particiones": {"entrenamiento": ["2024-09-01", "2026-01-01"], "validacion": ["2026-01-01", "2026-05-01"],
                            "prueba": ["2026-05-01", "2026-09-01"]}}
    (carpeta / "metadatos.json").write_text(json.dumps(meta), encoding="utf-8")
    assert carga.texto_periodo("entrenamiento") == "sep 2024 – dic 2025"
    assert carga.texto_periodo("prueba") == "may – ago 2026"


def test_metadatos_escritos_en_cp1252(carpeta):
    meta = {"particiones": {p: list(f) for p, f in FECHAS.items()}, "red": "Pequeña"}
    (carpeta / "metadatos.json").write_bytes(json.dumps(meta, ensure_ascii=False).encode("cp1252"))
    assert carga.metadatos()["red"] == "Pequeña"


def test_tablas_auxiliares(carpeta):
    assert list(carga.tabla_sensores()["nodo"]) == SENSORES
    assert carga.catalogo_fallas()["inicio"].dtype.kind == "M"


def test_sensores_columnas_y_formato_largo(red):
    assert carga.sensores(red) == SENSORES                    # en el orden del dataset
    assert carga.columnas(red, "PH") == [f"PH_{n}" for n in SENSORES]
    red.loc[red.index[0], "evento_J10"] = 1
    larga = carga.larga(red)
    assert len(larga) == len(red) * len(SENSORES)
    assert set(larga.columns) >= {"CL2", "TEMP", "TURB", "PH", "P", "sensor", "evento"}
    assert larga.loc[larga["sensor"] == "J10", "evento"].iloc[0] == "contaminacion"


def test_filtrar_sensores_y_evento_visible(red, monkeypatch):
    t = red.copy()
    t.loc[t.index[10:20], "evento_J20"] = 5
    t["evento_visible"] = t[["evento_J280", "evento_J10", "evento_J20"]].max(axis=1)
    f = carga.filtrar_sensores(t, ["J280", "J10"])
    assert carga.sensores(f) == ["J280", "J10"]
    assert "falla_CL2_J20" not in f and "evento_J20" not in f and "P_J20" not in f
    assert {"Q_R1", "lluvia_mm_h", "evento_red", "falla_CL2_J10"} <= set(f.columns)
    assert f["evento_visible"].max() == 0              # el evento solo lo veía J20
    with pytest.raises(ValueError):
        carga.filtrar_sensores(t, ["J280", "J99"])
    monkeypatch.setenv("SENSORES", "J280, J20")
    assert carga.sensores_sistema() == ["J280", "J20"]
    monkeypatch.delenv("SENSORES")
    assert carga.sensores_sistema() == list(carga.configuracion.seccion("red")["sensores"])
