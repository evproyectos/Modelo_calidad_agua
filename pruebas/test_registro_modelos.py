"""Versiones de los modelos en el Model Registry de MLflow (registro_modelos.py, RF-004.2)."""

import json

import joblib
import pytest

from calidad_agua import experimentos, registro_modelos as R
from conftest import red_sintetica


@pytest.fixture
def entorno(tmp_path, monkeypatch):
    """Datos mínimos (para la huella de la corrida), un MLflow propio y una carpeta de modelos."""
    datos = tmp_path / "datos"
    datos.mkdir()
    periodos = {"entrenamiento": ("2025-03-01", "2025-03-05"), "validacion": ("2025-03-05", "2025-03-07"),
                "prueba": ("2025-03-07", "2025-03-09")}
    for p, (ini, _) in periodos.items():
        red_sintetica(dias=1, inicio=ini).to_parquet(datos / f"{p}.parquet")
    (datos / "metadatos.json").write_text(json.dumps({"particiones": periodos}), encoding="utf-8")
    (datos / "pron.csv").write_text("a\n", encoding="utf-8")
    monkeypatch.setenv("DATOS_SENSORES", str(datos))
    monkeypatch.setenv("PRONOSTICO_LLUVIA", str(datos / "pron.csv"))
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    mlflow = experimentos.configurar(uri)
    for exp in ("detector_eventos", "fugas", "anticipacion_eventos"):
        mlflow.create_experiment(exp, artifact_location=(tmp_path / "mlartifacts" / exp).as_uri())
    carpeta = tmp_path / "modelos"
    carpeta.mkdir()
    return uri, carpeta


def entrenar(uri, carpeta, experimento, archivo, contenido, redes=()):
    """Lo que hace un notebook: una corrida, el .joblib con su referencia y los archivos subidos."""
    reg = experimentos.iniciar(experimento, "prueba", "test", uri=uri)
    for r in redes:
        (carpeta / r).write_bytes(f"red {r} {contenido}".encode())
        reg.archivo(carpeta / r)
    joblib.dump({**contenido, "mlflow": reg.referencia()}, carpeta / archivo)
    reg.archivo(carpeta / archivo)
    reg.terminar()
    return reg.id


def test_registra_la_version_1_con_alias_y_no_la_repite(entorno):
    uri, carpeta = entorno
    run = entrenar(uri, carpeta, "detector_eventos", "detector_eventos.joblib", {"umbral": 0.9})
    r = R.registrar(carpeta, uri)
    assert list(r) == ["detector_eventos"]                      # los demás no están en la carpeta
    assert r["detector_eventos"]["version"] == 1 and r["detector_eventos"]["nueva"]
    assert r["detector_eventos"]["run_id"] == run
    cliente = experimentos.configurar(uri).MlflowClient()
    v = cliente.get_model_version_by_alias("detector_eventos", R.ALIAS)
    assert str(v.version) == "1" and v.run_id == run
    assert v.tags["sha256.detector_eventos.joblib"] == r["detector_eventos"]["archivos"]["detector_eventos.joblib"]
    assert v.tags["notebook"] == "test" and v.tags["dataset_sha256"]
    resumen = json.loads((carpeta / R.VERSIONES).read_text(encoding="utf-8"))
    assert resumen["modelos"]["detector_eventos"]["version"] == 1
    otra = R.registrar(carpeta, uri)                             # sin reentrenar: misma versión
    assert otra["detector_eventos"]["version"] == 1 and not otra["detector_eventos"]["nueva"]
    assert len(cliente.search_model_versions("name = 'detector_eventos'")) == 1


def test_solo_sube_la_version_del_modelo_reentrenado(entorno):
    uri, carpeta = entorno
    entrenar(uri, carpeta, "detector_eventos", "detector_eventos.joblib", {"umbral": 0.9})
    entrenar(uri, carpeta, "fugas", "fugas.joblib", {"nombre": "SVM", "umbral": 0.5})
    R.registrar(carpeta, uri)
    entrenar(uri, carpeta, "fugas", "fugas.joblib", {"nombre": "SVM", "umbral": 0.6})
    r = R.registrar(carpeta, uri)
    assert (r["fugas"]["version"], r["fugas"]["nueva"]) == (2, True)
    assert (r["detector_eventos"]["version"], r["detector_eventos"]["nueva"]) == (1, False)
    filas = {(f["modelo"], f["version"]): f for f in R.lista(uri)}
    assert filas[("fugas", 2)]["en_produccion"] and not filas[("fugas", 1)]["en_produccion"]
    assert filas[("fugas", 2)]["algoritmo"] == "SVM"


def test_restaurar_una_version_anterior(entorno):
    uri, carpeta = entorno
    entrenar(uri, carpeta, "fugas", "fugas.joblib", {"nombre": "SVM", "umbral": 0.5})
    R.registrar(carpeta, uri)
    original = (carpeta / "fugas.joblib").read_bytes()
    entrenar(uri, carpeta, "fugas", "fugas.joblib", {"nombre": "SVM", "umbral": 0.6})
    R.registrar(carpeta, uri)
    r = R.restaurar("fugas", 1, carpeta, uri)
    assert (carpeta / "fugas.joblib").read_bytes() == original
    assert r["fugas"]["version"] == 1
    cliente = experimentos.configurar(uri).MlflowClient()
    assert str(cliente.get_model_version_by_alias("fugas", R.ALIAS).version) == "1"
    with pytest.raises(R.ErrorRegistro, match="no tiene la versión 7"):
        R.restaurar("fugas", 7, carpeta, uri)
    with pytest.raises(R.ErrorRegistro, match="No hay un modelo"):
        R.restaurar("otro", 1, carpeta, uri)


def test_las_redes_de_keras_van_en_la_misma_version(entorno):
    uri, carpeta = entorno
    redes = [f"anticipacion_lstm_{k}.keras" for k in range(2)]
    entrenar(uri, carpeta, "anticipacion_eventos", "anticipacion_24h.joblib", {"nombre": "LSTM", "redes": 2},
             redes=redes)
    r = R.registrar(carpeta, uri)
    assert list(r["anticipacion_eventos"]["archivos"]) == ["anticipacion_24h.joblib", *redes]
    (carpeta / redes[0]).unlink()
    with pytest.raises(R.ErrorRegistro, match="Faltan archivos"):
        R.registrar(carpeta, uri)


def test_archivo_que_no_es_el_de_su_corrida(entorno):
    uri, carpeta = entorno
    entrenar(uri, carpeta, "detector_eventos", "detector_eventos.joblib", {"umbral": 0.9})
    paquete = joblib.load(carpeta / "detector_eventos.joblib")
    joblib.dump({**paquete, "umbral": 0.1}, carpeta / "detector_eventos.joblib")   # editado a mano
    with pytest.raises(R.ErrorRegistro, match="no es el archivo que guardó la corrida"):
        R.registrar(carpeta, uri)


def test_modelo_sin_referencia_de_mlflow(entorno):
    uri, carpeta = entorno
    joblib.dump({"umbral": 0.9}, carpeta / "detector_eventos.joblib")
    with pytest.raises(R.ErrorRegistro, match="no dice qué corrida"):
        R.registrar(carpeta, uri)


def test_archivos_de_cada_modelo():
    assert R.archivos("pronostico_24h", {"redes": ["a.keras", "b.keras"]}) == [
        "pronostico_24h_final.joblib", "a.keras", "b.keras"]
    assert R.archivos("anticipacion_eventos", {"nombre": "Random Forest"}) == ["anticipacion_24h.joblib"]
    assert R.archivos("anticipacion_eventos", {"nombre": "LSTM", "redes": 3})[1:] == [
        "anticipacion_lstm_0.keras", "anticipacion_lstm_1.keras", "anticipacion_lstm_2.keras"]
    assert R.archivos("fugas", {"nombre": "Autoencoder"}) == ["fugas.joblib", "fugas_autoencoder.keras"]
    assert R.archivos("fugas", {"nombre": "SVM"}) == ["fugas.joblib"]
