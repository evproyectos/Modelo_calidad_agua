"""Hiperparámetros desde el YAML (configuracion.py) y registro en MLflow (experimentos.py)."""

import json

import matplotlib
import numpy as np
import pandas as pd
import pytest

from calidad_agua import configuracion, detector, experimentos, modelo
from conftest import red_sintetica

matplotlib.use("Agg")


def test_yaml_trae_los_mismos_hiperparametros_que_se_usaron_al_entrenar():
    # cambiar un valor obliga a reentrenar: esta prueba avisa si cambió sin querer
    assert modelo.HIPERPARAMETROS == dict(loss="absolute_error", max_iter=400, learning_rate=0.05, max_leaf_nodes=31,
                                          min_samples_leaf=50, l2_regularization=1.0, early_stopping=True,
                                          validation_fraction=0.1, random_state=0)
    assert detector.MODELO == "bosque"
    assert detector.HIPERPARAMETROS["bosque"] == dict(n_estimators=200, min_samples_leaf=20, max_features="sqrt",
                                                      class_weight="balanced_subsample", n_jobs=-1, random_state=0)
    assert detector.HIPERPARAMETROS["arboles"] == dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31,
                                                       min_samples_leaf=40, l2_regularization=1.0,
                                                       class_weight="balanced", early_stopping=True,
                                                       validation_fraction=0.1, random_state=0)
    lstm = configuracion.seccion("lstm")
    assert lstm["tasa_aprendizaje"] == 3e-4 and lstm["semillas"] == [0, 1, 2] and lstm["unidades"] == 48
    alarma = configuracion.seccion("eleccion_alarma")
    assert alarma["umbrales"] == [0.5, 0.6, 0.7, 0.8, 0.9, 0.95] and alarma["confirmaciones_h"] == [1, 2, 3]


def test_cada_seccion_es_una_copia():
    s = configuracion.seccion("lstm")
    s["unidades"] = 1
    assert configuracion.seccion("lstm")["unidades"] == 48


def test_otro_archivo_de_hiperparametros(tmp_path):
    f = tmp_path / "h.yaml"
    f.write_text("lstm:\n  unidades: 16\n", encoding="utf-8")
    assert configuracion.cargar(f) == {"lstm": {"unidades": 16}}


def test_nombres_validos_para_mlflow():
    assert experimentos.clave("% acierto (normal)") == "pct_acierto_normal"
    assert experimentos.clave("Validación ene–abr 2026") == "validacion_ene-abr_2026"
    assert experimentos.aplanar({"Árboles": {"CL2": 1, "lista": [1, 2]}}) == {"arboles.cl2": 1, "arboles.lista": [1, 2]}


def test_huella_de_los_datos(tmp_path):
    a, b = tmp_path / "a.parquet", tmp_path / "b.json"
    a.write_bytes(b"datos")
    b.write_bytes(b"{}")
    total, por_archivo = experimentos.huella_datos([a, b])
    assert por_archivo["a.parquet"] == experimentos.sha256(a) and len(total) == 64
    assert experimentos.huella_datos([b, a])[0] == total                # no depende del orden
    a.write_bytes(b"datos cambiados")
    assert experimentos.huella_datos([a, b])[0] != total


@pytest.fixture
def datos(tmp_path, monkeypatch):
    """Carpeta de datos mínima para que la corrida calcule la huella y los periodos."""
    for p, (ini, fin) in {"entrenamiento": ("2025-03-01", "2025-03-05"), "validacion": ("2025-03-05", "2025-03-07"),
                          "prueba": ("2025-03-07", "2025-03-09")}.items():
        red_sintetica(dias=1, inicio=ini).to_parquet(tmp_path / f"{p}.parquet")
    meta = {"particiones": {"entrenamiento": ["2025-03-01", "2025-03-05"], "validacion": ["2025-03-05", "2025-03-07"],
                            "prueba": ["2025-03-07", "2025-03-09"]}}
    (tmp_path / "metadatos.json").write_text(json.dumps(meta), encoding="utf-8")
    (tmp_path / "pron.csv").write_text("a\n", encoding="utf-8")
    monkeypatch.setenv("DATOS_SENSORES", str(tmp_path))
    monkeypatch.setenv("PRONOSTICO_LLUVIA", str(tmp_path / "pron.csv"))
    return tmp_path


def test_corrida_completa_en_mlflow(datos, tmp_path):
    import matplotlib.pyplot as plt
    import mlflow
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    modelo_ = tmp_path / "modelo.joblib"
    modelo_.write_bytes(b"modelo")
    with experimentos.corrida("prueba", "corrida de prueba", "test", uri=uri) as reg:
        reg.parametros({"variables": 90, "particiones": ["entrenamiento", "validacion"]})
        reg.metricas({"% acierto": {"CL2": 87.6, "TURB": float("nan")}, "texto": "no es número"}, "prueba")
        reg.tabla(pd.DataFrame({"x": [1, 2]}), "Resultados en prueba")
        fig, ax = plt.subplots()
        ax.plot([1, 2])
        reg.figura(fig, "curva")
        plt.close(fig)
        reg.archivo(modelo_)
        with reg.hija("variante") as hija:
            hija.metricas({"acierto": 80})
        ref = reg.referencia()
    assert ref["experimento"] == "prueba" and ref["dataset_sha256"] == experimentos.huella_datos()[0]
    cliente = mlflow.MlflowClient(uri)
    run = cliente.get_run(ref["run_id"])
    assert run.data.metrics == {"prueba.pct_acierto.cl2": 87.6}
    assert run.data.params["variables"] == "90" and run.data.params["particiones"] == "entrenamiento, validacion"
    assert run.data.params["hiperparametros.lstm.unidades"] == "48"
    assert run.data.params["periodo.prueba"] == "2025-03-07 a 2025-03-09"
    assert run.data.tags["notebook"] == "test" and run.data.tags["dataset_sha256"] == ref["dataset_sha256"]
    assert "sha256.entrenamiento.parquet" in run.data.tags and "version.scikit-learn" in run.data.tags
    artefactos = {a.path for carpeta in ["tablas", "figuras", "modelos"]
                  for a in cliente.list_artifacts(run.info.run_id, carpeta)}
    assert artefactos == {"tablas/resultados_en_prueba.csv", "figuras/curva.png", "modelos/modelo.joblib"}
    hijas = cliente.search_runs([run.info.experiment_id], f"tags.mlflow.parentRunId = '{run.info.run_id}'")
    assert len(hijas) == 1 and hijas[0].data.metrics["acierto"] == 80
    assert np.isfinite(run.info.start_time)


def test_corrida_abierta_en_un_notebook_y_tabla_de_resultados(datos, tmp_path):
    import mlflow
    uri = f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}"
    reg = experimentos.iniciar("prueba", "notebook", "06", uri=uri)
    otra = experimentos.iniciar("prueba", "se volvió a correr la primera celda", "06", uri=uri)
    tabla = pd.DataFrame({"% acierto modelo": [87.6, 40.0]},
                         index=pd.MultiIndex.from_tuples([("CL2", "normal"), ("CL2", "evento")]))
    otra.metricas_tabla(tabla, "prueba")
    otra.figuras_guardadas("no_hay_ninguna_")
    otra.terminar()
    cliente = mlflow.MlflowClient(uri)
    assert cliente.get_run(reg.id).info.status == "KILLED"
    r = cliente.get_run(otra.id)
    assert r.info.status == "FINISHED" and r.data.metrics == {"prueba.cl2.normal.pct_acierto_modelo": 87.6,
                                                              "prueba.cl2.evento.pct_acierto_modelo": 40.0}
    with pytest.raises(ValueError):
        with experimentos.corrida("prueba", "falla", "06", uri=uri) as fallida:
            raise ValueError("error en una celda")
    assert cliente.get_run(fallida.id).info.status == "FAILED"


def test_registro_propio_en_la_carpeta_del_proyecto(datos, tmp_path, monkeypatch):
    """Sin servidor configurado, base y archivos quedan en el proyecto (mlflow.db y mlartifacts/)."""
    import mlflow
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    monkeypatch.setattr(experimentos, "BASE", tmp_path / "mlflow.db")
    monkeypatch.setattr(experimentos, "ARTEFACTOS", tmp_path / "mlartifacts")
    with experimentos.corrida("propio", "corrida", "test") as reg:
        reg.metricas({"x": 1})
    exp = mlflow.MlflowClient(f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}").get_experiment_by_name("propio")
    assert exp.artifact_location == (tmp_path / "mlartifacts" / "propio").as_uri()
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)        # set_tracking_uri la define
