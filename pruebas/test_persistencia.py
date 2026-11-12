"""persistencia.guardar: los modelos se abren aunque joblib se importe después de dill (lo importa TensorFlow)."""

import functools
import pickle

import joblib
import pytest
from joblib import numpy_pickle
from sklearn.preprocessing import FunctionTransformer
from sklearn.utils.validation import check_array

from calidad_agua import persistencia

dill = pytest.importorskip("dill")


@pytest.fixture
def joblib_despues_de_dill(monkeypatch):
    """Como en el notebook 08: la tabla que copió joblib ya trae las funciones de dill."""
    dill.extend(True)
    monkeypatch.setattr(numpy_pickle.NumpyPickler, "dispatch", dict(pickle._Pickler.dispatch))


def objeto():
    # lo mismo que lleva dentro HistGradientBoosting: un FunctionTransformer con un functools.partial
    return {"t": FunctionTransformer(functools.partial(check_array, ensure_all_finite=False))}


def test_joblib_solo_queda_dependiendo_de_dill(tmp_path, joblib_despues_de_dill):
    joblib.dump(objeto(), tmp_path / "con_dill.joblib", compress=3)
    assert not persistencia.se_abre_sin_dill(tmp_path / "con_dill.joblib")


def test_guardar_se_abre_sin_dill_y_deja_joblib_como_estaba(tmp_path, joblib_despues_de_dill):
    antes = numpy_pickle.NumpyPickler.dispatch
    ruta = persistencia.guardar(objeto(), tmp_path / "modelo.joblib")
    assert persistencia.se_abre_sin_dill(ruta)
    assert joblib.load(ruta)["t"].func.keywords == {"ensure_all_finite": False}
    assert numpy_pickle.NumpyPickler.dispatch is antes
