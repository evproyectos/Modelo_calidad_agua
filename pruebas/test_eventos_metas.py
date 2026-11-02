"""Episodios de eventos (eventos.py) y criterio de acierto de la meta (metas.py)."""

import numpy as np
import pandas as pd

from calidad_agua import metas
from calidad_agua.eventos import episodios


def test_episodios_continuos_con_su_tipo_y_duracion():
    idx = pd.date_range("2025-03-01", periods=12, freq="15min")
    s = pd.Series([0, 1, 1, 1, 0, 0, 3, 3, 1, 0, 0, 2], index=idx)
    e = episodios(s)
    assert e["codigo"].tolist() == [1, 3, 1, 2]
    assert e["tipo"].tolist() == ["contaminacion", "turbidez_fuente", "contaminacion", "falla_cloro"]
    assert e.iloc[0]["inicio"] == idx[1] and e.iloc[0]["fin"] == idx[3]
    assert e["horas"].tolist() == [0.75, 0.5, 0.25, 0.25]
    assert episodios(pd.Series(0, index=idx)).empty


def test_tolerancia_absoluta_y_relativa():
    assert metas.tolerancia("CL2", [0.5, 0.9]).tolist() == [0.1, 0.1]
    assert np.allclose(metas.tolerancia("TURB", [0.5, 10.0, -3.0]), [0.2, 2.0, 0.6])


def test_acierto_dentro_de_la_tolerancia_y_nan_si_falta_un_dato():
    obs = [0.50, 0.50, np.nan, 10.0]
    pron = [0.59, 0.62, 0.5, 11.9]
    assert metas.acierto("CL2", obs[:3], pron[:3])[:2].tolist() == [1.0, 0.0]
    assert np.isnan(metas.acierto("CL2", obs[:3], pron[:3])[2])
    assert metas.acierto("TURB", [10.0], [11.9])[0] == 1.0     # ± 20 % de 10 = ± 2


def test_texto_de_las_tolerancias():
    assert metas.texto_tolerancia("PH") == "± 0.1"
    assert metas.texto_tolerancia("TURB") == "± 20% (mín. ± 0.2)"
