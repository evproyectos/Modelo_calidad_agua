"""Utilidades para trabajar con eventos (episodios continuos en las columnas evento_*)."""

import pandas as pd

from src.carga import TIPOS_EVENTO


def episodios(serie):
    """
    Tramos continuos con el mismo código de evento (> 0) en una serie de etiquetas.
    Devuelve una tabla con inicio, fin, tipo y duración en horas.
    """
    s = serie.values
    idx = serie.index
    paso_h = (idx[1] - idx[0]).total_seconds() / 3600
    filas, i = [], 0
    while i < len(s):
        if s[i] > 0:
            j = i
            while j + 1 < len(s) and s[j + 1] == s[i]:
                j += 1
            filas.append({"inicio": idx[i], "fin": idx[j], "codigo": int(s[i]),
                          "tipo": TIPOS_EVENTO[int(s[i])], "horas": (j - i + 1) * paso_h})
            i = j + 1
        else:
            i += 1
    return pd.DataFrame(filas, columns=["inicio", "fin", "codigo", "tipo", "horas"])
