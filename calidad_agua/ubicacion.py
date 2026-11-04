"""
Dónde van los sensores del sistema (notebook 05, sección de ubicación).

El sistema tiene tres puntos de medición, como el informe: la salida de la fuente (J280, con el
caudalímetro), que es obligatoria, y dos puntos de la red de distribución. Esos dos se eligen
con los datos: entre los sensores que exporta el simulador, el par que deja ver más eventos del
entrenamiento y la validación (la prueba no se usa para decidir) y, a igualdad, el que los ve
antes. Los eventos de la fuente (cloro, turbidez, pH) los ve siempre J280; lo que decide son las
contaminaciones y las fugas, que pasan dentro de la red.
"""

from itertools import combinations

import numpy as np
import pandas as pd

FUENTE = "J280"


def visibilidad(t, catalogo, horas_max=48):
    """
    Evento × sensor: horas desde el inicio del evento hasta que el sensor lo ve (evento_<n>
    con el código del evento), dentro de las `horas_max` siguientes al fin; NaN si no lo ve.
    `t`: lecturas cada 15 min con todos los sensores (carga.cargar_todo(todos=True)).
    """
    S = [c[7:] for c in t.columns if c.startswith("evento_") and c not in ("evento_visible", "evento_red")]
    E = t[[f"evento_{n}" for n in S]].set_axis(S, axis=1)
    filas = {}
    for _, ev in catalogo.iterrows():
        v = E.loc[ev["inicio"]: ev["fin"] + pd.Timedelta(hours=horas_max)]
        visto = v == ev["codigo"]
        primera = {n: (visto.index[visto[n].values][0] - ev["inicio"]).total_seconds() / 3600
                   for n in S if visto[n].any()}
        filas[ev["id"]] = {n: primera.get(n, np.nan) for n in S}
    return pd.DataFrame.from_dict(filas, orient="index").rename_axis("id")


def evaluar_conjunto(V, catalogo, nodos):
    """Eventos vistos por al menos uno de `nodos` y retraso hasta el primero que lo ve, por tipo."""
    r = V[nodos].min(axis=1)
    d = catalogo.set_index("id").loc[V.index, ["tipo"]].assign(visto=r.notna().values, retraso_h=r.values)
    return d.groupby("tipo").agg(eventos=("visto", "size"), vistos=("visto", "sum"),
                                 retraso_mediano_h=("retraso_h", "median"))


def mejores_pares(V, catalogo, fuente=FUENTE, candidatos=None):
    """
    Todos los pares de sensores de distribución junto con la fuente, ordenados por eventos
    vistos (más es mejor), después por fugas y contaminaciones vistas y por retraso mediano.
    """
    candidatos = [n for n in (candidatos or V.columns) if n != fuente]
    tipo = catalogo.set_index("id").loc[V.index, "tipo"]
    filas = []
    for a, b in combinations(candidatos, 2):
        r = V[[fuente, a, b]].min(axis=1)
        filas.append({"par": f"{a} y {b}", "a": a, "b": b, "vistos": int(r.notna().sum()),
                      "contaminaciones": int(r[tipo == "contaminacion"].notna().sum()),
                      "fugas": int(r[tipo == "fuga"].notna().sum()),
                      "retraso_mediano_h": float(r.median())})
    out = pd.DataFrame(filas)
    return out.sort_values(["vistos", "fugas", "contaminaciones", "retraso_mediano_h"],
                           ascending=[False, False, False, True]).reset_index(drop=True)
