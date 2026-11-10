"""
Dónde van los sensores del sistema (notebook 05, sección de ubicación).

El sistema tiene 10 puntos de medición (hiperparametros.yaml, red.cantidad): la salida de la
fuente (J280, con el caudalímetro), que es obligatoria, y 9 puntos de la red de distribución.
Mientras la ASADA no defina dónde van, se eligen con los datos (mejor_conjunto): entre los
sensores que exporta el simulador, uno por uno, el que más eventos nuevos deja ver del
entrenamiento y la validación (la prueba no se usa para decidir); a igualdad, el que ve más fugas
y contaminaciones y el que las ve antes. Los eventos de la fuente (cloro, turbidez, pH) los ve
siempre J280; lo que decide son las contaminaciones y las fugas, que pasan dentro de la red.
Cuando la ASADA dé la lista, se escribe en red.sensores y el notebook solo evalúa esa lista.
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


def _puntaje(V, tipo, nodos):
    r = V[nodos].min(axis=1)
    vistos = r.notna()
    return (int(vistos.sum()), int(vistos[tipo == "fuga"].sum()), int(vistos[tipo == "contaminacion"].sum()),
            -float(r.median()) if vistos.any() else -np.inf)


def mejor_conjunto(V, catalogo, n=10, fuente=FUENTE, candidatos=None):
    """
    Elección voraz de `n` sensores (la fuente y n - 1 más): en cada paso se agrega el que deja ver
    más eventos; a igualdad, más fugas, más contaminaciones y menos retraso mediano. Devuelve la
    lista elegida y una tabla con lo que aportó cada uno.
    """
    candidatos = [c for c in (candidatos or V.columns) if c != fuente]
    tipo = catalogo.set_index("id").loc[V.index, "tipo"]
    elegidos, filas = [fuente], []
    p = _puntaje(V, tipo, elegidos)
    filas.append({"sensor": fuente, "vistos": p[0], "fugas": p[1], "contaminaciones": p[2],
                  "retraso_mediano_h": -p[3], "motivo": "salida de la fuente (obligatorio)"})
    while len(elegidos) < n and candidatos:
        mejor = max(candidatos, key=lambda c: _puntaje(V, tipo, elegidos + [c]))
        p = _puntaje(V, tipo, elegidos + [mejor])
        elegidos.append(mejor)
        candidatos.remove(mejor)
        filas.append({"sensor": mejor, "vistos": p[0], "fugas": p[1], "contaminaciones": p[2],
                      "retraso_mediano_h": -p[3], "motivo": "el que más agrega"})
    tabla = pd.DataFrame(filas)
    tabla["eventos_nuevos"] = tabla["vistos"].diff().fillna(tabla["vistos"]).astype(int)
    return elegidos, tabla
