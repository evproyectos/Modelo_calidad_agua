"""
Detector de eventos (scikit-learn, HistGradientBoostingClassifier).

Una fila = un sensor en una hora. El modelo estima la probabilidad de que en esa hora haya un
evento visible en ese sensor, y de qué tipo. Solo usa información disponible hasta esa hora.

Variables (por lo encontrado en el notebook 05):
  - desviación de cada parámetro respecto a lo típico de ese sensor a esa hora del día;
  - cambios recientes y variabilidad de las últimas horas;
  - cuántos parámetros del sensor se desvían a la vez (un evento suele mover varios; una falla
    del sensor, uno solo);
  - cuántos sensores de la red se desvían a la vez y cómo está la fuente (los eventos de la
    fuente se ven en toda la red; una falla del sensor, en uno solo);
  - lluvia reciente (la turbidez sube con la lluvia sin que sea un evento) y hora del día.

Una alarma de red se enciende cuando algún sensor supera el umbral de probabilidad.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

from src import carga
from src.eventos import episodios

PARAMS = ["CL2", "TURB", "PH", "TEMP"]
UMBRAL_DESVIO = {"CL2": 0.1, "TURB": 0.5, "PH": 0.2, "TEMP": 1.0}
FUENTE = "J280"

HIPERPARAMETROS = dict(max_iter=300, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
                       l2_regularization=1.0, class_weight="balanced", early_stopping=True,
                       validation_fraction=0.1, random_state=0)


def horario(t15):
    num = t15[[c for c in t15.columns if c.split("_")[0] in PARAMS + ["P", "Q"] or c == "lluvia_mm_h"]]
    h = num.resample("1h").mean()
    for n in carga.sensores(t15):
        h[f"evento_{n}"] = t15[f"evento_{n}"].resample("1h").max()
        h[f"lecturas_{n}"] = t15[f"CL2_{n}"].notna().resample("1h").sum()
    h["particion"] = t15["particion"].resample("1h").first()
    return h


def tipicos_por_hora(h, S):
    """Mediana de cada sensor y parámetro por hora del día (operación normal, entrenamiento)."""
    entr = (h["particion"] == "entrenamiento").values
    tip = {}
    for n in S:
        normal = entr & (h[f"evento_{n}"] == 0).values
        for p in PARAMS:
            x = h[f"{p}_{n}"][normal]
            tip[p, n] = x.groupby(x.index.hour).median()
    return tip


def tabla(h, tip=None):
    S = [c[4:] for c in h.columns if c.startswith("CL2_")]
    if tip is None:
        tip = tipicos_por_hora(h, S)
    hora = h.index.hour

    # desviaciones respecto a lo típico, en formato ancho (hora × sensor) por parámetro
    D = {p: pd.DataFrame({n: h[f"{p}_{n}"].values - tip[p, n].reindex(hora).values for n in S},
                         index=h.index) for p in PARAMS}
    fuera = {p: (D[p].abs() > UMBRAL_DESVIO[p]) for p in PARAMS}

    comun = pd.DataFrame(index=h.index)
    for p in PARAMS:
        comun[f"red_sensores_desviados_{p}"] = fuera[p].sum(axis=1)
        comun[f"red_desvio_mediano_{p}"] = D[p].median(axis=1)
        comun[f"fuente_desvio_{p}"] = D[p][FUENTE]
        for k in [3, 6, 12]:
            comun[f"fuente_desvio_{p}_hace{k}h"] = D[p][FUENTE].shift(k)
    comun["lluvia_6h"] = h["lluvia_mm_h"].rolling(6, min_periods=1).sum()
    comun["lluvia_24h"] = h["lluvia_mm_h"].rolling(24, min_periods=1).sum()
    comun["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    comun["hora_cos"] = np.cos(2 * np.pi * hora / 24)

    partes = []
    for i, n in enumerate(S):
        d = comun.copy()
        d["sensor"] = i
        d["CL2_tipico"] = float(tip["CL2", n].median())
        desviados = 0
        for p in PARAMS:
            x = D[p][n]
            d[f"desvio_{p}"] = x
            d[f"desvio_{p}_media3h"] = x.rolling(3, min_periods=1).mean()
            d[f"desvio_{p}_media12h"] = x.rolling(12, min_periods=3).mean()
            d[f"cambio_{p}_1h"] = x.diff()
            d[f"cambio_{p}_3h"] = x - x.shift(3)
            d[f"desv_{p}_6h"] = h[f"{p}_{n}"].rolling(6, min_periods=3).std()
            d[f"desvio_{p}_vs_red"] = x - D[p].median(axis=1)
            desviados = desviados + fuera[p][n].astype(int)
        d["parametros_desviados"] = desviados
        d["lecturas_hora"] = h[f"lecturas_{n}"]
        d["P"] = h[f"P_{n}"]
        d["cambio_P_1h"] = h[f"P_{n}"].diff()
        d["clase"] = h[f"evento_{n}"].fillna(0).astype(int)
        d["particion"] = h["particion"]
        d["nodo"] = n
        partes.append(d)
    return pd.concat(partes).rename_axis("fecha").reset_index(), tip


def columnas_entrada(tabla_):
    return [c for c in tabla_.columns if c not in {"fecha", "particion", "nodo", "clase"}]


def entrenar(tabla_, columnas, filas):
    mod = HistGradientBoostingClassifier(categorical_features=[columnas.index("sensor")], **HIPERPARAMETROS)
    mod.fit(tabla_.loc[filas, columnas], tabla_.loc[filas, "clase"])
    return mod


def probabilidades(mod, tabla_, columnas):
    """Probabilidad de evento (1 − P(normal)) y tipo más probable, por fila."""
    pr = mod.predict_proba(tabla_[columnas])
    clases = list(mod.classes_)
    p_evento = 1 - pr[:, clases.index(0)]
    pr_ev = pr.copy()
    pr_ev[:, clases.index(0)] = -1
    tipo = np.array(clases)[pr_ev.argmax(axis=1)]
    return p_evento, tipo


# ---------------------------------------------------------------------------
# Alarmas y evaluación por episodio
# ---------------------------------------------------------------------------
def alarmas_sensor(tabla_, p_evento, umbral, confirmar_h=2):
    """
    Tabla ancha (hora × sensor) con True donde el sensor está en alarma: su probabilidad de
    evento supera el umbral durante `confirmar_h` horas seguidas.
    """
    d = pd.DataFrame({"fecha": tabla_["fecha"].values, "nodo": tabla_["nodo"].values,
                      "alto": p_evento >= umbral})
    ancho = d.pivot(index="fecha", columns="nodo", values="alto").fillna(False).astype(bool)
    if confirmar_h > 1:
        ancho = ancho.astype(int).rolling(confirmar_h, min_periods=1).sum() >= confirmar_h
    return ancho


def episodios_alarma(alarma_red, unir_h=2):
    """Tramos continuos de alarma de red; se unen los separados por menos de `unir_h` horas."""
    eps = episodios(alarma_red.astype(int))
    if eps.empty:
        return pd.DataFrame(columns=["inicio", "fin", "horas"])
    filas = [eps.iloc[0][["inicio", "fin"]].to_dict()]
    for _, e in eps.iloc[1:].iterrows():
        if (e["inicio"] - filas[-1]["fin"]).total_seconds() / 3600 <= unir_h:
            filas[-1]["fin"] = e["fin"]
        else:
            filas.append(e[["inicio", "fin"]].to_dict())
    out = pd.DataFrame(filas)
    out["horas"] = (out["fin"] - out["inicio"]).dt.total_seconds() / 3600 + 1
    return out


def evaluar(h, alarma, desde, hasta, tipo_predicho=None):
    """
    Evaluación en [desde, hasta), con `alarma` = tabla ancha hora × sensor (alarmas_sensor).

    - Un episodio de evento cuenta como DETECTADO si algún sensor que lo ve entra en alarma
      en una hora en que lo está viendo. Retraso: desde la primera hora visible hasta la
      primera alarma en un sensor que lo ve.
    - Un episodio de alarma es FALSO si ninguno de los sensores en alarma tiene un evento
      visible en esas horas.
    """
    S = list(alarma.columns)
    rango = (h.index >= desde) & (h.index < hasta)
    E = h.loc[rango, [f"evento_{n}" for n in S]].fillna(0).astype(int)
    E.columns = S
    A = alarma.reindex(index=E.index, columns=S).fillna(False).astype(bool)

    filas = []
    for _, ep in episodios(h.loc[rango, "evento_red"].fillna(0).astype(int)).iterrows():
        ventana = slice(ep.inicio, ep.fin + pd.Timedelta(hours=24))
        ve = (E.loc[ventana] == ep.codigo)
        if not ve.values.any():
            continue                                    # nadie lo ve: no se puede detectar
        acierto = ve & A.loc[ventana]
        horas_vis = ve.index[ve.any(axis=1)]
        detectado = bool(acierto.values.any())
        fila = {"inicio": ep.inicio, "tipo": ep.tipo, "sensores_que_lo_ven": int(ve.any().sum()),
                "detectado": detectado, "retraso_h": np.nan, "tipo_sugerido": None}
        if detectado:
            primera = acierto.index[acierto.any(axis=1)][0]
            fila["retraso_h"] = (primera - horas_vis[0]).total_seconds() / 3600
            if tipo_predicho is not None:
                tp = tipo_predicho.reindex(index=acierto.index, columns=S)[acierto]
                vals = pd.Series(tp.values.ravel()).dropna()
                if len(vals):
                    fila["tipo_sugerido"] = carga.TIPOS_EVENTO[int(vals.mode()[0])]
        filas.append(fila)
    det = pd.DataFrame(filas)

    red = A.any(axis=1)
    alarmas = episodios_alarma(red)
    verdadera = (A & (E > 0)).any(axis=1)
    alarmas["falsa"] = [not verdadera.loc[a.inicio: a.fin].any() for _, a in alarmas.iterrows()]
    semanas_normales = (~(E > 0).any(axis=1)).sum() / (24 * 7)
    resumen = {"episodios visibles": len(det), "detectados": int(det["detectado"].sum()) if len(det) else 0,
               "alarmas": len(alarmas), "falsas alarmas": int(alarmas["falsa"].sum()),
               "falsas por semana": alarmas["falsa"].sum() / max(semanas_normales, 1e-9)}
    return det, alarmas, resumen
