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
  - lluvia reciente (la turbidez sube con la lluvia sin que sea un evento) y hora del día;
  - para las fugas: caudal de la fuente y presión de cada sensor contra lo típico a esa hora
    (una fuga hace que la fuente entregue más y la presión baje), también en medias de 3 y 12 h,
    porque las bombas y los tanques los hacen subir y bajar a lo largo del día.

Una alarma de red se enciende cuando algún sensor supera el umbral de probabilidad.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier

from . import carga, configuracion
from .eventos import episodios

PARAMS = ["CL2", "TURB", "PH", "TEMP"]
UMBRAL_DESVIO = {"CL2": 0.1, "TURB": 0.5, "PH": 0.2, "TEMP": 1.0}
FUENTE = "J280"

HIPERPARAMETROS = configuracion.seccion("detector")                # hiperparametros.yaml
COMPARACION = configuracion.seccion("detector_comparacion")       # otros modelos (notebook 07)
VACIO = -999          # el bosque de scikit-learn no acepta vacíos en todas las versiones


def horario(t15):
    num = t15[[c for c in t15.columns if c.split("_")[0] in PARAMS + ["P", "Q"] or c == "lluvia_mm_h"]]
    h = num.resample("1h").mean()
    for n in carga.sensores(t15):
        h[f"evento_{n}"] = t15[f"evento_{n}"].resample("1h").max() if f"evento_{n}" in t15 else np.nan
        h[f"lecturas_{n}"] = t15[f"CL2_{n}"].notna().resample("1h").sum()
    h["particion"] = t15["particion"].resample("1h").first() if "particion" in t15 else None
    return h


def caudal(h):
    """Caudal que entregan las fuentes (L/s); NaN si el dataset no lo trae."""
    cols = [c for c in h.columns if c.startswith("Q_")]
    return h[cols].sum(axis=1, min_count=len(cols)) if cols else pd.Series(np.nan, index=h.index)


def tipicos_por_hora(h, S):
    """Mediana de cada sensor y parámetro (y del caudal de la fuente) por hora del día, en
    operación normal del entrenamiento."""
    entr = (h["particion"] == "entrenamiento").values
    tip = {}
    for n in S:
        normal = entr & (h[f"evento_{n}"] == 0).values
        for p in PARAMS + ["P"]:
            x = h[f"{p}_{n}"][normal]
            tip[p, n] = x.groupby(x.index.hour).median()
    normal = entr
    if "evento_visible" in h:
        normal = normal & (h["evento_visible"] == 0).values
    q = caudal(h)[normal]
    tip["Q", "fuente"] = q.groupby(q.index.hour).median()
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

    comun = {}                     # columnas en un diccionario y la tabla de una vez (más rápido)
    for p in PARAMS:
        comun[f"red_sensores_desviados_{p}"] = fuera[p].sum(axis=1)
        comun[f"red_desvio_mediano_{p}"] = D[p].median(axis=1)
        comun[f"fuente_desvio_{p}"] = D[p][FUENTE]
        for k in [3, 6, 12]:
            comun[f"fuente_desvio_{p}_hace{k}h"] = D[p][FUENTE].shift(k)
    # fugas: caudal de la fuente y presión mediana de los sensores contra lo típico a esa hora
    tip_q = tip.get(("Q", "fuente"))
    q_des = caudal(h) - (tip_q.reindex(hora).values if tip_q is not None else np.nan)
    comun["caudal"] = caudal(h)
    comun["caudal_desvio"] = q_des
    comun["caudal_desvio_media3h"] = q_des.rolling(3, min_periods=1).mean()
    comun["caudal_desvio_media12h"] = q_des.rolling(12, min_periods=3).mean()
    comun["caudal_cambio_3h"] = comun["caudal"] - comun["caudal"].shift(3)
    D_P = pd.DataFrame({n: h[f"P_{n}"].values - (tip["P", n].reindex(hora).values if ("P", n) in tip else np.nan)
                        for n in S}, index=h.index)
    comun["red_desvio_mediano_P"] = D_P.median(axis=1)
    for k in [6, 12, 24]:                     # lluvia acumulada (RF-001.5 d)
        comun[f"lluvia_{k}h"] = h["lluvia_mm_h"].rolling(k, min_periods=1).sum()
    comun["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    comun["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    comun = pd.DataFrame(comun, index=h.index)

    # ventanas móviles de todos los sensores a la vez (tabla hora × sensor por parámetro): da lo
    # mismo que calcularlas sensor por sensor y es varias veces más rápido
    V = {}
    for p in PARAMS:
        X = D[p]
        V[p] = {"media3h": X.rolling(3, min_periods=1).mean(),
                "media12h": X.rolling(12, min_periods=3).mean(),
                "cambio1h": X.diff(), "cambio3h": X - X.shift(3),
                "desv6h": h[[f"{p}_{n}" for n in S]].set_axis(S, axis=1).rolling(6, min_periods=3).std(),
                "vs_red": X.sub(X.median(axis=1), axis=0)}

    partes = []
    for i, n in enumerate(S):
        d = {}
        d["sensor"] = i
        d["CL2_tipico"] = float(tip["CL2", n].median())
        desviados = 0
        for p in PARAMS:
            v = V[p]
            d[f"desvio_{p}"] = D[p][n]
            d[f"desvio_{p}_media3h"] = v["media3h"][n]
            d[f"desvio_{p}_media12h"] = v["media12h"][n]
            d[f"cambio_{p}_1h"] = v["cambio1h"][n]
            d[f"cambio_{p}_3h"] = v["cambio3h"][n]
            d[f"desv_{p}_6h"] = v["desv6h"][n]
            d[f"desvio_{p}_vs_red"] = v["vs_red"][n]
            desviados = desviados + fuera[p][n].astype(int)
        d["parametros_desviados"] = desviados
        d["lecturas_hora"] = h[f"lecturas_{n}"]
        d["P"] = h[f"P_{n}"]
        d["cambio_P_1h"] = h[f"P_{n}"].diff()
        d["desvio_P"] = D_P[n]
        d["desvio_P_media3h"] = D_P[n].rolling(3, min_periods=1).mean()
        d["desvio_P_media12h"] = D_P[n].rolling(12, min_periods=3).mean()
        d["clase"] = h[f"evento_{n}"].fillna(0).astype(int)
        d["particion"] = h["particion"]
        d["nodo"] = n
        partes.append(pd.concat([comun, pd.DataFrame(d, index=h.index)], axis=1))
    return pd.concat(partes).rename_axis("fecha").reset_index(), tip


def columnas_entrada(tabla_):
    return [c for c in tabla_.columns if c not in {"fecha", "particion", "nodo", "clase"}]


def entrenar(tabla_, columnas, filas):
    mod = HistGradientBoostingClassifier(categorical_features=[columnas.index("sensor")], **HIPERPARAMETROS)
    mod.fit(tabla_.loc[filas, columnas], tabla_.loc[filas, "clase"])
    return mod


def entrenar_bosque(tabla_, columnas, filas):
    """Random Forest con las mismas variables y etiquetas que el detector (para compararlos)."""
    mod = RandomForestClassifier(**COMPARACION["bosque"])
    mod.fit(tabla_.loc[filas, columnas].fillna(VACIO), tabla_.loc[filas, "clase"])
    return mod


# variables que miden "qué tan raro está esto" (para el Isolation Forest mejorado): desvíos de lo
# típico y cambios recientes; sin el sensor, la hora, la lluvia, la presión ni el cloro típico
PREFIJOS_ANOMALIA = ("desvio_", "cambio_CL2", "cambio_TURB", "cambio_PH", "cambio_TEMP", "desv_",
                     "red_sensores_desviados_", "red_desvio_mediano_", "fuente_desvio_", "parametros_desviados",
                     "caudal_desvio")
LLUVIA = ["lluvia_6h", "lluvia_12h", "lluvia_24h"]


def columnas_anomalia(columnas):
    return [c for c in columnas if c.startswith(PREFIJOS_ANOMALIA)]


def operacion_limpia(tabla_, crudo):
    """
    True en las filas (sensor × hora) de operación normal de verdad: sin evento, sin falla del
    sensor (columnas falla_*) y sin el deterioro previo a una falla (precursor_red).
    """
    S = sorted(tabla_["nodo"].unique())
    falla = pd.DataFrame({n: (crudo[[c for c in crudo.columns if c.startswith("falla_") and c.endswith("_" + n)]] > 0)
                          .any(axis=1) for n in S}).resample("1h").max()
    if "precursor_red" in crudo:
        falla = falla | (crudo["precursor_red"] > 0).resample("1h").max().values[:, None]
    malo = falla.stack().reindex(pd.MultiIndex.from_arrays([tabla_["fecha"], tabla_["nodo"]])).fillna(False).values
    return (tabla_["clase"].values == 0) & ~malo.astype(bool)


def _ajustadas(mod, t):
    """Variables de anomalía, menos lo que la lluvia explica normalmente en ese sensor (con vacíos)."""
    Z = t[mod.columnas_].values.astype(float)
    if mod.lluvia_:
        L = np.log1p(t[LLUVIA].values.astype(float))
        for n, coef in mod.lluvia_.items():
            m = t["nodo"].values == n
            Z[m] -= L[m] @ coef
    return Z


def _sin_lluvia(mod, t):
    """Lo mismo, con los vacíos rellenados con la mediana de la operación normal."""
    Z = _ajustadas(mod, t)
    return np.where(np.isnan(Z), mod.medianas_, Z)


def entrenar_isolation(tabla_, columnas, filas, desvios=False, limpias=None, por_sensor=False, sin_lluvia=False):
    """
    Isolation Forest: aprende solo la operación normal y marca lo que se aparta. No usa las
    etiquetas de evento ni dice el tipo. El puntaje se lleva a 0-1: 0 es la mediana de la operación
    normal y 1, el 0,1 % más raro. Sin opciones, usa las mismas variables que el detector.

    Mejoras (notebook 07):
      desvios     solo las variables que miden rareza (columnas_anomalia);
      limpias     filas de operación normal de verdad (operacion_limpia) para entrenar;
      por_sensor  el puntaje se calibra con la operación normal de cada sensor (umbral por sensor);
      sin_lluvia  a cada variable se le resta lo que la lluvia explica normalmente en ese sensor
                  (regresión lineal sobre el logaritmo de la lluvia de 6 y 24 h).
    """
    normal = np.asarray(filas, bool) & (tabla_["clase"].values == 0)
    if limpias is not None:
        normal &= np.asarray(limpias, bool)
    Xn = tabla_.loc[normal]
    mod = IsolationForest(**COMPARACION["isolation"])
    mod.columnas_ = columnas_anomalia(columnas) if desvios else list(columnas)
    mod.lluvia_ = {}
    if sin_lluvia:
        for n, d in Xn.groupby("nodo"):
            Y = d[mod.columnas_].astype(float)
            A = np.column_stack([np.ones(len(d)), np.log1p(d[LLUVIA].values.astype(float))])
            coef = np.linalg.lstsq(A, Y.fillna(Y.median()).values, rcond=None)[0]
            mod.lluvia_[n] = coef[1:]                    # sin el intercepto: solo el efecto de la lluvia
    mod.medianas_ = np.nanmedian(_ajustadas(mod, Xn), axis=0)
    Zn = _sin_lluvia(mod, Xn)
    mod.fit(Zn)
    s = -mod.score_samples(Zn)
    grupos = Xn["nodo"].values if por_sensor else np.full(len(Xn), "")
    mod.rango_ = {g: tuple(np.quantile(s[grupos == g], (0.5, 0.999))) for g in np.unique(grupos)}
    mod.por_sensor_ = por_sensor
    return mod


def probabilidades(mod, tabla_, columnas):
    """Probabilidad de evento (1 − P(normal)) y tipo más probable, por fila. Con el Isolation
    Forest, el puntaje de anomalía en 0-1 y tipo 0 (no lo sabe)."""
    if isinstance(mod, IsolationForest):
        s = -mod.score_samples(_sin_lluvia(mod, tabla_))
        grupos = tabla_["nodo"].values if mod.por_sensor_ else np.full(len(tabla_), "")
        p = np.zeros(len(tabla_))
        for g, (lo, hi) in mod.rango_.items():
            m = grupos == g
            p[m] = np.clip((s[m] - lo) / (hi - lo), 0, 1)
        return p, np.zeros(len(tabla_), int)
    X = tabla_[columnas].fillna(VACIO) if isinstance(mod, RandomForestClassifier) else tabla_[columnas]
    pr = mod.predict_proba(X)
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
