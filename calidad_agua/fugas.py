"""
Detección y ubicación de fugas (RF-002.2, notebook 10).

Una fuga hace que la fuente entregue más agua de lo normal a esa hora y que baje la presión en
la red. Pero el caudal y la presión cambian mucho a lo largo del día y de la semana, con la
época del año y con la temperatura (se consume más con calor). Por eso el enfoque tiene dos pasos:

1. Lo esperado en operación normal, de dos formas:
   - regresión multivariable (scikit-learn, LinearRegression): para el caudal de la fuente y la
     presión de cada sensor, lo esperado según la hora del día, si es fin de semana, la
     temperatura del agua y la lluvia; se ajusta con horas sin fuga del entrenamiento;
   - línea base de la semana: la mediana de la misma hora en los 7 días anteriores (sigue sola
     los cambios de época y de consumo; una fuga dura horas o pocos días y casi no la mueve).
   El residuo (medido − esperado) es lo que la operación normal no explica.
2. Con los residuos (y sus medias de 3 a 24 h), un modelo que decide si hay fuga. Se comparan:
     - Random Forest (supervisado, con las etiquetas de fuga);
     - SVM (máquina de vectores de soporte, supervisada);
     - Autoencoder (TensorFlow/Keras, sin etiquetas): aprende a reconstruir la operación normal;
       una hora es anómala si el error de reconstrucción pasa el percentil 95 del error en la
       operación normal (umbral de RF-002.2).

Una fila = una hora (nivel de red). Clase = 1 mientras hay una fuga abierta (del catálogo: desde
que el tubo se rompe hasta que se repara). Para operar se necesitan HISTORIA_H horas de caudal y
presión hacia atrás (la línea base de la semana).

Ubicación: con 10 puntos de medición no se puede ubicar una fuga al metro. Se dice en qué tramo
está: entre qué dos sensores (los dos más cercanos por tubería al nodo de la fuga). Un Random
Forest aprende el tramo con el patrón de residuos de presión y caudal en las horas de fuga.
"""

import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from . import configuracion
from .eventos import episodios

CODIGO_FUGA = 5
FUENTE = "J280"
DIAS_BASE = 7                                 # línea base: misma hora de los 7 días anteriores
HISTORIA_H = 24 * DIAS_BASE + 24              # historia que necesita tabla() (con la media de 24 h)
VACIO = 0.0           # residuos vacíos (sin lectura): se rellenan con 0, "como lo esperado"
HIPERPARAMETROS = configuracion.seccion("fugas")


# ---------------------------------------------------------------------------- tramos
def tramos(distancias):
    """Tramo de cada nodo de consumo: 'A – B' con los dos sensores más cercanos por tubería,
    en el orden de las columnas de `distancias` (nodo × sensor, en metros)."""
    orden = {s: i for i, s in enumerate(distancias.columns)}

    def par(fila):
        a, b = fila.nsmallest(2).index
        a, b = sorted([a, b], key=orden.get)
        return f"{a} – {b}"
    return distancias.apply(par, axis=1)


# ---------------------------------------------------------------------------- variables
def _regresores(h):
    """Hora del día (una columna por hora), fin de semana por hora, temperatura y lluvia."""
    hora = h.index.hour
    finde = (h.index.dayofweek >= 5).astype(float)
    X = {f"h{k}": (hora == k).astype(float) for k in range(24)}
    X.update({f"finde_h{k}": ((hora == k) & (finde > 0)).astype(float) for k in range(24)})
    temp = h.get(f"TEMP_{FUENTE}", pd.Series(np.nan, index=h.index))
    X["temperatura"] = temp.ffill().bfill().fillna(25.0).values
    X["lluvia_24h"] = h["lluvia_mm_h"].rolling(24, min_periods=1).sum().fillna(0).values
    return pd.DataFrame(X, index=h.index)


def senales(h):
    """Series que se modelan: caudal de la fuente y presión de cada sensor (L/s y m)."""
    q = [c for c in h.columns if c.startswith("Q_")]
    S = [c[2:] for c in h.columns if c.startswith("P_")]
    d = {"caudal": h[q].sum(axis=1, min_count=len(q))}
    d.update({f"presion_{n}": h[f"P_{n}"] for n in S})
    return pd.DataFrame(d, index=h.index)


def clase(h, catalogo=None):
    """1 mientras hay una fuga abierta (con `catalogo`: de su inicio a su fin); sin catálogo, 1 si
    algún sensor ve una fuga en esa hora (evento_<n> == 5)."""
    if catalogo is not None:
        y = pd.Series(0, index=h.index)
        for _, ev in catalogo[catalogo["codigo"] == CODIGO_FUGA].iterrows():
            y[(h.index >= pd.Timestamp(ev["inicio"]).floor("1h")) & (h.index <= ev["fin"])] = 1
        return y
    E = h[[c for c in h.columns if c.startswith("evento_") and c not in ("evento_visible", "evento_red")]]
    return (E == CODIGO_FUGA).any(axis=1).astype(int)


def ajustar_normal(h, normal):
    """Regresión de la operación normal, una por señal: {señal: LinearRegression}."""
    X, Y = _regresores(h), senales(h)
    normal = np.asarray(normal, bool)
    modelos = {}
    for c in Y.columns:
        m = normal & Y[c].notna().values
        modelos[c] = LinearRegression().fit(X.values[m], Y[c].values[m])
    return modelos


def tabla(h, normal_modelos):
    """Residuos de cada señal contra lo esperado (regresión y línea base de la semana) y sus
    ventanas; hora del día. `h` debe ser horaria y continua, con HISTORIA_H horas de historia
    antes de la primera hora que se quiera evaluar."""
    h = h.asfreq("1h")
    X, Y = _regresores(h), senales(h)
    v = {}
    for c, mod in normal_modelos.items():
        r = Y[c] - mod.predict(X.values) if c in Y else pd.Series(np.nan, index=h.index)
        v[f"{c}_residuo"] = r
        for k in [3, 6, 12]:
            v[f"{c}_residuo_media{k}h"] = r.rolling(k, min_periods=1).mean()
        s = Y[c] if c in Y else pd.Series(np.nan, index=h.index)
        base = pd.concat([s.shift(24 * k) for k in range(1, DIAS_BASE + 1)], axis=1).median(axis=1)
        rb = s - base
        v[f"{c}_vs_semana"] = rb
        for k in [3, 6, 12, 24]:
            v[f"{c}_vs_semana_media{k}h"] = rb.rolling(k, min_periods=1).mean()
        v[f"{c}_cambio1h"] = Y[c].diff() if c in Y else np.nan
        v[f"{c}_cambio3h"] = Y[c] - Y[c].shift(3) if c in Y else np.nan
    v["hora_sin"] = np.sin(2 * np.pi * h.index.hour / 24)
    v["hora_cos"] = np.cos(2 * np.pi * h.index.hour / 24)
    return pd.DataFrame(v, index=h.index).astype("float32")


# ---------------------------------------------------------------------------- modelos
def entrenar_bosque(X, y):
    return RandomForestClassifier(**HIPERPARAMETROS["bosque"]).fit(X.fillna(VACIO), y)


def entrenar_svm(X, y):
    """SVM con escalado; la probabilidad sale de una calibración sigmoide (Platt)."""
    svm = CalibratedClassifierCV(SVC(**HIPERPARAMETROS["svm"]), method="sigmoid", cv=3, ensemble=False)
    return make_pipeline(StandardScaler(), svm).fit(X.fillna(VACIO), y)


def construir_autoencoder(n_variables, cfg=None):
    """Autoencoder denso: comprime la hora a pocas variables y la vuelve a armar."""
    import tensorflow as tf
    cfg = cfg or HIPERPARAMETROS["autoencoder"]
    k = tf.keras
    entrada = k.Input(shape=(n_variables,))
    x = entrada
    for u in cfg["capas"]:
        x = k.layers.Dense(u, activation="relu")(x)
    for u in reversed(cfg["capas"][:-1]):
        x = k.layers.Dense(u, activation="relu")(x)
    salida = k.layers.Dense(n_variables)(x)
    modelo = k.Model(entrada, salida)
    modelo.compile(optimizer=k.optimizers.Adam(cfg["tasa_aprendizaje"]), loss="mse")
    return modelo


class Autoencoder:
    """Autoencoder entrenado con operación normal; puntaje = error / umbral (percentil 95 del
    error en la operación normal), así que 1 es el umbral."""

    def __init__(self, cfg=None):
        self.cfg = cfg or HIPERPARAMETROS["autoencoder"]

    def fit(self, X_normal):
        import tensorflow as tf
        tf.keras.utils.set_random_seed(self.cfg["semilla"])
        self.escala_ = StandardScaler().fit(X_normal.fillna(VACIO))
        Z = self.escala_.transform(X_normal.fillna(VACIO))
        self.red_ = construir_autoencoder(Z.shape[1], self.cfg)
        parada = tf.keras.callbacks.EarlyStopping(patience=self.cfg["paciencia"], restore_best_weights=True)
        self.red_.fit(Z, Z, epochs=self.cfg["epocas"], batch_size=self.cfg["lote"], validation_split=0.1,
                      callbacks=[parada], verbose=0)
        self.umbral_ = float(np.percentile(self.error(X_normal), self.cfg["percentil"]))
        return self

    def error(self, X):
        Z = self.escala_.transform(X.fillna(VACIO))
        R = self.red_.predict(Z, batch_size=4096, verbose=0)
        return ((Z - R) ** 2).mean(axis=1)

    def puntaje(self, X):
        return self.error(X) / self.umbral_


def probabilidad(mod, X):
    """Probabilidad de fuga por hora (en el autoencoder, error / umbral: pasa de 1 en lo raro)."""
    if isinstance(mod, Autoencoder):
        return mod.puntaje(X)
    return mod.predict_proba(X.fillna(VACIO))[:, 1]


# ---------------------------------------------------------------------------- alarmas y evaluación
def alarma(p, umbral, confirmar_h=1):
    a = pd.Series(np.asarray(p) >= umbral)
    if confirmar_h > 1:
        a = a.astype(int).rolling(confirmar_h, min_periods=1).sum() >= confirmar_h
    return a.values


def evaluar(horas, alarma_, visible, desde, hasta, unir_h=2):
    """
    Evaluación por episodio en [desde, hasta):
      - una fuga (tramo continuo de horas con fuga visible) está DETECTADA si hay alarma en
        alguna de esas horas; retraso = desde la primera hora visible hasta la primera alarma;
      - un episodio de alarma es FALSO si no toca ninguna hora con fuga visible.
    Devuelve (tabla de fugas, resumen con precisión, sensibilidad y F1 por evento).
    """
    horas = pd.DatetimeIndex(horas)
    m = (horas >= desde) & (horas < hasta)
    a = pd.Series(np.asarray(alarma_, bool)[m], index=horas[m])
    vis = pd.Series(np.asarray(visible, int)[m], index=horas[m])
    filas = []
    for _, ep in episodios(vis).iterrows():
        tramo = a.loc[ep.inicio: ep.fin]
        detectado = bool(tramo.any())
        filas.append({"inicio": ep.inicio, "fin": ep.fin, "horas": ep.horas, "detectado": detectado,
                      "retraso_h": (tramo.index[tramo.values][0] - ep.inicio).total_seconds() / 3600
                      if detectado else np.nan})
    ev = pd.DataFrame(filas, columns=["inicio", "fin", "horas", "detectado", "retraso_h"])
    alarmas = episodios(a.astype(int))
    unidas = []
    for _, e in alarmas.iterrows():
        if unidas and (e.inicio - unidas[-1][1]).total_seconds() / 3600 <= unir_h:
            unidas[-1][1] = e.fin
        else:
            unidas.append([e.inicio, e.fin])
    falsas = sum(1 for i, f in unidas if not vis.loc[i:f].any())
    semanas = m.sum() / (24 * 7)
    detectadas = int(ev["detectado"].sum())
    precision = (len(unidas) - falsas) / len(unidas) if unidas else np.nan
    sensibilidad = detectadas / len(ev) if len(ev) else np.nan
    f1 = 2 * precision * sensibilidad / (precision + sensibilidad) if precision and sensibilidad else 0.0
    resumen = {"fugas": len(ev), "detectadas": detectadas, "alarmas": len(unidas), "falsas": falsas,
               "falsas_por_semana": falsas / semanas if semanas else np.nan,
               "precision": precision, "sensibilidad": sensibilidad, "f1": f1,
               "retraso_mediano_h": float(ev["retraso_h"].median()) if detectadas else np.nan}
    return ev, resumen


def elegir_umbral(horas, p, visible, bloques, umbrales, confirmaciones, max_falsas_semana):
    """Umbral y confirmación con la regla de los otros modelos: máximo de fugas detectadas con
    a lo sumo `max_falsas_semana` falsas alarmas por semana en CADA bloque; a igualdad, menos
    falsas en total y menos retraso. `bloques`: [(desde, hasta)], `p` y `visible` alineados
    con `horas`."""
    filas = []
    for u in umbrales:
        for c in confirmaciones:
            a = alarma(p, u, c)
            rs = [evaluar(horas, a, visible, d, hst)[1] for d, hst in bloques]
            filas.append({"umbral": u, "confirmacion_h": c, "detectadas": sum(r["detectadas"] for r in rs),
                          "fugas": sum(r["fugas"] for r in rs), "falsas": sum(r["falsas"] for r in rs),
                          "peor_falsas_semana": max(r["falsas_por_semana"] for r in rs),
                          "retraso_mediano_h": np.nanmedian([r["retraso_mediano_h"] for r in rs])})
    g = pd.DataFrame(filas)
    ok = g[g["peor_falsas_semana"] <= max_falsas_semana]
    if ok.empty:
        ok = g.nsmallest(1, "peor_falsas_semana")
    mejor = ok.sort_values(["detectadas", "falsas", "retraso_mediano_h"], ascending=[False, True, True]).iloc[0]
    return (float(mejor["umbral"]), int(mejor["confirmacion_h"])), g


# ---------------------------------------------------------------------------- ubicación
def entrenar_tramo(X, tramo):
    """Random Forest que dice el tramo con los residuos de las horas de fuga."""
    return RandomForestClassifier(**HIPERPARAMETROS["tramo"]).fit(X.fillna(VACIO), tramo)


def tramo_por_evento(mod, X, horas_evento):
    """Tramo más votado en las horas de una fuga (probabilidades promediadas)."""
    pr = mod.predict_proba(X.loc[horas_evento].fillna(VACIO)).mean(axis=0)
    return mod.classes_[int(np.argmax(pr))], float(pr.max())


def tramo_por_regla(X, sensores, horas_evento):
    """Línea base: los dos sensores con la mayor caída de presión (residuo más negativo)."""
    caida = {n: X.loc[horas_evento, f"presion_{n}_residuo_media3h"].mean() for n in sensores}
    a, b = sorted(sorted(caida, key=caida.get)[:2], key=sensores.index)
    return f"{a} – {b}"
