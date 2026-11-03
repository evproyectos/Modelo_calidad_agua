"""
Anticipación de eventos: probabilidad de que EMPIECE un evento en la red en las próximas 24 h.

El detector (detector.py) avisa cuando un evento ya se ve en los sensores. Este modelo avisa
ANTES, con lo que se puede ver de las causas:
  - lluvia medida y pronosticada (tormentas que saturan la planta -> turbidez en la fuente;
    suelo saturado + baja presión -> intrusión de contaminación);
  - deterioro gradual en la fuente (cloro que va bajando, pH que se va desviando) antes de una
    falla de cloración o de pH;
  - horas de baja presión en la red.

Una fila = una hora (nivel de red, no por sensor). Solo usa información disponible hasta esa
hora. Clase = tipo del primer evento que empieza en (t, t + 24 h]; 0 si ninguno.

Horas que no se usan para entrenar ni para contar falsas alarmas: las que tienen un evento en
curso en la red (evento_red > 0) y ninguno por empezar. Ahí manda el detector; el aviso de
anticipación no tiene sentido mientras el evento ya está pasando.

Modelos que se comparan (notebook 09), uno binario por tipo de evento salvo el Isolation Forest:
  - árboles: HistGradientBoostingClassifier (el mismo tipo de modelo que el detector);
  - bosque aleatorio: RandomForestClassifier;
  - Isolation Forest: sin etiquetas, aprende cómo es la operación normal y marca lo raro;
  - LSTM: secuencias de las últimas horas, una salida por tipo (Keras, opcional).
Cada tipo tiene su umbral y sus horas de confirmación (elegir_umbrales); el aviso dice qué tipo
de evento se espera.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, IsolationForest, RandomForestClassifier

from . import carga, clima, configuracion

FUENTE = "J280"
HORIZONTE_H = 24
PARAMS = ["CL2", "PH", "TURB", "TEMP"]
PRESION_BAJA_M = 15.0                   # m; por debajo puede entrar agua del suelo por una fuga
CLORO_MINIMO = 0.3                      # mg/L, norma (Decreto 38924-S)

HIPERPARAMETROS = configuracion.seccion("anticipacion")            # hiperparametros.yaml


# ---------------------------------------------------------------------------- objetivo
def objetivo(horas, catalogo, horizonte=HORIZONTE_H):
    """
    Para cada hora t: código del primer evento que empieza en (t, t + horizonte] (0 si ninguno),
    horas que faltan para ese inicio e id del evento.
    """
    horas = pd.DatetimeIndex(horas)
    cat = catalogo.sort_values("inicio")
    inicios = pd.DatetimeIndex(cat["inicio"]).ceil("1h")       # primera hora completa del evento
    # primer inicio estrictamente posterior a t
    k = np.searchsorted(inicios.values, horas.values, side="right")
    valido = k < len(cat)
    kk = np.clip(k, 0, len(cat) - 1)
    faltan = np.where(valido, (inicios.values[kk] - horas.values) / np.timedelta64(1, "h"), np.inf)
    dentro = valido & (faltan <= horizonte)
    return pd.DataFrame({
        "clase": np.where(dentro, cat["codigo"].values[kk], 0).astype(int),
        "horas_para_inicio": np.where(dentro, faltan, np.nan),
        "id_evento": np.where(dentro, cat["id"].values[kk], 0).astype(int),
    }, index=horas)


# ---------------------------------------------------------------------------- variables
def horario(t15):
    """Promedios horarios de las lecturas (ya limpiadas con calidad.lecturas_operativas)."""
    num = t15[[c for c in t15.columns if c.split("_")[0] in PARAMS + ["P", "Q"] or c == "lluvia_mm_h"]]
    h = num.resample("1h").mean()
    if "evento_red" in t15:
        h["evento_red"] = t15["evento_red"].resample("1h").max()
    if "particion" in t15:
        h["particion"] = t15["particion"].resample("1h").first()
    return h


def tipicos_por_hora(h, normal=None):
    """Mediana de la fuente (cada parámetro) y del caudal por hora del día, en operación normal."""
    normal = np.ones(len(h), bool) if normal is None else np.asarray(normal, bool)
    tip = {}
    for col in [f"{p}_{FUENTE}" for p in PARAMS] + ["Q_R1"]:
        x = h[col][normal]
        tip[col] = x.groupby(x.index.hour).median()
    return tip


def lluvia_30d(lluvia_horaria):
    """Lluvia acumulada en 30 días (sequía previa: lavado de la cuenca), por hora."""
    return lluvia_horaria.rolling(30 * 24, min_periods=24).sum()


def tabla(h, tipicos=None, pronostico=None, normal=None, lluvia_mes=None):
    """
    Tabla horaria de variables (una fila por hora). `pronostico`: lluvia pronosticada por tramos
    (clima.ventanas), indexada como `h`. `tipicos`: los guardados al entrenar (si es None se
    calculan con `normal`). `lluvia_mes`: lluvia de 30 días ya calculada (lluvia_30d) para cuando
    `h` tiene menos historia, como en el servicio en vivo; si es None se calcula con `h`.
    """
    if tipicos is None:
        tipicos = tipicos_por_hora(h, normal)
    S = carga.sensores(h)
    hora = h.index.hour
    v = {}
    # fuente: valor, desvío de lo típico a esa hora, tendencias (el deterioro previo a una falla)
    for p in PARAMS:
        col = f"{p}_{FUENTE}"
        x = h[col]
        d = x - tipicos[col].reindex(hora).values
        v[f"fuente_{p}"] = x
        v[f"fuente_{p}_desvio"] = d
        v[f"fuente_{p}_desvio_media6h"] = d.rolling(6, min_periods=3).mean()
        v[f"fuente_{p}_desvio_media24h"] = d.rolling(24, min_periods=12).mean()
        for k in [3, 6, 12, 24]:
            v[f"fuente_{p}_cambio{k}h"] = x - x.shift(k)
        # tendencia suave: media de las últimas 6 h contra la de 6 h que terminó 18 h antes
        v[f"fuente_{p}_tendencia"] = x.rolling(6, min_periods=3).mean() - x.shift(18).rolling(6, min_periods=3).mean()
        v[f"fuente_{p}_desv6h"] = x.rolling(6, min_periods=3).std()
    # red: cómo están los sensores
    for p in ["CL2", "PH", "TURB"]:
        red = h[[f"{p}_{n}" for n in S]]
        v[f"red_{p}_mediana"] = red.median(axis=1)
        v[f"red_{p}_mediana_cambio12h"] = v[f"red_{p}_mediana"] - v[f"red_{p}_mediana"].shift(12)
    cl2 = h[[f"CL2_{n}" for n in S]]
    v["red_sensores_cloro_bajo"] = (cl2 < CLORO_MINIMO).sum(axis=1)
    pres = h[[f"P_{n}" for n in S]]
    v["red_sensores_presion_baja"] = (pres < PRESION_BAJA_M).sum(axis=1)
    v["red_presion_minima"] = pres.min(axis=1)
    v["red_presion_baja_24h"] = v["red_sensores_presion_baja"].rolling(24, min_periods=1).sum()
    # caudal
    q = h["Q_R1"]
    v["caudal"] = q
    v["caudal_desvio"] = q - tipicos["Q_R1"].reindex(hora).values
    # lluvia: reciente, acumulada (suelo saturado), de 30 días (sequía: lavado) y pronosticada
    ll = h["lluvia_mm_h"]
    for k in [1, 6, 24, 48, 72]:
        v[f"lluvia_{k}h"] = ll.rolling(k, min_periods=1).sum()
    v["lluvia_30d"] = lluvia_30d(ll) if lluvia_mes is None else lluvia_mes.reindex(h.index)
    v["lluvia_6h_max_24h"] = v["lluvia_6h"].rolling(24, min_periods=1).max()
    v["hora_sin"] = np.sin(2 * np.pi * hora / 24)
    v["hora_cos"] = np.cos(2 * np.pi * hora / 24)
    t = pd.DataFrame(v, index=h.index)
    if pronostico is not None:
        pr = pronostico.reindex(h.index)
        t = t.join(pr)
        t["lluvia_pron_24h"] = pr[clima.COLUMNAS].sum(axis=1, min_count=len(clima.COLUMNAS))
    return t.astype("float32"), tipicos


def columnas_entrada(tabla_):
    return list(tabla_.columns)


def filas_utiles(h, obj):
    """Horas que cuentan: sin un evento en curso, salvo que haya uno por empezar."""
    en_curso = h["evento_red"].fillna(0).values > 0 if "evento_red" in h else np.zeros(len(h), bool)
    return (~en_curso) | (obj["clase"].values > 0)


# ---------------------------------------------------------------------------- modelos
# Un modelo binario POR TIPO de evento (¿empieza un evento de este tipo en 24 h?). Con un solo
# modelo multiclase, la contaminación al azar (la clase más numerosa y la que no tiene aviso
# posible) se lleva el ajuste y el modelo no aprende las señales de los otros tipos (notebook 09).
CODIGOS = [1, 2, 3, 4]


def _por_tipo(crear, X, y):
    """Un modelo binario por tipo; None si el tipo no tiene ningún ejemplo en estas filas."""
    y = np.asarray(y)
    return {k: crear().fit(X, (y == k).astype(int)) if (y == k).any() else None for k in CODIGOS}


def entrenar_arboles(X, y):
    return _por_tipo(lambda: HistGradientBoostingClassifier(**HIPERPARAMETROS["arboles"]), X, y)


def entrenar_bosque(X, y):
    # el bosque de scikit-learn no acepta vacíos en todas las versiones
    return _por_tipo(lambda: RandomForestClassifier(**HIPERPARAMETROS["bosque"]), X.fillna(-999), y)


def entrenar_isolation(X_normal):
    """Sin etiquetas: un solo modelo de cómo es la operación normal (no dice el tipo)."""
    mod = IsolationForest(**HIPERPARAMETROS["isolation"])
    mod.fit(X_normal.fillna(X_normal.median()))
    mod.medianas_ = X_normal.median()
    return mod


def calibrar_isolation(mod, X_normal, cuantiles=(0.5, 0.999)):
    """El puntaje mediano de la operación normal es 0 y el del 0,1 % más raro es 1."""
    s = -mod.score_samples(X_normal.fillna(mod.medianas_))
    mod.rango_ = tuple(np.quantile(s, cuantiles))
    return mod


def probabilidades(mods, X):
    """
    Tabla (horas × tipo) con la probabilidad de que empiece un evento de cada tipo en 24 h.
    Para el Isolation Forest, una sola columna (0) con el puntaje de anomalía llevado a 0-1.
    """
    if isinstance(mods, IsolationForest):
        s = -mods.score_samples(X.fillna(mods.medianas_))
        lo, hi = mods.rango_
        return pd.DataFrame({0: np.clip((s - lo) / (hi - lo), 0, 1)}, index=X.index)
    P = {}
    for k, m in mods.items():
        if m is None:
            P[k] = np.zeros(len(X))
            continue
        Xm = X.fillna(-999) if isinstance(m, RandomForestClassifier) else X
        P[k] = m.predict_proba(Xm)[:, list(m.classes_).index(1)]
    return pd.DataFrame(P, index=X.index)


def probabilidad_alguno(P):
    """Probabilidad de que empiece algún evento (si los tipos fueran independientes)."""
    return 1 - np.prod(1 - P.values, axis=1)


# ---------------------------------------------------------------------------- avisos y evaluación
def avisos(p, umbral, confirmar_h=1):
    """True donde la probabilidad supera el umbral `confirmar_h` horas seguidas."""
    alto = pd.Series(np.asarray(p) >= umbral)
    if confirmar_h > 1:
        alto = alto.astype(int).rolling(confirmar_h, min_periods=1).sum() >= confirmar_h
    return alto.values


def avisos_por_tipo(P, eleccion):
    """
    Avisos de cada tipo con su umbral y confirmación (`eleccion` = {tipo: (umbral, horas)}).
    Devuelve (aviso de algún tipo, tipo sugerido por hora): entre los tipos en aviso, el que más
    supera su umbral; 0 si no hay aviso.
    """
    activos = pd.DataFrame({k: avisos(P[k].values, u, c) for k, (u, c) in eleccion.items()}, index=P.index)
    razon = pd.DataFrame({k: P[k].values / u for k, (u, _) in eleccion.items()}, index=P.index).where(activos, -1)
    tipo = np.where(activos.any(axis=1).values, np.asarray(razon.columns)[razon.values.argmax(axis=1)], 0)
    return activos.any(axis=1).values, tipo.astype(int)


def _tramos(serie_bool, index):
    """Tramos continuos en True: lista de (inicio, fin)."""
    s = pd.Series(np.asarray(serie_bool, bool), index=index)
    cambio = s.ne(s.shift(fill_value=False))
    grupos = cambio.cumsum()[s]
    return [(g.index[0], g.index[-1]) for _, g in s[s].groupby(grupos)]


def evaluar(index, aviso, catalogo, utiles, desde, hasta, horizonte=HORIZONTE_H, tipo=None, catalogo_falsos=None):
    """
    Evaluación por evento y por aviso en [desde, hasta).

    - Un evento está ANTICIPADO si hay un aviso en alguna hora de las `horizonte` horas antes de
      su inicio. Anticipación: horas entre el primer aviso de esa ventana y el inicio.
    - Aviso FALSO: cada racha de horas seguidas en aviso en que no empieza ningún evento en las
      `horizonte` horas siguientes (un aviso encendido sin motivo cuenta, aunque después acierte).
      Los avisos durante un evento en curso no cuentan. `catalogo_falsos`: eventos con los que se
      decide si un aviso es falso (por defecto, `catalogo`; para evaluar un solo tipo se pasan
      todos, porque avisar de un evento que viene, aunque sea de otro tipo, no es falso).
    """
    index = pd.DatetimeIndex(index)
    desde, hasta = pd.Timestamp(desde), pd.Timestamp(hasta)
    en = (index >= desde) & (index < hasta)
    a = pd.Series(np.asarray(aviso, bool) & en, index=index)
    tp = pd.Series(tipo, index=index) if tipo is not None else None
    cat = catalogo[(catalogo["inicio"] >= desde + pd.Timedelta(hours=horizonte)) & (catalogo["inicio"] < hasta)]
    filas = []
    for _, ev in cat.iterrows():
        ini = pd.Timestamp(ev["inicio"]).ceil("1h")
        ventana = a[(a.index >= ini - pd.Timedelta(hours=horizonte)) & (a.index < ini)]
        anticipado = bool(ventana.any())
        fila = {"id": ev["id"], "inicio": ev["inicio"], "tipo": ev["tipo"],
                "causa": ev.get("causa", "aleatoria") if pd.notna(ev.get("causa", np.nan)) else "aleatoria",
                "anticipado": anticipado, "anticipacion_h": np.nan, "tipo_sugerido": None}
        if anticipado:
            primero = ventana.index[ventana.values][0]
            fila["anticipacion_h"] = (ini - primero).total_seconds() / 3600
            if tp is not None:
                vals = tp[ventana.index[ventana.values]]
                vals = vals[vals > 0]
                if len(vals):
                    fila["tipo_sugerido"] = carga.TIPOS_EVENTO[int(vals.mode()[0])]
        filas.append(fila)
    eventos = pd.DataFrame(filas, columns=["id", "inicio", "tipo", "causa", "anticipado", "anticipacion_h",
                                           "tipo_sugerido"])

    # tramos de aviso (solo horas útiles) y si fueron falsos
    obj = objetivo(index, catalogo if catalogo_falsos is None else catalogo_falsos, horizonte)
    a_util = a & pd.Series(np.asarray(utiles, bool), index=index)
    tramos = _tramos(a_util.values, index)
    falsos = _tramos((a_util & (obj["clase"] == 0)).values, index)
    av = pd.DataFrame({"inicio": [t[0] for t in tramos], "fin": [t[1] for t in tramos]})
    av["falso"] = [any(ini <= f0 <= fin for f0, _ in falsos) for ini, fin in zip(av["inicio"], av["fin"])]
    horas_normales = (en & np.asarray(utiles, bool) & (obj["clase"].values == 0)).sum()
    semanas = horas_normales / (24 * 7)
    resumen = {"eventos": len(eventos), "anticipados": int(eventos["anticipado"].sum()),
               "pct_anticipados": 100 * eventos["anticipado"].mean() if len(eventos) else np.nan,
               "anticipacion_mediana_h": eventos["anticipacion_h"].median(),
               "avisos": len(av), "falsos": len(falsos),
               "falsos_por_semana": len(falsos) / max(semanas, 1e-9)}
    return eventos, av, resumen


def metricas_hora(y, p, umbral, utiles):
    """Métricas usuales por hora (solo horas útiles): F1, precisión, sensibilidad, AUC."""
    from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score, roc_auc_score
    m = np.asarray(utiles, bool) & ~np.isnan(p)
    yb, pp = (np.asarray(y)[m] > 0), np.asarray(p)[m]
    pred = pp >= umbral
    dos_clases = yb.any() and (~yb).any()
    return {"precision": precision_score(yb, pred, zero_division=0),
            "sensibilidad": recall_score(yb, pred, zero_division=0),
            "f1": f1_score(yb, pred, zero_division=0),
            "auc_roc": roc_auc_score(yb, pp) if dos_clases else np.nan,
            "auc_pr": average_precision_score(yb, pp) if dos_clases else np.nan}


def elegir_umbrales(index, P, catalogo, utiles, bloques, umbrales, confirmaciones, max_falsos_semana):
    """
    Umbral y confirmación de cada tipo, con las probabilidades fuera de muestra de los `bloques`
    [(desde, hasta)]: el que más eventos de ese tipo anticipa con a lo sumo `max_falsos_semana`
    repartidos entre los tipos. Devuelve ({tipo: (umbral, horas)}, grilla).
    """
    tipos = list(P.columns)
    tope = max_falsos_semana / len(tipos)
    semanas = {(d, h): _semanas(index, utiles, catalogo, d, h) for d, h in bloques}
    filas = []
    for k in tipos:
        cat_k = catalogo if k == 0 else catalogo[catalogo["codigo"] == k]
        for c in confirmaciones:
            aviso_c = {u: avisos(np.nan_to_num(P[k].values), u, c) for u in umbrales}
            for u in umbrales:
                tot = {"eventos": 0, "anticipados": 0, "falsos": 0, "semanas": 0.0, "anticipacion": []}
                for desde, hasta in bloques:
                    ev, _, r = evaluar(index, aviso_c[u], cat_k, utiles, desde, hasta, catalogo_falsos=catalogo)
                    tot["eventos"] += r["eventos"]
                    tot["anticipados"] += r["anticipados"]
                    tot["falsos"] += r["falsos"]
                    tot["semanas"] += semanas[(desde, hasta)]
                    tot["anticipacion"] += ev["anticipacion_h"].dropna().tolist()
                filas.append({"tipo": k, "umbral": u, "confirmación (h)": c, "eventos": tot["eventos"],
                              "anticipados": tot["anticipados"], "falsos": tot["falsos"],
                              "falsos por semana": tot["falsos"] / max(tot["semanas"], 1e-9),
                              "anticipación mediana (h)": np.median(tot["anticipacion"]) if tot["anticipacion"]
                              else np.nan})
    grilla = pd.DataFrame(filas)
    eleccion = {}
    for k, g in grilla.groupby("tipo"):
        ok = g[g["falsos por semana"] <= tope]
        ok = ok if len(ok) else g.sort_values("falsos por semana").head(1)
        e = ok.sort_values(["anticipados", "falsos por semana", "anticipación mediana (h)"],
                           ascending=[False, True, False]).iloc[0]
        eleccion[int(k)] = (float(e["umbral"]), int(e["confirmación (h)"]))
    return eleccion, grilla


def _semanas(index, utiles, catalogo, desde, hasta, horizonte=HORIZONTE_H):
    """Semanas de horas útiles sin ningún evento por empezar en [desde, hasta)."""
    index = pd.DatetimeIndex(index)
    en = (index >= pd.Timestamp(desde)) & (index < pd.Timestamp(hasta))
    sin = objetivo(index, catalogo, horizonte)["clase"].values == 0
    return (en & np.asarray(utiles, bool) & sin).sum() / (24 * 7)


# ---------------------------------------------------------------------------- LSTM (opcional)
def secuencias(X, largo):
    """Ventanas deslizantes (n, largo, variables) que terminan en cada hora; NaN -> 0."""
    A = np.nan_to_num(np.asarray(X, dtype="float32"))
    n, k = A.shape
    salida = np.zeros((n, largo, k), dtype="float32")
    for j in range(largo):
        desfase = largo - 1 - j
        salida[desfase:, j, :] = A[: n - desfase]
    return salida


def construir_lstm(largo, n_variables, n_tipos=len(CODIGOS), cfg=None):
    """Una salida sigmoide por tipo: probabilidad de que empiece un evento de ese tipo en 24 h."""
    import tensorflow as tf
    cfg = cfg or HIPERPARAMETROS["lstm"]
    m = tf.keras.Sequential([
        tf.keras.layers.Input((largo, n_variables)),
        tf.keras.layers.LSTM(cfg["unidades"]),
        tf.keras.layers.Dropout(cfg["abandono"]),
        tf.keras.layers.Dense(n_tipos, activation="sigmoid"),
    ])
    m.compile(optimizer=tf.keras.optimizers.Adam(cfg["tasa_aprendizaje"]),
              loss="binary_crossentropy")
    return m


# ---------------------------------------------------------------------------- propagación
TIPOS_FUENTE = (2, 3, 4)            # falla de cloro, turbidez y falla de pH: empiezan en la fuente


def retrasos_propagacion(eventos_hora, catalogo, sensores, fuente=FUENTE):
    """
    Horas que tarda un evento de la fuente en hacerse visible en cada sensor, contadas desde que
    se ve en la salida de la fuente. `eventos_hora`: tabla horaria con columnas evento_<nodo>.
    Devuelve una tabla por sensor con la mediana, el mínimo, el máximo y en cuántos eventos se vio.
    """
    filas = []
    for _, ev in catalogo[catalogo["codigo"].isin(TIPOS_FUENTE)].iterrows():
        ventana = eventos_hora.loc[ev["inicio"] - pd.Timedelta(hours=1): ev["inicio"] + pd.Timedelta(hours=96)]
        en_fuente = ventana.index[(ventana[f"evento_{fuente}"] == ev["codigo"]).values]
        if not len(en_fuente):
            continue
        t0 = en_fuente[0]
        for n in sensores:
            vis = ventana.index[(ventana[f"evento_{n}"] == ev["codigo"]).values & (ventana.index >= t0)]
            if len(vis):
                filas.append({"sensor": n, "id": ev["id"], "horas": (vis[0] - t0).total_seconds() / 3600})
    d = pd.DataFrame(filas, columns=["sensor", "id", "horas"])
    r = d.groupby("sensor")["horas"].agg(mediana="median", minimo="min", maximo="max", eventos="size")
    return r.reindex(sensores)


def llegada_estimada(retrasos, desde):
    """Hora estimada de llegada a cada sensor de un evento que se ve en la fuente desde `desde`."""
    desde = pd.Timestamp(desde)
    return {n: desde + pd.Timedelta(hours=float(r)) for n, r in retrasos["mediana"].dropna().items()}
