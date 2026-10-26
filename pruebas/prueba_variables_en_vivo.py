"""
Prueba de compatibilidad: variables en vivo == variables del entrenamiento.

Para varias horas del periodo de prueba se toma solo una ventana de 7 días de lecturas,
SIN columnas de etiquetas (evento_*, falla_*) ni partición, como llegarían por la API,
y se arman las variables con las mismas funciones. La última hora de la ventana debe ser
idéntica a la fila de la tabla completa usada al entrenar.

Uso (desde la raíz del proyecto):  python pruebas/prueba_variables_en_vivo.py
"""
import numpy as np, pandas as pd
from calidad_agua import carga, calidad, variables, detector, red_neuronal as rn

crudo = carga.cargar_todo()
# tabla completa (como en los notebooks)
t15 = calidad.lecturas_operativas(crudo)
h = variables.horario(t15)
T, tip = variables.tabla(h)
hd = detector.horario(t15)
TD, tipd = detector.tabla(hd)
datos = rn.preparar(h, tip)
X, XD = variables.columnas_entrada(T), detector.columnas_entrada(TD)

solo_sensores = [c for c in crudo.columns if not c.startswith(("evento_", "falla_")) and c != "particion"]
diferencias = {"pronóstico": 0, "detector": 0, "LSTM": 0}
# unas horas al azar del periodo de prueba (a partir del día 10, para tener 7 días de historia)
horas = pd.date_range(carga.periodos()["prueba"][0] + pd.Timedelta(days=9, hours=7), periods=6, freq="17h")
for hora in horas:
    fin = hora + pd.Timedelta(minutes=45)                  # última lectura de esa hora
    ventana = crudo.loc[fin - pd.Timedelta(days=7) + pd.Timedelta(minutes=15): fin, solo_sensores]
    w15 = calidad.lecturas_operativas(ventana)

    hv = variables.horario(w15)
    Tv, _ = variables.tabla(hv, tipicos=tip)
    a = Tv[Tv.fecha == hora].set_index("nodo")[X].astype(float)
    b = T[T.fecha == hora].set_index("nodo")[X].astype(float)
    ok = np.isclose(a.values, b.loc[a.index].values, equal_nan=True, rtol=1e-6, atol=1e-9)
    diferencias["pronóstico"] += (~ok).sum()

    hdv = detector.horario(w15)
    TDv, _ = detector.tabla(hdv, tip=tipd)
    a = TDv[TDv.fecha == hora].set_index("nodo")[XD].astype(float)
    b = TD[TD.fecha == hora].set_index("nodo")[XD].astype(float)
    ok = np.isclose(a.values, b.loc[a.index].values, equal_nan=True, rtol=1e-6, atol=1e-9)
    diferencias["detector"] += (~ok).sum()

    dv = rn.preparar(hv, tip, normalizacion=datos["normalizacion"])
    k_v, k_b = len(hv) - 1, h.index.get_loc(hora)
    for i in range(len(dv["sensores"])):
        sv = dv["secuencias"][i][k_v - rn.VENTANA_H + 1: k_v + 1]
        sb = datos["secuencias"][i][k_b - rn.VENTANA_H + 1: k_b + 1]
        diferencias["LSTM"] += (~np.isclose(sv, sb, rtol=1e-5, atol=1e-6)).sum()
    print(hora, "ok")

print("Valores distintos entre vivo y entrenamiento:", diferencias)
assert all(v == 0 for v in diferencias.values()), "¡Las variables en vivo no coinciden!"
print("PRUEBA SUPERADA: las variables en vivo son idénticas a las del entrenamiento.")
