# Modelo de calidad de agua

Análisis exploratorio (EDA) y modelos de pronóstico de calidad de agua para una
red de distribución con 21 sensores. Cada sensor reporta cada 15 minutos:

| Parámetro | Columna | Unidad |
|---|---|---|
| Cloro residual libre | `CL2_<nodo>` | mg/L |
| Temperatura | `TEMP_<nodo>` | °C |
| Turbidez | `TURB_<nodo>` | UNT |
| pH | `PH_<nodo>` | — |
| Presión | `P_<nodo>` | m |

Además se registra el caudal de salida de la fuente (`Q_R1`, L/s) y la lluvia
(`lluvia_mm_h`).

Objetivo: pronosticar las próximas 24 h con al menos 80 % de exactitud,
usando también datos climáticos.

## Meta

Un pronóstico a 24 h se considera **acertado** si queda dentro de la tolerancia
del valor observado (promedio horario):

| Parámetro | Tolerancia |
|---|---|
| Cloro residual libre | ± 0,1 mg/L |
| Turbidez | ± 20 % del valor, mínimo ± 0,2 UNT |
| pH | ± 0,1 |

La meta es que el modelo acierte **al menos el 80 %** de los pronósticos en cada
uno de los tres parámetros, en operación normal, **y que supere al pronóstico
ingenuo** "igual que ayer a la misma hora" en el mismo periodo (notebook 03):

| Periodo | Cloro | Turbidez | pH |
|---|---|---|---|
| validación (ene–abr 2026, época más seca) | 81 % | 62 % | 84 % |
| prueba (may–ago 2026, época lluviosa) | 79 % | 43 % | 78 % |

El acierto durante eventos se reporta por separado. La definición está en
`calidad_agua/metas.py`.

## Datos

Los datos van de septiembre de 2024 a agosto de 2026, divididos en tres
periodos consecutivos:

| Partición | Periodo | Uso |
|---|---|---|
| entrenamiento | sep 2024 – dic 2025 | ajustar los modelos |
| validación | ene – abr 2026 (época seca) | elegir hiperparámetros y umbrales |
| prueba | may – ago 2026 (entrada de la época lluviosa) | evaluación final |

Las fechas de cada periodo vienen en `metadatos.json` junto con los datos y el código
las lee de ahí (`carga.periodos()`); no están escritas a mano en los notebooks.

Los datos no se versionan en este repositorio.

## Estructura

```
calidad_agua/ paquete de Python con el código compartido: carga, limpieza, variables,
              modelos (árboles, LSTM, detector), meta y figuras
notebooks/    EDA y modelos, numerados en el orden de lectura
pruebas/      pruebas del paquete
figuras/      figuras exportadas para el informe
modelos/      modelos entrenados (se generan con los notebooks 06, 07 y 08; no se versionan)
```

## Instalación

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

`requirements.txt` instala este proyecto como el paquete **`calidad_agua`** en modo
editable (`pip install -e .`), con todas sus dependencias. Así los notebooks y
otros proyectos (por ejemplo, el servicio de monitoreo) usan exactamente el mismo
código: `from calidad_agua import variables, detector`.

Para usarlo desde otro proyecto, en el entorno de ese proyecto:

```
pip install -e "D:\Practica_Profesional\Modelo_calidad_agua[red]"
```

### Prueba de variables en vivo

Comprueba que las variables calculadas con solo los últimos 7 días de lecturas y
sin etiquetas (como en operación) son idénticas a las usadas al entrenar:

```
python pruebas/prueba_variables_en_vivo.py
```

## Notebooks

| # | Notebook | Contenido |
|---|---|---|
| 01 | calidad_de_datos | Faltantes, cortes de comunicación, fallas de sensor |
| 02 | distribuciones | Rangos por parámetro y sensor, comparación con la norma |
| 03 | temporal | Ciclos diarios y semanales, estacionalidad, autocorrelación |
| 04 | clima | Lluvia de la zona (IMN, estación ITCR San Carlos), su efecto en turbidez, cloro, pH y temperatura, y cuánto aportaría un pronóstico de lluvia |
| 05 | eventos | Cómo se ven los eventos y las fallas en los datos |
| 06 | modelo_pronostico | Modelo de pronóstico a 24 h y evaluación contra la meta |
| 07 | detector_eventos | Detector de eventos: alarmas, tipo de evento y falsas alarmas |
| 08 | red_neuronal | Red LSTM (TensorFlow/Keras), comparación con los árboles y modelo final de pronóstico |
