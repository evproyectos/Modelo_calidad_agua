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
ingenuo** "igual que ayer a la misma hora", que logra 75 % en cloro, 52 % en
turbidez y 80 % en pH (notebook 03). El acierto durante eventos se reporta por
separado. La definición está en `src/metas.py`.

## Datos

Los datos van de enero de 2025 a diciembre de 2026, divididos en tres
periodos consecutivos:

| Partición | Periodo | Uso |
|---|---|---|
| entrenamiento | ene 2025 – abr 2026 | ajustar los modelos |
| validación | may – ago 2026 | elegir hiperparámetros y umbrales |
| prueba | sep – dic 2026 | evaluación final |

Los datos no se versionan en este repositorio.

## Estructura

```
src/          funciones compartidas (carga, limpieza, eventos, meta, figuras)
notebooks/    EDA y modelos, numerados en el orden de lectura
figuras/      figuras exportadas para el informe
modelos/      modelos entrenados (se generan con el notebook 06; no se versionan)
```

## Instalación

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Los notebooks agregan la raíz del proyecto al `sys.path`, así que
`from src import carga` funciona desde la carpeta `notebooks/`.

## Notebooks

| # | Notebook | Contenido |
|---|---|---|
| 01 | calidad_de_datos | Faltantes, cortes de comunicación, fallas de sensor |
| 02 | distribuciones | Rangos por parámetro y sensor, comparación con la norma |
| 03 | temporal | Ciclos diarios y semanales, estacionalidad, autocorrelación |
| 04 | clima | Relación de lluvia y temperatura con turbidez, cloro y pH |
| 05 | eventos | Cómo se ven los eventos y las fallas en los datos |
| 06 | modelo_pronostico | Modelo de pronóstico a 24 h y evaluación contra la meta |
