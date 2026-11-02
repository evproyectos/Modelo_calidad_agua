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
              pronóstico de lluvia, modelos (árboles, LSTM, detector), meta, figuras y
              registro de experimentos; hiperparametros.yaml tiene los hiperparámetros
clima/        pronósticos de lluvia pasados de Open-Meteo para la zona (se versionan)
notebooks/    EDA y modelos, numerados en el orden de lectura
pruebas/      pruebas unitarias (test_*.py) y la prueba de variables en vivo
.github/      integración continua en GitHub (estilo, pruebas, cobertura y secretos)
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

### Pruebas unitarias, cobertura y estilo

Las pruebas usan una red sintética pequeña creada en la propia prueba: no necesitan los datos
ni TensorFlow y corren en segundos. Cubren la carga, la limpieza (`calidad`), el pronóstico
de lluvia (`clima`), las variables del pronóstico (`variables`), el detector (`detector`), los
episodios y la meta. La cobertura de esos módulos debe ser de al menos 70 % (RNF-007);
`red_neuronal`, `modelo` y `graficos` quedan fuera de la medición.

```
pip install -e ".[dev]"
pytest --cov                 # pruebas y cobertura por módulo
flake8                       # estilo PEP 8 (líneas de hasta 120 caracteres)
```

En GitHub, el CI (`.github/workflows/ci.yml`) corre lo mismo en cada push y en cada pull
request, falla si la cobertura baja de 70 % y además busca secretos en el historial (gitleaks).

### Prueba de variables en vivo

Comprueba que las variables calculadas con solo los últimos 7 días de lecturas y
sin etiquetas (como en operación) son idénticas a las usadas al entrenar:

```
python pruebas/prueba_variables_en_vivo.py
```

## Hiperparámetros y experimentos (MLflow)

Los hiperparámetros de los tres modelos (árboles del pronóstico, detector y LSTM) y la grilla con
que se elige el umbral del detector están en **`calidad_agua/hiperparametros.yaml`**, la única
fuente: el código y los notebooks los leen de ahí. Si se cambia algo, hay que volver a correr los
notebooks 06, 07 y 08 y copiar los modelos nuevos al servicio.

Cada corrida de los notebooks 06, 07 y 08 queda registrada en **MLflow** (`calidad_agua/experimentos.py`):

- **parámetros:** los del YAML, los periodos de cada partición y las decisiones de la corrida
  (umbral y confirmación elegidos, modelo elegido por parámetro);
- **métricas:** acierto en validación y en prueba por parámetro, curvas de aprendizaje de la LSTM,
  eventos detectados, falsas alarmas y métricas de clasificación del detector (precisión,
  sensibilidad, F1 y AUC-ROC);
- **trazabilidad:** huella SHA-256 de los datos (cada partición, `metadatos.json` y el pronóstico
  de lluvia, más una huella combinada), commit de git y si había cambios sin guardar, fecha,
  máquina y versiones de las librerías;
- **archivos:** los modelos guardados, las figuras y las tablas de resultados (CSV).

Las variantes que se comparan dentro de un notebook (versiones de clima en el 06, combinaciones de
umbral y confirmación en el 07) quedan como corridas hijas, para compararlas en la interfaz.
Cada modelo guardado lleva el id de su corrida, y el servicio lo muestra en `/health`.

Los registros quedan en el proyecto, en `mlflow.db` y `mlartifacts/`, que no se versionan. Para
verlos, desde la raíz del proyecto y con el venv activado:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

(`--workers 1` hace falta en Windows: con varios procesos el servidor no arranca.)

y abrir **http://127.0.0.1:5000**. Experimentos: `pronostico_24h` (notebooks 06 y 08) y
`detector_eventos` (notebook 07).

## Pronóstico de lluvia

El pronóstico a 24 h usa la lluvia pronosticada para las próximas 24 h: el promedio de cuatro
modelos meteorológicos globales (ECMWF IFS, GFS, ICON y GEM) de Open-Meteo. Para entrenar se usan
los pronósticos **tal como estaban 24 h antes** de cada hora (Previous Runs API), guardados en
`clima/pronostico_lluvia_modelos.csv`; el enlace de descarga está en `calidad_agua/clima.py`. En
operación se usa el pronóstico vigente de los mismos cuatro modelos.

## Notebooks

| # | Notebook | Contenido |
|---|---|---|
| 01 | calidad_de_datos | Faltantes, cortes de comunicación, fallas de sensor |
| 02 | distribuciones | Rangos por parámetro y sensor, comparación con la norma |
| 03 | temporal | Ciclos diarios y semanales, estacionalidad, autocorrelación |
| 04 | clima | Lluvia de la zona (IMN, estación ITCR San Carlos), su efecto en turbidez, cloro, pH y temperatura, cuánto aportaría un pronóstico de lluvia y qué pronóstico usar |
| 05 | eventos | Cómo se ven los eventos y las fallas en los datos |
| 06 | modelo_pronostico | Modelo de pronóstico a 24 h (árboles), qué información de clima usar y evaluación contra la meta |
| 07 | detector_eventos | Detector de eventos: alarmas, tipo de evento y falsas alarmas |
| 08 | red_neuronal | Red LSTM (TensorFlow/Keras), comparación con los árboles y modelo final de pronóstico |
