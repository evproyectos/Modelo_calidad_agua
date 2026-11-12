# Modelo de calidad de agua

Análisis exploratorio (EDA) y modelos de calidad de agua y fugas para una red de distribución
con **10 puntos de medición**: la salida de la fuente (J280, con el caudalímetro) y 9 puntos de la
red (J237, J224, J439, J1058, J69, J345, J101, J284 y J144). **La ubicación es provisional:** la
eligieron los datos (notebook 05, sección 6) mientras la ASADA define dónde van los sensores.
El simulador exporta 21 sensores; el sistema usa solo esos 10 (sección `red` de
`calidad_agua/hiperparametros.yaml`). Para cambiarlos, ver **Cambiar los sensores** más abajo.
Cada punto reporta cada 15 minutos:

| Parámetro | Columna | Unidad |
|---|---|---|
| Cloro residual libre | `CL2_<nodo>` | mg/L |
| Temperatura | `TEMP_<nodo>` | °C |
| Turbidez | `TURB_<nodo>` | UNT |
| pH | `PH_<nodo>` | — |
| Presión | `P_<nodo>` | m |

Además se registra el caudal de salida de la fuente (`Q_R1`, L/s) y la lluvia
(`lluvia_mm_h`).

Objetivos: pronosticar las próximas 24 h con al menos 80 % de exactitud usando también datos
climáticos; detectar eventos de calidad (contaminación, falla de cloración, turbidez en la
fuente, falla de pH) y fugas, y decir en qué tramo está la fuga; anticipar los eventos 24 y
48 h antes.

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

Los datos no se versionan en este repositorio. El dataset actual es el **v6** del simulador
(`Simulador_sensor`): 2 años de la red D-Town con cinco tipos de evento, entre
ellos 56 fugas de 5 a 30 L/s (la mitad empieza como una fuga pequeña que crece). Además de las
particiones, `distancias_sensores.csv` da la distancia por tubería de cada nodo a cada sensor
(para el tramo de las fugas).

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
modelos/      modelos entrenados (se generan con los notebooks 06 a 10; no se versionan: están
              en un release de GitHub, ver Publicar los modelos)
artefactos.json  release de GitHub con los modelos y la historia de MLflow, y la huella de cada archivo
herramientas/ correr_notebooks.py: corre todos los notebooks en orden y los guarda;
              reentrenar.py: vuelve a entrenar todos los modelos;
              publicar_modelos.py: empaqueta los modelos y MLflow para el release;
              dataset_maestro.py: exporta el dataset maestro preprocesado (Parquet y CSV)
docs/         esquema del dataset (docs/datos) y variables de los modelos (docs/modelos)
registros/    registro del preprocesamiento (lo escriben los notebooks; no se versiona)
```

## Preprocesamiento y etiquetas (RF-001.3 y RF-001.6)

Antes de los modelos, las lecturas pasan por `calidad.lecturas_operativas`, que se usa igual al
entrenar y en el servicio en tiempo real:

1. Reglas de operación: presión mínima, rango físico y sensor pegado (notebook 01).
2. `preprocesamiento.py`: quita los **picos** de cloro, turbidez, pH y temperatura con el método
   **IQR** (k = 1,5) sobre la diferencia con las lecturas vecinas, **interpola** los huecos de hasta
   30 min y **marca** los más largos en la columna `flag_imputacion` (`original`, `interpolado` o
   `marcado`). Cada aplicación queda en `registros/preprocesamiento.log`. Parámetros en
   `hiperparametros.yaml`, sección `preprocesamiento`; la evaluación está en el notebook 01.

`etiquetas.py` arma el vector de etiquetas (`etiqueta` 0/1 y `fuente_etiqueta`) con los resultados
de laboratorio, los registros de fugas de la ASADA o, si no hay, 3 lecturas seguidas fuera de la
norma. Con el simulador, los modelos usan su verdad, y el notebook 05 compara la regla contra ella.

El dataset maestro preprocesado, con las etiquetas, se exporta con un comando:

```
python herramientas/dataset_maestro.py        # datos_maestro/dataset_maestro.parquet y .csv
```

Esquema: `docs/datos/esquema-dataset.md`. Variables de cada modelo: `docs/modelos/features.md`.

## Cambiar los sensores

1. Escribir la lista en `calidad_agua/hiperparametros.yaml`, `red.sensores` (la fuente primero).
2. Si algún punto no está entre los 21 del simulador, generar de nuevo el dataset en
   `Simulador_sensor` con esos nodos.
3. `python herramientas/correr_notebooks.py 05` para ver cuántos eventos ve la lista nueva, y
   `python herramientas/reentrenar.py` para entrenar todo con ella. El servicio de monitoreo
   carga los modelos nuevos solo.
4. Revisar en el notebook 07 (sección 6) si el detector elegido sigue ganando: si el otro modelo
   detecta más con la misma regla, cambiar `detector.modelo` en `hiperparametros.yaml` (`bosque` o
   `arboles`) y correr `python herramientas/reentrenar.py 07`.
5. Publicar los modelos nuevos (abajo), para que Docker los use en cualquier máquina.

## Publicar los modelos

`modelos/`, `mlflow.db` y `mlartifacts/` no se versionan (pesan unos 150 MB y cambian con cada
reentrenamiento). Se publican como un **release** de GitHub de este proyecto, y
`artefactos.json` (que sí se versiona) dice cuál y con qué huella SHA-256. El Docker de
`Monitoreo_calidad_agua` descarga de ahí lo que falte en una máquina nueva y comprueba la huella.
Después de reentrenar:

1. `python herramientas/publicar_modelos.py modelos-AAAA-MM-DD` (con la fecha de hoy): deja
   `artefactos/modelos.zip` y `artefactos/mlflow.zip` y actualiza `artefactos.json`.
2. En GitHub, **Releases → Draft a new release**: etiqueta `modelos-AAAA-MM-DD` (la misma), y
   adjuntar los dos `.zip` de `artefactos/`. O con la CLI de GitHub:
   `gh release create modelos-AAAA-MM-DD artefactos/modelos.zip artefactos/mlflow.zip --title modelos-AAAA-MM-DD`
3. Commit y push de `artefactos.json`.

Hasta que el release exista, Docker no puede descargar los modelos en otra máquina (en esta los
tiene, porque `modelos/` ya está). En la copia de `mlflow.db` que va en el release, las rutas de los
archivos de cada corrida apuntan a `/modelo/mlartifacts`, donde los ve el MLflow de Docker.

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

Los hiperparámetros de los modelos (árboles del pronóstico, detector, LSTM, anticipación de eventos y fugas)
y las grillas con que se eligen los umbrales del detector y del aviso de anticipación están en **`calidad_agua/hiperparametros.yaml`**, la única
fuente: el código y los notebooks los leen de ahí. Si se cambia algo, hay que volver a correr los
notebooks 06 a 10 y copiar los modelos nuevos al servicio. La sección `red` dice qué sensores usa
el sistema; la variable de entorno `SENSORES` la cambia sin tocar el archivo (`SENSORES=todos`
usa los 21 del simulador).

Cada corrida de los notebooks 06 a 10 queda registrada en **MLflow** (`calidad_agua/experimentos.py`):

- **parámetros:** los del YAML, los periodos de cada partición y las decisiones de la corrida
  (umbral y confirmación elegidos, modelo elegido por parámetro);
- **métricas:** acierto en validación y en prueba por parámetro, curvas de aprendizaje de la LSTM,
  eventos detectados, falsas alarmas y métricas de clasificación del detector (precisión,
  sensibilidad, F1 y AUC-ROC);
- **trazabilidad:** huella SHA-256 de los datos (cada partición, `metadatos.json`, el catálogo de
  eventos y el pronóstico de lluvia, más una huella combinada), commit de git y si había cambios sin guardar, fecha,
  máquina y versiones de las librerías;
- **archivos:** los modelos guardados, las figuras y las tablas de resultados (CSV).

Las variantes que se comparan dentro de un notebook (versiones de clima en el 06, combinaciones de
umbral y confirmación y los otros modelos del detector en el 07, los cuatro modelos de anticipación en el 09) quedan como corridas hijas, para compararlas en la interfaz.
Cada modelo guardado lleva el id de su corrida, y el servicio lo muestra en `/health`.

Los registros quedan en el proyecto, en `mlflow.db` y `mlartifacts/`, que no se versionan. Para
verlos, desde la raíz del proyecto y con el venv activado:

```
mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1
```

(`--workers 1` hace falta en Windows: con varios procesos el servidor no arranca.)

y abrir **http://127.0.0.1:5000**. Experimentos: `pronostico_24h` (notebooks 06 y 08),
`detector_eventos` (notebook 07), `anticipacion_eventos` (notebook 09) y `fugas` (notebook 10).

## Pronóstico de lluvia

El pronóstico a 24 h usa la lluvia pronosticada para las próximas 24 h: el promedio de cuatro
modelos meteorológicos globales (ECMWF IFS, GFS, ICON y GEM) de Open-Meteo. Para entrenar se usan
los pronósticos **tal como estaban 24 h antes** de cada hora (Previous Runs API), guardados en
`clima/pronostico_lluvia_modelos.csv`; el enlace de descarga está en `calidad_agua/clima.py`. En
operación se usa el pronóstico vigente de los mismos cuatro modelos.

## Notebooks

Para correrlos todos en orden (y guardarlos con sus resultados), desde la raíz con el venv activado:

```
python herramientas/correr_notebooks.py           # 01 a 10
python herramientas/correr_notebooks.py 06 07     # solo algunos
```

| # | Notebook | Contenido |
|---|---|---|
| 01 | calidad_de_datos | Faltantes, cortes de comunicación, fallas de sensor; preprocesamiento (IQR, interpolación y `flag_imputacion`) y su evaluación |
| 02 | distribuciones | Rangos por parámetro y sensor, comparación con la norma; correlaciones de Pearson y Spearman |
| 03 | temporal | Ciclos diarios y semanales, estacionalidad, autocorrelación; descomposición STL |
| 04 | clima | Lluvia de la zona (IMN, estación ITCR San Carlos), su efecto en turbidez, cloro, pH y temperatura, cuánto aportaría un pronóstico de lluvia y qué pronóstico usar |
| 05 | eventos | Cómo se ven los eventos (también las fugas) y las fallas en los datos; dónde van los 10 sensores del sistema; vector de etiquetas |
| 06 | modelo_pronostico | Modelo de pronóstico a 24 h (árboles), qué información de clima usar y evaluación contra la meta |
| 07 | detector_eventos | Detector de eventos (Random Forest): alarmas, tipo de evento y falsas alarmas; comparación con HistGradientBoosting e Isolation Forest |
| 08 | red_neuronal | Red LSTM (TensorFlow/Keras), comparación con los árboles y modelo final de pronóstico |
| 09 | anticipacion_eventos | Probabilidad de que empiece un evento en las próximas 24 h (lluvia, deterioro en la fuente, baja presión): árboles, bosque aleatorio, Isolation Forest y LSTM; anticipación y avisos falsos; horizonte de 48 h; llegada de los eventos de la fuente a cada sensor |
| 10 | fugas | Fugas con el caudal de la fuente y la presión de los sensores: regresión de la operación normal y línea base de la semana; Random Forest, SVM y Autoencoder (percentil 95); tramo de la fuga entre dos sensores (RF-002.2) |
