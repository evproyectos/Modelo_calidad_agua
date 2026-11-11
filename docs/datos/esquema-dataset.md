# Esquema del dataset maestro (RF-001.2)

El dataset maestro es una tabla de **lecturas cada 15 minutos** de los sensores de la red, con el
caudal de la fuente, la lluvia y las etiquetas de evento y de falla. Hoy lo genera el simulador
(`Simulador_sensor`, red D-Town, dataset v6); con los sensores reales de la ASADA tendrá la misma
forma, sin las columnas de etiquetas que solo existen en la simulación.

## Pipeline

| Etapa | Qué hace | Dónde |
|---|---|---|
| Extracción | Lee los Parquet del simulador (o, en operación, las lecturas que llegan a la API de datos) y el pronóstico de lluvia de Open-Meteo para la estación 69-701 del IMN (ITCR, San Carlos) | `Simulador_sensor`, `calidad_agua/carga.py`, `calidad_agua/clima.py` |
| Sincronización | Índice de tiempo común de 15 min para todas las variables; las tablas horarias de los modelos se arman con promedios por hora (`resample("1h")`) | `carga.py`, `detector.horario`, `variables.horario` |
| Selección de sensores | Deja solo los 10 puntos del sistema (`hiperparametros.yaml`, sección `red`) | `carga.filtrar_sensores` |
| Limpieza | Reglas de operación (presión mínima, rango físico, sensor pegado) y preprocesamiento (picos por IQR, interpolación de huecos de hasta 30 min, `flag_imputacion`) | `calidad.lecturas_operativas`, `preprocesamiento.py` |
| Etiquetas | Vector `etiqueta` / `fuente_etiqueta` (laboratorio, registros de la ASADA o umbral IoT) | `etiquetas.py` |
| Exportación | Dataset maestro en Parquet y CSV, con su resumen y huella SHA-256 | `herramientas/dataset_maestro.py` |

**Un solo comando** arma el dataset maestro: `python herramientas/dataset_maestro.py` (lo deja en
`datos_maestro/`). Es **idempotente**: con las mismas entradas da los mismos archivos, y
`dataset_maestro.json` guarda su huella SHA-256 para comprobarlo. Los modelos se entrenan con
`python herramientas/reentrenar.py` (usan `random_state` fijo en `hiperparametros.yaml`, y la
huella de los datos queda en MLflow con cada corrida).

## Archivos

| Archivo | Contenido |
|---|---|
| `entrenamiento.parquet` | 01/09/2024 – 31/12/2025 (16 meses), lecturas cada 15 min |
| `validacion.parquet` | 01/01/2026 – 30/04/2026 (4 meses, época seca) |
| `prueba.parquet` | 01/05/2026 – 31/08/2026 (4 meses, inicio de la época lluviosa). No se usa para decidir nada |
| `verdad_<partición>.parquet` | Valor de referencia de cada parámetro, sin ruido ni fallas (solo simulación) |
| `sensores.csv` | Los 21 puntos que exporta el simulador, con su orden de prioridad |
| `distancias_sensores.csv` | Distancia por tubería (m) de cada nodo de consumo a cada sensor: define los tramos de fuga |
| `eventos_catalogo.csv` | Cada evento simulado: tipo, inicio, fin, ubicación, intensidad, lluvia previa |
| `fallas_sensor_catalogo.csv` | Cada falla de sensor simulada: tipo, sensor, parámetro, inicio y fin |
| `metadatos.json` | Paso de tiempo, particiones, códigos, unidades y modelo de error de cada sensor |
| `clima/pronostico_lluvia_modelos.csv` | Lluvia pronosticada 24 h antes por cuatro modelos meteorológicos (Open-Meteo) |

## Columnas de la tabla de lecturas

Índice: `fecha` (datetime, hora de Costa Rica, UTC−6), una fila cada 15 min, sin saltos ni
duplicados. `<nodo>` es el identificador del punto de medición (p. ej. `J280`).

| Columna | Tipo | Unidad | Rango operativo | Descripción |
|---|---|---|---|---|
| `CL2_<nodo>` | float32 | mg/L | 0 – 5 | Cloro residual libre |
| `TEMP_<nodo>` | float32 | °C | 10 – 40 | Temperatura del agua |
| `TURB_<nodo>` | float32 | UNT | 0 – 100 | Turbidez |
| `PH_<nodo>` | float32 | — | 4 – 10 | pH |
| `P_<nodo>` | float32 | m (columna de agua) | 0 – 150 | Presión en el nodo |
| `Q_R1` | float32 | L/s | — | Caudal de salida de la fuente (caudalímetro) |
| `lluvia_mm_h` | float32 | mm/h | ≥ 0 | Lluvia en la estación del IMN |
| `falla_<PARAM>_<nodo>` | int8 | — | 0 – 5 | Falla del sensor: 0 ninguna, 1 pegado, 2 deriva, 3 picos, 4 ruido, 5 faltante (solo simulación) |
| `evento_<nodo>` | int8 | — | 0 – 5 | Evento visible en ese sensor: 0 normal, 1 contaminación, 2 falla de cloración, 3 turbidez en la fuente, 4 falla de pH, 5 fuga (solo simulación) |
| `evento_visible` | int8 | — | 0 – 5 | Evento visible en algún sensor del sistema (se recalcula con los sensores elegidos) |
| `evento_red` | int8 | — | 0 – 5 | Evento ocurriendo en la red, aunque ningún sensor lo vea |
| `precursor_red` | int8 | — | 0 / 1 | Horas previas a un evento con señales precursoras (solo simulación) |

Fuera de rango o con presión menor a 1 m, la lectura de calidad se trata como faltante
(`calidad.py`). Las normas del Decreto 38924-S que usa el sistema: cloro 0,3 – 1,0 mg/L,
turbidez ≤ 5 UNT (recomendado ≤ 1), pH 6,5 – 8,5.

## Columnas que agrega el preprocesamiento

| Columna | Valores | Descripción |
|---|---|---|
| `flag_imputacion` | `original`, `interpolado`, `marcado` | Por fila: todas las lecturas medidas; alguna rellenada (hueco ≤ 30 min o pico quitado); alguna vacía por un hueco más largo |
| `etiqueta` | 0 / 1 | Evento crítico (contaminación o fuga) en esa lectura, en algún sensor |
| `fuente_etiqueta` | `laboratorio`, `registro_asada`, `umbral_iot` | De dónde sale la etiqueta (ver `etiquetas.py`) |

Cada aplicación del preprocesamiento queda registrada, con la fecha, los parámetros y cuántas
lecturas tocó cada paso, en `registros/preprocesamiento.log` (una línea JSON por aplicación).

## Cambiar la cantidad o la ubicación de los sensores

Los 10 puntos actuales son **provisionales** (elegidos con los datos, notebook 05, sección 6).
Cuando la ASADA defina dónde van:

1. Escribir la lista en `calidad_agua/hiperparametros.yaml`, `red.sensores` (la fuente primero).
2. Si algún punto no está entre los 21 que exporta el simulador, generar de nuevo el dataset en
   `Simulador_sensor` con esos nodos (y sus distancias por tubería).
3. Correr `python herramientas/reentrenar.py`: vuelve a entrenar todos los modelos y deja todo en
   MLflow. El servicio de monitoreo recarga los modelos nuevos solo, sin reiniciarlo.
4. En el notebook 07 (sección 6), revisar si el detector elegido sigue ganando; si no, cambiar
   `detector.modelo` (`bosque` o `arboles`) y correr `python herramientas/reentrenar.py 07`.
