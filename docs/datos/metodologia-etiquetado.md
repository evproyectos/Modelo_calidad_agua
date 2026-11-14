# Metodología de etiquetado (RF-001.6)

Este documento explica cómo se arma el vector de etiquetas supervisadas: qué marca cada etiqueta,
de dónde sale y cuántas hay de cada clase. Hay dos conjuntos de etiquetas, y conviene no
confundirlos:

1. **Las etiquetas con que se entrenan los modelos**, que vienen de la verdad del simulador. Con la
   red simulada se sabe exactamente qué evento hubo, dónde y cuándo.
2. **El vector `etiqueta` / `fuente_etiqueta` que pide el RF-001.6**, armado con resultados de
   laboratorio, registros de la ASADA y una regla sobre la norma. Es el que servirá cuando lleguen
   datos reales, que no traen esa verdad. Con los datos simulados solo se puede probar la regla.

## 1. Lo que hay hoy: datos simulados

El dataset sale de `Simulador_sensor` (red D-Town, dataset v6: sep 2024 – ago 2026, una lectura
cada 15 min). Para la red simulada **no existen resultados de laboratorio del Ministerio de Salud
ni registros de fugas de la ASADA**, así que las fuentes (a) y (b) del RF-001.6 no tienen datos.
El simulador sí deja la verdad de cada evento:

| Archivo o columna | Qué tiene |
|---|---|
| `eventos_catalogo.csv` | Los 160 eventos simulados: tipo, causa, inicio, fin, punto de la red, intensidad y, si la tiene, la señal previa (`inicio_precursor`) |
| `evento_<nodo>` | Por lectura y sensor, el evento que ese sensor **ve** (0 = normal, 1 = contaminación, 2 = falla de cloración, 3 = turbidez en la fuente, 4 = falla de pH, 5 = fuga) |
| `evento_red` | El evento que hay en la red aunque ningún sensor lo vea |
| `falla_<parámetro>_<nodo>` | Fallas del instrumento (deriva, valor pegado, ruido, sin dato). **No son eventos**: sirven para que el modelo aprenda a no confundirlas |

Eventos del catálogo por tipo: contaminación 58, fuga 56, turbidez en la fuente 21, falla de
cloración 13 y falla de pH 12. Por partición: 108 en el entrenamiento (sep 2024 – dic 2025), 25 en
la validación (ene – abr 2026) y 27 en la prueba (may – ago 2026), según `carga.periodos`.

## 2. Etiquetas con que se entrenan los modelos

Cada modelo responde una pregunta distinta, así que cada uno tiene su propia etiqueta, construida
a partir de la verdad del simulador:

| Modelo | Etiqueta | Unidad | Dónde se construye |
|---|---|---|---|
| Detector de eventos (notebook 07) | `clase`: el tipo de evento que el sensor ve en esa hora (0 a 5) | hora × sensor | `detector.tabla`, a partir de `evento_<nodo>` |
| Anticipación a 24 h (notebook 09) | `clase`: el tipo de evento que **empieza** en la red en las próximas 24 h (0 si ninguno) | hora | `anticipacion.objetivo`, a partir del catálogo |
| Fugas (notebook 10) | 1 mientras hay una fuga abierta (del catálogo), 0 si no | hora | `fugas.clase` |

Además:

- **Anticipación.** Solo cuentan las horas útiles (`anticipacion.filas_utiles`): se quitan las
  horas en que ya hay un evento en curso, porque ahí ya no hay nada que anticipar.
- **Fugas.** La regresión de la operación normal se ajusta solo con horas lejos de una fuga. Se
  quitan la fuga pequeña que crece antes de la rotura y las 12 h después de la reparación,
  mientras los tanques se recuperan.

**Distribución de clases** (con los 10 sensores del sistema):

| Etiqueta | Entrenamiento | Validación | Prueba |
|---|---|---|---|
| Detector: horas × sensor con evento | 12 271 de 116 880 (10,5 %) | 3 045 de 28 800 (10,6 %) | 2 557 de 29 520 (8,7 %) |
| Anticipación: horas útiles con un evento que empieza en 24 h | 2 592 de 11 688 (22,2 %) | 600 de 2 880 (20,8 %) | 648 de 2 952 (22,0 %) |
| Fugas: horas con fuga abierta | 946 de 11 688 (8,1 %) | 318 de 2 880 (11,0 %) | 207 de 2 952 (7,0 %) |

Por lectura de 15 min y sensor, el 90,8 % es operación normal. El resto se reparte así: fuga
4,2 %, turbidez en la fuente 1,6 %, falla de pH 1,3 %, falla de cloración 1,1 % y contaminación
0,9 %. El EDA lo muestra en el notebook 05, sección 2, con los 21 sensores del simulador.

**Balanceo.** Cada tipo de evento está por debajo del 5 % de las lecturas. En vez de crear
lecturas sintéticas con SMOTE, que en una serie de tiempo inventaría estados de la red que no
ocurrieron, los modelos pesan cada clase según su frecuencia. Equivale a sobremuestrear los
eventos sin inventar datos:

| Modelo | Cómo se balancea |
|---|---|
| Detector | `class_weight: balanced` (`hiperparametros.yaml`) |
| Anticipación, modelos de árboles | `class_weight: balanced` |
| Anticipación, LSTM | Peso por muestra: horas normales / horas con evento |
| Fugas, SVM | Sin pesar las clases. La búsqueda de hiperparámetros (RF-002.4) eligió `class_weight=None`: pesar las clases la hacía avisar demasiado |

## 3. El vector `etiqueta` / `fuente_etiqueta` (para los datos reales)

`calidad_agua/etiquetas.py` arma, para cada lectura de 15 min, `etiqueta` (0 = operación normal,
1 = evento crítico) y `fuente_etiqueta`. Las fuentes van de la más confiable a la menos:

| Fuente | Regla | Datos que espera |
|---|---|---|
| `laboratorio` | Un muestreo del Ministerio de Salud que incumple el Decreto 38924-S marca con 1 las lecturas de ese sensor en la hora del muestreo. Uno que cumple las marca con 0, aunque la regla del umbral diga otra cosa | Tabla con `fecha`, `nodo`, `parametro`, `valor` |
| `registro_asada` | Una fuga reportada marca con 1 las lecturas entre su inicio y su fin, en su sensor o, si no se sabe, en toda la red | Tabla con `inicio`, `fin` y, si se sabe, `nodo` |
| `umbral_iot` | Si no hay laboratorio para ese momento: 3 o más lecturas seguidas de un sensor fuera de la norma (cloro < 0,3 mg/L, turbidez > 5 UNT o pH fuera de 6,5 – 8,5) | Las mismas lecturas |

Detalles de la regla:

- **Incumplimiento crónico.** Si un sensor pasa más de la mitad del tiempo fuera de la norma en un
  parámetro, ese parámetro no cuenta para la regla en ese sensor: ahí la norma no distingue un
  evento de lo de siempre. En los datos simulados pasa con el cloro de J144, al final de la red.
- **Varios sensores en la misma lectura.** La etiqueta de la fila es 1 si algún sensor la marca. La
  fuente de la fila es la más confiable entre los sensores que la deciden.
- **Dataset maestro.** `herramientas/dataset_maestro.py` agrega `etiqueta` y `fuente_etiqueta`
  (`docs/datos/esquema-dataset.md`).

**Volumen de cada fuente en los datos simulados** (70 077 lecturas de los 10 sensores):

| Fuente | Lecturas | Con etiqueta 1 |
|---|---|---|
| `laboratorio` | 0 (no hay datos) | — |
| `registro_asada` | 0 (no hay datos) | — |
| `umbral_iot` | 70 077 | 9 105 (13,0 %) |

**Qué tan bien funcionaría la regla sola**, comparada con la verdad del simulador en los 10
sensores (notebook 05, sección 7):

| Evento | Lecturas del evento | Marcadas por la regla | Aciertos | Precisión | Sensibilidad |
|---|---|---|---|---|---|
| Contaminación | 5 980 | 23 198 | 4 621 | 19,9 % | 77,3 % |
| Fuga | 29 620 | 23 198 | 87 | 0,4 % | 0,3 % |
| Cualquier evento de calidad | 34 561 | 23 198 | 19 677 | 84,8 % | 56,9 % |

La regla marca bien los eventos de calidad del agua: 85 % de lo que marca es de verdad un evento,
aunque se le escapa algo menos de la mitad. Las fugas no las ve en absoluto, porque una fuga baja
la presión y no saca al agua de la norma. Por eso **las fugas solo se pueden etiquetar con los
registros de la ASADA**: sin ellos, el modelo de fugas no tendría con qué entrenarse con datos
reales. Y como "contaminación" la regla marca también fallas de cloración, de pH y turbidez en la
fuente, que igual son incumplimientos de la norma.

## 4. Lo que falta para los datos reales

1. **Laboratorio.** Pedir a la ASADA los resultados de laboratorio del Ministerio de Salud de 2026,
   con fecha, punto de muestreo, parámetro y valor, en el formato de la tabla de arriba.
2. **Fugas.** Pedir el registro de fugas reportadas, con inicio, fin y lugar (aunque sea
   aproximado), en el formato de la tabla de arriba.
3. **Reentrenar.** Construir el vector con `etiquetas.construir(lecturas, laboratorio, registros)` y
   reentrenar los modelos con él (`herramientas/reentrenar.py`). Con datos reales no existe la
   verdad del simulador: estas etiquetas reemplazan a las de la sección 2.

## Criterios de aceptación del RF-001.6

| Criterio | Estado |
|---|---|
| (1) Columna `etiqueta` con valores 0 y 1 en el dataset maestro | Cumple: `herramientas/dataset_maestro.py`, con `fuente_etiqueta` |
| (2) Este documento, con la lógica y el volumen de cada fuente | Cumple |
| (3) Distribución de clases en el EDA antes del modelado | Cumple: notebook 05, sección 2 (desbalance) y sección 7 (vector de etiquetas) |
| (4) Aprobación de la metodología por el tutor académico | Pendiente |
