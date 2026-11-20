# Comparación de modelos de contaminación (RF-002.1)

Este reporte compara los modelos que vigilan la calidad del agua y dice cuáles se eligieron para el
servicio y cómo. Como el RF-002.1 pide dos cosas, el sistema las resuelve con dos modelos que
trabajan juntos:

| Qué | Modelo | Notebook | En el servicio |
|---|---|---|---|
| **Anticipar** un evento de calidad con 24 h de antelación | LSTM, uno por tipo de evento | 09 | Aviso amarillo ("puede venir un evento") |
| **Detectar** el evento cuando ya llega a los sensores | HistGradientBoosting | 07 | Alarma roja |

Tipos de evento de calidad:

- **Contaminación**: entrada de un contaminante en un punto de la red.
- **Falla de cloración**, **turbidez en la fuente** y **falla de pH**: empiezan en la planta y
  recorren la red. Son superaciones de la norma del Decreto 38924-S en cloro residual, turbidez y
  pH. Los sensores no miden conductividad.

Los dos modelos se entrenaron con los 10 sensores del sistema y la lluvia de Open-Meteo (RNF-009).
Los números son del periodo de prueba (may – ago 2026), que no se usó para decidir nada, y salen de
la corrida en Docker en la PC del proyecto (`docker compose run --rm reentrenar`, nov 2026): son los
modelos que usa el servicio, versión 1 en el Model Registry de MLflow. Todo quedó en MLflow:

- Anticipación: experimento `anticipacion_eventos`, corridas `09 · LSTM`, `09 · bosque aleatorio`,
  `09 · árboles` y `09 · Isolation Forest`.
- Detección: experimento `detector_eventos`, corridas `07 · …`.

## Cómo se mide

Cada métrica se calcula de dos formas:

- **Por evento**, que es lo que le importa a la ASADA. **Precisión**: avisos o alarmas que
  correspondían a un evento real. **Sensibilidad (recall)**: eventos avisados o detectados. **F1**:
  su promedio armónico.
- **Por hora**. En la anticipación, aviso encendido contra "empieza un evento en las próximas
  24 h". En la detección, alarma de un sensor contra "ese sensor ve un evento".

Por hora se reportan también la **exactitud (accuracy)** y el **AUC-ROC** de la probabilidad. La
exactitud por hora engaña con clases tan desbalanceadas: un modelo que nunca avisa tendría 78 % en
la anticipación y 91 % en la detección. Por eso las decisiones se toman con las métricas por evento.

**Cómo se eligió.** Cada modelo tiene su umbral (y horas de confirmación) elegido con validación por
bloques. El entrenamiento (sep 2024 – dic 2025) se parte en 4 bloques de 4 meses: cada bloque se
predice con un modelo entrenado con los demás. La validación oficial (ene – abr 2026) es el quinto
bloque. La regla quedó fijada antes de ver resultados:

- Anticipación: más eventos con causa anticipados con a lo sumo 2 avisos falsos por semana en cada
  bloque.
- Detección: más eventos detectados con a lo sumo 1 falsa alarma por semana en cada bloque.

Una sola validación de 4 meses (la oficial es la época seca, sin turbidez en la fuente) daba
umbrales que fallaban en la época lluviosa. En los dos casos el modelo elegido es también el de
mejor F1 en la validación, como pide el RF-002.1 (ver abajo).

## 1. Anticipación a 24 h (notebook 09)

Se comparan los tres algoritmos que pide el RF-002.1 y uno más, todos con las mismas 72 variables:

- **Random Forest** (bosque aleatorio), **HistGradientBoosting** (árboles) y **LSTM** (TensorFlow/
  Keras, secuencias de 48 h, tres redes con semillas distintas): uno por tipo de evento.
- **Isolation Forest**: sin etiquetas. No dice qué tipo de evento viene.

Los dos de árboles tienen los hiperparámetros de la búsqueda al azar del RF-002.4
(`docs/modelos/validacion-cruzada.md`).

**Validación por bloques** (20 meses, 133 eventos, 85 con causa):

| Modelo | Con causa anticipados | Avisos falsos por semana | Precisión por evento | Sensibilidad por evento | **F1 por evento** | AUC-ROC por hora |
|---|---|---|---|---|---|---|
| **LSTM (elegido)** | **31 de 85** | 1,11 | 50,0 % | 29,3 % | **37,0 %** | 0,58 |
| HistGradientBoosting | 28 de 85 | 0,91 | 54,8 % | 27,1 % | 36,2 % | 0,56 |
| Isolation Forest (sin tipo) | 40 de 85 | 1,77 | 34,7 % | 33,8 % | 34,3 % | 0,56 |
| Random Forest | 26 de 85 | 0,72 | 53,6 % | 24,8 % | 33,9 % | 0,57 |

**Prueba** (may – ago 2026, 27 eventos, 16 con causa):

| Modelo | Anticipados | Con causa | Avisos (falsos) | Exactitud por hora | Precisión por evento | Sensibilidad por evento | F1 por evento | AUC-ROC por hora |
|---|---|---|---|---|---|---|---|---|
| **LSTM (elegido)** | **12 de 27** | **11 de 16** | 24 (13) | 66,2 % | 45,8 % | **44,4 %** | 45,1 % | 0,64 |
| Random Forest | 10 de 27 | 8 de 16 | 14 (3) | 66,2 % | 78,6 % | 37,0 % | **50,3 %** | 0,60 |
| Isolation Forest | 10 de 27 | 9 de 16 | 17 (7) | 65,9 % | 58,8 % | 37,0 % | 45,5 % | 0,64 |
| HistGradientBoosting | 8 de 27 | 8 de 16 | 9 (1) | 67,1 % | **88,9 %** | 29,6 % | 44,4 % | 0,64 |

Por hora, la LSTM tiene precisión de 64,0 %, sensibilidad de 19,8 % y F1 de 30,2 %. Los demás
modelos quedan entre 74 % y 99 % de precisión, 11 % y 12 % de sensibilidad y 20 % y 21 % de F1.

**Por qué la LSTM:**

- Es la que más eventos con causa anticipa en la validación por bloques, que es la regla.
- Es también la de **mejor F1 en la validación** (37,0 %).
- El Isolation Forest anticipa más en los bloques, pero con casi el doble de avisos falsos y sin
  decir qué tipo de evento viene, así que no sirve para decidir qué revisar.
- En la prueba, la LSTM anticipa más eventos (12 de 27, 11 de 16 con causa), pero con más avisos
  falsos: 1,98 por semana, justo debajo del límite de 2 de la regla. El Random Forest tiene mejor
  F1 en la prueba (50 % contra 45 %) porque avisa menos y se equivoca menos. Las diferencias son de
  2 a 4 eventos de 27 y la elección estaba fija antes de mirar la prueba, así que se reporta tal
  cual.

**La LSTM varía un poco entre máquinas.** Los modelos de árboles dan exactamente lo mismo en
cualquier computadora. La LSTM no: con la misma semilla, TensorFlow suma en otro orden según el
procesador, y eso cambia algunos avisos. En otra máquina, la misma LSTM anticipó 11 de 27 eventos
(10 de 16 con causa) con 1,22 avisos falsos por semana, y F1 de 46 %. Los números de este reporte
son los de la PC del proyecto, que son los modelos en servicio. Por eso cada versión del Model
Registry guarda la huella de sus archivos: lo que se evalúa es exactamente lo que se usa.

**Lo que anticipa la LSTM en la prueba**, por tipo y causa:

| Tipo | Causa | Anticipados | Antelación mediana |
|---|---|---|---|
| Falla de cloración | Deterioro gradual | 2 de 2 | 20,5 h |
| Falla de pH | Deterioro gradual | 2 de 2 | 5,5 h |
| Turbidez en la fuente | Tormenta | 2 de 3 | 15,5 h |
| Contaminación | Lluvia + baja presión | 4 de 5 | 24 h |
| Contaminación | Al azar | 0 de 7 | — |
| Fuga | Deterioro gradual | 1 de 4 | 7 h |
| Fuga | Al azar | 1 de 4 | 4 h |

El tipo esperado es correcto en 7 de los 12 avisos que acertaron. Los eventos **al azar** (11 de
los 27) no tienen ninguna señal previa en el simulador, así que ningún modelo puede anticiparlos.
Aunque se anticiparan todos los que tienen causa, la sensibilidad no pasaría de 16 de 27 (59 %).

## 2. Detección cuando el evento llega a los sensores (notebook 07)

Se comparan HistGradientBoosting, Random Forest e Isolation Forest con las mismas variables (desvíos
de cada sensor contra lo típico, tendencias y lluvia). Del Isolation Forest se probaron cuatro
versiones para darle una oportunidad justa: solo desvíos, entrenamiento sin eventos, umbral por
sensor y sin el efecto de la lluvia. Una LSTM para esta tarea no aporta: el evento ya se ve en la
hora actual. La LSTM se compara donde hace falta mirar hacia atrás, que es la anticipación.

**Validación por bloques** (20 meses):

| Modelo | Detectados | Falsas alarmas por semana | Peor bloque |
|---|---|---|---|
| **HistGradientBoosting (elegido)** | **66 %** | **0,33** | 0,66 |
| Random Forest | 59 % | 0,42 | 0,88 |
| Isolation Forest mejorado | 25 % | 0,39 | 0,86 |
| Isolation Forest | 24 % | 0,42 | 0,86 |

**Prueba** (25 eventos visibles, 16 de calidad y 9 fugas):

| Modelo | Detectados | Falsas por semana | Exactitud por hora | Precisión por evento | Sensibilidad por evento | F1 por evento | AUC-ROC por hora |
|---|---|---|---|---|---|---|---|
| **HistGradientBoosting (elegido)** | **17 de 25** | 0,22 | **94,8 %** | 92,9 % | **68,0 %** | **78,5 %** | **0,87** |
| Random Forest | 16 de 25 | 0,00 | 93,6 % | 100 % | 64,0 % | 78,1 % | 0,87 |
| Isolation Forest mejorado | 7 de 25 | 0,00 | 91,8 % | 100 % | 28,0 % | 43,8 % | 0,71 |
| Isolation Forest | 6 de 25 | 0,07 | 91,5 % | 90,0 % | 24,0 % | 37,9 % | 0,73 |

Por hora y sensor, HistGradientBoosting tiene precisión de 88,4 %, sensibilidad de 46,0 % y F1 de
60,5 %. El Random Forest tiene 77,6 %, 35,8 % y 49,0 %.

**Por qué HistGradientBoosting:**

- En la validación por bloques detecta más eventos y con menos falsas alarmas que el Random
  Forest. Gana en las dos cosas, así que también tiene mejor F1 en la validación.
- En la prueba quedan prácticamente empatados. HistGradientBoosting detecta además una fuga y avisa
  antes (0 h de retraso mediano contra 2 h).
- Pesa 2,1 MB contra 19 MB del bosque.
- El Isolation Forest no aprende cómo se ve cada tipo de evento: se pierde la mayoría.

Los **16 eventos de calidad** se detectan todos, con el tipo correcto en 15. En eventos de calidad,
el F1 es de 96 % (precisión 93 %, sensibilidad 100 %). Las fugas las detecta el modelo de fugas
(`docs/modelos/comparacion-fugas.md`).

## Criterios de aceptación del RF-002.1

| Criterio | Estado |
|---|---|
| (1) Precisión ≥ 80 % en la prueba del modelo seleccionado | **Detección: cumple** (92,9 % por evento, 88,4 % por hora). **Anticipación a 24 h: no cumple** (45,8 % por evento, 64,0 % por hora). Ver abajo |
| (2) Resultados en MLflow con parámetros, métricas y artefactos | Cumple: una corrida por modelo comparado, con umbral, métricas de bloques y de prueba; y la corrida del modelo guardado con el archivo, tablas y figuras |
| (3) Este reporte en `/docs/modelos/comparacion-contaminacion.md` | Cumple |
| (4) El notebook de entrenamiento corre sin errores en Docker | Cumple: los notebooks 07 y 09 corrieron completos en Docker en la PC del proyecto (`docker compose run --rm reentrenar`, nov 2026) |
| Comparar al menos Random Forest, Isolation Forest y LSTM | Cumple en la anticipación a 24 h (y HistGradientBoosting). En la detección: Random Forest, HistGradientBoosting e Isolation Forest |
| Exactitud, precisión, recall, F1 y AUC-ROC | Cumple: tablas de arriba |
| Elegir el de mejor F1 en la validación | Cumple: LSTM (37,0 %) y HistGradientBoosting (más detectados y menos falsas que el Random Forest) |

**Sobre la precisión de la anticipación.** La anticipación es el modelo más débil del sistema:

- **Eventos al azar.** 4 de cada 10 eventos de la prueba no tienen ninguna señal previa.
- **Lluvia aproximada.** Las contaminaciones dependen de la lluvia que cae en La Lucha, y la de
  Open-Meteo solo la aproxima. Con la lluvia de la estación del IMN, el mismo notebook llegaba a
  F1 de 51 % (`docs/modelos/fuente-lluvia.md`).
- **Avisos amarillos.** Por eso sus avisos son amarillos y no alarmas: la alarma roja la da el
  detector, que sí cumple el criterio.
- **Margen de mejora.** Lo que más lo mejoraría es medir la lluvia en La Lucha con un pluviómetro
  (`FUENTE_LLUVIA=lecturas` en el servicio) y tener eventos reales registrados.
