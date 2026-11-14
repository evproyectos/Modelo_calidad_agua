# Comparación de modelos de fugas (RF-002.2)

Este reporte compara los modelos que detectan las fugas de la red con el caudal que entrega la
fuente y la presión de los sensores. Dice cuál se eligió para el servicio, cómo se eligió y en qué
tramo ubica cada fuga. Los números son del notebook 10 (`notebooks/10_fugas.ipynb`), entrenado con
los 10 sensores del sistema y la lluvia de Open-Meteo (RNF-009). Cada modelo quedó registrado en
MLflow, en el experimento `fugas`: corridas `10 · fugas`, `10 · Random Forest`, `10 · SVM` y
`10 · Autoencoder`.

## Diseño

1. **Lo esperado en operación normal: regresión multivariable** (scikit-learn, `fugas.ajustar_normal`).
   El caudal y la presión suben y bajan con la hora del día, el fin de semana, la época del año, el
   calor, la lluvia y los arranques de las bombas. Se ajusta una regresión por señal (caudal de la
   fuente y presión de cada sensor) con horas sin fuga del entrenamiento. Junto con la línea base
   de la semana (la mediana de la misma hora en los 7 días anteriores), da lo esperado. El
   **residuo** es lo que la operación normal no explica: es lo que separa una pérdida real de una
   variación operativa normal.
2. **Tres modelos con esos residuos** (123 variables):
   - **Random Forest**: supervisado, con las etiquetas de fuga.
   - **SVM** (kernel RBF): supervisado, con las etiquetas de fuga.
   - **Autoencoder** (TensorFlow/Keras): sin etiquetas. Aprende a reconstruir la operación normal y
     marca lo que no puede reconstruir. Su umbral es el percentil 95 del error en la operación
     normal (`docs/modelos/umbral-autoencoder.md`).
3. **Umbral y horas de confirmación de cada modelo** con validación por bloques. El entrenamiento se
   parte en 4 bloques de 4 meses y la validación oficial es el quinto bloque. La regla quedó fijada
   antes de ver resultados: **máximo de fugas detectadas con a lo sumo 1 falsa alarma por semana en
   cada bloque**; a igualdad, menos falsas y menos retraso. Se elige el modelo con más fugas
   detectadas bajo esa regla. La prueba no se usa para decidir.
4. **Tramo.** Un Random Forest aprende entre qué dos sensores está la fuga a partir del patrón de
   residuos. Se compara con la regla "los dos sensores donde más bajó la presión".

**Sobre la regresión multivariable.** El RF-002.2 la pide entre los modelos a evaluar. Aquí se usa
como primera etapa de los tres modelos, porque da lo que la operación normal explica. No es un
clasificador por sí misma. Qué tan bien explica cada señal en horas normales de la validación,
que no se usaron para ajustarla:

| Señal | R² | Error medio absoluto | Desviación del residuo |
|---|---|---|---|
| Caudal de la fuente | 0,44 | 14,0 L/s | 14,0 L/s |
| Presión J224 | 0,64 | 0,51 m | 0,62 m |
| Presión J439 | 0,58 | 6,70 m | 6,87 m |
| Presión J237 | 0,52 | 2,22 m | 3,67 m |
| Presión J1058 | 0,50 | 5,98 m | 6,04 m |
| Presión J101 | 0,43 | 4,66 m | 4,73 m |
| Presión J144 | 0,42 | 5,26 m | 5,33 m |
| Presión J345 | 0,30 | 3,89 m | 5,34 m |
| Presión J284 | 0,29 | 2,46 m | 5,14 m |
| Presión J69 | 0,29 | 1,59 m | 3,47 m |
| Presión J280 (fuente) | 0,00 | 0,08 m | 0,10 m |

El caudal de la fuente lo mandan sobre todo las bombas y los tanques, no solo el consumo, así que
la regresión explica menos de la mitad. Por eso se compara además contra la misma hora de la
semana anterior. En J280 la presión casi no varía: el R² es 0 porque no hay nada que explicar.

## Métricas

Por evento, como el detector de calidad:

| Métrica | Qué significa |
|---|---|
| Fugas detectadas (sensibilidad) | Fugas, desde que el tubo se rompe hasta que se repara, con alarma en alguna de esas horas |
| Precisión | Episodios de alarma que tocan una fuga, sobre todos los episodios de alarma |
| F1 | Promedio armónico de precisión y sensibilidad (RNF-003, meta ≥ 80 %) |
| Falsas alarmas por semana | Episodios de alarma que no tocan ninguna hora con fuga |
| Retraso | Horas desde que se rompe el tubo hasta la primera alarma |

## Resultados

**Validación por bloques** (sep 2024 – abr 2026, 20 meses, 48 fugas), con el umbral que eligió la
regla para cada modelo:

| Modelo | Umbral | Confirmación | Detectadas | Falsas por semana | Peor bloque | Precisión | F1 |
|---|---|---|---|---|---|---|---|
| **SVM (elegido)** | 0,5 | 2 h | **32 de 48 (67 %)** | 0,20 | 0,82 | 74 % | **70 %** |
| Random Forest | 0,6 | 3 h | 28 de 48 (58 %) | 0,28 | 0,58 | 64 % | 61 % |
| Autoencoder | 4 × percentil 95 | 3 h | 8 de 48 (17 %) | 0,09 | 0,47 | 53 % | 25 % |
| Autoencoder, percentil 95 sin ajustar | 1 × percentil 95 | 1 h | 39 de 48 (81 %) | 4,20 | 7,58 | 14 % | 24 % |

**Prueba** (may – ago 2026, 9 fugas). Cada modelo se entrena con todo el entrenamiento y se evalúa
una sola vez:

| Modelo | Detectadas | Alarmas | Falsas | Falsas por semana | Precisión | Sensibilidad | F1 | Retraso mediano |
|---|---|---|---|---|---|---|---|---|
| **SVM (elegido)** | **6 de 9** | 9 | 2 | 0,11 | **77,8 %** | 66,7 % | **71,8 %** | 4,5 h |
| Random Forest | 3 de 9 | 4 | 1 | 0,06 | 75,0 % | 33,3 % | 46,2 % | 18 h |
| Autoencoder | 1 de 9 | 1 | 0 | 0,00 | 100 % | 11,1 % | 20,0 % | 24 h |

Detalle de la SVM en la prueba, según el tamaño de la rotura:

| Caudal de la fuga | Fugas | Detectadas | Retraso mediano |
|---|---|---|---|
| 20 – 30 L/s | 3 | 3 | 4 h |
| 10 – 20 L/s | 3 | 2 | 23,5 h |
| 5 – 10 L/s | 2 | 0 | — |

## Tramo de la fuga

Con 10 puntos de medición no se puede ubicar una fuga al metro, pero sí decir entre qué dos
sensores está. Cada fuga del catálogo se asigna al tramo entre los dos sensores más cercanos por
tubería a su punto (`fugas.tramos`, con las distancias del simulador). Hay 7 tramos posibles. En
la prueba:

| Método | Tramo correcto |
|---|---|
| Random Forest de tramos (en el servicio) | 2 de 8 (25 %) |
| Regla "los dos sensores donde más bajó la presión" | 0 de 8 (0 %) |

El servicio lo informa así: "posible fuga entre J237 y J69". Acierta poco porque en el
entrenamiento hay pocas fugas por tramo (entre 46 y 329 horas de fuga según el tramo). Con
registros reales de la ASADA debería mejorar.

## Elección y criterios de aceptación

**Se eligió la SVM** (C = 10, gamma = 0,001, sin pesar las clases; hiperparámetros de la búsqueda
al azar del RF-002.4, `docs/modelos/validacion-cruzada.md`), con umbral 0,5 y 2 h de confirmación.
Es la que más fugas detecta en la validación por bloques sin pasar de 1 falsa alarma por semana en
ningún bloque, y la prueba lo confirma.

| Criterio del RF-002.2 | Estado |
|---|---|
| (1) Precisión ≥ 80 % en la prueba para la clase "fuga detectada" | **No se cumple por poco: 77,8 %** (7 de 9 alarmas eran una fuga; con una falsa alarma menos sería 87,5 %). Con 9 fugas en la prueba, cada alarma mueve más de 10 puntos |
| (2) Resultados registrados en MLflow | Cumple: experimento `fugas`, una corrida por modelo con su umbral y métricas, y la corrida del modelo guardado |
| (3) Indica el segmento afectado como tramo entre dos nodos | Cumple: "entre J237 y J69"; acierta 2 de 8 en la prueba |
| (4) Umbral del Autoencoder documentado y reproducible | Cumple: `docs/modelos/umbral-autoencoder.md` y `herramientas/umbral_autoencoder.py` |
| (5) Este reporte | Cumple |

**Antelación.** El RF-002.2 pide además detectar fugas con 24 h de antelación. Una fuga que
empieza pequeña y crece se intenta anticipar en el notebook 09, junto con los otros tipos de
evento, pero con los datos simulados no se anticipa ninguna en la prueba (0 de 4). El exceso de
caudal antes de la rotura es pequeño frente a la variación normal del consumo
(`docs/modelos/comparacion-contaminacion.md`). Este modelo detecta la fuga cuando el tubo ya se
rompió: 4,5 h después en mediana.

**Lo que más lo mejoraría:**

- **Fugas reales registradas.** El entrenamiento tiene solo 34 fugas simuladas, y cada sensor
  agrega 12 variables de presión.
- **Un medidor de caudal por sector.** Con solo el caudal de la fuente, una fuga pequeña se pierde
  en la variación del consumo.
