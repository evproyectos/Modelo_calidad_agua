# Variables de los modelos (RF-001.5)

Todas las variables se calculan con **información disponible hasta la hora t** (nada del futuro),
a partir de los promedios horarios de las lecturas ya limpias y preprocesadas
(`calidad.lecturas_operativas`). Las calcula el mismo código al entrenar y en el servicio en tiempo
real, así que se generan solas para cada dato nuevo. Notación: `x(t)` es el promedio horario de
un parámetro en un sensor; `típico(h)` es su mediana en operación normal a la hora del día `h`,
calculada en el entrenamiento.

## Lo que pide el requisito y dónde está

| Pedido (RF-001.5) | Variables | Modelos | Justificación |
|---|---|---|---|
| (a) Promedios móviles de 1, 6 y 24 h | Promedio horario de cada lectura (1 h); `<p>_media6h`, `<p>_media24h`; desvíos en medias de 3 y 12 h; residuos de fuga en medias de 3, 6, 12 y 24 h | pronóstico, detector, anticipación, fugas | Un evento o una fuga duran horas: el promedio separa un cambio sostenido del ruido de una lectura |
| (b) Tasa de cambio de turbidez, presión y pH | `cambio_<p>_1h`, `cambio_<p>_3h` = x(t) − x(t−k), para cloro, turbidez, pH y temperatura; `cambio_P_1h`; `caudal_cambio1h/3h`, `presion_<n>_cambio1h/3h` | detector, pronóstico, fugas | El inicio de un evento es un cambio brusco; la deriva de un sensor, uno lento |
| (c) Diferenciales de presión entre nodos | `presion_<n>_residuo` y `presion_<n>_vs_semana` por nodo (presión medida − esperada), y la mediana de la red `red_desvio_mediano_P`; el modelo de tramo los compara entre nodos | fugas, detector | Una fuga baja más la presión de los nodos cercanos: el patrón de caídas entre nodos dice en qué tramo está. Se usa la caída contra lo esperado y no la resta directa entre dos nodos, porque la diferencia de altura entre nodos ya separa sus presiones (decenas de metros) y taparía la caída de una fuga |
| (d) Lluvia acumulada de 6, 12 y 24 h | `lluvia_6h`, `lluvia_12h`, `lluvia_24h` (y 72 h, 48 h, 30 días y la lluvia pronosticada en las próximas 24 h por tramos) | detector, pronóstico, anticipación, fugas | La turbidez en la fuente sube con los aguaceros, y el suelo saturado favorece la intrusión de contaminación |
| (e) Indicadores cíclicos de tiempo | `hora_sin` = sin(2π·h/24), `hora_cos` = cos(2π·h/24); en fugas, la hora (24 indicadores) y si es fin de semana | todos | El consumo, la presión y el cloro tienen un ciclo diario fuerte (notebook 03). **No se usan el día de la semana ni el mes:** el caudal es casi igual todos los días (263–271 L/s, notebook 03, sin ciclo semanal), y con 16 meses de entrenamiento el modelo ve cada mes una sola vez y aprende cómo fue ese año en particular, no la época (notebook 06). La época entra por la lluvia y la temperatura |

## Variables por modelo

### Pronóstico de cloro, turbidez y pH a 24 h (`variables.py`, notebooks 06 y 08)

Una fila por sensor y hora; el objetivo es el valor en t + 24 h.

| Variable | Fórmula |
|---|---|
| `<p>` | x(t), para cloro, turbidez, pH y temperatura |
| `<p>_tipico` | mediana del sensor en operación normal (entrenamiento) |
| `<p>_hace24h`, `<p>_hace23h` | x(t−24), x(t−23): el mismo momento del ciclo diario |
| `<p>_cambio{1,3,6}h` | x(t) − x(t−k) |
| `<p>_media{6,24}h`, `<p>_desv{6,24}h` | media y desviación estándar de las últimas k horas |
| `<p>_vs_fuente` | x(t) − x_J280(t): cuánto cambió el agua desde la planta |
| `fuente_<p>`, `fuente_<p>_hace{3,6,12,24}h` | la fuente ahora y hace k horas (los cambios llegan a los sensores con 1 a 16 h de retraso) |
| `red_<p>_mediana` | mediana de los 10 sensores |
| `caudal`, `caudal_media24h` | caudal de la fuente y su media de 24 h |
| `lluvia_{6,12,24,72}h` | lluvia acumulada |
| lluvia pronosticada | por tramos de las próximas 24 h (`clima.py`) |
| `hora_sin`, `hora_cos` | de la hora objetivo (t + 24 h) |
| `P`, `lecturas_hora` | presión del sensor y lecturas válidas en la hora (calidad del dato) |

### Detector de eventos (`detector.py`, notebook 07)

Una fila por sensor y hora; la clase es el tipo de evento visible en ese sensor.

| Variable | Fórmula |
|---|---|
| `desvio_<p>` | x(t) − típico(h): cuánto se aparta de lo normal a esa hora |
| `desvio_<p>_media{3,12}h` | media móvil del desvío |
| `cambio_<p>_{1,3}h`, `desv_<p>_6h` | tasa de cambio y variabilidad |
| `desvio_<p>_vs_red` | desvío del sensor − mediana de los desvíos de la red |
| `parametros_desviados` | cuántos parámetros del sensor se apartan a la vez (un evento mueve varios; una falla del sensor, uno) |
| `red_sensores_desviados_<p>`, `red_desvio_mediano_<p>` | cuántos sensores se apartan y cuánto (los eventos de la fuente se ven en toda la red) |
| `fuente_desvio_<p>`, `..._hace{3,6,12}h` | la fuente ahora y antes |
| `caudal`, `caudal_desvio`, medias de 3 y 12 h, `caudal_cambio_3h` | la fuente entrega más con una fuga |
| `P`, `cambio_P_1h`, `desvio_P`, medias de 3 y 12 h, `red_desvio_mediano_P` | la presión baja con una fuga |
| `lluvia_{6,12,24}h`, `hora_sin`, `hora_cos`, `lecturas_hora`, `CL2_tipico`, `sensor` | contexto |

### Anticipación de eventos a 24 h (`anticipacion.py`, notebook 09)

Una fila por hora (toda la red); la clase es el tipo del primer evento que empieza en las
próximas 24 h. Variables de la fuente (valor, desvío, medias de 6 y 24 h, cambios de 3 a 24 h,
tendencia: media de las últimas 6 h − media de 6 h que terminó 18 h antes), de la red (medianas,
sensores con cloro bajo, sensores con presión baja, presión mínima), del caudal y la presión
(desvíos en medias de 6 y 24 h y su cambio contra el día anterior: una fuga que crece), de la lluvia
(1, 6, 12, 24, 48, 72 h, 30 días, máximo de 6 h en el día, pronóstico) y la hora.

### Fugas (`fugas.py`, notebook 10)

Una fila por hora. Primero se estima lo **esperado** de cada señal (caudal de la fuente y presión
de cada sensor) de dos formas: una regresión lineal con la hora del día, el fin de semana, la
temperatura y la lluvia de 24 h, ajustada solo con horas sin fuga, y la mediana de la misma hora en
los 7 días anteriores. Las variables son los **residuos** (medido − esperado) de cada señal y sus
medias de 3, 6, 12 y 24 h, los cambios de 1 y 3 h y la hora del día (46 variables con 3 sensores;
con 10 sensores, una serie más por cada sensor nuevo).

## Normalización y ventanas deslizantes (RF-001.3 c y d)

- Los modelos de árboles (bosques aleatorios, HistGradientBoosting) no necesitan normalizar.
- La SVM y el autoencoder de fugas usan `StandardScaler` dentro del modelo.
- Las redes LSTM (notebooks 08 y 09) estandarizan cada variable con la media y la desviación del
  entrenamiento y reciben **ventanas deslizantes** de las últimas 48 h. El largo se cambia en
  `hiperparametros.yaml` (`lstm.largo_h` en el pronóstico y `anticipacion.lstm.largo_h` en la
  anticipación: 24, 48 o 72 h).

## Importancia de las variables

La importancia de cada variable (permutación sobre la validación) está en los notebooks 06
(pronóstico), 07 (detector) y 09 (anticipación).
