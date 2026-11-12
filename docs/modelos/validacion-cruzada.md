# Validación cruzada y búsqueda de hiperparámetros (RF-002.3 y RF-002.4)

> Generado por `python herramientas/validacion_cruzada.py` el 11/11/2026 21:15 (95 min de cálculo) · commit `4cc0bf66` (con cambios sin commit) · datos sha256 `a0209416ebea`. No editar a mano: se vuelve a generar.

## Cómo se valida

**Time Series Split con k = 5** sobre las horas de entrenamiento y validación (sep 2024 – abr 2026). Cada pliegue entrena solo con el pasado (ventana creciente) y deja 24 h de separación antes del pliegue, para que las ventanas de las variables y la etiqueta de la anticipación (que mira 24 h adelante) no crucen. Lo que se ajusta con datos (valores típicos de cada sensor, regresión de la operación normal de las fugas) se recalcula en cada pliegue con su propio entrenamiento. El periodo de prueba (may–ago 2026) no se usa.

| Pliegue | Entrena | Valida | Meses de entrenamiento |
|---|---|---|---|
| 1 | 01/09/2024 – 10/12/2024 | 11/12/2024 – 22/03/2025 | 3.3 |
| 2 | 01/09/2024 – 21/03/2025 | 22/03/2025 – 01/07/2025 | 6.6 |
| 3 | 01/09/2024 – 30/06/2025 | 01/07/2025 – 10/10/2025 | 9.9 |
| 4 | 01/09/2024 – 09/10/2025 | 10/10/2025 – 19/01/2026 | 13.3 |
| 5 | 01/09/2024 – 18/01/2026 | 19/01/2026 – 01/05/2026 | 16.6 |

**Métrica: F1 por evento**, la de los notebooks. Precisión = alarmas (o avisos) que coinciden con un evento / alarmas; sensibilidad = eventos detectados (o anticipados) / eventos. El F1 depende del umbral y la confirmación: el del **modelo final** es el que usa el servicio; en la sensibilidad y la búsqueda cada configuración elige el suyo con la regla de los notebooks (el que más eventos detecta sin pasar del tope de falsas alarmas por semana en ningún pliegue), para comparar todas con las mismas reglas.

## Resumen

| Modelo | F1 por pliegue (media ± desv.) | Desv. ≤ 0.10 | F1 del modelo final (regla) | F1 de la mejor búsqueda | Mejora (puntos) | Decisión (RF-002.4) |
|---|---|---|---|---|---|---|
| Detector de eventos | 0.63 ± 0.06 | sí | 0.63 | 0.65 | +1.7 | el modelo final ya era óptimo |
| Anticipación a 24 h | 0.32 ± 0.08 | sí | 0.27 | 0.34 | +6.7 | se adopta la mejor configuración |
| Fugas | 0.49 ± 0.21 | no | 0.49 | 0.61 | +11.5 | se adopta la mejor configuración |

Criterios: RF-002.3 pide una desviación del F1 entre pliegues ≤ 0.10; RF-002.4, que la búsqueda mejore el F1 en al menos 2 puntos o que se documente que el modelo ya era óptimo. Los hiperparámetros del modelo final están en `calidad_agua/hiperparametros.yaml`.

![F1 por pliegue](../../figuras/validacion_pliegues.png)

## Detector de eventos

Hiperparámetros del modelo final: `{"n_estimators": "200", "min_samples_leaf": "20", "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": "-1", "random_state": "0"}`; umbral y confirmación: `(0.7, 3)`. Experimento de MLflow: `detector_eventos`.

### F1 por pliegue (modelo final)

| Pliegue | Eventos | Detectados | Alarmas | Falsas | Falsas/semana | Precisión | Sensibilidad | F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 20 | 8 | 16 | 2 | 0.18 | 88% | 40% | 0.55 |
| 2 | 20 | 13 | 32 | 10 | 0.88 | 69% | 65% | 0.67 |
| 3 | 24 | 14 | 24 | 1 | 0.09 | 96% | 58% | 0.73 |
| 4 | 18 | 10 | 22 | 6 | 0.51 | 73% | 56% | 0.63 |
| 5 | 23 | 13 | 22 | 8 | 0.69 | 64% | 57% | 0.60 |

Media 0.63, desviación estándar 0.06.

### Curva de aprendizaje

Último pliegue; se entrena con la parte más reciente del pasado.

| Meses | Desde | Eventos (entr.) | Error entrenamiento | Eventos (valid.) | Error validación |
|---|---|---|---|---|---|
| 3.3 | 2025-10-09 | 18 | 0.03 | 23 | 0.56 |
| 6.6 | 2025-06-30 | 41 | 0.05 | 23 | 0.58 |
| 9.9 | 2025-03-21 | 61 | 0.05 | 23 | 0.33 |
| 13.3 | 2024-12-11 | 80 | 0.05 | 23 | 0.34 |
| 16.6 | 2024-09-01 | 103 | 0.05 | 23 | 0.40 |

### Sensibilidad a los hiperparámetros

Un hiperparámetro a la vez; los demás, como en el modelo final.

| Hiperparámetro | Valor | F1 medio | Desv. | Detectados | De | Falsas/semana | Umbral elegido |
|---|---|---|---|---|---|---|---|
| n_estimators | 50 | 0.63 | 0.12 | 55 | 105 | 0.28 | (0.8, 2) |
| n_estimators | 100 | 0.63 | 0.10 | 54 | 105 | 0.23 | (0.8, 2) |
| n_estimators | 200 (final) | 0.63 | 0.06 | 58 | 105 | 0.47 | (0.7, 3) |
| n_estimators | 400 | 0.64 | 0.06 | 59 | 105 | 0.51 | (0.7, 3) |
| min_samples_leaf | 5 | 0.64 | 0.08 | 59 | 105 | 0.47 | (0.5, 3) |
| min_samples_leaf | 10 | 0.66 | 0.08 | 60 | 105 | 0.44 | (0.6, 3) |
| min_samples_leaf | 20 (final) | 0.63 | 0.06 | 58 | 105 | 0.47 | (0.7, 3) |
| min_samples_leaf | 40 | 0.59 | 0.15 | 53 | 105 | 0.40 | (0.8, 3) |
| min_samples_leaf | 80 | 0.64 | 0.17 | 58 | 105 | 0.33 | (0.9, 1) |
| max_features | sqrt (final) | 0.63 | 0.06 | 58 | 105 | 0.47 | (0.7, 3) |
| max_features | 0.25 | 0.64 | 0.09 | 56 | 105 | 0.38 | (0.8, 2) |
| max_features | 0.5 | 0.65 | 0.11 | 60 | 105 | 0.51 | (0.8, 2) |

### Búsqueda al azar (10 configuraciones y el modelo final)

| Configuración | n_estimators | min_samples_leaf | max_features | max_depth | class_weight | F1 medio | Desv. | Detectados | De | Falsas/semana | Segundos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| #6 | 100 | 20 | 0.33 | 15 | balanced | 0.65 | 0.13 | 59 | 105 | 0.42 | 76 |
| #7 | 200 | 80 | 0.33 | 30 | balanced | 0.65 | 0.14 | 63 | 105 | 0.65 | 137 |
| #3 | 300 | 10 | 0.2 | None | balanced | 0.63 | 0.09 | 56 | 105 | 0.35 | 169 |
| modelo final | — | — | — | — | — | 0.63 | 0.06 | 58 | 105 | 0.47 | 5 |
| #1 | 200 | 10 | sqrt | None | balanced | 0.63 | 0.11 | 59 | 105 | 0.49 | 72 |
| #2 | 200 | 40 | sqrt | None | balanced | 0.63 | 0.17 | 58 | 105 | 0.40 | 64 |
| #5 | 200 | 40 | 0.2 | 15 | balanced | 0.62 | 0.14 | 57 | 105 | 0.44 | 92 |
| #10 | 100 | 80 | 0.33 | 15 | balanced_subsample | 0.62 | 0.13 | 54 | 105 | 0.26 | 171 |
| #8 | 300 | 5 | 0.33 | None | balanced_subsample | 0.62 | 0.10 | 55 | 105 | 0.42 | 657 |
| #9 | 100 | 40 | 0.33 | 15 | balanced_subsample | 0.61 | 0.13 | 53 | 105 | 0.31 | 176 |
| #4 | 200 | 5 | sqrt | 15 | balanced | 0.55 | 0.19 | 46 | 105 | 0.19 | 63 |

Mejor: #6 (F1 0.65); modelo final con la misma regla: 0.63 → diferencia +1.7 puntos.

## Anticipación a 24 h

Hiperparámetros del modelo final: `{"n_estimators": "300", "min_samples_leaf": "10", "max_features": "sqrt", "class_weight": "balanced_subsample", "n_jobs": "-1", "random_state": "0"}`; umbral y confirmación: `{1: (0.6, 1), 2: (0.6, 2), 3: (0.5, 2), 4: (0.2, 3), 5: (0.5, 3)}`. Experimento de MLflow: `anticipacion_eventos`.

### F1 por pliegue (modelo final)

| Pliegue | Eventos | Detectados | Alarmas | Falsas | Falsas/semana | Precisión | Sensibilidad | F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 22 | 3 | 13 | 10 | 1.46 | 23% | 14% | 0.17 |
| 2 | 22 | 7 | 18 | 11 | 1.95 | 39% | 32% | 0.35 |
| 3 | 24 | 9 | 26 | 17 | 3.27 | 35% | 38% | 0.36 |
| 4 | 22 | 5 | 13 | 7 | 1.12 | 46% | 23% | 0.30 |
| 5 | 20 | 9 | 25 | 16 | 2.16 | 36% | 45% | 0.40 |

Media 0.32, desviación estándar 0.08.

### Curva de aprendizaje

Último pliegue; se entrena con la parte más reciente del pasado.

| Meses | Desde | Eventos (entr.) | Error entrenamiento | Eventos (valid.) | Error validación |
|---|---|---|---|---|---|
| 3.3 | 2025-10-09 | 21 | 0.07 | 20 | 0.88 |
| 6.6 | 2025-06-30 | 45 | 0.13 | 20 | 0.80 |
| 9.9 | 2025-03-21 | 67 | 0.13 | 20 | 0.68 |
| 13.3 | 2024-12-11 | 89 | 0.11 | 20 | 0.60 |
| 16.6 | 2024-09-01 | 112 | 0.09 | 20 | 0.60 |

### Sensibilidad a los hiperparámetros

Un hiperparámetro a la vez; los demás, como en el modelo final.

| Hiperparámetro | Valor | F1 medio | Desv. | Detectados | De | Falsas/semana | Umbral elegido |
|---|---|---|---|---|---|---|---|
| n_estimators | 100 | 0.29 | 0.10 | 25 | 110 | 1.18 | {1: (0.6, 1), 2: (0.3, 3), 3: (0.6, 1), 4: (0.6, 1), 5: (0.6, 3)} |
| n_estimators | 300 (final) | 0.27 | 0.11 | 26 | 110 | 1.47 | {1: (0.6, 2), 2: (0.3, 3), 3: (0.5, 1), 4: (0.3, 3), 5: (0.6, 3)} |
| n_estimators | 500 | 0.24 | 0.10 | 22 | 110 | 1.39 | {1: (0.6, 2), 2: (0.3, 3), 3: (0.6, 1), 4: (0.3, 3), 5: (0.6, 3)} |
| min_samples_leaf | 3 | 0.29 | 0.12 | 26 | 110 | 1.33 | {1: (0.5, 1), 2: (0.2, 3), 3: (0.4, 1), 4: (0.2, 3), 5: (0.6, 2)} |
| min_samples_leaf | 5 | 0.30 | 0.13 | 28 | 110 | 1.41 | {1: (0.5, 3), 2: (0.2, 3), 3: (0.4, 2), 4: (0.2, 3), 5: (0.6, 2)} |
| min_samples_leaf | 10 (final) | 0.27 | 0.11 | 26 | 110 | 1.47 | {1: (0.6, 2), 2: (0.3, 3), 3: (0.5, 1), 4: (0.3, 3), 5: (0.6, 3)} |
| min_samples_leaf | 20 | 0.28 | 0.08 | 25 | 110 | 1.18 | {1: (0.6, 2), 2: (0.4, 3), 3: (0.6, 2), 4: (0.5, 2), 5: (0.7, 3)} |
| min_samples_leaf | 40 | 0.21 | 0.12 | 18 | 110 | 0.81 | {1: (0.7, 1), 2: (0.5, 3), 3: (0.7, 2), 4: (0.6, 2), 5: (0.9, 1)} |
| max_features | sqrt (final) | 0.27 | 0.11 | 26 | 110 | 1.47 | {1: (0.6, 2), 2: (0.3, 3), 3: (0.5, 1), 4: (0.3, 3), 5: (0.6, 3)} |
| max_features | 0.25 | 0.31 | 0.11 | 26 | 110 | 1.13 | {1: (0.6, 1), 2: (0.3, 3), 3: (0.5, 3), 4: (0.7, 1), 5: (0.7, 1)} |
| max_features | 0.5 | 0.30 | 0.13 | 25 | 110 | 1.00 | {1: (0.6, 1), 2: (0.7, 3), 3: (0.6, 1), 4: (0.6, 3), 5: (0.8, 3)} |

### Búsqueda al azar (10 configuraciones y el modelo final)

| Configuración | n_estimators | min_samples_leaf | max_features | max_depth | class_weight | F1 medio | Desv. | Detectados | De | Falsas/semana | Segundos |
|---|---|---|---|---|---|---|---|---|---|---|---|
| #8 | 500 | 3 | 0.33 | None | balanced_subsample | 0.34 | 0.12 | 31 | 110 | 1.49 | 153 |
| #3 | 500 | 5 | 0.2 | None | balanced | 0.33 | 0.13 | 29 | 110 | 1.10 | 80 |
| #5 | 300 | 20 | 0.2 | 10 | balanced | 0.32 | 0.12 | 26 | 110 | 0.72 | 44 |
| #7 | 300 | 40 | 0.33 | 20 | balanced | 0.31 | 0.11 | 28 | 110 | 1.29 | 59 |
| #10 | 100 | 40 | 0.33 | 10 | balanced_subsample | 0.30 | 0.12 | 25 | 110 | 0.92 | 30 |
| #1 | 300 | 5 | sqrt | None | balanced | 0.30 | 0.09 | 28 | 110 | 1.30 | 35 |
| #9 | 100 | 20 | 0.33 | 10 | balanced_subsample | 0.30 | 0.12 | 27 | 110 | 1.31 | 30 |
| modelo final | — | — | — | — | — | 0.27 | 0.11 | 26 | 110 | 1.47 | 4 |
| #6 | 100 | 10 | 0.33 | 10 | balanced | 0.27 | 0.18 | 25 | 110 | 1.05 | 25 |
| #4 | 300 | 3 | sqrt | 10 | balanced | 0.23 | 0.15 | 22 | 110 | 1.09 | 34 |
| #2 | 300 | 20 | sqrt | None | balanced | 0.22 | 0.09 | 20 | 110 | 0.99 | 33 |

Mejor: #8 (F1 0.34); modelo final con la misma regla: 0.27 → diferencia +6.7 puntos.

## Fugas

Hiperparámetros del modelo final: `{"C": "1.0", "kernel": "rbf", "gamma": "scale", "class_weight": "balanced", "random_state": "0"}`; umbral y confirmación: `(0.3, 3)`. Experimento de MLflow: `fugas`.

### F1 por pliegue (modelo final)

| Pliegue | Eventos | Detectados | Alarmas | Falsas | Falsas/semana | Precisión | Sensibilidad | F1 |
|---|---|---|---|---|---|---|---|---|
| 1 | 11 | 1 | 1 | 0 | 0.00 | 100% | 9% | 0.17 |
| 2 | 7 | 2 | 9 | 6 | 0.42 | 33% | 29% | 0.31 |
| 3 | 7 | 5 | 15 | 6 | 0.42 | 60% | 71% | 0.65 |
| 4 | 7 | 4 | 8 | 2 | 0.14 | 75% | 57% | 0.65 |
| 5 | 11 | 9 | 26 | 11 | 0.76 | 58% | 82% | 0.68 |

Media 0.49, desviación estándar 0.21.

### Curva de aprendizaje

Último pliegue; se entrena con la parte más reciente del pasado.

| Meses | Desde | Eventos (entr.) | Error entrenamiento | Eventos (valid.) | Error validación |
|---|---|---|---|---|---|
| 3.3 | 2025-10-09 | 7 | 0.07 | 11 | 0.70 |
| 6.6 | 2025-06-30 | 14 | 0.03 | 11 | 0.63 |
| 9.9 | 2025-03-21 | 21 | 0.09 | 11 | 0.33 |
| 13.3 | 2024-12-11 | 31 | 0.08 | 11 | 0.27 |
| 16.6 | 2024-09-01 | 37 | 0.14 | 11 | 0.32 |

### Sensibilidad a los hiperparámetros

Un hiperparámetro a la vez; los demás, como en el modelo final.

| Hiperparámetro | Valor | F1 medio | Desv. | Detectados | De | Falsas/semana | Umbral elegido |
|---|---|---|---|---|---|---|---|
| C | 0.1 | 0.48 | 0.26 | 19 | 43 | 0.22 | (0.4, 2) |
| C | 0.3 | 0.44 | 0.24 | 19 | 43 | 0.36 | (0.3, 3) |
| C | 1.0 (final) | 0.49 | 0.21 | 21 | 43 | 0.35 | (0.3, 3) |
| C | 3.0 | 0.58 | 0.14 | 29 | 43 | 0.55 | (0.3, 2) |
| C | 10.0 | 0.61 | 0.15 | 30 | 43 | 0.54 | (0.3, 2) |
| gamma | scale (final) | 0.49 | 0.21 | 21 | 43 | 0.35 | (0.3, 3) |
| gamma | 0.001 | 0.57 | 0.18 | 22 | 43 | 0.18 | (0.3, 2) |
| gamma | 0.003 | 0.60 | 0.21 | 26 | 43 | 0.33 | (0.3, 2) |
| gamma | 0.01 | 0.51 | 0.16 | 22 | 43 | 0.37 | (0.3, 3) |
| gamma | 0.03 | 0.36 | 0.23 | 16 | 43 | 0.29 | (0.5, 1) |

### Búsqueda al azar (10 configuraciones y el modelo final)

| Configuración | C | gamma | class_weight | F1 medio | Desv. | Detectados | De | Falsas/semana | Segundos |
|---|---|---|---|---|---|---|---|---|---|
| #10 | 10.0 | 0.001 | None | 0.61 | 0.14 | 27 | 43 | 0.33 | 12 |
| #3 | 0.03 | scale | None | 0.58 | 0.11 | 22 | 43 | 0.25 | 16 |
| #8 | 3.0 | 0.0003 | balanced | 0.51 | 0.22 | 19 | 43 | 0.10 | 43 |
| modelo final | — | — | — | 0.49 | 0.21 | 21 | 43 | 0.35 | 1 |
| #6 | 0.3 | scale | balanced | 0.44 | 0.24 | 19 | 43 | 0.36 | 38 |
| #4 | 0.1 | 0.001 | balanced | 0.43 | 0.22 | 16 | 43 | 0.22 | 50 |
| #2 | 0.03 | 0.003 | balanced | 0.42 | 0.22 | 14 | 43 | 0.12 | 54 |
| #1 | 0.03 | 0.0003 | balanced | 0.41 | 0.12 | 18 | 43 | 0.42 | 62 |
| #9 | 10.0 | 0.03 | balanced | 0.36 | 0.19 | 15 | 43 | 0.30 | 36 |
| #7 | 1.0 | 0.03 | balanced | 0.36 | 0.23 | 16 | 43 | 0.29 | 36 |
| #5 | 0.1 | 0.003 | None | 0.31 | 0.25 | 13 | 43 | 0.21 | 14 |

Mejor: #10 (F1 0.61); modelo final con la misma regla: 0.49 → diferencia +11.5 puntos.

## Curvas de aprendizaje y sensibilidad

![Curvas de aprendizaje](../../figuras/validacion_curvas_aprendizaje.png)

![Sensibilidad](../../figuras/validacion_sensibilidad.png)

## Lectura de los resultados

Esta sección se escribe a mano (`docs/modelos/validacion-cruzada-lectura.md`) y el generador la agrega
al reporte. Corresponde a la corrida del 11/11/2026.

**Cómo leer las diferencias.** Con 5 pliegues y una desviación de 0,06 a 0,21 entre ellos, el error
estándar de un F1 medio es de 0,03 a 0,09: diferencias de pocos puntos entre configuraciones están
dentro del ruido. Además, quedarse con la mejor de 10 configuraciones favorece a la que tuvo suerte
en los pliegues. Por eso el resultado de la prueba (may–ago 2026), que la búsqueda no vio, es el que
confirma o no cada cambio.

**Detector de eventos (Random Forest).** Es el más estable: F1 de 0,63 ± 0,06 entre pliegues
(cumple la desviación ≤ 0,10). Es menor que en la prueba (78 %) porque los primeros pliegues
entrenan con pocos meses. La curva de aprendizaje lo muestra: con 3 a 7 meses el error de validación
es 0,56–0,58; desde unos 10 meses baja a 0,33–0,40 y ya no mejora mucho más. El error de
entrenamiento es casi cero (el bosque se aprende el entrenamiento), así que lo que limita es la
variedad de eventos, no el tamaño del modelo. La sensibilidad es plana: de 50 a 400 árboles el F1
queda en 0,63–0,64; con hojas de 5 a 20 lecturas, 0,63–0,66, y con 40 baja a 0,60 y varía más; con
25 % a 50 % de las variables por corte, 0,64–0,65. La mejor configuración de la búsqueda gana 1,7
puntos con el doble de desviación (0,13 contra 0,06): no llega a los 2 puntos, así que **el modelo
final ya era óptimo** y se mantiene (umbral 0,70, confirmación 3 h).

**Anticipación a 24 h (Random Forest por tipo).** Con el umbral que usa el servicio, F1 de 0,32 ±
0,08 (cumple la desviación). Con la regla de elección sobre los pliegues el modelo anterior da 0,27, y
la mejor configuración (500 árboles, hojas de 3, 33 % de las variables por corte) 0,34: **+6,7
puntos, se adoptó**. En la prueba cambia aciertos al azar por menos avisos falsos: anticipa 11 de 16
eventos con causa (antes 12) y 0 de 11 al azar (antes 3), con 0,91 avisos falsos por semana (antes
1,82); su F1 en la prueba baja de 59 % a 51 %. Se dejó la nueva configuración, por decisión del
proyecto: se respeta la regla fijada antes de mirar la prueba, los eventos al azar no tienen señal
previa (acertarlos era más por avisar seguido que por anticiparlos) y los avisos falsos eran el
problema pendiente de este modelo. La curva de aprendizaje todavía baja (error de 0,88 con 3 meses a
0,60 con 13 o más): más historia ayudaría, porque cada tipo tiene pocos eventos. La sensibilidad es
ruidosa: usar 25 % a 50 % de las variables por corte da 0,30–0,31 contra 0,27 con la raíz cuadrada,
y cambiar un solo hiperparámetro a la vez no siempre ayuda (500 árboles con hojas de 10 da 0,24): los
hiperparámetros interactúan.

**Fugas (SVM).** El modelo anterior es inestable: F1 de 0,49 ± 0,21 (no cumple). El primer pliegue
entrena con 3 meses y unas pocas fugas, y detecta 1 de 11; la curva de aprendizaje dice lo mismo
(error de 0,70 con 3 meses y de 0,27 a 0,33 desde 10 meses). La mejor configuración (C = 10, gamma =
0,001, sin pesar las clases) da 0,61 ± 0,14: **+11,5 puntos, se adoptó**, y la prueba lo confirma:
6 de 9 fugas con 1 falsa alarma en 4 meses, F1 de 76 % (antes 5 de 9 con 6 falsas, 50 %). Pesar las
clases hacía que la SVM avisara demasiado. Sigue sin cumplir la desviación ≤ 0,10 (0,14) y debajo de
la meta de F1 de 80 %: con pocas fugas para entrenar, los primeros pliegues detectan poco. Con datos
reales y más fugas registradas debería estabilizarse.

**Lo que queda para los requisitos.**

- RF-002.3: la validación cruzada temporal, las curvas de aprendizaje y la sensibilidad están en
  este reporte, que se genera solo y se puede repetir con Docker. La desviación ≤ 0,10 se cumple en
  el detector y la anticipación; en fugas no (0,14 con la configuración nueva).
- RF-002.4: en la anticipación y las fugas la búsqueda mejoró el F1 en más de 2 puntos y la
  configuración nueva está en `hiperparametros.yaml`; en el detector se documenta que ya era óptimo.
  Cada configuración probada quedó en MLflow, en el experimento de su modelo, con sus
  hiperparámetros, F1 y duración.
- Las secciones **F1 por pliegue (modelo final)** de la anticipación y las fugas son de los modelos
  antes de adoptar la búsqueda; los adoptados son las configuraciones #8 (anticipación) y #10
  (fugas) de cada tabla de búsqueda.

## Cómo repetirlo

Con el entorno de este proyecto: `python herramientas/validacion_cruzada.py`. Con Docker, desde esta carpeta: `docker compose run --rm validacion` (ver README, **Docker**). Pliegues, grillas y semilla: `hiperparametros.yaml`, sección `validacion`.
