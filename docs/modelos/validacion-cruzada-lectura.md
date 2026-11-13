## Lectura de los resultados

Esta sección se escribe a mano (`docs/modelos/validacion-cruzada-lectura.md`) y el generador la agrega
al reporte. Corresponde a las corridas del 12/11/2026, con la lluvia de Open-Meteo (RNF-009,
`docs/modelos/fuente-lluvia.md`), con la que se reentrenaron todos los modelos.

**Cómo leer las diferencias.** Con 5 pliegues y una desviación de 0,08 a 0,16 entre ellos, el error
estándar de un F1 medio es de 0,04 a 0,07: diferencias de pocos puntos entre configuraciones están
dentro del ruido. Además, quedarse con la mejor de 10 configuraciones favorece a la que tuvo suerte
en los pliegues. Por eso cada cambio que sugirió la búsqueda se miró también en la prueba
(may–ago 2026), que la búsqueda no vio, antes de adoptarlo.

**Las tablas son de los modelos antes de adoptar la búsqueda.** Las secciones *F1 por pliegue
(modelo final)*, la curva de aprendizaje y la sensibilidad de cada modelo se calcularon con la
configuración que tenía el modelo cuando se corrió la validación; las configuraciones adoptadas son
las mejores de cada tabla de búsqueda, y ya están en `calidad_agua/hiperparametros.yaml`.

**Detector de eventos (HistGradientBoosting).** Con la lluvia de Open-Meteo la regla del notebook 07
elige HistGradientBoosting en lugar del Random Forest, y la validación se hizo con él (grilla
`detector_arboles`). Es el más estable: F1 de 0,67 ± 0,08 entre pliegues (cumple la desviación
≤ 0,10). Es menor que en la prueba porque los primeros pliegues entrenan con pocos meses: la curva
de aprendizaje baja de un error de validación de 0,53 con 3 meses a 0,39–0,45 desde 13 meses. El
error de entrenamiento es casi cero, así que lo que limita es la variedad de eventos, no el tamaño
del modelo. La sensibilidad es plana (0,67–0,69 con cualquier valor de iteraciones, tasa de
aprendizaje o mínimo por hoja). La mejor configuración de la búsqueda (#9: 150 iteraciones, hasta 63
hojas, mínimo 80 por hoja, l2 de 3) gana **+3,1 puntos y se adoptó**. En la prueba detecta los mismos
eventos con menos falsas alarmas: con el procedimiento de esta validación, 16 de 25 con 2 falsas
alarmas contra 5 (F1 76 % contra 74 %); en el notebook 07, que elige además el umbral y la
confirmación por bloques, 17 de 25 con 3 falsas alarmas en 4 meses y F1 de 78,5 % (96 % en eventos
de calidad).

**Anticipación a 24 h.** Fue el modelo que más cambió:

1. Con el modelo que había al empezar (HistGradientBoosting por tipo) el F1 entre pliegues fue de
   0,31 ± 0,13, y la mejor configuración de la búsqueda (300 iteraciones, tasa 0,03, 7 hojas,
   mínimo 20 por hoja, sin l2) dio 0,37: **+5,7 puntos, se adoptó**. En la prueba anticipaba 11 de
   27 eventos con 0,61 avisos falsos por semana, contra 9 de 27 con 2,43.
2. Al reentrenar con esa configuración, la regla del notebook 09 eligió el **bosque aleatorio**, que
   todavía tenía la configuración de la búsqueda hecha con la lluvia del IMN. Se repitió la
   validación con el bosque (es la sección *Anticipación a 24 h* de este reporte): 0,30 ± 0,12 entre
   pliegues con su umbral, 0,25 con la regla, y la mejor configuración de la búsqueda (#7: 300
   árboles, mínimo 40 por hoja, profundidad 20, 33 % de las variables por corte, clases pesadas) dio
   0,36: **+10,2 puntos, se adoptó**. En la prueba, con el procedimiento de esta validación, anticipa
   13 de 27 eventos con 1,22 avisos falsos por semana (F1 57 %), contra 9 de 27 (48 %).
3. Con los dos modelos de árboles ya ajustados con la lluvia de Open-Meteo se corrió el notebook 09
   una última vez, decidido de antemano que lo que eligiera la regla quedaba como final (sin otra
   vuelta de búsqueda). La regla eligió la **LSTM**: 30 de 85 eventos con causa anticipados en los
   bloques, contra 28 de HistGradientBoosting y 26 del bosque. En la prueba anticipa 11 de 27 (10
   de 16 con causa) con 1,22 avisos falsos por semana: F1 de 46 %. La LSTM no entra en esta
   búsqueda (`validacion.py` la omite); sus hiperparámetros son los de la sección `anticipacion.lstm`.

La desviación entre pliegues no cumple ≤ 0,10 en ninguna variante (0,12–0,13): el primer pliegue
entrena con 3 meses y anticipa 1 evento de 22. La curva de aprendizaje todavía baja (error de 0,84
con 3 meses a 0,57 con 16): más historia ayudaría, porque cada tipo tiene pocos eventos. La
sensibilidad es ruidosa y los hiperparámetros interactúan: en el bosque, subir el mínimo por hoja
a 20 da 0,35 y a 40 baja a 0,28, mientras que la configuración #7 combina 40 con profundidad 20 y
clases pesadas. El F1 de este modelo tiene un techo bajo porque los eventos al azar (40 % de los de
la prueba) no tienen señal previa.

**Fugas (SVM).** La configuración que salió de la búsqueda con la lluvia del IMN (C = 10, gamma =
0,001, sin pesar las clases) sigue siendo la mejor con la de Open-Meteo: F1 de 0,65 con la regla, y
ninguna de las 10 configuraciones la supera, así que **el modelo final ya era óptimo**. Con el umbral
guardado da 0,58 ± 0,16 entre pliegues, que no cumple la desviación ≤ 0,10: los primeros pliegues
tienen pocas fugas para entrenar y detectan 2 o 3 de 7 a 11, mientras que el último detecta 9 de 11.
La curva de aprendizaje lo confirma (error de 0,34 a 0,61 con 3 a 7 meses y de 0,10 a 0,16 desde
10 meses). En la prueba detecta 6 de 9 fugas con 2 falsas alarmas (F1 72 %, notebook 10).

**Lo que queda para los requisitos.**

- RF-002.3: la validación cruzada temporal, las curvas de aprendizaje y la sensibilidad están en
  este reporte, que se genera solo y se puede repetir con Docker. La desviación ≤ 0,10 se cumple en
  el detector; en la anticipación (0,12) y en fugas (0,16) no, por los pocos eventos de los primeros
  pliegues.
- RF-002.4: en el detector y en los dos modelos de árboles de la anticipación la búsqueda mejoró el
  F1 en más de 2 puntos y la configuración nueva está en `hiperparametros.yaml`; en fugas se
  documenta que ya era óptimo. Cada configuración probada quedó en MLflow, en el experimento de su
  modelo, con sus hiperparámetros, F1 y duración. El modelo de anticipación que usa el servicio es
  la LSTM, que se eligió con la validación por bloques del notebook 09 entre los modelos ya ajustados.
