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
