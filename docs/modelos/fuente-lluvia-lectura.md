## Lectura

**Por qué hace falta otra fuente.** Los sensores de la red no miden lluvia. La del dataset es la de
la estación 69-701 del IMN, que el simulador mete en las lecturas para poder entrenar; con la API de
datos definitiva no va a venir. El servicio de monitoreo la consigue de una fuente elegida con
`FUENTE_LLUVIA` (adaptadores en `Monitoreo_calidad_agua/backend/monitoreo/fuentes_lluvia.py`), y la
que se puede consultar en vivo para La Lucha es Open-Meteo.

**Concordancia.** Hora por hora, Open-Meteo casi no coincide con la estación (r ≈ 0,15–0,25): la
lluvia de la zona es convectiva y muy local, y una cuadrícula de 9 a 25 km no acierta la hora
exacta de un aguacero en un punto. La lluvia acumulada sí coincide: r ≈ 0,9 a 30 días. El mejor
modelo meteorológico es ECMWF IFS (r 0,89 a 30 días, 19 % menos lluvia que la estación), seguido
del promedio de los cuatro; ICON y GFS se quedan cortos y GEM se pasa. Es la serie que usa el
adaptador por defecto (`LLUVIA_MODELOS=ecmwf_ifs025`). El faltante no es parejo: en julio y agosto
de 2026, justo el periodo de prueba, Open-Meteo vio cerca de la mitad de la lluvia de la estación.

**Qué pasa con los modelos.** Los cuatro casos se calculan con el mismo procedimiento (entrenamiento
+ validación, prueba may–ago 2026), así que se comparan entre sí; pueden diferir un poco de los
notebooks, que entrenan y eligen el punto a su manera.

- **Detector y fugas no dependen de la fuente.** Si se cambia la lluvia en vivo sin reentrenar, su
  probabilidad cambia en promedio 0,014 y 0,003 y detectan lo mismo (18/25 y 4/9, sin falsas
  alarmas). Reentrenados con Open-Meteo quedan igual o mejor que con el IMN con la misma regla del
  punto de operación: detector F1 81 % contra 84 %; fugas 7/9 con 1 falsa contra 7/9 con 4 falsas
  (con 9 fugas en la prueba, la diferencia es de pocos casos).
- **La anticipación sí depende.** Usa la lluvia de 30 días, y con la de Open-Meteo, que es menor, el
  modelo entrenado con el IMN "ve" meses más secos de lo que fueron: cambiar la fuente sin reentrenar
  la hunde de 10 a 3 eventos anticipados de 27 (F1 51 % → 20 %). Reentrenada con Open-Meteo recupera
  buena parte: 7/27 con 3 falsos, F1 38 %, contra 9/27 con 4 falsos (F1 46 %) del IMN con la misma
  regla. Quedan unos 8 puntos de F1 de diferencia, el costo de usar una lluvia modelada en lugar de
  medida en el sitio.

**Decisión.** La arquitectura cumple RNF-009: la fuente se cambia con un adaptador y una variable
de entorno, y los modelos funcionan con la lluvia nueva sin tocar ningún módulo de modelado. Para
operar con `FUENTE_LLUVIA=open-meteo`:

1. Hay que reentrenar los modelos con la lluvia de Open-Meteo (al menos la anticipación). La
   Historical Forecast API da la misma serie desde 2024 (ECMWF), así que el historial alcanza.
2. Si la ASADA puede instalar un pluviómetro en La Lucha y mandarlo junto con las lecturas
   (`FUENTE_LLUVIA=lecturas`), los modelos tendrían lluvia medida, como la de entrenamiento; es la
   opción que mejor conserva la anticipación.

## Evidencia de que no se tocaron los módulos de modelado

En `Modelo_calidad_agua`, `git status` después de esta prueba solo muestra archivos nuevos y el
README; `calidad_agua/` (preprocesamiento, variables, detector, anticipación, fugas, validación)
no cambió:

```
$ git status --short
 M README.md
?? clima/lluvia_open_meteo_historica.csv
?? docs/modelos/fuente-lluvia-lectura.md
?? docs/modelos/fuente-lluvia.json
?? docs/modelos/fuente-lluvia.md
?? figuras/fuente_lluvia.png
?? herramientas/fuente_lluvia.py
?? pruebas/test_fuente_lluvia.py
$ git diff --stat -- calidad_agua/
(sin cambios)
```

En `Monitoreo_calidad_agua`, el motor (`motor.py`) y la carga de los modelos (`modelos.py`) no
cambiaron. Los adaptadores son un archivo nuevo de la capa de ingesta (`fuentes_lluvia.py`); el
servicio solo se tocó una vez, para envolver al cliente de la API de datos con la fuente elegida y
mostrar su estado en `/health` (10 líneas):

```
$ git diff --stat
 .env.ejemplo                  |  2 ++
 README.md                     | 21 +++++++++++++++++++++
 backend/monitoreo/config.py   |  4 ++++
 backend/monitoreo/servicio.py | 10 ++++++++--
$ git status --short backend/monitoreo/motor.py backend/monitoreo/modelos.py
(sin cambios)
```

De aquí en adelante, una fuente más (otra API, una estación propia) es solo una clase nueva
registrada en `FUENTES`. La prueba `test_una_fuente_nueva_es_solo_una_clase`
(`backend/pruebas/test_fuentes_lluvia.py`) lo comprueba: agrega un "pluviómetro" sin tocar nada
más, y el motor recibe su lluvia en cada lectura.
