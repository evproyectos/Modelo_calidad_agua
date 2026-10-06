# Modelo de calidad de agua

Análisis exploratorio (EDA) y modelos de pronóstico para los datos del
proyecto **Simulador_sensor** (red D-Town, Costa Rica): cloro residual libre,
turbidez, pH, temperatura y presión en 21 sensores, cada 15 minutos.

Objetivo: pronosticar las próximas 24 h con al menos 80 % de exactitud
(definición en el notebook de modelos), usando también datos climáticos.

## Relación con el simulador

```
Simulador_sensor  ──(exporta datos/dtown/final/*.parquet)──►  Modelo_calidad_agua
```

Este proyecto no copia los datos: los lee de la carpeta del simulador. Si el
dataset se regenera (por ejemplo, con el clima del IMN), solo hay que volver a
correr los notebooks.

Por defecto busca los datos en `../Simulador_sensor/datos/dtown/final`. Si el
simulador está en otro lugar, copia `.env.ejemplo` como `.env` y ajusta la ruta.

## Estructura

```
src/          funciones compartidas (carga de datos)
notebooks/    EDA y modelos, numerados en el orden de lectura
figuras/      figuras exportadas para el informe
```

## Instalación

```
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook
```

Los notebooks agregan la raíz del proyecto al `sys.path`, así que
`from src import carga` funciona desde la carpeta `notebooks/`.

## Notebooks

| # | Notebook | Contenido |
|---|---|---|
| 01 | calidad_de_datos | Faltantes, cortes de comunicación, fallas de sensor |
| 02 | distribuciones | Rangos por parámetro y sensor, comparación con la norma |
| 03 | temporal | Ciclos diarios y semanales, estacionalidad, autocorrelación |
| 04 | clima | Relación de lluvia y temperatura con turbidez, cloro y pH |
| 05 | eventos | Cómo se ven los eventos y las fallas en los datos |
