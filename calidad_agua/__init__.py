"""
calidad_agua: limpieza de datos, variables y modelos para sensores de calidad de agua.

Módulos:
  carga         lectura del dataset de los sensores
  calidad       limpieza de lecturas (con etiquetas de falla y solo con reglas de operación)
  preprocesamiento  picos por IQR, interpolación de huecos cortos y flag_imputacion (RF-001.3)
  etiquetas     vector de etiquetas: laboratorio, registros de la ASADA o umbral IoT (RF-001.6)
  ubicacion     dónde van los sensores del sistema
  clima         pronóstico de lluvia (Open-Meteo)
  fugas         detección de fugas y tramo de la red
  deriva        deriva de los datos respecto del entrenamiento
  metas         meta del proyecto: tolerancias y criterio de acierto
  configuracion hiperparámetros de los modelos (hiperparametros.yaml)
  experimentos  registro de los experimentos en MLflow
  variables     variables de entrada del pronóstico a 24 h
  modelo        pronóstico con árboles (scikit-learn)
  red_neuronal  pronóstico con LSTM (TensorFlow/Keras)
  detector      detector de eventos
  anticipacion  probabilidad de que empiece un evento en las próximas 24 h
  validacion    validación cruzada temporal y búsqueda de hiperparámetros (RF-002.3 y RF-002.4)
  eventos       episodios a partir de etiquetas
  graficos      estilo común de las figuras
"""

__version__ = "0.2.0"
