"""
calidad_agua: limpieza de datos, variables y modelos para sensores de calidad de agua.

Módulos:
  carga         lectura del dataset de los sensores
  calidad       limpieza de lecturas (con etiquetas de falla y solo con reglas de operación)
  metas         meta del proyecto: tolerancias y criterio de acierto
  variables     variables de entrada del pronóstico a 24 h
  modelo        pronóstico con árboles (scikit-learn)
  red_neuronal  pronóstico con LSTM (TensorFlow/Keras)
  detector      detector de eventos
  eventos       episodios a partir de etiquetas
  graficos      estilo común de las figuras
"""

__version__ = "0.1.0"
