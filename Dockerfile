# Entorno de los notebooks, el reentrenamiento y la validación cruzada (RF-001.4, RF-002.3, RNF-008).
# Se usa con docker-compose.yml de esta carpeta; el código, los notebooks, los modelos y MLflow
# quedan en la carpeta del proyecto (montada en /modelo), no dentro de la imagen.
FROM python:3.13-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TF_CPP_MIN_LOG_LEVEL=3 \
    MLFLOW_DISABLE_TELEMETRY=true DO_NOT_TRACK=true MLFLOW_DISABLE_AGENT_HINT=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
WORKDIR /modelo
# las mismas versiones con las que se entrenaron los modelos (y que usa el servicio de monitoreo)
COPY pyproject.toml README.md ./
COPY calidad_agua calidad_agua
RUN pip install --no-cache-dir "scikit-learn==1.9.1" "numpy==2.5.3" "pandas==3.0.6" "pyarrow==25.0.1" \
        "joblib==1.6.0" "tensorflow==2.21.0" "keras==3.15.1" "mlflow==3.17.2" \
    && pip install --no-cache-dir -e ".[red,notebooks]"
# git dentro del contenedor: la carpeta montada es de otro usuario
RUN git config --global --add safe.directory /modelo
CMD ["python", "herramientas/validacion_cruzada.py"]
