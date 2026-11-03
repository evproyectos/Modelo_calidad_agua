"""
Registro de los experimentos en MLflow (RF-002, RNF-005).

Cada corrida de los notebooks 06, 07, 08 y 09 queda registrada con:
  - los hiperparámetros (hiperparametros.yaml) y las demás decisiones de la corrida;
  - las métricas de validación y de prueba;
  - la huella SHA-256 de los datos usados: las tres particiones, metadatos.json, el catálogo de
    eventos y el archivo de pronóstico de lluvia, uno por uno y combinados en `dataset_sha256`;
  - el commit de git del código y si el paquete tenía cambios sin guardar;
  - la fecha, la máquina y las versiones de Python y de las librerías;
  - los modelos guardados, las figuras y las tablas de resultados.

Con eso, cualquier resultado se puede reproducir: mismo commit, mismos datos (se comprueba con la
huella) y mismos hiperparámetros.

Los registros quedan en el propio proyecto, en mlflow.db (la base) y mlartifacts/ (los archivos,
una carpeta por experimento); no se versionan en git. No se envía telemetría de uso de MLflow.
Para verlos, desde la raíz del proyecto:

    mlflow ui --backend-store-uri sqlite:///mlflow.db --workers 1    → http://127.0.0.1:5000

(`--workers 1` hace falta en Windows: con varios procesos el servidor no arranca.)

Con la variable de entorno MLFLOW_TRACKING_URI se puede usar otro servidor de MLflow.
"""

import hashlib
import math
import os
import platform
import subprocess
import sys
import tempfile
import unicodedata
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path

from . import carga, clima

RAIZ = Path(__file__).resolve().parents[1]
BASE = RAIZ / "mlflow.db"
ARTEFACTOS = RAIZ / "mlartifacts"


# ---------------------------------------------------------------------------- huella de los datos
def sha256(ruta, bloque=1 << 20):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for parte in iter(lambda: f.read(bloque), b""):
            h.update(parte)
    return h.hexdigest()


def archivos_datos():
    """Los archivos de los que salen los modelos: particiones, metadatos, catálogo de eventos (si
    lo hay; el notebook 09 lo usa) y pronóstico de lluvia."""
    d = carga.carpeta_datos()
    extra = [d / "eventos_catalogo.csv"] if (d / "eventos_catalogo.csv").exists() else []
    return [d / f"{p}.parquet" for p in carga.PARTICIONES] + [d / "metadatos.json"] + extra + [clima.ruta_archivo()]


def huella_datos(archivos=None):
    """(huella combinada, {archivo: SHA-256}). La combinada identifica el conjunto de datos."""
    por_archivo = {Path(a).name: sha256(a) for a in (archivos or archivos_datos())}
    texto = "".join(f"{nombre}:{h}\n" for nombre, h in sorted(por_archivo.items()))
    return hashlib.sha256(texto.encode()).hexdigest(), por_archivo


# ---------------------------------------------------------------------------- contexto de la corrida
def estado_git(raiz=RAIZ):
    """Commit actual y si el código del paquete tiene cambios sin guardar en git."""
    def git(*args):
        return subprocess.run(["git", *args], cwd=raiz, capture_output=True, text=True, timeout=15).stdout.strip()
    try:
        commit = git("rev-parse", "HEAD")
        cambios = git("status", "--porcelain", "--", "calidad_agua", "pyproject.toml")
    except (OSError, subprocess.SubprocessError):
        commit, cambios = "", "?"
    return {"git_commit": commit or "desconocido", "git_cambios_sin_guardar": "no" if not cambios else "sí"}


def versiones():
    import numpy
    import pandas
    import sklearn
    v = {"python": platform.python_version(), "numpy": numpy.__version__, "pandas": pandas.__version__,
         "scikit-learn": sklearn.__version__}
    if "tensorflow" in sys.modules:                       # solo si la corrida lo usa
        v["tensorflow"] = sys.modules["tensorflow"].__version__
    return v


def clave(texto):
    """Nombre válido para MLflow: sin tildes ni símbolos ('% acierto (normal)' → 'pct_acierto_normal')."""
    t = str(texto).replace("–", "-").replace("—", "-")
    t = unicodedata.normalize("NFKD", t).encode("ascii", "ignore").decode()
    t = t.replace("%", "pct").replace("≥", "ge").replace("+", "mas")
    t = "".join(c if c.isalnum() or c in "_-./" else "_" for c in t.strip().lower())
    while "__" in t:
        t = t.replace("__", "_")
    return t.strip("_")


def aplanar(d, prefijo=""):
    """{'a': {'b': 1}} → {'a.b': 1}; las listas quedan como texto."""
    plano = {}
    for k, v in d.items():
        nombre = f"{prefijo}.{clave(k)}" if prefijo else clave(k)
        if isinstance(v, dict):
            plano.update(aplanar(v, nombre))
        else:
            plano[nombre] = v
    return plano


# ---------------------------------------------------------------------------- registro
class Registro:
    """Lo que se puede registrar durante una corrida (y en sus corridas hijas)."""

    def __init__(self, mlflow, run, dataset):
        self.mlflow, self.run, self.dataset = mlflow, run, dataset

    @property
    def id(self):
        return self.run.info.run_id

    def parametros(self, d, prefijo=""):
        valores = {k: (", ".join(map(str, v)) if isinstance(v, (list, tuple)) else v)
                   for k, v in aplanar(d, prefijo).items()}
        self.mlflow.log_params(valores, run_id=self.id)

    def metricas(self, d, prefijo="", paso=None):
        """Registra los valores numéricos (los NaN se omiten)."""
        valores = {}
        for k, v in aplanar(d, prefijo).items():
            try:
                v = float(v)
            except (TypeError, ValueError):
                continue
            if not math.isnan(v):
                valores[k] = v
        if valores:
            self.mlflow.log_metrics(valores, step=paso, run_id=self.id)

    def tabla(self, df, nombre):
        """Tabla de resultados como CSV (se abre en Excel)."""
        with tempfile.TemporaryDirectory() as d:
            ruta = Path(d) / f"{clave(nombre)}.csv"
            df.to_csv(ruta, encoding="utf-8-sig")
            self.mlflow.log_artifact(str(ruta), artifact_path="tablas", run_id=self.id)

    def metricas_tabla(self, df, prefijo):
        """Cada celda numérica de una tabla de resultados: '<prefijo>.<fila>.<columna>'."""
        valores = {}
        for fila, datos in df.iterrows():
            nombre = ".".join(clave(x) for x in (fila if isinstance(fila, tuple) else (fila,)))
            for columna, v in datos.items():
                valores[f"{nombre}.{clave(columna)}"] = v
        self.metricas(valores, prefijo)

    def figura(self, fig, nombre):
        self.mlflow.MlflowClient().log_figure(self.id, fig, f"figuras/{nombre}.png")

    def figuras_guardadas(self, prefijo):
        """Las figuras ya exportadas a figuras/ cuyo nombre empieza con `prefijo` (p. ej. '06_')."""
        for f in sorted((RAIZ / "figuras").glob(f"{prefijo}*.png")):
            self.archivo(f, "figuras")

    def archivo(self, ruta, carpeta="modelos"):
        self.mlflow.log_artifact(str(ruta), artifact_path=carpeta, run_id=self.id)

    def referencia(self):
        """Lo que se guarda junto al modelo para encontrar su experimento (lo muestra /health)."""
        exp = self.mlflow.get_experiment(self.run.info.experiment_id).name
        return {"run_id": self.id, "experimento": exp, "dataset_sha256": self.dataset}

    def terminar(self, estado="FINISHED"):
        self.mlflow.MlflowClient().set_terminated(self.id, estado)
        if self.mlflow.active_run() and self.mlflow.active_run().info.run_id == self.id:
            self.mlflow.end_run(estado)

    @contextmanager
    def hija(self, nombre):
        """Corrida anidada, para cada variante de un experimento (versiones, bloques...)."""
        with self.mlflow.start_run(run_name=nombre, nested=True,
                                   parent_run_id=self.id, experiment_id=self.run.info.experiment_id) as r:
            yield Registro(self.mlflow, r, self.dataset)


def configurar(uri=None):
    # sin telemetría de uso de MLflow: nada del proyecto sale de la máquina
    os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
    os.environ.setdefault("DO_NOT_TRACK", "true")
    import mlflow
    mlflow.set_tracking_uri(uri or os.environ.get("MLFLOW_TRACKING_URI") or f"sqlite:///{BASE.as_posix()}")
    return mlflow


def iniciar(experimento, nombre, notebook, uri=None, archivos=None):
    """
    Abre una corrida de MLflow con todo el contexto ya registrado: huella de los datos, git,
    versiones, hiperparámetros del YAML y periodos de cada partición. En un notebook:

        reg = experimentos.iniciar("pronostico_24h", "06 · árboles", "06_modelo_pronostico")
        ...
        reg.metricas({...})
        reg.terminar()
    """
    from . import configuracion
    # ¿registro propio del proyecto? (se decide antes de configurar: set_tracking_uri define
    # MLFLOW_TRACKING_URI en el entorno)
    propia = uri is None and not os.environ.get("MLFLOW_TRACKING_URI")
    mlflow = configurar(uri)
    if mlflow.active_run():                      # una corrida anterior que quedó abierta (celda con error)
        mlflow.end_run("KILLED")
    if mlflow.get_experiment_by_name(experimento) is None:
        mlflow.create_experiment(experimento, artifact_location=(ARTEFACTOS / experimento).as_uri() if propia else None)
    mlflow.set_experiment(experimento)
    dataset, por_archivo = huella_datos(archivos)
    run = mlflow.start_run(run_name=nombre)
    reg = Registro(mlflow, run, dataset)
    mlflow.set_tags({"notebook": notebook, "dataset_sha256": dataset, "maquina": platform.node(),
                     "fecha": datetime.now().isoformat(timespec="seconds"), **estado_git(),
                     **{f"version.{k}": v for k, v in versiones().items()},
                     **{f"sha256.{k}": v for k, v in por_archivo.items()}})
    reg.parametros(configuracion.cargar(), "hiperparametros")
    try:
        reg.parametros({p: f"{a.date()} a {b.date()}" for p, (a, b) in carga.periodos().items()}, "periodo")
    except (FileNotFoundError, KeyError):
        pass
    return reg


@contextmanager
def corrida(experimento, nombre, notebook, uri=None, archivos=None):
    """Lo mismo que iniciar(), como bloque `with`: si algo falla, la corrida queda como FAILED."""
    reg = iniciar(experimento, nombre, notebook, uri, archivos)
    try:
        yield reg
    except BaseException:
        reg.terminar("FAILED")
        raise
    reg.terminar()
