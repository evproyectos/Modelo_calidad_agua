"""
Versiones de los modelos en el Model Registry de MLflow (RF-004.2, criterio 3).

Cada modelo que usa el servicio de monitoreo es un modelo registrado con versiones numeradas:

  pronostico_24h         modelos/pronostico_24h_final.joblib y las redes lstm_24h_semilla*.keras (08)
  detector_eventos       modelos/detector_eventos.joblib (07)
  anticipacion_eventos   modelos/anticipacion_24h.joblib y, si ganó la LSTM, anticipacion_lstm_*.keras (09)
  fugas                  modelos/fugas.joblib y, si ganó el Autoencoder, fugas_autoencoder.keras (10)

registrar() mira los archivos de modelos/ y, para cada modelo, la corrida de MLflow que lo entrenó
(la referencia que el notebook guarda dentro del .joblib). Si esa corrida todavía no tiene versión,
crea la siguiente (1, 2, 3...) con los archivos de la corrida y le pone el alias `produccion`. Si ya
la tiene, no hace nada: correrlo dos veces no sube la versión, solo un reentrenamiento la sube.
Cada versión guarda en sus etiquetas el SHA-256 de cada archivo, la huella de los datos y el
commit, y modelos/versiones.json resume las versiones que están en modelos/ (lo lee el servicio
de monitoreo para mostrarlas en /health).

restaurar() vuelve a poner en modelos/ los archivos de una versión anterior (y le pasa el alias
`produccion`). El servicio de monitoreo los carga solo, como después de un reentrenamiento.

Uso: herramientas/registrar_modelos.py (reentrenar.py lo llama al final).
"""

import hashlib
import json
import shutil
from datetime import datetime
from pathlib import Path

import joblib

from . import experimentos

RAIZ = Path(__file__).resolve().parents[1]
CARPETA = RAIZ / "modelos"
ALIAS = "produccion"
SUBCARPETA = "modelos"                    # donde los notebooks dejan los modelos dentro de la corrida
VERSIONES = "versiones.json"

# nombre registrado: (archivo principal, descripción)
MODELOS = {
    "pronostico_24h": ("pronostico_24h_final.joblib",
                       "Pronóstico a 24 h de cloro libre, turbidez y pH por sensor (notebook 08)"),
    "detector_eventos": ("detector_eventos.joblib",
                         "Detector de eventos de calidad del agua por sensor y hora (notebook 07)"),
    "anticipacion_eventos": ("anticipacion_24h.joblib",
                             "Anticipación de eventos en las próximas 24 y 48 h (notebook 09)"),
    "fugas": ("fugas.joblib", "Detección de fugas y tramo más probable (notebook 10)"),
}


class ErrorRegistro(Exception):
    pass


def _sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def archivos(nombre, paquete):
    """Los archivos de un modelo: el .joblib y las redes de Keras que carga con él."""
    principal = MODELOS[nombre][0]
    extra = []
    if nombre == "pronostico_24h":
        extra = list(paquete.get("redes", []))
    elif nombre == "anticipacion_eventos" and paquete.get("nombre") == "LSTM":
        extra = [f"anticipacion_lstm_{k}.keras" for k in range(int(paquete.get("redes", 0)))]
    elif nombre == "fugas" and paquete.get("nombre") == "Autoencoder":
        extra = ["fugas_autoencoder.keras"]
    return [principal, *extra]


def _carpeta_corrida(run):
    """Carpeta local de los archivos de una corrida (también si se creó en otra máquina)."""
    uri = run.info.artifact_uri
    otra = experimentos.destino_local(uri)
    if otra is not None:
        return otra
    if not uri.startswith("file:"):
        raise ErrorRegistro(f"Los archivos de la corrida {run.info.run_id} no están en esta máquina ({uri}).")
    from mlflow.utils.file_utils import local_file_uri_to_path
    return Path(local_file_uri_to_path(uri))


def _subir_faltantes(mlflow, run, carpeta, nombres):
    """Copia a la corrida los archivos que no tenga; si tiene uno distinto del de modelos/, falla."""
    destino = _carpeta_corrida(run) / SUBCARPETA
    for n in nombres:
        local, remoto = carpeta / n, destino / n
        if remoto.exists():
            if _sha256(remoto) != _sha256(local):
                raise ErrorRegistro(f"modelos/{n} no es el archivo que guardó la corrida {run.info.run_id}: "
                                    "vuelve a correr su notebook (o reentrenar.py) antes de registrarlo.")
            continue
        if experimentos.destino_local(run.info.artifact_uri) is None:
            mlflow.MlflowClient().log_artifact(run.info.run_id, str(local), SUBCARPETA)
        else:
            destino.mkdir(parents=True, exist_ok=True)
            shutil.copy2(local, remoto)


def _version_de_corrida(cliente, nombre, run_id):
    for v in cliente.search_model_versions(f"name = '{nombre}'"):
        if v.run_id == run_id:
            return v
    return None


def registrar(carpeta=CARPETA, uri=None):
    """
    Registra en el Model Registry los modelos de `carpeta` que todavía no tengan versión. Devuelve
    {nombre: {"version", "nueva", "run_id", "archivos": {archivo: sha256}}} y escribe versiones.json.
    """
    carpeta = Path(carpeta)
    mlflow = experimentos.configurar(uri)
    cliente = mlflow.MlflowClient()
    resumen = {}
    for nombre, (principal, descripcion) in MODELOS.items():
        ruta = carpeta / principal
        if not ruta.exists():
            continue
        paquete = joblib.load(ruta)
        ref = paquete.get("mlflow") if isinstance(paquete, dict) else None
        if not ref or not ref.get("run_id"):
            raise ErrorRegistro(f"modelos/{principal} no dice qué corrida de MLflow lo entrenó: "
                                "vuelve a correr su notebook.")
        nombres = archivos(nombre, paquete)
        faltan = [n for n in nombres if not (carpeta / n).exists()]
        if faltan:
            raise ErrorRegistro(f"Faltan archivos de {nombre} en modelos/: {', '.join(faltan)}")
        huellas = {n: _sha256(carpeta / n) for n in nombres}
        try:
            run = cliente.get_run(ref["run_id"])
        except Exception as e:
            raise ErrorRegistro(f"La corrida {ref['run_id']} de {nombre} no está en este MLflow: {e}") from e

        v = _version_de_corrida(cliente, nombre, run.info.run_id)
        nueva = v is None
        if nueva:
            _subir_faltantes(mlflow, run, carpeta, nombres)
            try:
                cliente.get_registered_model(nombre)
            except Exception:
                cliente.create_registered_model(nombre, description=descripcion)
            etiquetas = {f"sha256.{n}": h for n, h in huellas.items()}
            etiquetas.update({"archivos": ", ".join(nombres), "experimento": ref.get("experimento", ""),
                              "dataset_sha256": ref.get("dataset_sha256") or run.data.tags.get("dataset_sha256", ""),
                              "git_commit": run.data.tags.get("git_commit", ""),
                              "notebook": run.data.tags.get("notebook", "")})
            if paquete.get("nombre"):
                etiquetas["algoritmo"] = str(paquete["nombre"])
            v = cliente.create_model_version(
                nombre, source=f"{run.info.artifact_uri}/{SUBCARPETA}", run_id=run.info.run_id,
                tags=etiquetas, description=f"Entrenado el {run.data.tags.get('fecha', '?')} "
                                            f"(corrida {run.info.run_name})")
        if nueva or not _tiene_alias(cliente, nombre, v.version):
            cliente.set_registered_model_alias(nombre, ALIAS, v.version)
        resumen[nombre] = {"version": int(v.version), "nueva": nueva, "run_id": run.info.run_id,
                           "experimento": ref.get("experimento"), "archivos": huellas}
    escribir_versiones(carpeta, resumen)
    return resumen


def _tiene_alias(cliente, nombre, version):
    try:
        return str(cliente.get_model_version_by_alias(nombre, ALIAS).version) == str(version)
    except Exception:
        return False


def escribir_versiones(carpeta, resumen):
    datos = {"registrado": datetime.now().isoformat(timespec="seconds"), "alias": ALIAS,
             "modelos": {n: {k: v for k, v in r.items() if k != "nueva"} for n, r in resumen.items()}}
    (Path(carpeta) / VERSIONES).write_text(json.dumps(datos, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def restaurar(nombre, version, carpeta=CARPETA, uri=None):
    """Pone en `carpeta` los archivos de la versión `version` de `nombre` y le pasa el alias."""
    if nombre not in MODELOS:
        raise ErrorRegistro(f"No hay un modelo llamado {nombre}. Son: {', '.join(MODELOS)}")
    carpeta = Path(carpeta)
    mlflow = experimentos.configurar(uri)
    cliente = mlflow.MlflowClient()
    try:
        v = cliente.get_model_version(nombre, str(version))
    except Exception as e:
        raise ErrorRegistro(f"{nombre} no tiene la versión {version}.") from e
    origen = _carpeta_corrida(cliente.get_run(v.run_id)) / SUBCARPETA
    nombres = [n.strip() for n in v.tags.get("archivos", "").split(",") if n.strip()]
    for n in nombres:
        if not (origen / n).exists():
            raise ErrorRegistro(f"La versión {version} de {nombre} no tiene {n} en MLflow ({origen}).")
        if v.tags.get(f"sha256.{n}") and _sha256(origen / n) != v.tags[f"sha256.{n}"]:
            raise ErrorRegistro(f"{n} de la versión {version} no coincide con la huella registrada.")
    # primero las redes y al final el .joblib: el servicio recarga cuando ve el conjunto completo
    for n in sorted(nombres, key=lambda x: x.endswith(".joblib")):
        shutil.copy2(origen / n, carpeta / n)
    cliente.set_registered_model_alias(nombre, ALIAS, v.version)
    return registrar(carpeta, uri)


def lista(uri=None):
    """Todas las versiones de cada modelo, de la más nueva a la más vieja."""
    mlflow = experimentos.configurar(uri)
    cliente = mlflow.MlflowClient()
    filas = []
    for nombre in MODELOS:
        try:
            alias = str(cliente.get_model_version_by_alias(nombre, ALIAS).version)
        except Exception:
            alias = None
        versiones = sorted(cliente.search_model_versions(f"name = '{nombre}'"), key=lambda x: -int(x.version))
        for v in versiones:
            filas.append({"modelo": nombre, "version": int(v.version), "en_produccion": str(v.version) == alias,
                          "creada": datetime.fromtimestamp(v.creation_timestamp / 1000).isoformat(timespec="seconds"),
                          "algoritmo": v.tags.get("algoritmo", ""), "run_id": v.run_id})
    return filas
