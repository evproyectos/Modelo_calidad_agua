"""
Empaqueta los modelos entrenados y la historia de MLflow para publicarlos como un release de
GitHub (RNF-008). modelos/, mlflow.db y mlartifacts/ no se versionan (pesan ~150 MB y cambian con
cada reentrenamiento); el release los guarda y Docker los descarga solo en una máquina nueva
(Monitoreo_calidad_agua/docker/preparar.py).

Uso, desde la raíz del proyecto y después de reentrenar (herramientas/reentrenar.py):

    python herramientas/publicar_modelos.py modelos-2026-11-10

Deja en artefactos/ (no se versiona):
  modelos.zip   los archivos de modelos/
  mlflow.zip    mlflow.db y mlartifacts/; en la copia de la base, las rutas de los archivos de cada
                corrida apuntan a /modelo/mlartifacts, donde los ve el MLflow de Docker
y actualiza artefactos.json (sí se versiona): el nombre del release y la huella SHA-256 de cada
archivo, para que la descarga se verifique. Después hay que:
  1. crear el release con ese nombre en GitHub y adjuntarle los dos .zip;
  2. hacer commit de artefactos.json.

Los .zip son reproducibles: con los mismos modelos dan los mismos bytes (fechas fijas y orden fijo).
"""

import hashlib
import json
import shutil
import sqlite3
import sys
import tempfile
import zipfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
REPOSITORIO = "evproyectos/Modelo_calidad_agua"
SALIDA = RAIZ / "artefactos"
MANIFIESTO = RAIZ / "artefactos.json"
ARTEFACTOS_DOCKER = "file:///modelo/mlartifacts"      # donde los monta el servicio mlflow de Docker
FECHA_FIJA = (1980, 1, 1, 0, 0, 0)


def sha256(ruta):
    h = hashlib.sha256()
    with open(ruta, "rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def comprimir(destino, archivos):
    """archivos: {nombre dentro del zip: ruta}. Orden y fechas fijas: mismo contenido, mismo zip."""
    with zipfile.ZipFile(destino, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for nombre in sorted(archivos):
            info = zipfile.ZipInfo(nombre, FECHA_FIJA)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            with open(archivos[nombre], "rb") as f, z.open(info, "w") as dentro:
                shutil.copyfileobj(f, dentro, 1 << 20)


def base_para_docker(origen, destino):
    """Copia mlflow.db y cambia las rutas de los archivos (…/mlartifacts/…) a las de Docker."""
    shutil.copyfile(origen, destino)
    con = sqlite3.connect(destino)
    try:
        tablas = [t for (t,) in con.execute("select name from sqlite_master where type = 'table'")]
        for t in tablas:
            for col in con.execute(f'pragma table_info("{t}")').fetchall():
                nombre, tipo = col[1], (col[2] or "").upper()
                if "CHAR" not in tipo and "TEXT" not in tipo:
                    continue
                con.execute(f'update "{t}" set "{nombre}" = ? || substr("{nombre}", instr("{nombre}", '
                            f"'/mlartifacts') + 12) where \"{nombre}\" like 'file:%/mlartifacts%'",
                            (ARTEFACTOS_DOCKER,))
        # el experimento Default (sin corridas) apunta a una carpeta de quien entrenó
        con.execute("update experiments set artifact_location = ? where artifact_location not like 'file:///modelo/%'",
                    (f"{ARTEFACTOS_DOCKER}/0",))
        con.commit()
        con.execute("vacuum")
    finally:
        con.close()


def empaquetar(release):
    SALIDA.mkdir(exist_ok=True)
    modelos = {p.name: p for p in sorted((RAIZ / "modelos").iterdir())
               if p.is_file() and not p.name.startswith(".")}
    if not modelos:
        sys.exit("modelos/ está vacía: primero hay que entrenar (python herramientas/reentrenar.py).")
    comprimir(SALIDA / "modelos.zip", modelos)

    archivos = [{"nombre": "modelos.zip", "destino": "modelos", "obligatorio": True,
                 "contenido": {n: sha256(p) for n, p in modelos.items()}}]
    if (RAIZ / "mlflow.db").exists():
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "mlflow.db"
            base_para_docker(RAIZ / "mlflow.db", base)
            contenido = {"mlflow.db": base}
            for p in sorted((RAIZ / "mlartifacts").rglob("*")):
                if p.is_file():
                    contenido[p.relative_to(RAIZ).as_posix()] = p
            comprimir(SALIDA / "mlflow.zip", contenido)
        # la historia de MLflow no hace falta para que el servicio funcione
        archivos.append({"nombre": "mlflow.zip", "destino": ".", "obligatorio": False,
                         "contenido": {"mlflow.db": None}})
    for a in archivos:
        zip_ = SALIDA / a["nombre"]
        a.update(sha256=sha256(zip_), bytes=zip_.stat().st_size)
    manifiesto = {"repositorio": REPOSITORIO, "release": release,
                  "url": f"https://github.com/{REPOSITORIO}/releases/download/{release}",
                  "archivos": archivos}
    MANIFIESTO.write_text(json.dumps(manifiesto, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifiesto


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    m = empaquetar(sys.argv[1])
    for a in m["archivos"]:
        print(f"artefactos/{a['nombre']}: {a['bytes'] / 1e6:.1f} MB · sha256 {a['sha256'][:12]}")
    print(f"\nartefactos.json actualizado (release {m['release']}). Falta:")
    print(f"  1. Crear el release '{m['release']}' en https://github.com/{m['repositorio']}/releases/new")
    print("     y adjuntarle los archivos de artefactos/ (o, con la CLI de GitHub:")
    print(f"     gh release create {m['release']} artefactos/*.zip --repo {m['repositorio']} --title {m['release']})")
    print("  2. git add artefactos.json && git commit")
