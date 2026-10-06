"""Estilo común de las figuras (matplotlib) para que todos los notebooks se vean igual."""

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

RAIZ = Path(__file__).resolve().parents[1]
FIGURAS = RAIZ / "figuras"

# paleta categórica en orden fijo (validada para daltonismo); gris para referencias
COLORES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
GRIS = "#8a8984"
TEXTO = "#0b0b0b"
TEXTO2 = "#52514e"
REJILLA = "#ecebe7"
FONDO = "#fcfcfb"


def estilo():
    mpl.rcParams.update({
        "figure.facecolor": FONDO, "axes.facecolor": FONDO, "savefig.facecolor": FONDO,
        "figure.dpi": 110, "savefig.dpi": 150, "savefig.bbox": "tight",
        "axes.prop_cycle": mpl.cycler(color=COLORES),
        "axes.edgecolor": REJILLA, "axes.labelcolor": TEXTO2, "axes.titlecolor": TEXTO,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.grid": True, "grid.color": REJILLA, "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False,
        "xtick.color": TEXTO2, "ytick.color": TEXTO2, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.fontsize": 8, "legend.frameon": False,
        "lines.linewidth": 1.5, "font.size": 9,
    })


def guardar(fig, nombre):
    """Guarda la figura en figuras/ (para el informe) además de mostrarla en el notebook."""
    FIGURAS.mkdir(exist_ok=True)
    fig.savefig(FIGURAS / f"{nombre}.png")
