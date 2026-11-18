"""
plot_style.py — Estilo das figuras do TCC: o mesmo das figuras do artigo.

Por que um módulo: as figuras dos resultados (``src.visualization``), das análises
complementares (``src.visualization_analyses``) e da EDA (``src.eda``) usam o mesmo
estilo das figuras do texto: paleta ColorBrewer Set2 (segura para daltonismo), fonte e
parâmetros padrão do matplotlib, título no topo, bordas de cima e da direita removidas,
grade leve e valores escritos nas barras. Os nomes e cores são os mesmos que o
``visualization`` já usava (Pipeline A verde, B laranja, B gene novo azul-lavanda,
homologia cinza).

Cada gráfico é salvo em PNG (300 DPI) e PDF, sem data embutida no PDF (o arquivo só
muda quando a figura muda). As figuras com vários painéis têm também uma versão de um
gráfico por arquivo (pasta ``individuais/``), para entrar direto no texto.

Números com vírgula decimal, como no texto: ``num`` formata os rótulos escritos nas
figuras, e ``save`` troca o formatador dos eixos numéricos antes de salvar (sem isso, o
matplotlib escreve "0.2" no eixo ao lado de um rótulo "0,25" na barra).

Critério de conteúdo: cada figura mostra só o necessário para a sua mensagem, de forma
fácil de ler. O detalhe (cada modelo, cada intervalo) fica nas tabelas e nos CSVs.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import ScalarFormatter  # noqa: E402

from src.config import DPI_FIGURES  # noqa: E402

logger = logging.getLogger(__name__)
logging.getLogger("fontTools").setLevel(logging.ERROR)

# Paleta ColorBrewer Set2 (segura para daltonismo), a mesma das figuras do artigo.
ACCESSIBLE_PALETTE = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
COLOR_A, COLOR_B = ACCESSIBLE_PALETTE[0], ACCESSIBLE_PALETTE[1]
COLOR_NEW_GENE = ACCESSIBLE_PALETTE[2]  # Pipeline B no cenário "gene novo"
COLOR_HOMOLOGY = "0.35"  # referência por homologia (sem aprendizado de máquina)

MECHANISMS = [
    "antibiotic inactivation", "antibiotic target alteration", "antibiotic efflux",
    "antibiotic target protection", "antibiotic target replacement",
]
MECHANISM_COLORS = dict(zip(MECHANISMS, ACCESSIBLE_PALETTE[:5]))

GROUP_LABELS: dict[str, str] = {
    "fisico_quimicas": "Físico-químicas (4)",
    "aminoacidos": "Aminoácidos (20)",
    "pares_de_grupos": "Pares de grupos (49)",
    "estruturais": "ARO: estruturais (3)",
    "classes_de_antibiotico": "ARO: classes de antibiótico (38)",
}
GROUP_COLORS: dict[str, str] = {
    "fisico_quimicas": ACCESSIBLE_PALETTE[0], "aminoacidos": ACCESSIBLE_PALETTE[4], "pares_de_grupos": ACCESSIBLE_PALETTE[5],
    "estruturais": ACCESSIBLE_PALETTE[2], "classes_de_antibiotico": ACCESSIBLE_PALETTE[1],
}


def short(c: str) -> str:
    """'antibiotic target protection' -> 'target protection' (rótulos curtos, como nas figuras do artigo)."""
    return str(c).replace("antibiotic ", "").replace(" to antibiotic", "")


def num(v: float, d: int = 3, sign: bool = False) -> str:
    """Número com vírgula decimal: num(0.2361) -> '0,236'; num(0.13, 2, sign=True) -> '+0,13'."""
    return (f"{v:+.{d}f}" if sign else f"{v:.{d}f}").replace(".", ",")


class _DecimalComma(ScalarFormatter):
    """O formatador padrão do matplotlib, com vírgula no lugar do ponto."""

    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")

    def get_offset(self):
        return super().get_offset().replace(".", ",")


def decimal_comma(fig) -> None:
    """Vírgula decimal nos eixos numéricos de todos os painéis (inclusive barras de cor e detalhes)."""
    for ax in [a for top in fig.axes for a in (top, *top.child_axes)]:  # child_axes: detalhes ampliados (inset)
        for axis in (ax.xaxis, ax.yaxis):
            if type(axis.get_major_formatter()) is ScalarFormatter:
                axis.set_major_formatter(_DecimalComma())


def save(fig, path: Path, dpi: int = DPI_FIGURES) -> None:
    """PNG e PDF, com vírgula decimal nos eixos; sem data de criação no PDF (o matplotlib gravava a data a cada execução)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    decimal_comma(fig)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", metadata={"CreationDate": None})
    plt.close(fig)
    logger.info("Figura salva: %s.png/.pdf", path)


def clean_axes(ax, grid_axis: str | None = "y") -> None:
    """Bordas de cima e da direita removidas e grade leve, como nas figuras do artigo."""
    ax.spines[["top", "right"]].set_visible(False)
    if grid_axis:
        ax.grid(axis=grid_axis, alpha=0.2)
        ax.set_axisbelow(True)
