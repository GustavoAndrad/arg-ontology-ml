"""
visualization.py — Figuras do TCC (Figuras 4.2, 5.1, 5.2 e 5.3) e matrizes de confusão.

Saídas (PNG e PDF, 300 DPI):
    plots/eda/class_distribution_experimento        — distribuição das classes do experimento (Fig. 4.2; 5 no TCC2)
    plots/comparacao_pipelines/f1_macro_comparison  — F1 macro A vs B (Fig. 5.1)
    plots/comparacao_pipelines/f1_delta_percentual  — ganho percentual (Fig. 5.2)
    plots/roc_curves/roc_all_models_grid            — ROC macro OvR, 2x2 (Fig. 5.3)
    plots/roc_curves/roc_{modelo}                   — ROC por modelo
    plots/confusion_matrices/cm_{modelo}_{a|b}      — matrizes normalizadas por linha
    plots/interpretabilidade/feature_importance_rf_b — top-20 Gini (se existir o CSV)
    plots/comparacao_pipelines/curva_limiar        — F1 por divisão (aleatória e grupos 90/70/50%)

Uso:
    python -m src.visualization
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score, roc_curve  # noqa: E402

from src.config import DPI_FIGURES  # noqa: E402
from src.experiment_config import MODEL_NAMES, PLOTS_DIR, RESULTS_DIR, ROC_GRID_POINTS, TARGET, load_experiment_dataset  # noqa: E402
from src.pipeline_builder import MODEL_LABELS  # noqa: E402

logger = logging.getLogger(__name__)
# Paleta ColorBrewer Set2 (segura para daltonismo)
ACCESSIBLE_PALETTE = ["#66c2a5", "#fc8d62", "#8da0cb", "#e78ac3", "#a6d854", "#ffd92f"]
COLOR_A, COLOR_B = ACCESSIBLE_PALETTE[0], ACCESSIBLE_PALETTE[1]
PREDICTIONS_DIR = RESULTS_DIR / "predicoes"


def _save(fig, path: Path, dpi: int = DPI_FIGURES) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path.with_suffix(".png"), dpi=dpi, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    logger.info("Figura salva: %s.png/.pdf", path)


def plot_class_distribution(counts: pd.Series, output_path: Path) -> None:
    counts = counts.sort_values()
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(counts.index, counts.values, color=ACCESSIBLE_PALETTE[: len(counts)][::-1])
    for b, v in zip(bars, counts.values):
        ax.text(
            b.get_width() + total * 0.005,
            b.get_y() + b.get_height() / 2,
            f"{v} ({v / total:.1%})",
            va="center",
            fontsize=8,
        )
    ax.set_xlabel("Número de Instâncias")
    ax.set_ylabel("Mecanismo de Resistência")
    ax.set_title("Distribuição de Classes — Mecanismo de Resistência")
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, counts.max() * 1.18)
    _save(fig, output_path)


def plot_f1_comparison_bar(metrics_a: pd.DataFrame, metrics_b: pd.DataFrame, output_path: Path) -> None:
    """
    Barras de F1-macro no teste, A vs B, por modelo (Fig. 5.1).

    O eixo começa em 0 (barras não devem ser cortadas) e vai até 1,1 só para dar
    espaço aos rótulos acima das barras; as marcas param em 1,0. A legenda fica
    abaixo do gráfico: dentro dele, com F1 perto de 1, ela cobria barras e valores.
    """
    fa = metrics_a[metrics_a.metrica == "f1_macro"].set_index("modelo").loc[MODEL_NAMES, "valor_teste"]
    fb = metrics_b[metrics_b.metrica == "f1_macro"].set_index("modelo").loc[MODEL_NAMES, "valor_teste"]
    x = np.arange(len(MODEL_NAMES))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 6))
    for off, vals, color, label in (
        (-w / 2, fa, COLOR_A, "Pipeline A (Sequência)"),
        (w / 2, fb, COLOR_B, "Pipeline B (Sequência + Ontologia)"),
    ):
        bars = ax.bar(x + off, vals.values, w, color=color, edgecolor="black", linewidth=0.5, label=label)
        for b, v in zip(bars, vals.values):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, f"{v:.3f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_NAMES], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("F1-score macro (teste)")
    ax.set_title("Comparação do F1-score macro — Pipeline A vs Pipeline B")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=11)
    _save(fig, output_path)


def plot_f1_delta_percent(metrics_a: pd.DataFrame, metrics_b: pd.DataFrame, output_path: Path) -> None:
    """Ganho relativo do B sobre o A no F1-macro de teste (Fig. 5.2), com o ganho absoluto entre parênteses."""
    fa = metrics_a[metrics_a.metrica == "f1_macro"].set_index("modelo")["valor_teste"].loc[MODEL_NAMES]
    fb = metrics_b[metrics_b.metrica == "f1_macro"].set_index("modelo")["valor_teste"].loc[MODEL_NAMES]
    delta = (fb - fa) / fa * 100
    labels = [MODEL_LABELS[m] for m in MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(labels, delta.values, color=ACCESSIBLE_PALETTE[:4], edgecolor="black", linewidth=0.5, height=0.6)
    for b, v, d in zip(bars, delta.values, (fb - fa).values):
        ax.text(
            b.get_width() + 0.2,
            b.get_y() + b.get_height() / 2,
            f"{v:+.1f}% ({d:+.3f})",
            va="center",
            fontsize=9,
            fontweight="bold",
        )
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(delta.max() * 1.3, 1))
    ax.set_xlabel("Ganho Percentual do Pipeline B vs Pipeline A (%)")
    ax.set_title("Melhoria Relativa em F1-score Macro por Modelo")
    _save(fig, output_path)


def macro_roc_curve(y_true: np.ndarray, proba: np.ndarray, n_points: int = ROC_GRID_POINTS):
    """Curva ROC macro (OvR): TPR interpolado em grade comum de FPR e média entre classes."""
    grid = np.linspace(0, 1, n_points)
    tprs = []
    for c in range(proba.shape[1]):
        fpr, tpr, _ = roc_curve((y_true == c).astype(int), proba[:, c])
        tprs.append(np.interp(grid, fpr, tpr))
    auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=list(range(proba.shape[1])))
    return grid, np.mean(tprs, axis=0), auc


def _load_pred(key: str, model: str):
    return np.load(PREDICTIONS_DIR / f"pipeline_{key}_{model}.npz")


# Recorte ampliado das curvas ROC: com AUC perto de 1, as curvas de A e B ficam coladas
# no canto superior esquerdo e a diferença só aparece ampliando essa região.
ROC_ZOOM_FPR: tuple[float, float] = (0.0, 0.10)
ROC_ZOOM_TPR: tuple[float, float] = (0.75, 1.005)


def _roc_axes(ax, model: str) -> None:
    curves = []
    for key, color, style in (("a", "#d95f02", "--"), ("b", "#1b9e77", "-")):
        d = _load_pred(key, model)
        grid, tpr, auc = macro_roc_curve(d["y_true"], d["proba"])
        ax.plot(grid, tpr, style, color=color, lw=2, label=f"Pipeline {key.upper()} (AUC = {auc:.4f})")
        curves.append((grid, tpr, style, color))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.5, alpha=0.6)
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.2)
    # Detalhe ampliado (fica acima da legenda, longe da diagonal).
    inset = ax.inset_axes([0.38, 0.22, 0.58, 0.55])
    for grid, tpr, style, color in curves:
        inset.plot(grid, tpr, style, color=color, lw=2)
    inset.set_xlim(*ROC_ZOOM_FPR)
    inset.set_ylim(*ROC_ZOOM_TPR)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.2)
    inset.set_title(f"detalhe: FPR ≤ {ROC_ZOOM_FPR[1]:.2f}", fontsize=8)
    # Só o retângulo da região ampliada; as linhas de ligação cruzavam as curvas.
    _, connectors = ax.indicate_inset_zoom(inset, edgecolor="grey")
    for line in connectors:
        line.set_visible(False)


def plot_roc_curves(output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (letter, model) in zip(axes.ravel(), zip("abcd", MODEL_NAMES)):
        _roc_axes(ax, model)
        ax.set_title(f"({letter}) {MODEL_LABELS[model]}", fontweight="bold")
    fig.tight_layout()
    _save(fig, output_dir / "roc_all_models_grid")
    for model in MODEL_NAMES:
        fig, ax = plt.subplots(figsize=(7, 6))
        _roc_axes(ax, model)
        ax.set_title(f"ROC macro (OvR) — {MODEL_LABELS[model]}")
        _save(fig, output_dir / f"roc_{model}")


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, output_path: Path) -> None:
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(
                j,
                i,
                f"{norm[i, j]:.2f}\n(n={cm[i, j]})",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if norm[i, j] > 0.5 else "black",
            )
    short = [c.replace("antibiotic ", "").replace(" to antibiotic", "") for c in class_names]
    ax.set_xticks(range(len(short)), short, rotation=30, ha="right")
    ax.set_yticks(range(len(short)), short)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Verdadeiro")
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    _save(fig, output_path)


def plot_feature_importance(importance: pd.DataFrame, output_path: Path, top_n: int = 20) -> None:
    top = importance.nlargest(top_n, "importancia").iloc[::-1]
    colors = [COLOR_B if t == "ontologica" else COLOR_A for t in top["tipo"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    ax.barh(top["feature"], top["importancia"], color=colors, edgecolor="black", linewidth=0.4)
    ax.set_xlabel("Gini importance")
    ax.set_title(f"Top {top_n} features — Random Forest (Pipeline B)")
    from matplotlib.patches import Patch

    ax.legend(
        handles=[Patch(color=COLOR_A, label="sequência"), Patch(color=COLOR_B, label="ontológica")], loc="lower right"
    )
    _save(fig, output_path)


def plot_threshold_curve(curve: pd.DataFrame, partitions: pd.DataFrame, output_path: Path) -> None:
    """
    F1-macro no teste por divisão (aleatória → grupos de 90, 70 e 50%), A e B, um painel por modelo.

    Mostra se a vantagem da ontologia cresce quando o teste fica menos parecido com o
    treino. Cenários que não atendem o critério de entrada das classes (menos de 5
    genes de uma classe num fold) aparecem com marcador vazio.
    """
    order = ["aleatoria", "90", "70", "50"]
    labels = ["aleatória", "grupos 90%", "grupos 70%", "grupos 50%"]
    ok = partitions.set_index("cenario")["atende_criterio"].astype(bool)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4.2), sharey=True)
    for ax, model in zip(axes, MODEL_NAMES):
        sub = curve[curve.modelo == model].set_index("cenario").loc[order]
        x = np.arange(len(order))
        for key, color, label in (("a", COLOR_A, "Pipeline A"), ("b", COLOR_B, "Pipeline B")):
            vals = sub[f"f1_teste_{key}"].values
            ax.plot(x, vals, "-", color=color, lw=2, label=label)
            for xi, v, c in zip(x, vals, order):
                ax.plot(xi, v, "o", ms=7, color=color, mfc=color if ok[str(c)] else "white", mew=2)
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1-score macro (teste)")
    axes[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Desempenho por divisão treino/teste (marcador vazio: não atende o critério de genes por fold)", fontsize=11)
    fig.tight_layout()
    _save(fig, output_path)


def generate_all_figures() -> None:
    data = load_experiment_dataset()
    plot_class_distribution(data[TARGET].value_counts(), PLOTS_DIR / "eda" / "class_distribution_experimento")
    ma = pd.read_csv(RESULTS_DIR / "metrics_pipeline_a.csv")
    mb = pd.read_csv(RESULTS_DIR / "metrics_pipeline_b.csv")
    plot_f1_comparison_bar(ma, mb, PLOTS_DIR / "comparacao_pipelines" / "f1_macro_comparison")
    plot_f1_delta_percent(ma, mb, PLOTS_DIR / "comparacao_pipelines" / "f1_delta_percentual")
    plot_roc_curves(PLOTS_DIR / "roc_curves")
    classes = json.loads((RESULTS_DIR / "classes.json").read_text(encoding="utf-8"))
    for model in MODEL_NAMES:
        for key in ("a", "b"):
            d = _load_pred(key, model)
            plot_confusion_matrix(
                d["confusion_matrix"],
                classes,
                f"{MODEL_LABELS[model]} — Pipeline {key.upper()}",
                PLOTS_DIR / "confusion_matrices" / f"cm_{model}_{key}",
            )
    curve_path = RESULTS_DIR / "curva_limiar.csv"
    if curve_path.exists():
        plot_threshold_curve(
            pd.read_csv(curve_path),
            pd.read_csv(RESULTS_DIR / "curva_limiar_particoes.csv", dtype={"cenario": str}),
            PLOTS_DIR / "comparacao_pipelines" / "curva_limiar",
        )
    imp_path = RESULTS_DIR / "feature_importance_gini_rf_b.csv"
    if imp_path.exists():
        plot_feature_importance(pd.read_csv(imp_path), PLOTS_DIR / "interpretabilidade" / "feature_importance_rf_b")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    generate_all_figures()
