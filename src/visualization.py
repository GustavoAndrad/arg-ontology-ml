"""
visualization.py — Figuras dos resultados do TCC e matrizes de confusão.

Saídas (PNG e PDF, 300 DPI):
    plots/eda/class_distribution_experimento        — genes e % de cada uma das 5 classes do experimento (Figura 4.2)
    plots/comparacao_pipelines/f1_macro_comparison  — F1 macro A vs B no teste
    plots/comparacao_pipelines/f1_delta_percentual  — ganho percentual do B no F1 de teste (com o absoluto)
    plots/comparacao_pipelines/ganho_cv            — B − A no F1-macro na CV, 8 testes confirmatórios, IC corrigido
    plots/comparacao_pipelines/ganho_por_classe    — B − A por classe e modelo no teste (precisão, revocação, F1)
    plots/comparacao_pipelines/ganho_por_classe_cv — B − A no F1 de cada classe na CV (curado e gene novo)
    plots/comparacao_pipelines/cenarios_gene_novo  — F1 no teste de A, B curado, B gene novo e homologia
    plots/comparacao_pipelines/curva_limiar        — ganho B − A por divisão (aleatória e grupos 90/70/50%)
    plots/comparacao_pipelines/curva_limiar_f1     — F1 na CV por divisão: A, B curado, B gene novo e homologia
    plots/roc_pr/individuais/roc_{modelo}          — ROC de um modelo, com o detalhe dos poucos falsos positivos
    plots/comparacao_pipelines/curva_aprendizado    — F1 por fração dos grupos do treino
    plots/comparacao_pipelines/overfitting          — F1 no treino, na validação e no teste
    plots/roc_pr/roc_pr_all_models                 — ROC e precisão-revocação lado a lado
    plots/confusion_matrices/cm_{modelo}_{a|b}      — matrizes normalizadas por linha
    plots/interpretabilidade/importancia_grupos     — queda do F1 ao embaralhar cada grupo (CV, IC 95%), A e B
    plots/interpretabilidade/importancia_por_classe — idem, por mecanismo (Pipeline B, anotação curada)
    plots/interpretabilidade/top_atributos          — 15 atributos mais importantes por modelo e pipeline
    plots/interpretabilidade/shap_{xgboost|random_forest}_b — SHAP por mecanismo (Pipeline B; direção)
    plots/*/individuais/                           — um gráfico por arquivo das figuras com vários painéis
    (figuras dos dados e das análises complementares: src.visualization_analyses; catálogo: results/figuras_catalogo.json)

As ROC por modelo da primeira versão do experimento (``roc_curves/``) passaram para
``roc_pr/individuais/roc_{modelo}``, com as cores das outras figuras; a grade 2 × 2 não é
mais gerada, porque repetia a coluna da esquerda de ``roc_pr_all_models``. A figura
antiga do Gini (``interpretabilidade/feature_importance_rf_b``) também não é mais gerada.

ROC e precisão-revocação lado a lado: com uma classe
muito grande (inativação), a ROC fica perto de 1 mesmo quando muitos alarmes das
classes pequenas são falsos; a curva de precisão-revocação mostra isso. As curvas
macro são a média, entre as 5 classes, das curvas de cada classe contra as outras,
numa grade comum (FPR na ROC; revocação na PR, com a precisão interpolada pelo
máximo à direita, a forma usual da curva PR). O número da legenda é a ROC-AUC macro
e a PR-AUC macro (precisão média), os mesmos das tabelas.

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
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve  # noqa: E402

from src.config import DPI_FIGURES  # noqa: E402
from src.experiment_config import MODEL_NAMES, PLOTS_DIR, RESULTS_DIR, ROC_GRID_POINTS, TARGET, load_experiment_dataset  # noqa: E402
from src.pipeline_builder import MODEL_LABELS  # noqa: E402

from src.plot_style import (  # noqa: E402
    ACCESSIBLE_PALETTE, COLOR_A, COLOR_B, COLOR_HOMOLOGY, COLOR_NEW_GENE, MECHANISM_COLORS, MECHANISMS, num, save, short,
)

logger = logging.getLogger(__name__)
INDIVIDUAL = "individuais"  # subpasta das versões de um gráfico por arquivo
PREDICTIONS_DIR = RESULTS_DIR / "predicoes"


def _save(fig, path: Path, dpi: int = DPI_FIGURES) -> None:
    """PNG e PDF pelo ``plot_style.save`` (vírgula decimal nos eixos; PDF sem data, para não mudar a cada execução)."""
    save(fig, path, dpi)


def _in_order(classes: list[str]) -> list[str]:
    """As classes na ordem de tamanho usada em todas as figuras (inativação primeiro), e não em ordem alfabética."""
    return [c for c in MECHANISMS if c in classes] + [c for c in classes if c not in MECHANISMS]


def plot_class_distribution(counts: pd.Series, output_path: Path) -> None:
    """
    Genes por mecanismo nas 5 classes do experimento, com a porcentagem (Figura 4.2).

    Complementa ``eda/class_distribution`` (os 10 rótulos e o destino de cada um): aqui
    estão só as classes que o modelo vê, numa escala em que as menores aparecem.
    """
    counts = counts.reindex([m for m in MECHANISMS if m in counts.index]).iloc[::-1]
    total = counts.sum()
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.barh([short(m) for m in counts.index], counts.values, color=[MECHANISM_COLORS[m] for m in counts.index], height=0.65)
    for b, v in zip(bars, counts.values):
        ax.text(b.get_width() + total * 0.006, b.get_y() + b.get_height() / 2, f"{v} ({num(100 * v / total, 1)}%)",
                va="center", fontsize=10)
    ax.set_xlabel("Número de genes")
    ax.set_title(f"Genes por mecanismo nas 5 classes do experimento ({total:_} genes)".replace("_", "."), fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", alpha=0.2)
    ax.set_axisbelow(True)
    ax.set_xlim(0, counts.max() * 1.2)
    _save(fig, output_path)


def plot_f1_delta_percent(metrics_a: pd.DataFrame, metrics_b: pd.DataFrame, output_path: Path) -> None:
    """
    Ganho relativo do B sobre o A no F1-macro de teste, com o ganho absoluto entre parênteses.

    Leitura: o percentual depende do ponto de partida (cresce quando o A é pior), por isso o
    rótulo traz também a diferença absoluta. É uma figura do teste; o argumento principal é
    o ganho na validação cruzada (``ganho_cv``).
    """
    fa = metrics_a[metrics_a.metrica == "f1_macro"].set_index("modelo")["valor_teste"].loc[MODEL_NAMES]
    fb = metrics_b[metrics_b.metrica == "f1_macro"].set_index("modelo")["valor_teste"].loc[MODEL_NAMES]
    delta = (fb - fa) / fa * 100
    labels = [MODEL_LABELS[m] for m in MODEL_NAMES]
    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.barh(labels, delta.values, color=ACCESSIBLE_PALETTE[:4], edgecolor="black", linewidth=0.5, height=0.6)
    for b, v, d in zip(bars, delta.values, (fb - fa).values):
        ax.text(b.get_width() + 0.6, b.get_y() + b.get_height() / 2, f"{num(v, 1, sign=True)}% ({num(d, 3, sign=True)})",
                va="center", fontsize=10)
    ax.invert_yaxis()
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_xlim(0, max(delta.max() * 1.3, 1))
    ax.set_xlabel("Ganho do Pipeline B sobre o A no F1-macro de teste (%)")
    ax.set_title("Ganho relativo do Pipeline B no F1-macro (teste)", fontweight="bold")
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
            ax.text(b.get_x() + b.get_width() / 2, v + 0.012, num(v), ha="center", va="bottom", fontsize=10)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_NAMES], fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.set_ylabel("F1-macro no teste")
    ax.set_title("F1-macro no teste — Pipeline A e Pipeline B", fontweight="bold")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.08), ncol=2, frameon=False, fontsize=11)
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


# Recorte ampliado da ROC: com AUC perto de 1, as curvas de A e B ficam coladas no canto
# superior esquerdo, e a diferença só aparece ampliando essa região.
ROC_ZOOM_FPR: tuple[float, float] = (0.0, 0.10)
ROC_ZOOM_TPR: tuple[float, float] = (0.75, 1.005)


def plot_roc_curve(model: str, output_path: Path) -> None:
    """
    ROC macro de A e B de um modelo, com o detalhe ampliado da região de poucos falsos
    positivos. É a ROC num arquivo próprio (na figura ``roc_pr_all_models`` ela não tem o
    detalhe). Cores como nas outras figuras: A verde tracejado, B laranja.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    curves = []
    for key, color, style in (("a", COLOR_A, "--"), ("b", COLOR_B, "-")):
        d = _load_pred(key, model)
        grid, tpr, auc = macro_roc_curve(d["y_true"], d["proba"])
        ax.plot(grid, tpr, style, color=color, lw=2, label=f"Pipeline {key.upper()} (ROC-AUC = {num(auc)})")
        curves.append((grid, tpr, style, color))
    ax.plot([0, 1], [0, 1], color="grey", lw=0.5, alpha=0.6)
    ax.set_xlabel("Taxa de falsos positivos")
    ax.set_ylabel("Revocação (TPR)")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    inset = ax.inset_axes([0.38, 0.22, 0.58, 0.55])
    for grid, tpr, style, color in curves:
        inset.plot(grid, tpr, style, color=color, lw=2)
    inset.set_xlim(*ROC_ZOOM_FPR)
    inset.set_ylim(*ROC_ZOOM_TPR)
    inset.tick_params(labelsize=7)
    inset.grid(alpha=0.2)
    inset.set_title(f"detalhe: falsos positivos ≤ {num(ROC_ZOOM_FPR[1], 2)}", fontsize=8)
    _, connectors = ax.indicate_inset_zoom(inset, edgecolor="grey")
    for line in connectors:  # só o retângulo; as linhas de ligação cruzavam as curvas
        line.set_visible(False)
    ax.set_title(f"ROC macro — {MODEL_LABELS[model]}", fontweight="bold")
    _save(fig, output_path)


def macro_pr_curve(y_true: np.ndarray, proba: np.ndarray, n_points: int = ROC_GRID_POINTS):
    """
    Curva precisão-revocação macro: em cada classe (contra as outras), precisão interpolada
    (máximo da precisão para revocação ≥ r) numa grade comum de revocação; média entre classes.
    Devolve também a PR-AUC macro (média da precisão média de cada classe).
    """
    grid = np.linspace(0, 1, n_points)
    precisions, aps = [], []
    for c in range(proba.shape[1]):
        positive = (y_true == c).astype(int)
        precision, recall, _ = precision_recall_curve(positive, proba[:, c])
        # Precisão interpolada pela definição: em cada revocação r da grade, a maior
        # precisão entre os pontos com revocação >= r (pontos empatados incluídos).
        precisions.append(np.array([precision[recall >= r].max() for r in grid]))
        aps.append(average_precision_score(positive, proba[:, c]))
    return grid, np.mean(precisions, axis=0), float(np.mean(aps))


def plot_roc_pr_curves(output_path: Path) -> None:
    """Uma linha por modelo: ROC macro à esquerda e precisão-revocação macro à direita, A e B."""
    fig, axes = plt.subplots(len(MODEL_NAMES), 2, figsize=(12, 4.2 * len(MODEL_NAMES)))
    for row, model in zip(axes, MODEL_NAMES):
        for key, color, style in (("a", COLOR_A, "--"), ("b", COLOR_B, "-")):
            d = _load_pred(key, model)
            grid, tpr, auc = macro_roc_curve(d["y_true"], d["proba"])
            row[0].plot(grid, tpr, style, color=color, lw=2, label=f"Pipeline {key.upper()} (ROC-AUC = {num(auc)})")
            rgrid, prec, ap = macro_pr_curve(d["y_true"], d["proba"])
            row[1].plot(rgrid, prec, style, color=color, lw=2, label=f"Pipeline {key.upper()} (PR-AUC = {num(ap)})")
        row[0].plot([0, 1], [0, 1], color="grey", lw=0.5, alpha=0.6)
        row[0].set_xlabel("Taxa de falsos positivos")
        row[0].set_ylabel("Revocação (TPR)")
        row[0].legend(loc="lower right", fontsize=8)
        row[1].set_xlabel("Revocação")
        row[1].set_ylabel("Precisão")
        row[1].legend(loc="lower left", fontsize=8)
        for ax, kind in zip(row, ("ROC", "Precisão-revocação")):
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1.02)
            ax.grid(alpha=0.2)
            ax.set_title(f"{MODEL_LABELS[model]} — {kind} (macro)", fontweight="bold", fontsize=10)
    fig.tight_layout()
    _save(fig, output_path)


def plot_gain_per_class(boot: pd.DataFrame, output_path: Path, only: str | None = None) -> None:
    """
    Mapa de calor de B − A por classe (linhas) e modelo (colunas), para precisão, revocação
    e F1 no teste. Descritivo (sem teste). Responde "o B ganha porque deixa de perder genes
    (revocação) ou porque dá menos alarmes falsos (precisão)?".

    Cada célula traz só o valor; a incerteza aparece como contorno (IC 95% do bootstrap de
    grupos acima de zero), como na figura da CV. Os intervalos ficam no CSV.
    """
    from matplotlib.patches import Rectangle

    diff = boot[(boot["comparacao"] == "b_menos_a") & (boot["classe"] != "macro")]
    classes = _in_order(list(dict.fromkeys(diff["classe"])))
    short = [c.replace("antibiotic ", "") for c in classes]
    metrics = [("precisao", "Precisão"), ("revocacao", "Revocação"), ("f1", "F1")]
    if only:  # um gráfico por arquivo
        metrics = [m for m in metrics if m[0] == only]
    fig, axes = plt.subplots(1, len(metrics), figsize=(16 if len(metrics) == 3 else 7, 4.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (metric, title) in zip(axes, metrics):
        sub = diff[diff["metrica"] == metric].set_index(["classe", "modelo"])
        values = np.array([[sub.loc[(c, m), "valor"] for m in MODEL_NAMES] for c in classes])
        im = ax.imshow(values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        for i, c in enumerate(classes):
            for j, m in enumerate(MODEL_NAMES):
                r = sub.loc[(c, m)]
                ax.text(j, i, num(r["valor"], 2, sign=True), ha="center", va="center", fontsize=10,
                        color="white" if abs(r["valor"]) > 0.6 else "black")
                if r["ic95_inf"] > 0:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.6))
        ax.set_xticks(range(len(MODEL_NAMES)), [MODEL_LABELS[m] for m in MODEL_NAMES], rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{title}: B − A no teste", fontweight="bold", fontsize=11)
    axes[0].set_yticks(range(len(short)), short)
    fig.colorbar(im, ax=axes, fraction=0.02 if len(metrics) == 3 else 0.04, pad=0.01, label="B − A (contorno: IC 95% acima de zero)")
    _save(fig, output_path)


def plot_cv_gain(tests: pd.DataFrame, output_path: Path) -> None:
    """
    Ganho B − A no F1-macro na validação cruzada, com o IC 95% corrigido de Nadeau e
    Bengio, nos 8 testes confirmatórios (2 cenários × 4 modelos), com o p de Holm.

    Por que: as conclusões do trabalho se apoiam na CV (50 estimativas pareadas), e não no
    teste, que é uma partição única dominada pela família PDC. Esta é a figura do
    resultado principal; a de F1 no teste fica como complemento. Os p de Holm (todos
    abaixo de 0,001) ficam em ``tabela_teste_t.tex``: na figura, bastam o ganho e o
    intervalo. Fonte: ``results/testes_estatisticos.csv``.
    """
    labels = {"anotacao_curada": "B (anotação curada) − A", "gene_novo": "B (gene novo) − A"}
    colors = {"anotacao_curada": COLOR_B, "gene_novo": COLOR_NEW_GENE}
    fig, ax = plt.subplots(figsize=(9, 4.8))
    y = np.arange(len(MODEL_NAMES))
    for scen, off in (("anotacao_curada", -0.15), ("gene_novo", 0.15)):
        sub = tests[tests["cenario"] == scen].set_index("modelo").loc[MODEL_NAMES]
        d = sub["diferenca_media"].to_numpy()
        err = np.vstack([d - sub["ic95_inf"].to_numpy(), sub["ic95_sup"].to_numpy() - d])
        ax.errorbar(d, y + off, xerr=err, fmt="o", color=colors[scen], capsize=4, lw=2, ms=7, label=labels[scen])
        for yi, (di, hi) in enumerate(zip(d, sub["ic95_sup"])):
            ax.text(hi + 0.01, yi + off, num(di, 2, sign=True), va="center", fontsize=9)
    ax.axvline(0, color="0.4", lw=1)
    ax.set_yticks(y, [MODEL_LABELS[m] for m in MODEL_NAMES])
    ax.invert_yaxis()
    ax.set_xlim(-0.05, float(tests["ic95_sup"].max()) + 0.1)
    ax.set_xlabel("Ganho no F1-macro (média dos 50 folds e IC 95% corrigido)")
    ax.set_title("Ganho B − A no F1-macro na validação cruzada", fontweight="bold")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9, frameon=False)
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output_path)


def plot_cv_gain_per_class(gain: pd.DataFrame, output_path: Path, only: str | None = None) -> None:
    """
    B − A no F1 de cada classe na CV (anotação curada e gene novo), com o IC corrigido.
    Célula com contorno = IC acima de zero. ``only`` = um cenário só (um gráfico por arquivo).

    Por que: o ganho por classe só existia no bootstrap do teste (``ganho_por_classe``);
    na CV, que é a base das conclusões, ele é comprovado na substituição e na alteração do
    alvo, mais fraco no efluxo e não comprovado na proteção. Descritivo (sem correção para
    comparações múltiplas). Fonte: ``results/ganho_por_classe_cv.csv`` (``src.data_characterization``).
    """
    from matplotlib.patches import Rectangle

    classes = _in_order(list(dict.fromkeys(gain["classe"])))
    scenarios = [("anotacao_curada", "Anotação curada"), ("gene_novo", "Gene novo")]
    if only:
        scenarios = [s_ for s_ in scenarios if s_[0] == only]
    fig, axes = plt.subplots(1, len(scenarios), figsize=(12 if len(scenarios) == 2 else 7, 4.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, (scen, title) in zip(axes, scenarios):
        sub = gain[gain["cenario"] == scen].set_index(["classe", "modelo"])
        values = np.array([[sub.loc[(c, m), "dif_cv"] for m in MODEL_NAMES] for c in classes])
        im = ax.imshow(values, cmap="RdBu", vmin=-1, vmax=1, aspect="auto")
        for i, c in enumerate(classes):
            for j, m in enumerate(MODEL_NAMES):
                r = sub.loc[(c, m)]
                ax.text(j, i, num(r["dif_cv"], 2, sign=True), ha="center", va="center",
                        fontsize=10, color="white" if abs(r["dif_cv"]) > 0.6 else "black")
                if r["ic_acima_de_zero"]:
                    ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", lw=1.6))
        ax.set_xticks(range(len(MODEL_NAMES)), [MODEL_LABELS[m] for m in MODEL_NAMES], rotation=20, ha="right", fontsize=9)
        ax.set_title(f"{title}: B − A no F1 de cada classe (CV)", fontweight="bold", fontsize=11)
    axes[0].set_yticks(range(len(classes)), [c.replace("antibiotic ", "") for c in classes])
    fig.colorbar(im, ax=axes, fraction=0.02 if len(scenarios) == 2 else 0.04, pad=0.01, label="B − A (contorno: IC 95% acima de zero)")
    _save(fig, output_path)


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str, output_path: Path) -> None:
    """Matriz normalizada por linha (fração e número de genes), com as classes na ordem das outras figuras."""
    order = [class_names.index(c) for c in _in_order(list(class_names))]
    cm = np.asarray(cm)[np.ix_(order, order)]
    class_names = [class_names[i] for i in order]
    norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=1)
    for i in range(norm.shape[0]):
        for j in range(norm.shape[1]):
            ax.text(
                j,
                i,
                f"{num(norm[i, j], 2)}\n(n={cm[i, j]})",
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
    ax.set_title(title, fontweight="bold")
    fig.colorbar(im, ax=ax, fraction=0.046, label="Fração dos genes da classe verdadeira")
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


def plot_threshold_curve(curve: pd.DataFrame, partitions: pd.DataFrame, output_path: Path,
                         models: list[str] = MODEL_NAMES, rows: tuple[str, ...] = ("f1", "ganho")) -> None:
    """
    F1-macro no teste por divisão (aleatória → grupos de 90, 70 e 50%), A e B, um painel por modelo
    (linha de cima), e o ganho B − A (linha de baixo).

    Mostra se a vantagem da ontologia cresce quando o teste fica menos parecido com o
    treino. Cenários que não atendem o critério de entrada das classes (menos de 5
    genes de uma classe num fold) aparecem com marcador vazio.

    Linha de baixo: ganho médio na validação cruzada repetida 10 × 5,
    com o IC 95% do teste t corrigido (``ganho_cv_ic95_inf/sup`` de ``curva_limiar.csv``),
    e o ganho no teste como referência. A linha de cima vem de uma única partição de
    teste por cenário; o intervalo da CV mostra quanto o ganho varia entre divisões.
    Os intervalos de cenários diferentes não são uma comparação pareada (cada cenário
    tem outra partição): a leitura é descritiva, sem teste.

    Entram também o Pipeline B no cenário "gene novo" (tracejado) e a referência por
    homologia (pontilhado) na linha de cima, e o ganho do B gene novo sobre o A, com o
    intervalo corrigido, na linha de baixo.
    """
    order = ["aleatoria", "90", "70", "50"]
    labels = ["aleatória", "grupos 90%", "grupos 70%", "grupos 50%"]
    ok = partitions.set_index("cenario")["atende_criterio"].astype(bool)
    has_ci = {"ganho_cv", "ganho_cv_ic95_inf", "ganho_cv_ic95_sup"} <= set(curve.columns)
    draw_f1, draw_gain = "f1" in rows, has_ci and "ganho" in rows
    n_rows = int(draw_f1) + int(draw_gain)
    full = len(models) == 4
    fig, grid = plt.subplots(n_rows, len(models), figsize=(16 if full else 7, 4.2 * n_rows), sharey="row", squeeze=False)
    axes = grid[0] if draw_f1 else []
    g_row = 1 if draw_f1 else 0
    if draw_gain:
        for ax, model in zip(grid[g_row], models):
            sub = curve[curve.modelo == model].set_index("cenario").loc[order]
            x = np.arange(len(order))
            g = sub["ganho_cv"].values
            err = np.vstack([g - sub["ganho_cv_ic95_inf"].values, sub["ganho_cv_ic95_sup"].values - g])
            ax.errorbar(x, g, yerr=err, fmt="-", color="0.25", lw=2, capsize=4, label="CV 10 × 5 (IC 95% corrigido)")
            for xi, v, c in zip(x, g, order):
                ax.plot(xi, v, "o", ms=7, color="0.25", mfc="0.25" if ok[str(c)] else "white", mew=2)
            ax.plot(x, sub["ganho_teste"].values, "D", ms=6, color=COLOR_B, label="teste")
            if "ganho_cv_gene_novo" in sub:
                gn = sub["ganho_cv_gene_novo"].values
                err_n = np.vstack([gn - sub["ganho_cv_gene_novo_ic95_inf"].values, sub["ganho_cv_gene_novo_ic95_sup"].values - gn])
                ax.errorbar(x + 0.12, gn, yerr=err_n, fmt="--", color=COLOR_NEW_GENE, lw=2, capsize=4,
                            label="gene novo: CV 10 × 5 (IC 95%)")
            ax.axhline(0, color="0.5", lw=0.8)
            ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=9)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        grid[g_row][0].set_ylabel("Ganho B − A no F1-macro")
        grid[g_row][0].legend(loc="upper left", fontsize=8)
    for ax, model in zip(axes, models):
        sub = curve[curve.modelo == model].set_index("cenario").loc[order]
        x = np.arange(len(order))
        series = [("a", COLOR_A, "Pipeline A", "-"), ("b", COLOR_B, "Pipeline B (anotação curada)", "-"),
                  ("b_gene_novo", COLOR_NEW_GENE, "Pipeline B (gene novo)", "--"),
                  ("homologia", COLOR_HOMOLOGY, "Homologia (sem ML)", ":")]
        for key, color, label, style in series:
            if f"f1_teste_{key}" not in sub:
                continue
            vals = sub[f"f1_teste_{key}"].values
            ax.plot(x, vals, style, color=color, lw=2, label=label)
            for xi, v, c in zip(x, vals, order):
                ax.plot(xi, v, "o", ms=7, color=color, mfc=color if ok[str(c)] else "white", mew=2)
        ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=9)
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    if len(axes):
        axes[0].set_ylabel("F1-score macro (teste)")
        axes[0].legend(loc="lower left", fontsize=8)
    if full:
        fig.suptitle("Desempenho por divisão treino/teste (marcador vazio: não atende o critério de genes por fold)", fontsize=11)
    else:
        what = "F1-macro no teste" if draw_f1 else "Ganho B − A"
        grid[0][0].set_title(f"{what} por divisão treino/teste — {MODEL_LABELS[models[0]]}\n"
                             "(marcador vazio: não atende o critério de genes por fold)", fontweight="bold", fontsize=11)
    fig.tight_layout()
    _save(fig, output_path)


def plot_threshold_gain_summary(curve: pd.DataFrame, partitions: pd.DataFrame, output_path: Path) -> None:
    """
    Ganho B − A no F1-macro (média da CV 10 × 5) por divisão treino/teste, uma linha por modelo.

    Por que um gráfico só: a mensagem é "o ganho da ontologia cresce quando o teste fica
    menos parecido com o treino". Os 8 painéis por modelo (F1 de cada pipeline, teste,
    gene novo, intervalos) continuam em ``individuais/curva_limiar_{f1|ganho}_{modelo}``.
    Marcador vazio: a divisão não atende o critério de genes por fold.
    """
    from matplotlib.lines import Line2D

    order = ["aleatoria", "90", "70", "50"]
    ok = partitions.set_index("cenario")["atende_criterio"].astype(bool)
    x = np.arange(len(order))
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for model, color in zip(MODEL_NAMES, ACCESSIBLE_PALETTE):
        g = curve[curve.modelo == model].set_index("cenario").loc[order, "ganho_cv"].to_numpy()
        ax.plot(x, g, "-", color=color, lw=2, label=MODEL_LABELS[model])
        for xi, v, c in zip(x, g, order):
            ax.plot(xi, v, "o", ms=7, color=color, mfc=color if ok[str(c)] else "white", mew=2)
    ax.axhline(0, color="0.5", lw=0.8)
    ax.set_xticks(x, ["aleatória", "grupos de 90%", "grupos de 70%\n(experimento)", "grupos de 50%"])
    ax.set_ylim(0, None)
    ax.set_xlabel("Divisão treino/teste (da mais parecida à menos parecida)")
    ax.set_ylabel("Ganho B − A no F1-macro (CV)")
    ax.set_title("Ganho da ontologia por tipo de divisão treino/teste", fontweight="bold")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="none", color="0.3", mfc="white", mew=2, ms=7,
                          label="não atende o critério de genes por fold"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3, frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output_path)


def plot_threshold_f1_summary(curve: pd.DataFrame, partitions: pd.DataFrame, output_path: Path) -> None:
    """
    F1-macro na validação cruzada (10 × 5) por divisão treino/teste: A, B (anotação curada),
    B (gene novo) e a referência por homologia. A e os B são a média dos 4 modelos; a
    homologia não tem modelo. Usa a CV, base das conclusões (o teste é uma partição só por
    divisão); o F1 no teste, por modelo, está em ``individuais/curva_limiar_f1_{modelo}``.

    Complementa ``curva_limiar`` (só o ganho): mostra o nível de cada um e que a homologia
    e o B gene novo, que dependem do vizinho mais parecido, caem quando o teste fica
    distante do treino, enquanto o B curado se mantém. Marcador vazio: a divisão não
    atende o critério de genes por fold.
    """
    from matplotlib.lines import Line2D

    order = ["aleatoria", "90", "70", "50"]
    ok = partitions.set_index("cenario")["atende_criterio"].astype(bool)
    x = np.arange(len(order))
    series = [("a", COLOR_A, "-", "Pipeline A"), ("b", COLOR_B, "-", "Pipeline B (anotação curada)"),
              ("b_gene_novo", COLOR_NEW_GENE, "--", "Pipeline B (gene novo)"),
              ("homologia", COLOR_HOMOLOGY, ":", "Homologia (sem aprendizado de máquina)")]
    fig, ax = plt.subplots(figsize=(8.5, 4.8))
    for key, color, style, label in series:
        col = f"f1_cv_media_{key}"
        if col not in curve:
            continue
        vals = curve.pivot_table(index="cenario", columns="modelo", values=col).loc[order].mean(axis=1).to_numpy()
        ax.plot(x, vals, style, color=color, lw=2.2, label=label)
        for xi, v, c in zip(x, vals, order):
            ax.plot(xi, v, "o", ms=7, color=color, mfc=color if ok[str(c)] else "white", mew=2)
    ax.set_xticks(x, ["aleatória", "grupos de 90%", "grupos de 70%\n(experimento)", "grupos de 50%"])
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Divisão treino/teste (da mais parecida à menos parecida)")
    ax.set_ylabel("F1-macro na CV (média dos 4 modelos)")
    ax.set_title("F1-macro na validação cruzada por tipo de divisão treino/teste", fontweight="bold")
    handles, _ = ax.get_legend_handles_labels()
    handles.append(Line2D([], [], marker="o", ls="none", color="0.3", mfc="white", mew=2, ms=7,
                          label="não atende o critério de genes por fold"))
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, fontsize=9)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    _save(fig, output_path)


def plot_scenarios(table: pd.DataFrame, output_path: Path) -> None:
    """
    F1-macro no teste do A, do B (anotação curada e gene novo) e da referência por
    homologia, com o IC 95% do bootstrap de grupos (``results/tabela_cenarios.csv``).
    Descritivo. A homologia é a mesma em todos os grupos de barras (não tem modelo).
    """
    series = [("a", COLOR_A, "Pipeline A"), ("b", COLOR_B, "Pipeline B (anotação curada)"),
              ("b_gene_novo", COLOR_NEW_GENE, "Pipeline B (gene novo)"), ("homologia", COLOR_HOMOLOGY, "Homologia (sem aprendizado de máquina)")]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.2
    x = np.arange(len(MODEL_NAMES))
    t = table.set_index("modelo").loc[MODEL_NAMES]
    for i, (key, color, label) in enumerate(series):
        v = t[f"f1_teste_{key}"].values
        err = np.vstack([v - t[f"ic95_inf_{key}"].values, t[f"ic95_sup_{key}"].values - v])
        ax.bar(x + (i - 1.5) * w, v, w, color=color, label=label, yerr=err, capsize=3, edgecolor="white")
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_NAMES])
    ax.set_ylabel("F1-macro no teste (IC 95%)")
    ax.set_ylim(0, 1.05)
    ax.set_title("F1-macro no teste: anotação curada, gene novo e homologia", fontweight="bold")
    ax.legend(ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), fontsize=9, frameon=False)
    ax.grid(axis="y", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    _save(fig, output_path)


# ── Interpretação e overfitting ─────────────────────────────────────────────────
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
# Contextos mostrados na figura de importância por grupo: A e B (anotação curada). O
# contexto "gene novo" fica no CSV e na tabela; na figura, uma terceira barra por grupo
# dobrava a informação sem mudar a leitura.
CONTEXT_STYLE: dict[str, tuple[str, str]] = {
    "a": (COLOR_A, "Pipeline A"), "b": (COLOR_B, "Pipeline B"),
}


def _short(c: str) -> str:
    return c.replace("antibiotic ", "")


def plot_group_importance(summary: pd.DataFrame, output_path: Path, models: list[str] = MODEL_NAMES) -> None:
    """
    Queda do F1-macro ao embaralhar cada grupo de atributos (média nos 50 folds, IC 95%
    corrigido), por modelo, nos pipelines A e B (anotação curada). Descritivo.
    """
    groups = list(GROUP_LABELS)
    fig, axes = plt.subplots(1, len(models), figsize=(18 if len(models) == 4 else 7, 4.8), sharex=True, sharey=True, squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        sub = summary[summary["modelo"] == model]
        h = 0.36
        for i, (ctx, (color, label)) in enumerate(CONTEXT_STYLE.items()):
            s_ctx = sub[sub["contexto"] == ctx].set_index("conjunto")
            for j, g in enumerate(groups):
                if g not in s_ctx.index:
                    continue
                r = s_ctx.loc[g]
                y = j + (i - 0.5) * h
                ax.barh(y, r["cv_media_macro"], h, color=color, label=label if j == 0 else None)
                ax.errorbar(r["cv_media_macro"], y, xerr=[[r["cv_media_macro"] - r["cv_ic95_inf_macro"]],
                            [r["cv_ic95_sup_macro"] - r["cv_media_macro"]]], fmt="none", ecolor="0.2", capsize=2, lw=1)
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_yticks(range(len(groups)), [GROUP_LABELS[g] for g in groups], fontsize=9)
        ax.invert_yaxis()
        ax.set_title(MODEL_LABELS[model] if len(models) == 4 else f"Importância por permutação dos grupos — {MODEL_LABELS[model]}",
                     fontweight="bold")
        ax.set_xlabel("Queda do F1-macro (CV, IC 95%)")
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    if len(models) == 4:
        axes[0].legend(loc="lower right", fontsize=8)
    else:
        axes[0].legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout()
    _save(fig, output_path)


def plot_group_importance_per_class(summary: pd.DataFrame, classes: list[str], output_path: Path, context: str = "b",
                                    models: list[str] = MODEL_NAMES) -> None:
    """Mapa de calor grupo × mecanismo da queda do F1 de cada classe (média da CV), por modelo."""
    groups = list(GROUP_LABELS)
    sub_all = summary[summary["contexto"] == context]
    vmax = max(0.05, float(np.nanmax([sub_all[f"cv_media_{c}"].max() for c in classes])))
    fig, axes = plt.subplots(1, len(models), figsize=(20 if len(models) == 4 else 7.5, 4.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        sub = sub_all[sub_all["modelo"] == model].set_index("conjunto").reindex(groups)
        values = sub[[f"cv_media_{c}" for c in classes]].to_numpy(dtype=float)
        im = ax.imshow(values, cmap="Oranges", vmin=0, vmax=vmax, aspect="auto")
        for i in range(len(groups)):
            for j in range(len(classes)):
                v = values[i, j]
                ax.text(j, i, num(v, 2).replace("-0,00", "0,00"), ha="center", va="center", fontsize=8, color="white" if v > 0.6 * vmax else "black")
        ax.set_xticks(range(len(classes)), [_short(c) for c in classes], rotation=30, ha="right", fontsize=8)
        ax.set_title(MODEL_LABELS[model], fontweight="bold")
    axes[0].set_yticks(range(len(groups)), [GROUP_LABELS[g] for g in groups], fontsize=9)
    fig.colorbar(im, ax=axes, fraction=0.015, pad=0.01, label="Queda do F1 da classe (CV)")
    _save(fig, output_path)


def plot_top_features(summary: pd.DataFrame, output_path: Path, top_n: int = 15,
                      models: list[str] = MODEL_NAMES, contexts: tuple[str, ...] = ("a", "b")) -> None:
    """
    Os ``top_n`` atributos com maior queda média do F1-macro na CV, por modelo (colunas)
    e pipeline (linhas), com IC 95%; cor = grupo; barra clara = IC inclui zero (não consistente).
    """
    from matplotlib.patches import Patch

    from src.interpretability import feature_group

    full = len(models) == 4 and len(contexts) == 2
    fig, grid = plt.subplots(len(contexts), len(models), figsize=(22, 10) if full else (6.5 * len(models), 5.5 * len(contexts)), squeeze=False)
    for row, ctx in enumerate(contexts):
        for ax, model in zip(grid[row], models):
            sub = summary[(summary["contexto"] == ctx) & (summary["modelo"] == model)].nsmallest(top_n, "posicao_cv")
            y = np.arange(len(sub))
            for yi, (_, r) in zip(y, sub.iterrows()):
                color = GROUP_COLORS[feature_group(r["conjunto"])]
                ax.barh(yi, r["cv_media_macro"], color=color, alpha=1.0 if r["consistente"] else 0.35)
                ax.errorbar(r["cv_media_macro"], yi, xerr=[[r["cv_media_macro"] - r["cv_ic95_inf_macro"]],
                            [r["cv_ic95_sup_macro"] - r["cv_media_macro"]]], fmt="none", ecolor="0.2", capsize=2, lw=1)
            ax.set_yticks(y, sub["conjunto"], fontsize=8)
            ax.invert_yaxis()
            ax.axvline(0, color="0.5", lw=0.8)
            ax.set_title(f"{MODEL_LABELS[model]} — Pipeline {ctx.upper()}", fontweight="bold", fontsize=10)
            ax.grid(axis="x", alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        grid[row][0].set_xlabel("Queda do F1-macro (CV, IC 95%)")
    fig.legend(handles=[Patch(color=c, label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()]
               + [Patch(color="0.6", alpha=0.35, label="barra clara: IC inclui zero")],
               loc="lower center", ncol=6 if full else 2, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, 0.04 if full else 0.14, 1, 1))
    _save(fig, output_path)


def plot_shap_per_class(shap: pd.DataFrame, classes: list[str], output_path: Path, model: str = "xgboost",
                        pipeline: str = "b", top_n: int = 10) -> None:
    """
    Para os genes de cada mecanismo (teste), os ``top_n`` atributos com maior |SHAP| médio
    na saída daquele mecanismo. Barra = SHAP médio com sinal (à direita: empurra para o
    mecanismo); cor = grupo do atributo. O |SHAP| médio, que define a ordem, fica no CSV:
    desenhado junto, eram duas medidas no mesmo eixo.
    """
    from matplotlib.patches import Patch

    sub = shap[(shap["modelo"] == model) & (shap["pipeline"] == pipeline) & (shap["nivel"] == "atributo")]
    unit = "log-odds" if model == "xgboost" else "probabilidade"
    fig, grid = plt.subplots(2, 3, figsize=(19, 9.5))
    axes = grid.ravel()
    for ax, c in zip(axes, classes):
        s = sub[sub["classe"] == c].nlargest(top_n, "media_abs_shap")
        y = np.arange(len(s))
        ax.barh(y, s["media_shap"], color=[GROUP_COLORS[g] for g in s["grupo"]])
        ax.set_yticks(y, s["nome"], fontsize=8.5)
        ax.invert_yaxis()
        ax.axvline(0, color="0.5", lw=0.8)
        ax.set_title(_short(c), fontweight="bold", fontsize=11)
        ax.set_xlabel(f"SHAP médio nos genes da classe ({unit})")
        ax.grid(axis="x", alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[len(classes):]:
        ax.axis("off")
    handles = [Patch(color=c, label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()]
    axes[-1].legend(handles=handles, loc="center", fontsize=10, frameon=False, title="Grupo do atributo")
    fig.suptitle(f"SHAP por mecanismo — {MODEL_LABELS[model]}, Pipeline {pipeline.upper()} (teste)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save(fig, output_path)


def plot_shap_one_class(shap: pd.DataFrame, cls: str, output_path: Path, model: str = "xgboost",
                        pipeline: str = "b", top_n: int = 10) -> None:
    """O painel de um mecanismo da figura SHAP por mecanismo, sozinho (com a legenda dos grupos)."""
    from matplotlib.patches import Patch

    sub = shap[(shap["modelo"] == model) & (shap["pipeline"] == pipeline) & (shap["nivel"] == "atributo")]
    unit = "log-odds" if model == "xgboost" else "probabilidade"
    fig, ax = plt.subplots(figsize=(7.5, 5))
    s = sub[sub["classe"] == cls].nlargest(top_n, "media_abs_shap")
    y = np.arange(len(s))
    present = list(dict.fromkeys(s["grupo"]))
    ax.barh(y, s["media_shap"], color=[GROUP_COLORS[g] for g in s["grupo"]])
    ax.set_yticks(y, s["nome"], fontsize=8.5)
    ax.invert_yaxis()
    ax.axvline(0, color="0.5", lw=0.8)
    ax.set_title(f"SHAP — {_short(cls)} ({MODEL_LABELS[model]}, Pipeline {pipeline.upper()})", fontweight="bold", fontsize=11)
    ax.set_xlabel(f"SHAP médio nos genes da classe ({unit})")
    ax.grid(axis="x", alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    # Só os grupos que aparecem neste painel.
    ax.legend(handles=[Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g]) for g in GROUP_COLORS if g in present],
              loc="lower right", fontsize=8)
    fig.tight_layout()
    _save(fig, output_path)


def plot_pr_curve(model: str, output_path: Path) -> None:
    """Precisão-revocação macro de A e B de um modelo (o painel direito da figura ROC × PR, sozinho)."""
    fig, ax = plt.subplots(figsize=(7, 6))
    for key, color, style in (("a", COLOR_A, "--"), ("b", COLOR_B, "-")):
        d = _load_pred(key, model)
        rgrid, prec, ap = macro_pr_curve(d["y_true"], d["proba"])
        ax.plot(rgrid, prec, style, color=color, lw=2, label=f"Pipeline {key.upper()} (PR-AUC = {num(ap)})")
    ax.set_xlabel("Revocação")
    ax.set_ylabel("Precisão")
    ax.legend(loc="lower left", fontsize=9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.grid(alpha=0.2)
    ax.spines[["top", "right"]].set_visible(False)
    ax.set_title(f"Precisão-revocação macro — {MODEL_LABELS[model]}", fontweight="bold")
    _save(fig, output_path)


def plot_learning_curve(curve: pd.DataFrame, output_path: Path, models: list[str] = MODEL_NAMES) -> None:
    """F1-macro médio na validação (50 folds, IC 95%) por fração dos grupos do treino, A e B; marcador vazio = não atende o critério."""
    fig, axes = plt.subplots(1, len(models), figsize=(17 if len(models) == 4 else 6.5, 4.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        for key, color in (("a", COLOR_A), ("b", COLOR_B)):
            sub = curve[(curve["modelo"] == model) & (curve["pipeline"] == key)].sort_values("fracao")
            x = 100 * sub["fracao"].to_numpy()
            ax.fill_between(x, sub["ic95_inf_macro"], sub["ic95_sup_macro"], color=color, alpha=0.2)
            ax.plot(x, sub["media_macro"], "-", color=color, lw=2, label=f"Pipeline {key.upper()}")
            for xi, v, ok in zip(x, sub["media_macro"], sub["atende_criterio"]):
                ax.plot(xi, v, "o", ms=7, color=color, mfc=color if ok else "white", mew=2)
        ax.set_title(MODEL_LABELS[model] if len(models) == 4 else
                     f"Curva de aprendizado — {MODEL_LABELS[model]}\n(marcador vazio: alguma classe com < 5 genes no treino em algum fold)",
                     fontweight="bold", **({} if len(models) == 4 else {"fontsize": 10}))
        ax.set_xlabel("Fração dos grupos do treino (%)")
        ax.set_xticks([20, 40, 60, 80, 100])
        ax.set_ylim(0, 1.02)
        ax.grid(alpha=0.2)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1-macro na validação (CV, IC 95%)")
    axes[0].legend(loc="lower right", fontsize=9)
    if len(models) == 4:
        fig.suptitle("Curva de aprendizado (marcador vazio: alguma classe com menos de 5 genes no treino em algum fold)", fontsize=11)
    fig.tight_layout()
    _save(fig, output_path)


def plot_overfitting(ov: pd.DataFrame, output_path: Path, models: list[str] = MODEL_NAMES) -> None:
    """F1-macro no treino (folds), na validação (folds) e no teste, por modelo e pipeline."""
    sub = ov[ov["classe"] == "macro"]
    cols = [("f1_treino_cv", "Treino (folds)"), ("f1_validacao_cv", "Validação (CV)"), ("f1_teste", "Teste")]
    fig, axes = plt.subplots(1, len(models), figsize=(17 if len(models) == 4 else 6, 4.2), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, model in zip(axes, models):
        for i, key in enumerate(("a", "b")):
            r = sub[(sub["modelo"] == model) & (sub["pipeline"] == key)].iloc[0]
            for j, (col, label) in enumerate(cols):
                x = i * 4 + j
                ax.bar(x, r[col], color=(COLOR_A if key == "a" else COLOR_B), alpha=[1.0, 0.7, 0.45][j], edgecolor="0.3")
                ax.text(x, r[col] + 0.01, num(r[col], 2), ha="center", fontsize=7 if len(models) == 4 else 9)
        ax.set_xticks([0, 1, 2, 4, 5, 6], ["treino", "valid.", "teste"] * 2, fontsize=8)
        ax.text(1, -0.14, "Pipeline A", ha="center", transform=ax.get_xaxis_transform(), fontsize=9)
        ax.text(5, -0.14, "Pipeline B", ha="center", transform=ax.get_xaxis_transform(), fontsize=9)
        ax.set_ylim(0, 1.08)
        ax.set_title(MODEL_LABELS[model] if len(models) == 4 else f"F1-macro no treino, na validação e no teste — {MODEL_LABELS[model]}",
                     fontweight="bold", **({} if len(models) == 4 else {"fontsize": 10}))
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("F1-macro")
    fig.tight_layout()
    _save(fig, output_path)


def _available(name: str, figure: str, module: str) -> bool:
    """
    Confere se o arquivo de entrada de uma figura existe; se não, avisa em vez de pular em
    silêncio. Algumas figuras dependem de etapas que rodam depois desta (por exemplo,
    ``report_generator`` e ``threshold_curve``); numa execução do zero, rode a
    visualização de novo no fim.
    """
    if (RESULTS_DIR / name).exists():
        return True
    logger.warning("Figura %s não gerada: falta results/%s (rode python -m src.%s e depois a visualização de novo).",
                   figure, name, module)
    return False


def generate_all_figures() -> None:
    """
    Figuras do estilo do artigo. As que têm vários painéis são geradas como antes e também
    com um gráfico por arquivo, na subpasta ``individuais/`` de cada pasta, para entrar
    direto no texto.
    """
    data = load_experiment_dataset()
    plot_class_distribution(data[TARGET].value_counts(), PLOTS_DIR / "eda" / "class_distribution_experimento")
    ma = pd.read_csv(RESULTS_DIR / "metrics_pipeline_a.csv")
    mb = pd.read_csv(RESULTS_DIR / "metrics_pipeline_b.csv")
    comp = PLOTS_DIR / "comparacao_pipelines"
    plot_f1_comparison_bar(ma, mb, comp / "f1_macro_comparison")
    plot_f1_delta_percent(ma, mb, comp / "f1_delta_percentual")
    plot_roc_pr_curves(PLOTS_DIR / "roc_pr" / "roc_pr_all_models")
    for model in MODEL_NAMES:
        plot_pr_curve(model, PLOTS_DIR / "roc_pr" / INDIVIDUAL / f"pr_{model}")
        plot_roc_curve(model, PLOTS_DIR / "roc_pr" / INDIVIDUAL / f"roc_{model}")
    if _available("testes_estatisticos.csv", "ganho_cv", "statistical_tests"):
        plot_cv_gain(pd.read_csv(RESULTS_DIR / "testes_estatisticos.csv"), comp / "ganho_cv")
    if _available("bootstrap_teste.csv", "ganho_por_classe", "bootstrap_test"):
        boot = pd.read_csv(RESULTS_DIR / "bootstrap_teste.csv")
        plot_gain_per_class(boot, comp / "ganho_por_classe")
        for metric in ("precisao", "revocacao", "f1"):
            plot_gain_per_class(boot, comp / INDIVIDUAL / f"ganho_por_classe_{metric}", only=metric)
    if _available("ganho_por_classe_cv.csv", "ganho_por_classe_cv", "data_characterization"):
        gain = pd.read_csv(RESULTS_DIR / "ganho_por_classe_cv.csv")
        plot_cv_gain_per_class(gain, comp / "ganho_por_classe_cv")
        for scen in ("anotacao_curada", "gene_novo"):
            plot_cv_gain_per_class(gain, comp / INDIVIDUAL / f"ganho_por_classe_cv_{scen}", only=scen)
    if _available("tabela_cenarios.csv", "cenarios_gene_novo", "report_generator"):
        plot_scenarios(pd.read_csv(RESULTS_DIR / "tabela_cenarios.csv"), comp / "cenarios_gene_novo")
    classes = json.loads((RESULTS_DIR / "classes.json").read_text(encoding="utf-8"))  # ordem das colunas das matrizes salvas
    ordered = _in_order(classes)  # ordem das figuras
    for model in MODEL_NAMES:
        for key in ("a", "b"):
            d = _load_pred(key, model)
            plot_confusion_matrix(
                d["confusion_matrix"],
                classes,
                f"{MODEL_LABELS[model]} — Pipeline {key.upper()}",
                PLOTS_DIR / "confusion_matrices" / f"cm_{model}_{key}",
            )
    if _available("curva_limiar.csv", "curva_limiar", "threshold_curve"):
        curve = pd.read_csv(RESULTS_DIR / "curva_limiar.csv")
        parts = pd.read_csv(RESULTS_DIR / "curva_limiar_particoes.csv", dtype={"cenario": str})
        plot_threshold_gain_summary(curve, parts, comp / "curva_limiar")
        plot_threshold_f1_summary(curve, parts, comp / "curva_limiar_f1")
        for model in MODEL_NAMES:
            plot_threshold_curve(curve, parts, comp / INDIVIDUAL / f"curva_limiar_f1_{model}", models=[model], rows=("f1",))
            plot_threshold_curve(curve, parts, comp / INDIVIDUAL / f"curva_limiar_ganho_{model}", models=[model], rows=("ganho",))
    interp = PLOTS_DIR / "interpretabilidade"
    if _available("importancia_grupos.csv", "importancia_grupos", "interpretability"):
        groups = pd.read_csv(RESULTS_DIR / "importancia_grupos.csv")
        feats = pd.read_csv(RESULTS_DIR / "importancia_atributos.csv")
        plot_group_importance(groups, interp / "importancia_grupos")
        plot_group_importance_per_class(groups, ordered, interp / "importancia_por_classe")
        plot_top_features(feats, interp / "top_atributos")
        for model in MODEL_NAMES:
            plot_group_importance(groups, interp / INDIVIDUAL / f"importancia_grupos_{model}", models=[model])
            plot_group_importance_per_class(groups, ordered, interp / INDIVIDUAL / f"importancia_por_classe_{model}", models=[model])
            for ctx in ("a", "b"):
                plot_top_features(feats, interp / INDIVIDUAL / f"top_atributos_{model}_{ctx}", models=[model], contexts=(ctx,))
    if _available("shap_por_classe.csv", "shap_*_b", "interpretability"):
        shap = pd.read_csv(RESULTS_DIR / "shap_por_classe.csv")
        for model in ("xgboost", "random_forest"):
            plot_shap_per_class(shap, ordered, interp / f"shap_{model}_b", model=model)
            for c in ordered:
                plot_shap_one_class(shap, c, interp / INDIVIDUAL / f"shap_{model}_b_{_short(c).replace(' ', '_')}", model=model)
    if _available("curva_aprendizado.csv", "curva_aprendizado e overfitting", "learning_curve"):
        lc = pd.read_csv(RESULTS_DIR / "curva_aprendizado.csv")
        ov = pd.read_csv(RESULTS_DIR / "overfitting.csv")
        plot_learning_curve(lc, comp / "curva_aprendizado")
        plot_overfitting(ov, comp / "overfitting")
        for model in MODEL_NAMES:
            plot_learning_curve(lc, comp / INDIVIDUAL / f"curva_aprendizado_{model}", models=[model])
            plot_overfitting(ov, comp / INDIVIDUAL / f"overfitting_{model}", models=[model])
    from src.visualization_analyses import generate_analysis_figures, write_figure_catalog

    generate_analysis_figures()
    write_figure_catalog()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    generate_all_figures()
