"""
report_generator.py — Tabelas comparativas, tabelas LaTeX e texto de conclusões.

Saídas:
    results/tabela_comparativa.csv                 — F1 A vs B (Tabela 5.3)
    reports/tabelas_latex/tabela_pipeline_a.tex    — Tabela 5.1
    reports/tabelas_latex/tabela_pipeline_b.tex    — Tabela 5.2
    reports/tabelas_latex/tabela_ganho.tex         — Tabela 5.3
    reports/tabelas_latex/tabela_roc_auc.tex       — Tabela 5.4
    reports/tabelas_latex/tabela_cv.tex            — Tabela 5.5
    reports/tabelas_latex/tabela_wilcoxon.tex      — Tabela 5.6
    results/conclusoes_experimento.txt

Uso:
    python -m src.report_generator
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.experiment_config import ALPHA_STAT, MODEL_NAMES, REPORTS_DIR, RESULTS_DIR
from src.pipeline_builder import MODEL_LABELS

logger = logging.getLogger(__name__)
LATEX_DIR = REPORTS_DIR / "tabelas_latex"


def _br(v: float, casas: int = 4) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def _latex_table(header: list[str], rows: list[list[str]], caption: str, label: str) -> str:
    cols = "l" + "c" * (len(header) - 1)
    body = "\n".join("    " + " & ".join(r) + r" \\" for r in rows)
    return (
        "\\begin{table}[htbp]\n  \\centering\n"
        f"  \\caption{{{caption}}}\n  \\label{{{label}}}\n"
        f"  \\begin{{tabular}}{{{cols}}}\n    \\toprule\n    " + " & ".join(header) + r" \\" + "\n    \\midrule\n"
        f"{body}\n    \\bottomrule\n  \\end{{tabular}}\n\\end{{table}}\n"
    )


def _write(name: str, text: str) -> None:
    LATEX_DIR.mkdir(parents=True, exist_ok=True)
    (LATEX_DIR / name).write_text(text, encoding="utf-8")
    logger.info("LaTeX: %s", LATEX_DIR / name)


def load_metrics() -> tuple[pd.DataFrame, pd.DataFrame]:
    return (
        pd.read_csv(RESULTS_DIR / "metrics_pipeline_a.csv").set_index(["modelo", "metrica"]),
        pd.read_csv(RESULTS_DIR / "metrics_pipeline_b.csv").set_index(["modelo", "metrica"]),
    )


def generate_comparative_table(ma: pd.DataFrame, mb: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODEL_NAMES:
        a, b = ma.loc[(m, "f1_macro"), "valor_teste"], mb.loc[(m, "f1_macro"), "valor_teste"]
        rows.append(
            {
                "modelo": m,
                "f1_macro_a": a,
                "f1_macro_b": b,
                "delta_absoluto": b - a,
                "delta_percentual": (b - a) / a * 100,
                "roc_auc_a": ma.loc[(m, "roc_auc_ovr"), "valor_teste"],
                "roc_auc_b": mb.loc[(m, "roc_auc_ovr"), "valor_teste"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tabela_comparativa.csv", index=False)
    return df


def generate_latex_tables(ma: pd.DataFrame, mb: pd.DataFrame, comp: pd.DataFrame, wil: pd.DataFrame) -> None:
    n_folds = int(wil["n_samples"].iloc[0])  # número de folds da validação cruzada
    for key, m, cap in (
        ("a", ma, "Desempenho no conjunto de teste no Pipeline A (apenas features de sequência)."),
        ("b", mb, "Desempenho no conjunto de teste no Pipeline B (sequência + ontologia)."),
    ):
        rows = [
            [MODEL_LABELS[n]]
            + [_br(m.loc[(n, x), "valor_teste"]) for x in ("accuracy", "precision_macro", "recall_macro", "f1_macro")]
            for n in MODEL_NAMES
        ]
        _write(
            f"tabela_pipeline_{key}.tex",
            _latex_table(
                ["Modelo", "Acurácia", "Precisão (macro)", "Revocação (macro)", "F1 (macro)"],
                rows,
                cap,
                f"tab:pipeline_{key}",
            ),
        )
    rows = [
        [
            MODEL_LABELS[r.modelo],
            _br(r.f1_macro_a),
            _br(r.f1_macro_b),
            "+" + _br(r.delta_absoluto),
            f"+{_br(r.delta_percentual, 2)}\\%",
        ]
        for r in comp.itertuples()
    ]
    _write(
        "tabela_ganho.tex",
        _latex_table(
            ["Modelo", "F1 macro (A)", "F1 macro (B)", r"$\Delta$ Absoluto", r"$\Delta$ Percentual"],
            rows,
            "Ganho de desempenho do Pipeline B em relação ao Pipeline A.",
            "tab:ganho",
        ),
    )
    rows = [
        [
            MODEL_LABELS[r.modelo],
            _br(r.roc_auc_a),
            _br(r.roc_auc_b),
            f"{r.roc_auc_b - r.roc_auc_a:+.4f}".replace(".", ","),
        ]
        for r in comp.itertuples()
    ]
    _write(
        "tabela_roc_auc.tex",
        _latex_table(
            ["Modelo", "ROC-AUC (A)", "ROC-AUC (B)", r"$\Delta$"],
            rows,
            "ROC-AUC macro (OvR) no conjunto de teste.",
            "tab:roc_auc",
        ),
    )
    rows = [
        [
            MODEL_LABELS[n],
            f"{_br(ma.loc[(n, 'f1_macro'), 'media_cv'])} $\\pm$ {_br(ma.loc[(n, 'f1_macro'), 'std_cv'])}",
            f"{_br(mb.loc[(n, 'f1_macro'), 'media_cv'])} $\\pm$ {_br(mb.loc[(n, 'f1_macro'), 'std_cv'])}",
        ]
        for n in MODEL_NAMES
    ]
    _write(
        "tabela_cv.tex",
        _latex_table(
            ["Modelo", "Pipeline A", "Pipeline B"],
            rows,
            f"F1-score macro (média $\\pm$ desvio padrão) na validação cruzada por grupos de {n_folds} folds.",
            "tab:cv",
        ),
    )
    w = wil[wil.metrica == "f1_macro"].set_index("modelo")
    rows = [
        [
            MODEL_LABELS[n],
            _br(w.loc[n, "statistic"], 1),
            _br(w.loc[n, "p_value"]),
            "Rejeita $H_0$" if w.loc[n, "decision"] == "reject_H0" else "Falha em rejeitar $H_0$",
        ]
        for n in MODEL_NAMES
    ]
    _write(
        "tabela_wilcoxon.tex",
        _latex_table(
            ["Modelo", "Estatística $W$", "valor-$p$", "Decisão"],
            rows,
            f"Resultado do teste de Wilcoxon Signed-Rank (F1-macro, A vs. B, $N = {n_folds}$ folds, "
            f"$\\alpha = 0{{,}}05$; menor $p$ possível com {n_folds} pares $= {_br(2 / 2 ** n_folds)}$).",
            "tab:wilcoxon",
        ),
    )


def generate_conclusions_text(
    comp: pd.DataFrame, wil: pd.DataFrame, output_path: Path = RESULTS_DIR / "conclusoes_experimento.txt"
) -> str:
    w = wil[wil.metrica == "f1_macro"].set_index("modelo")
    n_folds = int(w["n_samples"].iloc[0])
    best = comp.loc[comp.f1_macro_b.idxmax()]
    rejected = [MODEL_LABELS[m] for m in MODEL_NAMES if w.loc[m, "decision"] == "reject_H0"]
    not_rej = [MODEL_LABELS[m] for m in MODEL_NAMES if w.loc[m, "decision"] != "reject_H0"]
    lines = [
        "CONCLUSÕES DO EXPERIMENTO — Pipeline A (sequência) vs Pipeline B (sequência + ontologia ARO)",
        "",
        f"Ganho médio de F1-macro no teste: {comp.delta_percentual.mean():+.2f}% "
        f"(de {comp.delta_percentual.min():+.2f}% a {comp.delta_percentual.max():+.2f}%).",
        f"Melhor modelo: {MODEL_LABELS[best.modelo]} no Pipeline B (F1-macro = {best.f1_macro_b:.4f}).",
        f"Pipeline B superou o A no F1-macro de teste em {(comp.delta_absoluto > 0).sum()} de {len(comp)} modelos.",
        "",
        f"Teste de Wilcoxon ({n_folds} folds, alfa = {ALPHA_STAT}; "
        f"menor p possível com {n_folds} pares = {2 / 2 ** n_folds:.4f}):",
    ]
    for m in MODEL_NAMES:
        r = w.loc[m]
        lines.append(
            f"  - {MODEL_LABELS[m]}: W = {r.statistic:.1f}, p = {r.p_value:.4f}, "
            f"B > A em {r.folds_b_maior}/{n_folds} folds -> "
            + ("rejeita H0" if r.decision == "reject_H0" else "falha em rejeitar H0")
        )
    lines += [
        "",
        f"H0 rejeitada para: {', '.join(rejected) or 'nenhum modelo'}.",
        f"Sem significância estatística: {', '.join(not_rej) or 'nenhum'}.",
    ]
    text = "\n".join(lines) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    ma, mb = load_metrics()
    comp = generate_comparative_table(ma, mb)
    wil = pd.read_csv(RESULTS_DIR / "testes_estatisticos.csv")
    generate_latex_tables(ma, mb, comp, wil)
    print(generate_conclusions_text(comp, wil))
