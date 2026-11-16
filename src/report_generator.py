"""
report_generator.py — Tabelas comparativas, tabelas LaTeX e texto de conclusões.

Saídas:
    results/tabela_comparativa.csv                 — F1 A vs B no teste, com IC 95% (Tabela 5.3)
    reports/tabelas_latex/tabela_pipeline_a.tex    — Tabela 5.1
    reports/tabelas_latex/tabela_pipeline_b.tex    — Tabela 5.2
    reports/tabelas_latex/tabela_ganho.tex         — Tabela 5.3 (B − A com IC 95% do bootstrap)
    reports/tabelas_latex/tabela_roc_auc.tex       — Tabela 5.4 (ROC-AUC e PR-AUC lado a lado)
    reports/tabelas_latex/tabela_cv.tex            — Tabela 5.5 (CV 10 × 5, IC corrigido)
    reports/tabelas_latex/tabela_teste_t.tex       — Tabela 5.6 (teste t corrigido + Holm)
    reports/tabelas_latex/tabela_por_classe.tex    — precisão, revocação e F1 por classe (teste)
    results/conclusoes_experimento.txt

Intervalos (TCC2, ponto 5): no teste, IC 95% por bootstrap de grupos
(``src.bootstrap_test``); na validação cruzada, IC 95% com o erro padrão corrigido
de Nadeau e Bengio (``src.statistical_tests``). A decisão sobre H0 usa o p corrigido
por Holm.

Uso (depois de ``src.statistical_tests`` e ``src.bootstrap_test``):
    python -m src.report_generator
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import pandas as pd

from src.bootstrap_test import N_BOOTSTRAP
from src.experiment_config import ALPHA_STAT, MODEL_NAMES, N_CV_REPEATS, N_CV_SPLITS, REPORTS_DIR, RESULTS_DIR
from src.pipeline_builder import MODEL_LABELS

logger = logging.getLogger(__name__)
LATEX_DIR = REPORTS_DIR / "tabelas_latex"


def _br(v: float, casas: int = 4) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def _ic(low: float, high: float, casas: int = 3) -> str:
    return f"[{_br(low, casas)}; {_br(high, casas)}]"


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


def load_bootstrap() -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / "bootstrap_teste.csv").set_index(["modelo", "comparacao", "metrica", "classe"])


def generate_comparative_table(ma: pd.DataFrame, mb: pd.DataFrame, boot: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for m in MODEL_NAMES:
        a, b = ma.loc[(m, "f1_macro"), "valor_teste"], mb.loc[(m, "f1_macro"), "valor_teste"]
        ic = {k: boot.loc[(m, k, "f1_macro", "macro")] for k in ("a", "b", "b_menos_a")}
        rows.append(
            {
                "modelo": m,
                "f1_macro_a": a,
                "f1_macro_a_ic95_inf": ic["a"]["ic95_inf"],
                "f1_macro_a_ic95_sup": ic["a"]["ic95_sup"],
                "f1_macro_b": b,
                "f1_macro_b_ic95_inf": ic["b"]["ic95_inf"],
                "f1_macro_b_ic95_sup": ic["b"]["ic95_sup"],
                "delta_absoluto": b - a,
                "delta_ic95_inf": ic["b_menos_a"]["ic95_inf"],
                "delta_ic95_sup": ic["b_menos_a"]["ic95_sup"],
                "delta_percentual": (b - a) / a * 100,
                "roc_auc_a": ma.loc[(m, "roc_auc_ovr"), "valor_teste"],
                "roc_auc_b": mb.loc[(m, "roc_auc_ovr"), "valor_teste"],
                "pr_auc_a": ma.loc[(m, "pr_auc_macro"), "valor_teste"],
                "pr_auc_b": mb.loc[(m, "pr_auc_macro"), "valor_teste"],
            }
        )
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tabela_comparativa.csv", index=False)
    return df


def generate_latex_tables(ma: pd.DataFrame, mb: pd.DataFrame, comp: pd.DataFrame, tests: pd.DataFrame) -> None:
    n_est = int(tests["n_estimativas"].iloc[0])
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
            f"{_br(r.f1_macro_a)} {_ic(r.f1_macro_a_ic95_inf, r.f1_macro_a_ic95_sup)}",
            f"{_br(r.f1_macro_b)} {_ic(r.f1_macro_b_ic95_inf, r.f1_macro_b_ic95_sup)}",
            f"{r.delta_absoluto:+.4f}".replace(".", ",") + f" {_ic(r.delta_ic95_inf, r.delta_ic95_sup)}",
            f"{r.delta_percentual:+.2f}".replace(".", ",") + "\\%",
        ]
        for r in comp.itertuples()
    ]
    _write(
        "tabela_ganho.tex",
        _latex_table(
            ["Modelo", "F1 macro (A)", "F1 macro (B)", r"$\Delta$ Absoluto", r"$\Delta$ Percentual"],
            rows,
            "F1-macro no conjunto de teste e ganho do Pipeline B, com intervalo de 95\\% por bootstrap "
            f"de grupos ({N_BOOTSTRAP:_} reamostragens, estratificado por classe, pareado entre A e B).".replace("_", "."),
            "tab:ganho",
        ),
    )
    rows = [
        [
            MODEL_LABELS[r.modelo],
            _br(r.roc_auc_a),
            _br(r.pr_auc_a),
            _br(r.roc_auc_b),
            _br(r.pr_auc_b),
        ]
        for r in comp.itertuples()
    ]
    _write(
        "tabela_roc_auc.tex",
        _latex_table(
            ["Modelo", "ROC-AUC (A)", "PR-AUC (A)", "ROC-AUC (B)", "PR-AUC (B)"],
            rows,
            "ROC-AUC e PR-AUC (precisão média) macro, uma classe contra as outras, no conjunto de teste.",
            "tab:roc_auc",
        ),
    )
    t = tests.set_index("modelo")
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
            f"F1-score macro (média $\\pm$ desvio padrão) na validação cruzada por grupos repetida "
            f"({n_est} estimativas: {N_CV_REPEATS} repetições de {N_CV_SPLITS} folds).",
            "tab:cv",
        ),
    )
    rows = [
        [
            MODEL_LABELS[n],
            f"{t.loc[n, 'diferenca_media']:+.4f}".replace(".", ","),
            _ic(t.loc[n, "ic95_inf"], t.loc[n, "ic95_sup"]),
            _br(t.loc[n, "t"], 2),
            f"{t.loc[n, 'p_value']:.2e}".replace(".", ","),
            f"{t.loc[n, 'p_holm']:.2e}".replace(".", ","),
            "Rejeita $H_0$" if t.loc[n, "decisao"] == "rejeita_H0" else "Não rejeita $H_0$",
        ]
        for n in MODEL_NAMES
    ]
    _write(
        "tabela_teste_t.tex",
        _latex_table(
            ["Modelo", r"$\bar{d}$ (B $-$ A)", "IC 95\\%", "$t$", "valor-$p$", "$p$ (Holm)", "Decisão"],
            rows,
            f"Teste t corrigido para validação cruzada repetida (Nadeau e Bengio), F1-macro, B vs. A, "
            f"{n_est} estimativas pareadas, {n_est - 1} graus de liberdade; correção de Holm sobre "
            f"{int(t['testes_na_familia'].iloc[0])} comparações; $\\alpha = 0{{,}}05$.",
            "tab:teste_t",
        ),
    )


def generate_per_class_table(per_class: pd.DataFrame) -> None:
    """Precisão, revocação e F1 de cada classe no teste, A e B (ponto 10, decisão 1)."""
    rows = []
    for model in MODEL_NAMES:
        sub = per_class[per_class["modelo"] == model]
        for i, cls in enumerate(sub["classe"].unique()):
            a = sub[(sub["pipeline"] == "a") & (sub["classe"] == cls)].iloc[0]
            b = sub[(sub["pipeline"] == "b") & (sub["classe"] == cls)].iloc[0]
            rows.append(
                [MODEL_LABELS[model] if i == 0 else "", cls.replace("antibiotic ", ""), str(int(a["n"]))]
                + [_br(a[x], 2) for x in ("precisao", "revocacao", "f1")]
                + [_br(b[x], 2) for x in ("precisao", "revocacao", "f1")]
            )
    _write(
        "tabela_por_classe.tex",
        _latex_table(
            ["Modelo", "Mecanismo", "$n$", "P (A)", "R (A)", "F1 (A)", "P (B)", "R (B)", "F1 (B)"],
            rows,
            "Precisão (P), revocação (R) e F1 de cada mecanismo no conjunto de teste. "
            "Intervalos de 95\\% em \\texttt{results/bootstrap\\_teste.csv}.",
            "tab:por_classe",
        ),
    )


def generate_conclusions_text(
    comp: pd.DataFrame, tests: pd.DataFrame, output_path: Path = RESULTS_DIR / "conclusoes_experimento.txt"
) -> str:
    t = tests.set_index("modelo")
    n_est = int(t["n_estimativas"].iloc[0])
    best = comp.loc[comp.f1_macro_b.idxmax()]
    rejected = [MODEL_LABELS[m] for m in MODEL_NAMES if t.loc[m, "decisao"] == "rejeita_H0"]
    not_rej = [MODEL_LABELS[m] for m in MODEL_NAMES if t.loc[m, "decisao"] != "rejeita_H0"]
    lines = [
        "CONCLUSÕES DO EXPERIMENTO — Pipeline A (sequência) vs Pipeline B (sequência + ontologia ARO)",
        "",
        "Conjunto de teste (IC 95% por bootstrap de grupos):",
    ]
    for r in comp.itertuples():
        lines.append(
            f"  - {MODEL_LABELS[r.modelo]}: A = {r.f1_macro_a:.4f} [{r.f1_macro_a_ic95_inf:.4f}; {r.f1_macro_a_ic95_sup:.4f}], "
            f"B = {r.f1_macro_b:.4f} [{r.f1_macro_b_ic95_inf:.4f}; {r.f1_macro_b_ic95_sup:.4f}], "
            f"B − A = {r.delta_absoluto:+.4f} [{r.delta_ic95_inf:+.4f}; {r.delta_ic95_sup:+.4f}]"
        )
    lines += [
        f"Melhor modelo: {MODEL_LABELS[best.modelo]} no Pipeline B (F1-macro = {best.f1_macro_b:.4f}).",
        f"Pipeline B superou o A no F1-macro de teste em {(comp.delta_absoluto > 0).sum()} de {len(comp)} modelos.",
        "",
        f"Teste t corrigido (validação cruzada {N_CV_REPEATS} × {N_CV_SPLITS}, {n_est} estimativas pareadas; Holm sobre "
        f"{int(t['testes_na_familia'].iloc[0])} comparações; alfa = {ALPHA_STAT}):",
    ]
    for m in MODEL_NAMES:
        r = t.loc[m]
        lines.append(
            f"  - {MODEL_LABELS[m]}: B − A = {r.diferenca_media:+.4f} [{r.ic95_inf:+.4f}; {r.ic95_sup:+.4f}], "
            f"t = {r.t:.2f}, p = {r.p_value:.2e}, p_Holm = {r.p_holm:.2e}, "
            f"B > A em {r.estimativas_b_maior}/{n_est} -> "
            + ("rejeita H0" if r.decisao == "rejeita_H0" else "não rejeita H0")
        )
    lines += [
        "",
        f"H0 rejeitada para: {', '.join(rejected) or 'nenhum modelo'}.",
        f"Sem significância estatística: {', '.join(not_rej) or 'nenhum'}.",
        "Família confirmatória provisória: o cenário 'gene novo' (ponto 4) acrescentará 4 testes à correção de Holm.",
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
    comp = generate_comparative_table(ma, mb, load_bootstrap())
    tests = pd.read_csv(RESULTS_DIR / "testes_estatisticos.csv")
    generate_latex_tables(ma, mb, comp, tests)
    generate_per_class_table(pd.read_csv(RESULTS_DIR / "metricas_por_classe_teste.csv"))
    print(generate_conclusions_text(comp, tests))
