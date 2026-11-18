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
    results/tabela_cenarios.csv                    — A, B curado, B gene novo e homologia no teste
    results/cenarios_cv.csv                        — diferenças entre os cenários na CV, IC corrigido (descritivo)
    reports/tabelas_latex/tabela_cenarios.tex      — idem, em LaTeX
    reports/tabelas_latex/tabela_ablacao.tex       — ablação das features ontológicas (CV, IC corrigido)
    reports/tabelas_latex/tabela_importancia_grupos.tex — queda do F1 ao embaralhar cada grupo (CV)
    reports/tabelas_latex/tabela_top_atributos_{a,b}.tex — 15 atributos mais importantes por modelo
    reports/tabelas_latex/tabela_dicionario_atributos.tex — todos os atributos e o significado (apêndice; longtable)
    reports/tabelas_latex/tabela_overfitting.tex   — F1 no treino, na validação e no teste
    reports/tabelas_latex/tabela_curva_aprendizado.tex — F1 na validação por fração dos grupos do treino
    reports/tabelas_latex/tabela_{composicao,grupos,familias_teste,ganho_por_classe_cv,recortes_teste}.tex
    reports/tabelas_latex/tabela_sensibilidade_{representacao,balanceamento,ajuste}.tex, tabela_estudo_caso.tex
    results/conclusoes_experimento.txt

Intervalos: no teste, IC 95% por bootstrap de grupos
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
    t = curated_tests(tests)
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
            SCENARIO_LABELS[r.cenario],
            MODEL_LABELS[r.modelo],
            f"{r.diferenca_media:+.4f}".replace(".", ","),
            _ic(r.ic95_inf, r.ic95_sup),
            _br(r.t, 2),
            f"{r.p_value:.2e}".replace(".", ","),
            f"{r.p_holm:.2e}".replace(".", ","),
            "Rejeita $H_0$" if r.decisao == "rejeita_H0" else "Não rejeita $H_0$",
        ]
        for r in tests.itertuples()
    ]
    _write(
        "tabela_teste_t.tex",
        _latex_table(
            ["Cenário", "Modelo", r"$\bar{d}$ (B $-$ A)", "IC 95\\%", "$t$", "valor-$p$", "$p$ (Holm)", "Decisão"],
            rows,
            f"Teste t corrigido para validação cruzada repetida (Nadeau e Bengio), F1-macro, B vs. A, "
            f"{n_est} estimativas pareadas, {n_est - 1} graus de liberdade; correção de Holm sobre "
            f"{int(tests['testes_na_familia'].iloc[0])} comparações (2 cenários $\\times$ 4 modelos); $\\alpha = 0{{,}}05$.",
            "tab:teste_t",
        ),
    )


SCENARIO_LABELS: dict[str, str] = {"anotacao_curada": "Anotação curada", "gene_novo": "Gene novo"}


def curated_tests(tests: pd.DataFrame) -> pd.DataFrame:
    """Linhas do cenário "anotação curada", indexadas pelo modelo (tabelas do experimento A × B)."""
    return tests[tests["cenario"] == "anotacao_curada"].set_index("modelo")


def generate_scenario_table(boot: pd.DataFrame) -> pd.DataFrame:
    """
    F1-macro no teste (IC 95% por bootstrap de grupos) do A, do B nos dois cenários e da
    referência por homologia, e as diferenças pareadas. Média da CV ao lado.
    """
    keys = ("a", "b", "b_gene_novo", "b_gene_novo_zeros", "b_gene_novo_piso40")
    cv = {key: pd.read_csv(RESULTS_DIR / f"metrics_pipeline_{key}.csv").set_index(["modelo", "metrica"]) for key in keys}
    cv_hom = {
        k: pd.read_csv(RESULTS_DIR / f"metrics_pipeline_referencia_{k}.csv").set_index(["modelo", "metrica"])
        for k in ("homologia", "homologia_piso40")
    }
    rows = []
    for m in MODEL_NAMES:
        row = {"modelo": m}
        for key in keys:
            r = boot.loc[(m, key, "f1_macro", "macro")]
            row |= {f"f1_teste_{key}": r["valor"], f"ic95_inf_{key}": r["ic95_inf"], f"ic95_sup_{key}": r["ic95_sup"],
                    f"f1_cv_{key}": cv[key].loc[(m, "f1_macro"), "media_cv"]}
        for k in ("homologia", "homologia_piso40"):
            hom = boot.loc[("homologia", k, "f1_macro", "macro")]
            row |= {f"f1_teste_{k}": hom["valor"], f"ic95_inf_{k}": hom["ic95_inf"], f"ic95_sup_{k}": hom["ic95_sup"],
                    f"f1_cv_{k}": cv_hom[k].loc[("homologia", "f1_macro"), "media_cv"]}
        for comp in ("b_gene_novo_menos_a", "b_gene_novo_menos_b", "b_gene_novo_zeros_menos_b_gene_novo",
                     "homologia_menos_b", "homologia_menos_b_gene_novo", "b_gene_novo_piso40_menos_b_gene_novo"):
            r = boot.loc[(m, comp, "f1_macro", "macro")]
            row |= {comp: r["valor"], f"{comp}_ic95_inf": r["ic95_inf"], f"{comp}_ic95_sup": r["ic95_sup"]}
        rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / "tabela_cenarios.csv", index=False)
    body = [
        [MODEL_LABELS[r["modelo"]]]
        + [f"{_br(r[f'f1_teste_{k}'])} {_ic(r[f'ic95_inf_{k}'], r[f'ic95_sup_{k}'])}" for k in ("a", "b", "b_gene_novo", "homologia")]
        for r in rows
    ]
    _write(
        "tabela_cenarios.tex",
        _latex_table(
            ["Modelo", "A", "B (anotação curada)", "B (gene novo)", "Homologia (sem ML)"],
            body,
            "F1-macro no conjunto de teste nos cenários do Pipeline B e na referência por homologia (mecanismo do "
            "gene mais parecido do treino, na busca do MMseqs2 sem piso de identidade), com intervalo de 95\\% por "
            "bootstrap de grupos. A referência por homologia "
            "não depende do modelo.",
            "tab:cenarios",
        ),
    )
    return df


CV_COMPARISONS: list[tuple[str, str, str]] = [
    # (nome, minuendo, subtraendo) — diferenças pareadas por fold, descritivas (sem teste).
    ("b_gene_novo_menos_b", "b_gene_novo", "b"),
    ("homologia_menos_b", "homologia", "b"),
    ("homologia_menos_b_gene_novo", "homologia", "b_gene_novo"),
    ("b_gene_novo_zeros_menos_b_gene_novo", "b_gene_novo_zeros", "b_gene_novo"),
    ("b_gene_novo_piso40_menos_b_gene_novo", "b_gene_novo_piso40", "b_gene_novo"),
    ("homologia_piso40_menos_homologia", "homologia_piso40", "homologia"),
]


def generate_scenario_cv_table(cv_folds: pd.DataFrame) -> pd.DataFrame:
    """
    Diferenças entre os cenários de dependência da anotação na validação cruzada (mesmos 50 folds), com o
    intervalo de 95% corrigido de Nadeau e Bengio (o mesmo do teste t), sem teste: são
    comparações descritivas. A referência por homologia, que não
    tem modelo, é pareada com cada modelo pelo fold.
    """
    from src.statistical_tests import corrected_interval, test_train_ratio

    ratio = test_train_ratio(cv_folds)
    f1 = cv_folds.pivot_table(index=["repeticao", "fold"], columns=["pipeline", "modelo"], values="f1_macro")
    rows = []
    for m in MODEL_NAMES:
        for name, minuend, subtrahend in CV_COMPARISONS:
            col = lambda key: f1[(key, "homologia" if key.startswith("homologia") else m)]
            d = (col(minuend) - col(subtrahend)).to_numpy()
            low, high = corrected_interval(d, ratio)
            rows.append({"modelo": m, "comparacao": name, "media": d.mean(), "ic95_inf": low, "ic95_sup": high,
                         "folds_minuendo_maior": int((d > 0).sum()), "folds_subtraendo_maior": int((d < 0).sum())})
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "cenarios_cv.csv", index=False)
    return out


def generate_ablation_table() -> pd.DataFrame | None:
    """Ablação por grupo de features ontológicas (``src.leakage_check``), média da CV com IC corrigido."""
    path = RESULTS_DIR / "ablacao_ontologia.csv"
    if not path.exists():
        logger.warning("Sem %s: rode python -m src.leakage_check.", path)
        return None
    abl = pd.read_csv(path)
    configs = list(dict.fromkeys(abl["config"]))
    body = []
    for scen in ("anotacao_curada", "gene_novo"):
        for i, cfg in enumerate(configs):
            sub = abl[(abl["config"] == cfg) & (abl["cenario"] == scen)].set_index("modelo")
            body.append(
                [SCENARIO_LABELS[scen] if i == 0 else "", cfg]
                + [f"{_br(sub.loc[m, 'f1_cv_media'], 3)} {_ic(sub.loc[m, 'f1_cv_ic95_inf'], sub.loc[m, 'f1_cv_ic95_sup'])}"
                   for m in MODEL_NAMES]
            )
    _write(
        "tabela_ablacao.tex",
        _latex_table(
            ["Cenário", "Configuração"] + [MODEL_LABELS[m] for m in MODEL_NAMES],
            body,
            f"Ablação das features ontológicas: F1-macro médio na validação cruzada por grupos "
            f"({N_CV_REPEATS} $\\times$ {N_CV_SPLITS}) com intervalo de 95\\% corrigido (Nadeau e Bengio). "
            "Estruturais: profundidade, ancestrais e filhos na ARO; classes: 38 classes de antibiótico.",
            "tab:ablacao",
        ),
    )
    return abl


GROUP_NAMES: dict[str, str] = {
    "fisico_quimicas": "Físico-químicas (4)", "aminoacidos": "Aminoácidos (20)", "pares_de_grupos": "Pares de grupos (49)",
    "estruturais": "ARO: estruturais (3)", "classes_de_antibiotico": "ARO: classes de antibiótico (38)",
}
CONTEXT_NAMES: dict[str, str] = {"a": "Pipeline A", "b": "Pipeline B (curado)", "b_gene_novo": "Pipeline B (gene novo)"}


def _tex(text: str) -> str:
    return str(text).replace("_", "\\_").replace("%", "\\%").replace("&", "\\&")


def generate_interpretation_tables() -> None:
    """Tabelas de interpretação (importância por grupo, top-15 e dicionário), se os resultados existirem."""
    path = RESULTS_DIR / "importancia_grupos.csv"
    if not path.exists():
        logger.warning("Sem %s: rode python -m src.interpretability.", path)
        return
    g = pd.read_csv(path)
    body = []
    for ctx, ctx_label in CONTEXT_NAMES.items():
        sub = g[g["contexto"] == ctx]
        for i, grp in enumerate([x for x in GROUP_NAMES if x in set(sub["conjunto"])]):
            row = sub[sub["conjunto"] == grp].set_index("modelo")
            body.append([ctx_label if i == 0 else "", GROUP_NAMES[grp]] + [
                f"{_br(row.loc[m, 'cv_media_macro'], 3)} {_ic(row.loc[m, 'cv_ic95_inf_macro'], row.loc[m, 'cv_ic95_sup_macro'])}"
                for m in MODEL_NAMES])
    _write("tabela_importancia_grupos.tex", _latex_table(
        ["Contexto", "Grupo de atributos"] + [MODEL_LABELS[m] for m in MODEL_NAMES], body,
        f"Importância por permutação de cada grupo de atributos: queda do F1-macro ao embaralhar o grupo inteiro, "
        f"média nos {N_CV_REPEATS * N_CV_SPLITS} folds da validação cruzada, com IC 95\\% corrigido (Nadeau e Bengio). "
        "No cenário gene novo, entre os genes de validação com vizinho no treino.",
        "tab:importancia_grupos"))
    f = pd.read_csv(RESULTS_DIR / "importancia_atributos.csv")
    for ctx in ("a", "b"):
        body = []
        for pos in range(1, 16):
            cells = []
            for m in MODEL_NAMES:
                r = f[(f["contexto"] == ctx) & (f["modelo"] == m) & (f["posicao_cv"] == pos)].iloc[0]
                mark = "" if r["consistente"] else "$^\\dagger$"
                cells.append(f"\\texttt{{{_tex(r['conjunto'])}}} ({_br(r['cv_media_macro'], 3)}){mark}")
            body.append([str(pos)] + cells)
        _write(f"tabela_top_atributos_{ctx}.tex", _latex_table(
            ["\\#"] + [MODEL_LABELS[m] for m in MODEL_NAMES], body,
            f"Os 15 atributos mais importantes no Pipeline {ctx.upper()} (queda média do F1-macro ao embaralhar o atributo, "
            "50 folds da validação cruzada). $^\\dagger$: IC 95\\% corrigido inclui zero (não consistente).",
            f"tab:top_atributos_{ctx}"))
    d = pd.read_csv(RESULTS_DIR / "dicionario_atributos.csv")
    lines = ["\\begin{longtable}{p{4.2cm}p{2.6cm}p{8cm}}",
             "  \\caption{Atributos usados pelos pipelines e o que cada um significa.}\\label{tab:dicionario_atributos} \\\\",
             "  \\toprule", "  Atributo & Grupo & Significado \\\\", "  \\midrule", "  \\endhead"]
    for r in d.itertuples():
        lines.append(f"  \\texttt{{{_tex(r.atributo)}}} & {GROUP_NAMES[r.grupo]} & {_tex(r.significado)} \\\\")
    lines += ["  \\bottomrule", "\\end{longtable}"]
    _write("tabela_dicionario_atributos.tex", "% requer \\usepackage{longtable} e \\usepackage{booktabs}\n" + "\n".join(lines) + "\n")


def generate_overfitting_tables() -> None:
    """Tabelas de overfitting e da curva de aprendizado, se os resultados existirem."""
    path = RESULTS_DIR / "overfitting.csv"
    if not path.exists():
        logger.warning("Sem %s: rode python -m src.learning_curve.", path)
        return
    ov = pd.read_csv(path)
    macro = ov[ov["classe"] == "macro"]
    body = []
    for m in MODEL_NAMES:
        for key in ("a", "b"):
            r = macro[(macro["modelo"] == m) & (macro["pipeline"] == key)].iloc[0]
            body.append([MODEL_LABELS[m] if key == "a" else "", key.upper(), _br(r["f1_treino_cv"], 3),
                         _br(r["f1_validacao_cv"], 3), _br(r["f1_teste"], 3), _br(r["queda_treino_validacao"], 3)])
    _write("tabela_overfitting.tex", _latex_table(
        ["Modelo", "Pipeline", "Treino (folds)", "Validação (CV)", "Teste", "Queda treino $\\to$ validação"], body,
        "F1-macro no treino (o modelo avaliado nos próprios genes de ajuste, média dos 50 folds), na validação cruzada "
        "e no teste. Por classe: \\texttt{results/overfitting.csv}.",
        "tab:overfitting"))
    curve = pd.read_csv(RESULTS_DIR / "curva_aprendizado.csv")
    fracs = sorted(curve["fracao"].unique())
    body = []
    for m in MODEL_NAMES:
        for key in ("a", "b"):
            sub = curve[(curve["modelo"] == m) & (curve["pipeline"] == key)].set_index("fracao")
            body.append([MODEL_LABELS[m] if key == "a" else "", key.upper()] + [
                _br(sub.loc[f, "media_macro"], 3) + ("" if sub.loc[f, "atende_criterio"] else "$^\\dagger$") for f in fracs])
    _write("tabela_curva_aprendizado.tex", _latex_table(
        ["Modelo", "Pipeline"] + [f"{int(round(100 * f))}\\%" for f in fracs], body,
        "Curva de aprendizado: F1-macro médio na validação cruzada (50 folds) com o modelo treinado numa fração "
        "dos grupos de similaridade de cada classe do conjunto de ajuste (grupos inteiros). $^\\dagger$: alguma classe "
        "com menos de 5 genes no treino em algum fold. Intervalos em \\texttt{results/curva\\_aprendizado.csv}.",
        "tab:curva_aprendizado"))


def generate_per_class_table(per_class: pd.DataFrame) -> None:
    """Precisão, revocação e F1 de cada classe no teste, A e B."""
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
    comp: pd.DataFrame,
    tests: pd.DataFrame,
    output_path: Path = RESULTS_DIR / "conclusoes_experimento.txt",
    scenarios: pd.DataFrame | None = None,
) -> str:
    t = curated_tests(tests)
    n_est = int(t["n_estimativas"].iloc[0])
    best = comp.loc[comp.f1_macro_b.idxmax()]
    def label(r):
        return f"{MODEL_LABELS[r.modelo]} ({SCENARIO_LABELS[r.cenario].lower()})"

    rejected = [label(r) for r in tests.itertuples() if r.decisao == "rejeita_H0"]
    not_rej = [label(r) for r in tests.itertuples() if r.decisao != "rejeita_H0"]
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
        f"{int(t['testes_na_familia'].iloc[0])} comparações: 2 cenários × 4 modelos; alfa = {ALPHA_STAT}):",
    ]
    for r in tests.itertuples():
        lines.append(
            f"  - {SCENARIO_LABELS[r.cenario]}, {MODEL_LABELS[r.modelo]}: B − A = {r.diferenca_media:+.4f} [{r.ic95_inf:+.4f}; {r.ic95_sup:+.4f}], "
            f"t = {r.t:.2f}, p = {r.p_value:.2e}, p_Holm = {r.p_holm:.2e}, "
            f"B > A em {r.estimativas_b_maior}/{n_est} -> "
            + ("rejeita H0" if r.decisao == "rejeita_H0" else "não rejeita H0")
        )
    lines += [
        "",
        f"H0 rejeitada para: {', '.join(rejected) or 'nenhum modelo'}.",
        f"Sem significância estatística: {', '.join(not_rej) or 'nenhum'}.",
    ]
    if scenarios is not None:
        lines += ["", "Cenários do Pipeline B e referência por homologia (teste, IC 95% por bootstrap de grupos):"]
        for r in scenarios.itertuples():
            lines.append(
                f"  - {MODEL_LABELS[r.modelo]}: B curado = {r.f1_teste_b:.4f}, B gene novo = {r.f1_teste_b_gene_novo:.4f} "
                f"(gene novo − A = {r.b_gene_novo_menos_a:+.4f} [{r.b_gene_novo_menos_a_ic95_inf:+.4f}; "
                f"{r.b_gene_novo_menos_a_ic95_sup:+.4f}]), homologia = {r.f1_teste_homologia:.4f} "
                f"(homologia − B gene novo = {r.homologia_menos_b_gene_novo:+.4f} "
                f"[{r.homologia_menos_b_gene_novo_ic95_inf:+.4f}; {r.homologia_menos_b_gene_novo_ic95_sup:+.4f}])"
            )
    text = "\n".join(lines) + "\n"
    output_path.write_text(text, encoding="utf-8")
    return text


# ── Tabelas de caracterização e de sensibilidade ─────────────────────────────────
# Cada tabela formata um arquivo de results/ já gerado e justificado pelo módulo de
# origem (ver results/caracterizacao_metadados.json para as do data_characterization).
SHORT = lambda c: str(c).replace("antibiotic ", "").replace(" to antibiotic", "")
DESTINATIONS = {"experimento": "experimento", "estudo_de_caso": "estudo de caso", "multirrotulo": "fora (multirrótulo)"}


def _exists(name: str, module: str) -> bool:
    if (RESULTS_DIR / name).exists():
        return True
    logger.warning("Tabela não gerada: falta results/%s (rode python -m src.%s).", name, module)
    return False


def generate_extra_tables() -> None:
    """
    Tabelas que faltavam para o texto: composição do dataset (nova Tabela 4.1), grupos
    por limiar, famílias no treino e no teste (R2), ganho por classe na CV (R19), recortes
    do teste, as três sensibilidades e o estudo de caso. Só formatam arquivos de results/.
    """
    if _exists("composicao_dataset.csv", "train_evaluate"):
        c = pd.read_csv(RESULTS_DIR / "composicao_dataset.csv")
        rows = [[_tex(r.mecanismo), f"{int(r.genes):_}".replace("_", "."), DESTINATIONS.get(r.destino, r.destino)] for r in c.itertuples()]
        rows.append(["Total", f"{int(c.genes.sum()):_}".replace("_", "."), ""])
        _write("tabela_composicao.tex", _latex_table(["Mecanismo (rótulo do CARD)", "Genes", "Destino"], rows,
               "Mecanismos de resistência do dataset integrado (CARD 4.0.1, modelo homólogo) e o destino de cada um. "
               "Entram no experimento os mecanismos com genes suficientes para cerca de 5 genes por fold.", "tab:composicao"))
    if _exists("grupos_resumo.csv", "homology_groups"):
        g = pd.read_csv(RESULTS_DIR / "grupos_resumo.csv")
        g = g[~g["mecanismo"].astype(str).str.startswith("(")]
        mechs = list(dict.fromkeys(g["mecanismo"]))
        rows = []
        for t, sub in g.groupby("limiar"):
            sub = sub.set_index("mecanismo")
            rows.append([f"{int(t)}\\%"] + [f"{int(sub.loc[m, 'grupos'])} ({_br(sub.loc[m, 'fracao_maior_grupo'], 2)})" for m in mechs])
        _write("tabela_grupos.tex", _latex_table(["Limiar"] + [SHORT(m) for m in mechs], rows,
               "Grupos de similaridade por limiar de identidade (cobertura $\\geq$ 0,5; componentes conexos) e, entre "
               "parênteses, a fração da classe no maior grupo. A divisão principal usa 70\\%.", "tab:grupos"))
    if _exists("familias_treino_teste.csv", "data_characterization"):
        f = pd.read_csv(RESULTS_DIR / "familias_treino_teste.csv")
        rows = [[SHORT(r.mecanismo), str(int(r.genes_teste)), f"{int(r.genes_teste_familia_no_treino)} ({_br(100 * r.fracao_familia_no_treino, 0)}\\%)",
                 f"{int(r.familias_no_teste_ausentes_do_treino)} de {int(r.familias_no_teste)}"] for r in f.itertuples()]
        _write("tabela_familias_teste.tex", _latex_table(["Mecanismo", "Genes no teste", "Família com genes no treino", "Famílias do teste ausentes do treino"], rows,
               "Genes de teste cuja família AMR também tem genes no treino. A divisão por grupos separa sequências parecidas, "
               "não famílias: nas classes menores, o teste mede variantes novas de famílias conhecidas.", "tab:familias_teste"))
    if _exists("ganho_por_classe_cv.csv", "data_characterization"):
        gc = pd.read_csv(RESULTS_DIR / "ganho_por_classe_cv.csv")
        gc = gc[gc["cenario"] == "anotacao_curada"]
        rows = []
        for c, sub in gc.groupby("classe"):
            sub = sub.set_index("modelo")
            rows.append([SHORT(c)] + [f"{_br(sub.loc[m, 'dif_cv'], 2)} {_ic(sub.loc[m, 'ic95_inf'], sub.loc[m, 'ic95_sup'], 2)}"
                                      + ("" if sub.loc[m, "ic_acima_de_zero"] else "$^\\dagger$") for m in MODEL_NAMES])
        _write("tabela_ganho_por_classe_cv.tex", _latex_table(["Mecanismo"] + [MODEL_LABELS[m] for m in MODEL_NAMES], rows,
               f"Ganho B $-$ A no F1 de cada mecanismo na validação cruzada ({N_CV_REPEATS} $\\times$ {N_CV_SPLITS}, folds pareados), "
               "com IC 95\\% corrigido (Nadeau e Bengio); descritivo, sem correção para comparações múltiplas. "
               "$^\\dagger$: IC inclui zero.", "tab:ganho_por_classe_cv"))
    if _exists("recortes_teste.csv", "data_characterization"):
        rc = pd.read_csv(RESULTS_DIR / "recortes_teste.csv")
        names = {"com_parente_no_treino": "Com parente no treino", "distantes_sem_parente_40": "Distantes (sem parente $\\geq$ 40\\%)",
                 "sem_nenhum_parente": "Sem nenhum parente"}
        rows = []
        for r_name, sub in rc.groupby("recorte", sort=False):
            acc = lambda k: [sub[(sub.pipeline == k) & (sub.modelo == m)].acuracia.iloc[0] for m in MODEL_NAMES]
            hom = sub[sub.pipeline == "homologia"].acuracia.iloc[0]
            rows.append([f"{names[r_name]} ({int(sub.genes.iloc[0])})", _br(hom, 2),
                         f"{_br(min(acc('a')), 2)}--{_br(max(acc('a')), 2)}", f"{_br(min(acc('b_gene_novo')), 2)}--{_br(max(acc('b_gene_novo')), 2)}"])
        _write("tabela_recortes_teste.tex", _latex_table(["Recorte do teste (genes)", "Homologia", "A (4 modelos)", "B gene novo (4 modelos)"], rows,
               "Acurácia no teste por recorte, segundo o parentesco com o treino (busca do MMseqs2). Usa-se a acurácia porque os "
               "recortes pequenos têm classes com 1 gene.", "tab:recortes_teste"))
    if _exists("sensibilidade_representacao.csv", "sensitivity_representation"):
        sr = pd.read_csv(RESULTS_DIR / "sensibilidade_representacao.csv")
        rows = [[_tex(r.config) if i == 0 else "", MODEL_LABELS[r.modelo], _br(r.f1_cv_media_a, 3), _br(r.f1_cv_media_b, 3),
                 f"{_br(r.ganho_cv, 3)} {_ic(r.ganho_cv_ic95_inf, r.ganho_cv_ic95_sup, 3)}"]
                for _, grp in sr.groupby("config", sort=False) for i, r in enumerate(grp.itertuples())]
        _write("tabela_sensibilidade_representacao.tex", _latex_table(["Representação", "Modelo", "A (CV)", "B (CV)", "B $-$ A (CV) [IC 95\\%]"], rows,
               "Sensibilidade à representação da sequência: F1-macro médio na validação cruzada e ganho do B com IC corrigido. "
               "Descritivo. reduzida\\_73 é a representação do experimento.", "tab:sens_representacao"))
    if _exists("sensibilidade_balanceamento.csv", "sensitivity_balancing"):
        sb = pd.read_csv(RESULTS_DIR / "sensibilidade_balanceamento.csv")
        rows = [[_tex(r.estrategia) if i == 0 else "", MODEL_LABELS[r.modelo], _br(r.f1_cv_media_a, 3), _br(r.f1_cv_media_b, 3),
                 f"{_br(r.ganho_cv, 3)} {_ic(r.ganho_cv_ic95_inf, r.ganho_cv_ic95_sup, 3)}"]
                for _, grp in sb.groupby("estrategia", sort=False) for i, r in enumerate(grp.itertuples())]
        _write("tabela_sensibilidade_balanceamento.tex", _latex_table(["Estratégia", "Modelo", "A (CV)", "B (CV)", "B $-$ A (CV) [IC 95\\%]"], rows,
               "Sensibilidade à estratégia de balanceamento: F1-macro médio na validação cruzada e ganho do B com IC corrigido. "
               "Descritivo. \"peso\" é a estratégia do experimento.", "tab:sens_balanceamento"))
    if _exists("sensibilidade_ajuste.csv", "sensitivity_tuning"):
        st = pd.read_csv(RESULTS_DIR / "sensibilidade_ajuste.csv")
        rows = [[MODEL_LABELS[r.modelo] if r.pipeline == "a" else "", r.pipeline.upper(), _br(r.f1_cv_padrao, 3), _br(r.f1_cv_ajustado, 3),
                 _br(r.f1_teste_padrao, 3), _br(r.f1_teste_ajustado, 3)] for r in st.itertuples()]
        _write("tabela_sensibilidade_ajuste.tex", _latex_table(["Modelo", "Pipeline", "CV padrão", "CV ajustado", "Teste padrão", "Teste ajustado"], rows,
               "Sensibilidade ao ajuste de hiperparâmetros (validação cruzada aninhada por grupos, 5 folds externos e 3 internos; "
               "grade definida pelos autores em torno do padrão). Descritivo.", "tab:sens_ajuste"))
    if _exists("cenarios_cv.csv", "report_generator"):
        # Substitui a figura analises/cenarios_cv: são comparações técnicas (custo de perder a
        # anotação curada, homologia, variantes do gene novo), lidas melhor como números.
        cc = pd.read_csv(RESULTS_DIR / "cenarios_cv.csv")
        names = {"b_gene_novo_menos_b": "B gene novo $-$ B curado", "homologia_menos_b": "Homologia $-$ B curado",
                 "homologia_menos_b_gene_novo": "Homologia $-$ B gene novo",
                 "b_gene_novo_zeros_menos_b_gene_novo": "Gene sem vizinho: atributos zerados $-$ usar o A",
                 "b_gene_novo_piso40_menos_b_gene_novo": "B gene novo: piso de 40\\% $-$ sem piso",
                 "homologia_piso40_menos_homologia": "Homologia: piso de 40\\% $-$ sem piso"}
        rows = []
        for comp, sub in cc.groupby("comparacao", sort=False):
            sub = sub.set_index("modelo")
            rows.append([names.get(comp, _tex(comp))] + [f"{_br(sub.loc[m, 'media'], 3)} {_ic(sub.loc[m, 'ic95_inf'], sub.loc[m, 'ic95_sup'], 3)}"
                                                        for m in MODEL_NAMES])
        _write("tabela_cenarios_cv.tex", _latex_table(["Comparação"] + [MODEL_LABELS[m] for m in MODEL_NAMES], rows,
               f"Diferenças no F1-macro entre os cenários na validação cruzada ({N_CV_REPEATS} $\\times$ {N_CV_SPLITS}, mesmos folds), "
               "com IC 95\\% corrigido (Nadeau e Bengio). Descritivo, sem teste. A homologia não tem modelo: nas linhas só "
               "de homologia, o valor é o mesmo nas quatro colunas.", "tab:cenarios_cv"))
    if _exists("vetores_ontologicos.csv", "data_characterization"):
        # Substitui a figura dados/vetores_ontologicos: um identificador de vetor ("vetor 12")
        # não diz nada ao leitor; a mensagem é quanto cada mecanismo fica em vetores compartilhados.
        v = pd.read_csv(RESULTS_DIR / "vetores_ontologicos.csv")
        amb = v[v["n_classes"] > 1]
        rows = []
        for m in [c for c in v.columns if c.startswith("genes_") and c != "genes_na_majoritaria"]:
            total, shared = int(v[m].sum()), int(amb[m].sum())
            rows.append([_tex(m.replace("genes_", "").replace("_", " ")), str(total), f"{shared} ({_br(100 * shared / total, 1)}\\%)"])
        rows.sort(key=lambda r: -int(r[1]))
        purity = v["genes_na_majoritaria"].sum() / v["genes"].sum()
        rows.append(["Total", str(int(v["genes"].sum())), f"{int(amb['genes'].sum())} ({_br(100 * amb['genes'].sum() / v['genes'].sum(), 1)}\\%)"])
        _write("tabela_vetores_ontologicos.tex", _latex_table(["Mecanismo", "Genes", "Em vetor compartilhado com outro mecanismo"], rows,
               f"Vetores ontológicos: combinações distintas dos 41 atributos da ARO ({len(v)} vetores; {len(amb)} com genes de mais "
               f"de um mecanismo). Pureza global (genes na classe majoritária do seu vetor): {_br(100 * purity, 1)}\\%, calculada no "
               "próprio conjunto, portanto otimista.", "tab:vetores_ontologicos"))
    if _exists("estudo_caso_resumo.csv", "case_study"):
        ec = pd.read_csv(RESULTS_DIR / "estudo_caso_resumo.csv")
        cols = [f"{k}_{m}" for k in ("A", "B") for m in MODEL_NAMES]
        abbrev = {"inactivation": "inat.", "efflux": "eflux.", "target alteration": "alt.", "target protection": "prot.", "target replacement": "subst."}
        rows = [[r["ARO_Accession"], _tex(SHORT(r["Resistance_Mechanism"]))] + [abbrev.get(r[c], _tex(r[c])) for c in cols] for _, r in ec.iterrows()]
        header = ["Gene", "Mecanismo"] + [f"{k}-{MODEL_LABELS[m].split(' ')[0]}" for k in ("A", "B") for m in MODEL_NAMES]
        _write("tabela_estudo_caso.tex", _latex_table(header, rows,
               "Estudo de caso: classe prevista (predict) pelos 8 modelos finais para os genes dos mecanismos raros, que não "
               "estão entre as classes aprendidas. Qualitativo, sem F1.", "tab:estudo_caso"))


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    ma, mb = load_metrics()
    boot = load_bootstrap()
    comp = generate_comparative_table(ma, mb, boot)
    tests = pd.read_csv(RESULTS_DIR / "testes_estatisticos.csv")
    generate_latex_tables(ma, mb, comp, tests)
    generate_per_class_table(pd.read_csv(RESULTS_DIR / "metricas_por_classe_teste.csv"))
    scenarios = generate_scenario_table(boot)
    generate_scenario_cv_table(pd.read_csv(RESULTS_DIR / "cv_folds.csv"))
    generate_ablation_table()
    generate_interpretation_tables()
    generate_overfitting_tables()
    generate_extra_tables()
    print(generate_conclusions_text(comp, tests, scenarios=scenarios))
