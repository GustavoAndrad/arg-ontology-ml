"""
compare_with_tcc.py — Confronta os resultados gerados com os valores publicados no TCC.

**Legado da primeira apresentação do TCC à banca.** Este script compara o código com
o texto daquela versão (4 classes, Wilcoxon, divisão aleatória) e só faz sentido no
commit ``2b0247f``, onde o código reproduz 73/73 itens do PDF. Fica no repositório
como registro do gabarito usado para validar a reimplementação. O experimento atual
é outro (5 classes, divisão por grupos, teste t corrigido), então fora daquele commit
o script aponta diferenças por construção e termina com erro: ``results/testes_estatisticos.csv``
não tem mais as colunas do Wilcoxon, e ``results/sanidade_leakage.json`` não é mais
gerado.

Os valores de referência abaixo foram transcritos do PDF (Tabelas 4.1, 4.2, 5.1–5.6 e
Seção 4.6.1). Nenhum deles é usado para gerar resultados: servem apenas de gabarito.
Um item é considerado igual quando coincide no número de casas decimais publicado.

Saída: results/comparacao_tcc.csv

Uso:
    python -m src.compare_with_tcc
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd

from src.experiment_config import PROCESSED_DIR, RESULTS_DIR, TARGET, load_experiment_dataset

logger = logging.getLogger(__name__)

TCC = {
    "test": {  # Tabelas 5.1, 5.2 e 5.4: (acurácia, precisão, revocação, F1, ROC-AUC)
        ("a", "random_forest"): (0.9699, 0.7366, 0.6028, 0.6550, 0.9297),
        ("a", "xgboost"): (0.9767, 0.7180, 0.6462, 0.6787, 0.9916),
        ("a", "logistic_regression"): (0.9552, 0.6002, 0.6654, 0.6271, 0.9702),
        ("a", "svm"): (0.9707, 0.6557, 0.6694, 0.6605, 0.9886),
        ("b", "random_forest"): (0.9897, 0.7427, 0.7057, 0.7228, 0.9943),
        ("b", "xgboost"): (0.9914, 0.7388, 0.7227, 0.7305, 0.9979),
        ("b", "logistic_regression"): (0.9862, 0.7075, 0.7298, 0.7182, 0.9938),
        ("b", "svm"): (0.9897, 0.7182, 0.7310, 0.7243, 0.9979),
    },
    "cv": {  # Tabela 5.5: (média, desvio) do F1 macro em 10 folds
        ("a", "random_forest"): (0.7095, 0.0952),
        ("b", "random_forest"): (0.7980, 0.1237),
        ("a", "xgboost"): (0.7624, 0.1215),
        ("b", "xgboost"): (0.8266, 0.1181),
        ("a", "logistic_regression"): (0.6848, 0.0947),
        ("b", "logistic_regression"): (0.7527, 0.0998),
        ("a", "svm"): (0.7513, 0.1092),
        ("b", "svm"): (0.8206, 0.1208),
    },
    "wilcoxon": {  # Tabela 5.6: (W, p)
        "random_forest": (0.0, 0.0020),
        "xgboost": (0.0, 0.0020),
        "logistic_regression": (9.0, 0.0645),
        "svm": (0.0, 0.0020),
    },
    "classes": {
        "antibiotic inactivation": 5239,
        "antibiotic target alteration": 285,
        "antibiotic efflux": 269,
        "reduced permeability to antibiotic": 8,
    },
    "sanity_f1": 0.8011,
    "n_features": {"sequencia": 426, "ontologicas": 39, "pipeline_b": 465},
}
TEST_METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "roc_auc_ovr"]


def _row(secao, item, tcc, obtido, casas=4):
    igual = (
        bool(np.isclose(round(float(obtido), casas), tcc, atol=10 ** -(casas + 2)))
        if casas is not None
        else obtido == tcc
    )
    return {
        "secao": secao,
        "item": item,
        "valor_tcc": tcc,
        "valor_obtido": round(float(obtido), 6),
        "diferenca": round(float(obtido) - float(tcc), 6),
        "igual": igual,
    }


def build_comparison() -> pd.DataFrame:
    rows = []
    data = load_experiment_dataset()
    counts = data[TARGET].value_counts()
    rows.append(_row("Tab. 4.1", "total de instâncias", 5801, len(data), 0))
    for c, n in TCC["classes"].items():
        rows.append(_row("Tab. 4.1", f"instâncias: {c}", n, counts.get(c, 0), 0))
    seq = pd.read_csv(PROCESSED_DIR / "features_sequencia.csv", nrows=1).shape[1] - 1
    onto = pd.read_csv(PROCESSED_DIR / "features_ontologicas.csv", nrows=1).shape[1] - 1
    rows += [
        _row("Tab. 4.2", "features de sequência", 426, seq, 0),
        _row("Seç. 4.5", "features ontológicas", 39, onto, 0),
        _row("Seç. 4.7", "features Pipeline B", 465, seq + onto, 0),
    ]
    sanity = json.loads((RESULTS_DIR / "sanidade_leakage.json").read_text(encoding="utf-8"))
    rows.append(_row("Seç. 4.6.1", "F1 somente ontologia", TCC["sanity_f1"], sanity["f1_macro_somente_ontologia"]))

    for key in ("a", "b"):
        m = pd.read_csv(RESULTS_DIR / f"metrics_pipeline_{key}.csv").set_index(["modelo", "metrica"])
        for (k, model), ref in TCC["test"].items():
            if k != key:
                continue
            for metric, v in zip(TEST_METRICS, ref):
                tab = "Tab. 5.4" if metric == "roc_auc_ovr" else ("Tab. 5.1" if key == "a" else "Tab. 5.2")
                rows.append(
                    _row(tab, f"{key.upper()} {model} {metric} (teste)", v, m.loc[(model, metric), "valor_teste"])
                )
            mean, std = TCC["cv"][(key, model)]
            rows.append(
                _row("Tab. 5.5", f"{key.upper()} {model} F1 CV média", mean, m.loc[(model, "f1_macro"), "media_cv"])
            )
            rows.append(
                _row("Tab. 5.5", f"{key.upper()} {model} F1 CV desvio", std, m.loc[(model, "f1_macro"), "std_cv"])
            )
    w = pd.read_csv(RESULTS_DIR / "testes_estatisticos.csv")
    w = w[w.metrica == "f1_macro"].set_index("modelo")
    for model, (stat, p) in TCC["wilcoxon"].items():
        rows.append(_row("Tab. 5.6", f"{model} Wilcoxon W", stat, w.loc[model, "statistic"], 1))
        rows.append(_row("Tab. 5.6", f"{model} Wilcoxon p", p, w.loc[model, "p_value"]))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s", handlers=[logging.StreamHandler(sys.stdout)])
    comp = build_comparison()
    comp.to_csv(RESULTS_DIR / "comparacao_tcc.csv", index=False)
    pd.set_option("display.width", 200, "display.max_rows", 200)
    diff = comp[~comp.igual]
    logger.info("Itens comparados: %d | idênticos ao TCC: %d | diferentes: %d", len(comp), comp.igual.sum(), len(diff))
    if len(diff):
        logger.info("\n%s", diff.to_string(index=False))
