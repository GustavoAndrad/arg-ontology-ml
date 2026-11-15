"""
statistical_tests.py — Teste de Wilcoxon Signed-Rank (Seções 4.12 e 5.5.1).

Compara, para cada modelo, os valores pareados dos folds de CV entre os
Pipelines A e B. H0: F1_A = F1_B; H1: F1_A ≠ F1_B (bilateral), α = 0,05.

Limite importante: com n pares, o menor p bilateral possível do Wilcoxon exato é
2 / 2**n. Com os 5 folds da validação cruzada por grupos, isso dá 0,0625, então o
teste **não pode** rejeitar H0 a 5%, mesmo que B vença em todos os folds. A coluna
``p_minimo_possivel`` registra isso, para que "não rejeita" não seja lido como
ausência de efeito. O teste é substituído pelo teste t corrigido para validação
cruzada repetida (TCC2, ponto 5).

Uso:
    python -m src.statistical_tests
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

from src.experiment_config import ALPHA_STAT, MODEL_NAMES, RESULTS_DIR

logger = logging.getLogger(__name__)
TESTED_METRICS = ["f1_macro", "accuracy", "precision_macro", "recall_macro", "balanced_accuracy"]


def run_wilcoxon_test(scores_a, scores_b, alpha: float = ALPHA_STAT) -> dict:
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Amostras pareadas com tamanhos diferentes.")
    if np.allclose(a, b):
        stat, p = float("nan"), 1.0
    else:
        res = wilcoxon(a, b)
        stat, p = float(res.statistic), float(res.pvalue)
    return {
        "statistic": stat,
        "p_value": p,
        "decision": "reject_H0" if p < alpha else "fail_to_reject",
        "alpha_used": alpha,
        "n_samples": len(a),
        "p_minimo_possivel": 2 / 2 ** len(a),
        "folds_b_maior": int((b > a).sum()),
        "folds_a_maior": int((a > b).sum()),
        "media_a": float(a.mean()),
        "media_b": float(b.mean()),
    }


def run_all_wilcoxon_tests(cv_folds: pd.DataFrame, alpha: float = ALPHA_STAT, metrics=TESTED_METRICS) -> pd.DataFrame:
    rows = []
    for model in MODEL_NAMES:
        for metric in metrics:
            sub = cv_folds[cv_folds["modelo"] == model]
            a = sub[sub["pipeline"] == "a"].sort_values("fold")[metric].to_numpy()
            b = sub[sub["pipeline"] == "b"].sort_values("fold")[metric].to_numpy()
            rows.append({"modelo": model, "metrica": metric, **run_wilcoxon_test(a, b, alpha)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    folds = pd.read_csv(RESULTS_DIR / "cv_folds.csv")
    result = run_all_wilcoxon_tests(folds)
    result.to_csv(RESULTS_DIR / "testes_estatisticos.csv", index=False)
    for _, r in result[result["metrica"] == "f1_macro"].iterrows():
        logger.info(
            "%-20s W=%.1f p=%.4f (mínimo possível %.4f) -> %s (B>A em %d/%d folds)",
            r["modelo"],
            r["statistic"],
            r["p_value"],
            r["p_minimo_possivel"],
            r["decision"],
            r["folds_b_maior"],
            r["n_samples"],
        )
