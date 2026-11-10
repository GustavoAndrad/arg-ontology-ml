"""
interpretability.py — Importância de features dos modelos finais.

* Gini importance do Random Forest (Pipelines A e B);
* Permutation importance (n_repeats=10, random_state=42) no conjunto de teste,
  métrica F1-macro, restrita às features escolhidas pelo SelectKBest (as demais
  não chegam ao classificador e têm importância zero por construção);
* SHAP (TreeExplainer) para o Random Forest do Pipeline B, se ``shap`` estiver instalado.

Uso:
    python -m src.interpretability
"""

from __future__ import annotations

import logging
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.experiment_config import (
    MODEL_NAMES,
    MODELS_DIR,
    N_PERMUTATION_REPEATS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
)
from src.train_evaluate import load_feature_matrices

logger = logging.getLogger(__name__)
SEQ_PREFIXES = (
    "length",
    "molecular_weight",
    "isoelectric_point",
    "aromaticity",
    "instability_index",
    "charge_at_pH7",
    "aa_comp_",
    "dipep_",
)


def feature_type(name: str) -> str:
    return "sequencia" if name.startswith(SEQ_PREFIXES) else "ontologica"


def get_feature_names_after_pipeline(pipeline, feature_names_in: list[str]) -> list[str]:
    names = np.asarray(feature_names_in)
    names = names[pipeline.named_steps["variance"].get_support()]
    return list(names[pipeline.named_steps["select"].get_support()])


def compute_gini_importance(pipeline, feature_names: list[str]) -> pd.DataFrame:
    selected = get_feature_names_after_pipeline(pipeline, feature_names)
    imp = pipeline.named_steps["clf"].feature_importances_
    df = pd.DataFrame({"feature": selected, "importancia": imp})
    df["tipo"] = df["feature"].map(feature_type)
    return df.sort_values("importancia", ascending=False).reset_index(drop=True)


def compute_permutation_importance(
    pipeline, X_test: np.ndarray, y_test: np.ndarray, feature_names: list[str]
) -> pd.DataFrame:
    """# SEED FIXA"""
    pre = pipeline[:-1]
    Xt = pre.transform(X_test)
    selected = get_feature_names_after_pipeline(pipeline, feature_names)
    r = permutation_importance(
        pipeline.named_steps["clf"],
        Xt,
        y_test,
        scoring="f1_macro",
        n_repeats=N_PERMUTATION_REPEATS,
        random_state=RANDOM_STATE,
    )
    df = pd.DataFrame(
        {"feature": selected, "importancia_media": r.importances_mean, "importancia_std": r.importances_std}
    )
    df["tipo"] = df["feature"].map(feature_type)
    return df.sort_values("importancia_media", ascending=False).reset_index(drop=True)


def compute_shap_values(pipeline, X_test: np.ndarray, feature_names: list[str]):
    try:
        import shap
    except ImportError:
        logger.warning("shap não instalado; valores SHAP não calculados.")
        return None, None
    Xt = pipeline[:-1].transform(X_test)
    values = shap.TreeExplainer(pipeline.named_steps["clf"]).shap_values(Xt)
    return values, get_feature_names_after_pipeline(pipeline, feature_names)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    X_a, X_b, y, _ = load_feature_matrices()
    test_idx = np.load(PROCESSED_DIR / "test_indices.npy")
    perm_rows = []
    for key, X_df in (("a", X_a), ("b", X_b)):
        names = list(X_df.columns)
        X = X_df.to_numpy(dtype=float)
        for model in MODEL_NAMES:
            pipe = joblib.load(MODELS_DIR / f"pipeline_{key}_{model}.joblib")
            if model == "random_forest":
                gini = compute_gini_importance(pipe, names)
                gini.to_csv(RESULTS_DIR / f"feature_importance_gini_rf_{key}.csv", index=False)
                top = gini.head(20)
                logger.info(
                    "RF Pipeline %s: %d features ontológicas no top-20 (Gini).",
                    key.upper(),
                    (top.tipo == "ontologica").sum(),
                )
            perm = compute_permutation_importance(pipe, X[test_idx], y[test_idx], names)
            perm.insert(0, "modelo", model)
            perm.insert(0, "pipeline", key)
            perm_rows.append(perm)
            logger.info("Permutation importance: pipeline %s / %s concluído.", key.upper(), model)
    pd.concat(perm_rows).to_csv(RESULTS_DIR / "permutation_importance.csv", index=False)
