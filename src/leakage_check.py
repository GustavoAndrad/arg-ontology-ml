"""
leakage_check.py — Controle de leakage semântico (Seção 4.6).

1. Correlação de Spearman |ρ| entre cada feature ontológica e o rótulo codificado;
   features com |ρ| ≥ 0,95 são removidas.
2. Experimento de sanidade (Seção 4.6.1): Random Forest (mesmos hiperparâmetros)
   treinado só com as features ontológicas e avaliado no conjunto de teste.
   Regra: F1-macro ≤ 0,85 → sem evidência de leakage direto.

Uso:
    python -m src.leakage_check
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.feature_selection import VarianceThreshold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler

from src.experiment_config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    SANITY_LEAKAGE_F1_THRESHOLD,
    SPEARMAN_LEAKAGE_THRESHOLD,
    TARGET,
    encode_target,
    load_experiment_dataset,
    load_groups,
    split_indices,
)
from src.pipeline_builder import build_classifier

logger = logging.getLogger(__name__)


def label_encode_target(series: pd.Series) -> np.ndarray:
    return encode_target(series)[0]


def compute_spearman_correlations(features: pd.DataFrame, encoded_target: np.ndarray) -> pd.Series:
    """|ρ| de Spearman por feature (0 para features constantes)."""
    out = {}
    for col in features.columns:
        values = features[col].to_numpy()
        out[col] = 0.0 if np.unique(values).size < 2 else abs(spearmanr(values, encoded_target).statistic)
    return pd.Series(out, name="abs_spearman_rho").sort_values(ascending=False)


def identify_leaky_features(correlations: pd.Series, threshold: float = SPEARMAN_LEAKAGE_THRESHOLD) -> list[str]:
    return correlations[correlations >= threshold].index.tolist()


def remove_leaky_features(features: pd.DataFrame, leaky: list[str], log_path: Path) -> pd.DataFrame:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with open(log_path, "w", encoding="utf-8") as fh:
        if leaky:
            for f in leaky:
                fh.write(f"REMOVIDA: {f} (|rho| >= {SPEARMAN_LEAKAGE_THRESHOLD})\n")
        else:
            fh.write(f"Nenhuma feature ontológica com |rho| >= {SPEARMAN_LEAKAGE_THRESHOLD}; nenhuma removida.\n")
    return features.drop(columns=leaky)


def run_leakage_check(
    ontological: pd.DataFrame,
    target: pd.Series,
    threshold: float = SPEARMAN_LEAKAGE_THRESHOLD,
    log_path: Path = RESULTS_DIR / "leakage_check.log",
) -> tuple[pd.DataFrame, pd.Series]:
    """# ANTI-LEAKAGE: ponto central de verificação."""
    y = label_encode_target(target)
    corr = compute_spearman_correlations(ontological, y)
    leaky = identify_leaky_features(corr, threshold)
    if len(leaky) == ontological.shape[1]:
        logger.warning("Todas as features ontológicas foram removidas: Pipeline B degrada para Pipeline A.")
    logger.info("Spearman: máx |ρ| = %.4f (%s); removidas: %s", corr.iloc[0], corr.index[0], leaky or "nenhuma")
    return remove_leaky_features(ontological, leaky, log_path), corr


def run_sanity_experiment(ontological: pd.DataFrame, target: pd.Series, groups=None) -> dict:
    """RF treinado apenas com features ontológicas; métricas no conjunto de teste (mesma partição do experimento)."""
    y, classes = encode_target(target)
    train_idx, test_idx = split_indices(y, groups)
    model = Pipeline(
        [("scaler", MinMaxScaler()), ("variance", VarianceThreshold(0.0)), ("clf", build_classifier("random_forest"))]
    )
    X = ontological.to_numpy(dtype=float)
    model.fit(X[train_idx], y[train_idx])
    pred = model.predict(X[test_idx])
    f1 = f1_score(y[test_idx], pred, average="macro", zero_division=0)
    per_class = f1_score(y[test_idx], pred, average=None, labels=list(range(len(classes))), zero_division=0)
    result = {
        "f1_macro_somente_ontologia": float(f1),
        "accuracy": float(accuracy_score(y[test_idx], pred)),
        "f1_por_classe": dict(zip(classes, map(float, per_class))),
        "limiar": SANITY_LEAKAGE_F1_THRESHOLD,
        "decisao": "manter features (sem evidência de leakage direto)"
        if f1 <= SANITY_LEAKAGE_F1_THRESHOLD
        else "investigar leakage (F1 acima do limiar)",
    }
    logger.info("Experimento de sanidade: F1-macro = %.4f -> %s", f1, result["decisao"])
    return result


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    data = load_experiment_dataset()
    path = PROCESSED_DIR / "features_ontologicas.csv"
    onto = pd.read_csv(path)
    assert (onto["ARO_Accession"].values == data["ARO_Accession"].values).all(), "Ordem das instâncias divergente"
    feats = onto.drop(columns=["ARO_Accession"])
    clean, corr = run_leakage_check(feats, data[TARGET])
    sanity = run_sanity_experiment(clean, data[TARGET], load_groups(data=data))
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    corr.to_csv(RESULTS_DIR / "spearman_features_ontologicas.csv", header=True)
    (RESULTS_DIR / "sanidade_leakage.json").write_text(
        json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    clean.insert(0, "ARO_Accession", onto["ARO_Accession"])
    clean.to_csv(path, index=False)
    logger.info("Features ontológicas verificadas salvas em %s (%d features).", path, clean.shape[1] - 1)
