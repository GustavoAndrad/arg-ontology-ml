"""
experiment_config.py — Parâmetros do experimento conforme o TCC 1 (Capítulo 4).

Complementa ``src/config.py`` sem alterá-lo. Onde os dois divergem, este módulo
segue o texto do TCC:

* validação cruzada estratificada com **10 folds** (Seção 4.8.1) — ``config.py``
  ainda registra 5;
* **4 classes efetivas** (Seção 4.3.1, Tabela 4.1), resultando em 5.801 instâncias.

Também concentra o carregamento do dataset do experimento, para que todos os
módulos usem exatamente as mesmas instâncias, na mesma ordem.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    ALPHA_STAT,
    K_FEATURES,
    MODELS_DIR,
    PLOTS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    RESULTS_DIR,
    SPEARMAN_LEAKAGE_THRESHOLD,
    TEST_SIZE,
    VARIANCE_THRESHOLD,
)

logger = logging.getLogger(__name__)

__all__ = [
    "ALPHA_STAT",
    "K_FEATURES",
    "MODELS_DIR",
    "PLOTS_DIR",
    "PROCESSED_DIR",
    "RANDOM_STATE",
    "REPORTS_DIR",
    "RESULTS_DIR",
    "SPEARMAN_LEAKAGE_THRESHOLD",
    "TEST_SIZE",
    "VARIANCE_THRESHOLD",
    "N_CV_SPLITS_TCC",
    "EFFECTIVE_CLASSES",
    "TARGET",
    "MODEL_NAMES",
    "PIPELINES",
    "SANITY_LEAKAGE_F1_THRESHOLD",
    "N_PERMUTATION_REPEATS",
    "ROC_GRID_POINTS",
    "load_experiment_dataset",
]

# SEED FIXA — StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
N_CV_SPLITS_TCC: int = 10

# Seção 4.3.1: as quatro classes efetivas do dataset final (5.801 instâncias).
# Instâncias com outros rótulos no aro_index (p.ex. "antibiotic target protection",
# "antibiotic target replacement" e rótulos múltiplos separados por ';') ficam fora.
EFFECTIVE_CLASSES: list[str] = [
    "antibiotic inactivation",
    "antibiotic target alteration",
    "antibiotic efflux",
    "reduced permeability to antibiotic",
]

TARGET: str = "Resistance_Mechanism"
MODEL_NAMES: list[str] = ["random_forest", "xgboost", "logistic_regression", "svm"]
PIPELINES: dict[str, str] = {"a": "Pipeline A (sequência)", "b": "Pipeline B (sequência + ontologia)"}

# Seção 4.6.1: regra de decisão do experimento de sanidade.
SANITY_LEAKAGE_F1_THRESHOLD: float = 0.85
N_PERMUTATION_REPEATS: int = 10
# Seção 5.4: curvas ROC interpoladas em grade comum de 500 pontos de FPR.
ROC_GRID_POINTS: int = 500


def encode_target(labels: pd.Series):
    """LabelEncoder (ordem alfabética das classes). Retorna (y, classes)."""
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    y = le.fit_transform(labels)
    return y, list(le.classes_)


def split_indices(y):
    """
    Partição estratificada 80/20 (Seção 4.8), idêntica para os Pipelines A e B.
    # SEED FIXA  # ANTI-LEAKAGE: índices de treino e teste disjuntos
    """
    import numpy as np
    from sklearn.model_selection import train_test_split

    idx = np.arange(len(y))
    train_idx, test_idx = train_test_split(idx, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE)
    assert len(set(train_idx) & set(test_idx)) == 0
    return train_idx, test_idx


def cv_splitter():
    """StratifiedKFold de 10 folds com embaralhamento (Seção 4.8.1).  # SEED FIXA"""
    from sklearn.model_selection import StratifiedKFold

    return StratifiedKFold(n_splits=N_CV_SPLITS_TCC, shuffle=True, random_state=RANDOM_STATE)


def load_experiment_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Carrega ``dataset_integrado.csv`` e mantém apenas as 4 classes efetivas.

    A ordem original das linhas é preservada (é dela que dependem a partição
    treino/teste e os folds). Retorna 5.801 instâncias com os dados do CARD 4.0.1.
    """
    path = Path(path) if path else PROCESSED_DIR / "dataset_integrado.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Execute `python -m src.data_loader` primeiro.")
    df = pd.read_csv(path, dtype=str)
    n_total = len(df)
    df = df[df[TARGET].isin(EFFECTIVE_CLASSES)].reset_index(drop=True)
    logger.info(
        "Dataset do experimento: %d de %d instâncias (4 classes efetivas; %d com outros rótulos excluídas).",
        len(df),
        n_total,
        n_total - len(df),
    )
    return df
