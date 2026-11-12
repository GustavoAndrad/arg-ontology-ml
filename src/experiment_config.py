"""
experiment_config.py — Parâmetros e dados do experimento.

Concentra o carregamento do dataset do experimento, para que todos os módulos
usem exatamente as mesmas instâncias, na mesma ordem.

Classes (TCC2, ponto 2): cada gene do ``dataset_integrado.csv`` (6.052) cai em
exatamente um de três destinos, e a soma é conferida ao carregar:

* **experimento** — as 5 classes de ``config.TARGET_CLASSES`` (6.033 genes);
* **estudo de caso** — ``config.CASE_STUDY_CLASSES`` (9 genes), só para análise
  qualitativa com os modelos já treinados;
* **multirrótulo** — genes com dois mecanismos (10 genes), fora do trabalho.

Um mecanismo que não esteja em nenhuma lista gera erro, em vez de ser descartado
em silêncio como acontecia no TCC1 (foi assim que duas classes sumiram sem
ninguém perceber).

Ainda iguais ao TCC1 (mudam nos pontos seguintes): validação cruzada estratificada
com **10 folds** (Seção 4.8.1; ponto 5 muda para 5) e divisão aleatória 80/20
(ponto 3 muda para divisão por grupos).

O TCC1 (4 classes, 5.801 genes) é reproduzido pelo commit ``2b0247f``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from src.config import (
    ALPHA_STAT,
    CASE_STUDY_CLASSES,
    K_FEATURES,
    MULTI_LABEL_SEPARATOR,
    MODELS_DIR,
    PLOTS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    REPORTS_DIR,
    RESULTS_DIR,
    SPEARMAN_LEAKAGE_THRESHOLD,
    TARGET_CLASSES,
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
    "MIN_GENES_PER_FOLD",
    "EFFECTIVE_CLASSES",
    "CASE_STUDY_CLASSES",
    "SPLIT_EXPERIMENT",
    "SPLIT_CASE_STUDY",
    "SPLIT_MULTI_LABEL",
    "TARGET",
    "MODEL_NAMES",
    "PIPELINES",
    "SANITY_LEAKAGE_F1_THRESHOLD",
    "N_PERMUTATION_REPEATS",
    "ROC_GRID_POINTS",
    "assign_split",
    "partition_dataset",
    "dataset_composition",
    "load_integrated_dataset",
    "load_experiment_dataset",
    "load_case_study_dataset",
]

# SEED FIXA — StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
N_CV_SPLITS_TCC: int = 10

# Critério de entrada das classes (ponto 2): cerca de 5 genes de cada classe em cada
# fold de validação. Abaixo disso o código avisa; com 0 genes o F1 da classe não
# existe naquele fold, e o código para com erro.
MIN_GENES_PER_FOLD: int = 5

# As 5 classes do experimento (6.033 genes). A lista mora em config.py, para
# existir um lugar só; o nome antigo é mantido porque os demais módulos o usam.
EFFECTIVE_CLASSES: list[str] = list(TARGET_CLASSES)

# Destinos possíveis de cada gene do dataset integrado (ver docstring do módulo).
SPLIT_EXPERIMENT: str = "experimento"
SPLIT_CASE_STUDY: str = "estudo_de_caso"
SPLIT_MULTI_LABEL: str = "multirrotulo"

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


def assign_split(mechanism: str) -> str:
    """
    Diz para onde vai um gene, a partir do seu mecanismo de resistência.

    Retorna ``SPLIT_EXPERIMENT``, ``SPLIT_CASE_STUDY`` ou ``SPLIT_MULTI_LABEL``.
    Levanta ``ValueError`` para um mecanismo de rótulo único que não esteja em
    nenhuma lista: uma classe nova precisa de uma decisão explícita, e não de um
    descarte silencioso.
    """
    label = str(mechanism).strip()
    if MULTI_LABEL_SEPARATOR in label:
        return SPLIT_MULTI_LABEL
    if label in EFFECTIVE_CLASSES:
        return SPLIT_EXPERIMENT
    if label in CASE_STUDY_CLASSES:
        return SPLIT_CASE_STUDY
    raise ValueError(
        f"Mecanismo desconhecido: {label!r}. Inclua-o em TARGET_CLASSES ou CASE_STUDY_CLASSES "
        "(src/config.py) com a justificativa."
    )


def partition_dataset(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """
    Separa o dataset integrado nos três destinos, preservando a ordem original.

    A ordem importa: é dela que dependem a partição treino/teste e os folds.
    Confere que cada gene caiu em exatamente um destino (a soma tem de dar o total).
    """
    splits = df[TARGET].map(assign_split)
    parts = {
        name: df[splits == name].reset_index(drop=True)
        for name in (SPLIT_EXPERIMENT, SPLIT_CASE_STUDY, SPLIT_MULTI_LABEL)
    }
    total = sum(len(p) for p in parts.values())
    if total != len(df):  # só falharia se assign_split ganhasse um destino novo sem entrar aqui
        raise AssertionError(f"Partição não fecha: {total} != {len(df)}")
    return parts


def dataset_composition(df: pd.DataFrame) -> pd.DataFrame:
    """
    Tabela mecanismo → número de genes → destino, para o texto (Tabela 4.1 e
    Seção 4.3.3) e para conferir a conta 6.033 + 9 + 10 = 6.052.
    """
    counts = df[TARGET].value_counts()
    out = pd.DataFrame({"mecanismo": counts.index, "genes": counts.values})
    out["destino"] = out["mecanismo"].map(assign_split)
    out["_ordem"] = out["destino"].map({SPLIT_EXPERIMENT: 0, SPLIT_CASE_STUDY: 1, SPLIT_MULTI_LABEL: 2})
    out = out.sort_values(["_ordem", "genes"], ascending=[True, False])
    return out.drop(columns="_ordem").reset_index(drop=True)


def load_integrated_dataset(path: Path | None = None) -> pd.DataFrame:
    """Carrega ``dataset_integrado.csv`` (6.052 genes do CARD 4.0.1), sem filtro."""
    path = Path(path) if path else PROCESSED_DIR / "dataset_integrado.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Execute `python -m src.data_loader` primeiro.")
    return pd.read_csv(path, dtype=str)


def load_experiment_dataset(path: Path | None = None) -> pd.DataFrame:
    """
    Genes das 5 classes do experimento (6.033 com o CARD 4.0.1), na ordem original.
    """
    df = load_integrated_dataset(path)
    parts = partition_dataset(df)
    logger.info(
        "Dataset do experimento: %d de %d genes (%d classes) | estudo de caso: %d | multirrótulo (fora): %d.",
        len(parts[SPLIT_EXPERIMENT]),
        len(df),
        len(EFFECTIVE_CLASSES),
        len(parts[SPLIT_CASE_STUDY]),
        len(parts[SPLIT_MULTI_LABEL]),
    )
    return parts[SPLIT_EXPERIMENT]


def load_case_study_dataset(path: Path | None = None) -> pd.DataFrame:
    """Genes do estudo de caso qualitativo (9 com o CARD 4.0.1), na ordem original."""
    return partition_dataset(load_integrated_dataset(path))[SPLIT_CASE_STUDY]
