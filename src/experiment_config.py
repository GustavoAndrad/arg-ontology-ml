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

Divisão treino/teste e validação cruzada **por grupos de similaridade** (TCC2,
ponto 3): genes com identidade ≥ ``SIMILARITY_THRESHOLD`` (70%) e cobertura ≥ 50%
ficam no mesmo grupo (``src.homology_groups``), e o grupo inteiro vai para o treino
ou para o teste. Assim o teste não tem variantes quase idênticas de genes do treino.

* Teste: um dos 5 folds de um ``StratifiedGroupKFold`` (~20%), escolhido **só pela
  proporção das classes** entre sementes candidatas (``choose_test_partition``),
  nunca pelo desempenho.
* Validação cruzada: ``StratifiedGroupKFold`` com **5 folds** no treino. Com grupos,
  10 folds deixariam a substituição do alvo com cerca de 3 genes por fold, abaixo
  do critério de entrada (antecipa a decisão 1 do ponto 5). A semente dos folds
  também é escolhida só pela distribuição das classes (``choose_cv_seed``): com
  grupos grandes, uma semente qualquer pode deixar uma classe pequena com 1 a 3
  genes num fold (com a semente 42, a substituição do alvo ficava com 3).
* ``groups=None`` (ou ``random_groups``) reproduz a divisão aleatória como caso
  particular: cada gene é o seu próprio grupo.

O TCC1 (4 classes, 5.801 genes) é reproduzido pelo commit ``2b0247f``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    ALPHA_STAT,
    CASE_STUDY_CLASSES,
    K_FEATURES,
    MULTI_LABEL_SEPARATOR,
    MODELS_DIR,
    N_CV_SPLITS,
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
    "N_CV_SPLITS",
    "SIMILARITY_THRESHOLD",
    "PARTITION_SEED_CANDIDATES",
    "load_groups",
    "random_groups",
    "choose_test_partition",
    "choose_cv_seed",
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

# Limiar de identidade (%) dos grupos da avaliação principal (ponto 3).
SIMILARITY_THRESHOLD: int = 70
# Sementes testadas para escolher a partição de teste pela proporção das classes.  # SEED FIXA
PARTITION_SEED_CANDIDATES: range = range(100)
TEST_FRACTION_TARGET: float = 0.2

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


def random_groups(n: int) -> np.ndarray:
    """Cada gene no seu próprio grupo: a divisão por grupos vira a divisão aleatória."""
    return np.arange(n)


def load_groups(threshold: int | None = SIMILARITY_THRESHOLD, data: pd.DataFrame | None = None) -> np.ndarray:
    """
    Grupo de cada gene do experimento (na ordem de ``load_experiment_dataset``).

    ``threshold=None`` devolve grupos unitários (divisão aleatória). Os grupos vêm de
    ``results/grupos_similaridade.csv``, gerado por ``python -m src.homology_groups``.
    """
    data = load_experiment_dataset() if data is None else data
    if threshold is None:
        return random_groups(len(data))
    path = RESULTS_DIR / "grupos_similaridade.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} não encontrado. Execute `python -m src.homology_groups` primeiro.")
    table = pd.read_csv(path)
    if table["ARO_Accession"].tolist() != data["ARO_Accession"].tolist():
        raise ValueError("Grupos desalinhados ao dataset do experimento; reexecute `python -m src.homology_groups`.")
    return table[f"grupo_{threshold}"].to_numpy()


def _test_fraction_by_class(y: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
    return np.bincount(y[test_idx], minlength=y.max() + 1) / np.bincount(y, minlength=y.max() + 1)


def choose_test_partition(y: np.ndarray, groups: np.ndarray, seeds=PARTITION_SEED_CANDIDATES):
    """
    Escolhe a partição treino/teste por grupos mais próxima de 20% de cada classe no teste.

    Candidatas: os 5 folds de ``StratifiedGroupKFold(5, shuffle=True, random_state=s)``
    para cada semente s. Critério, fixado antes de ver qualquer resultado: menor desvio
    máximo |fração da classe no teste − 0,20| entre as classes; empate pelo desvio do
    tamanho total do teste e depois pela menor semente. Retorna (train_idx, test_idx, info).
    # SEED FIXA  # ANTI-LEAKAGE: grupos inteiros de um lado só
    """
    from sklearn.model_selection import StratifiedGroupKFold

    n_folds = round(1 / TEST_FRACTION_TARGET)
    best = None
    for seed in seeds:
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for fold, (train_idx, test_idx) in enumerate(cv.split(np.zeros(len(y)), y, groups)):
            frac = _test_fraction_by_class(y, test_idx)
            key = (
                float(np.abs(frac - TEST_FRACTION_TARGET).max()),
                abs(len(test_idx) / len(y) - TEST_FRACTION_TARGET),
                seed,
                fold,
            )
            if best is None or key < best[0]:
                best = (key, train_idx, test_idx, frac)
    (max_dev, size_dev, seed, fold), train_idx, test_idx, frac = best
    assert not set(groups[train_idx]) & set(groups[test_idx]), "grupo nos dois lados"
    info = {
        "semente": int(seed),
        "fold": int(fold),
        "n_folds": n_folds,
        "sementes_testadas": [min(seeds), max(seeds)],
        "desvio_maximo_da_fracao_por_classe": max_dev,
        "fracao_teste_por_classe": [float(f) for f in frac],
        "n_treino": int(len(train_idx)),
        "n_teste": int(len(test_idx)),
        "grupos_treino": int(len(set(groups[train_idx]))),
        "grupos_teste": int(len(set(groups[test_idx]))),
    }
    return np.sort(train_idx), np.sort(test_idx), info


def split_indices(y, groups: np.ndarray | None = None):
    """
    Partição treino/teste do experimento, idêntica para os Pipelines A e B.

    Com ``groups`` (o padrão do experimento é ``load_groups()``), usa ``choose_test_partition``.
    Sem ``groups``, cada gene é um grupo (divisão aleatória estratificada).
    # SEED FIXA  # ANTI-LEAKAGE: índices de treino e teste disjuntos
    """
    y = np.asarray(y)
    groups = random_groups(len(y)) if groups is None else np.asarray(groups)
    train_idx, test_idx, _ = choose_test_partition(y, groups)
    assert len(set(train_idx) & set(test_idx)) == 0
    return train_idx, test_idx


def cv_splitter(random_state: int = RANDOM_STATE):
    """
    Validação cruzada por grupos: ``StratifiedGroupKFold(5, shuffle=True, random_state)``.
    No experimento, ``random_state`` vem de ``choose_cv_seed``. Use ``.split(X, y, groups)``.  # SEED FIXA
    """
    from sklearn.model_selection import StratifiedGroupKFold

    return StratifiedGroupKFold(n_splits=N_CV_SPLITS, shuffle=True, random_state=random_state)


def choose_cv_seed(y_train: np.ndarray, groups_train: np.ndarray, seeds=PARTITION_SEED_CANDIDATES):
    """
    Escolhe a semente dos folds da validação cruzada só pela distribuição das classes.

    Critério, fixado antes de ver qualquer resultado: maior número mínimo de genes de
    uma classe num fold (o critério de entrada das classes é cerca de 5 por fold);
    empate pelo menor desvio máximo |fração da classe no fold − 1/5|; depois pela menor
    semente. Retorna (semente, info).  # SEED FIXA
    """
    y_train = np.asarray(y_train)
    n_classes = y_train.max() + 1
    totals = np.bincount(y_train, minlength=n_classes)
    best = None
    for seed in seeds:
        counts = np.array(
            [
                np.bincount(y_train[val], minlength=n_classes)
                for _, val in cv_splitter(seed).split(np.zeros(len(y_train)), y_train, groups_train)
            ]
        )
        key = (-int(counts.min()), float(np.abs(counts / totals - 1 / N_CV_SPLITS).max()), seed)
        if best is None or key < best[0]:
            best = (key, counts)
    (neg_min, max_dev, seed), counts = best
    info = {
        "semente_cv": int(seed),
        "sementes_cv_testadas": [min(seeds), max(seeds)],
        "minimo_genes_classe_fold": int(-neg_min),
        "minimo_por_classe": [int(c) for c in counts.min(axis=0)],
        "desvio_maximo_fracao_fold": max_dev,
    }
    return int(seed), info


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
