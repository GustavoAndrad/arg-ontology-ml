"""
pipeline_builder.py — Pipelines scikit-learn (Seções 4.7, 4.9 e 4.10).

Experimento:

    MinMaxScaler → classificador

Todos os atributos entram no modelo (73 no Pipeline A, 114 no B), sem seleção.
Por que não há mais SelectKBest nem VarianceThreshold (TCC2, ponto 7):
    * o k = 100 do TCC1 era arbitrário, e o tamanho do ganho do B dependia dele;
    * no Pipeline B, a seleção conjunta deixava entrar só 18 dos 41 atributos
      ontológicos e mudava também quais atributos de sequência entravam, de modo
      que A e B não diferiam só pela ontologia (Seção 4.7);
    * o VarianceThreshold só remove colunas constantes; sem ele, todas as colunas
      entram em todo fold (duas flags de classe de antibiótico têm 1 gene no treino
      e poderiam ficar constantes num fold, mudando o número de atributos do B).

A seleção do TCC1 continua disponível via ``selection`` (``FeatureSelection``) só
para a análise de sensibilidade à representação (``src.sensitivity_representation``):
    * escopo "todos": MinMaxScaler → VarianceThreshold → SelectKBest(k) → clf
      (exatamente o pipeline do TCC1);
    * escopo "sequencia": a seleção é aplicada só às colunas de sequência e as
      ontológicas passam inteiras (B = A_k + 41).

Classificadores (random_state=42; hiperparâmetros fixos e iguais para A e B):
    * Random Forest — n_estimators=100, class_weight='balanced'
    * XGBoost       — n_estimators=100, eval_metric='mlogloss', peso 'balanced' por amostra
    * Regressão Logística — max_iter=1000, class_weight='balanced'
    * SVM (RBF)     — probability=True, class_weight='balanced'
Os demais hiperparâmetros são os padrão das bibliotecas; a lista completa, lida
com ``get_params()``, fica em ``models/hyperparameters.json`` e em
``results/hiperparametros.csv`` (``src.experiment_record``).

Peso por classe em todos os modelos (TCC2, ponto 6): o XGBoost não tem
``class_weight``; o peso vai como ``sample_weight`` no ``fit``. Em vez de passar o
peso em cada lugar que treina um modelo (CV, modelo final, curva por limiar,
análises de sensibilidade) — e arriscar esquecer um —, ``BalancedXGBClassifier``
calcula o peso sozinho a cada ``fit``, com a mesma fórmula do
``class_weight='balanced'`` do scikit-learn: n_amostras / (n_classes × n_da_classe).
Assim os 4 modelos usam exatamente a mesma regra de balanceamento.

Hiperparâmetros fixos (TCC2, ponto 13): o objetivo é comparar conjuntos de
atributos nas mesmas condições, não achar o melhor modelo possível. O ajuste entra
só como análise de sensibilidade (``src.sensitivity_tuning``).

Uso:
    python -m src.pipeline_builder   # grava models/hyperparameters.json
"""

from __future__ import annotations

import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier

from src.experiment_config import MODEL_NAMES, MODELS_DIR, RANDOM_STATE, VARIANCE_THRESHOLD

logger = logging.getLogger(__name__)


class BalancedXGBClassifier(XGBClassifier):
    """
    XGBoost com peso por classe "balanced" calculado em cada ``fit``.

    Se ``sample_weight`` não for passado, usa
    ``compute_sample_weight("balanced", y)``: cada gene recebe
    n_amostras / (n_classes × n_genes_da_classe), a mesma regra do
    ``class_weight='balanced'`` dos outros três modelos. Os hiperparâmetros são os
    do ``XGBClassifier`` (``get_params()`` não muda). Os rótulos chegam codificados
    como inteiros 0..n_classes-1.
    """

    def fit(self, X, y, *, sample_weight=None, **kwargs):
        if sample_weight is None:
            sample_weight = compute_sample_weight("balanced", y)
        return super().fit(X, y, sample_weight=sample_weight, **kwargs)


# Como cada modelo compensa o desbalanceamento (registrado no hyperparameters.json).
CLASS_BALANCING: dict[str, str] = {
    "random_forest": "class_weight='balanced'",
    "xgboost": "sample_weight 'balanced' em cada fit (BalancedXGBClassifier)",
    "logistic_regression": "class_weight='balanced'",
    "svm": "class_weight='balanced'",
}

# Nomes exibidos em figuras e tabelas. "Regressão Logística" em português, como no
# resto do texto do TCC (ponto 12 do documento de decisões); os demais são nomes próprios.
MODEL_LABELS: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Regressão Logística",
    "svm": "SVM (RBF)",
}


def _model_params() -> dict[str, dict]:
    """Hiperparâmetros definidos à mão; os demais ficam no padrão da biblioteca.  # SEED FIXA"""
    return {
        "random_forest": {"n_estimators": 100, "class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1},
        "xgboost": {"n_estimators": 100, "eval_metric": "mlogloss", "random_state": RANDOM_STATE, "n_jobs": -1},
        "logistic_regression": {"max_iter": 1000, "class_weight": "balanced", "random_state": RANDOM_STATE},
        "svm": {"kernel": "rbf", "probability": True, "class_weight": "balanced", "random_state": RANDOM_STATE},
    }


@dataclass(frozen=True)
class FeatureSelection:
    """
    Seleção de atributos do TCC1, usada só na análise de sensibilidade.

    ``k``: número de atributos mantidos pelo SelectKBest (ANOVA F).
    ``scope``: "todos" (seleção conjunta, como no TCC1) ou "sequencia" (seleção só
    nas primeiras ``n_sequence`` colunas, que são as de sequência; as demais passam).
    """

    k: int
    scope: str = "todos"
    n_sequence: int | None = None

    def __post_init__(self) -> None:
        if self.scope not in ("todos", "sequencia"):
            raise ValueError(f"Escopo de seleção desconhecido: {self.scope}")
        if self.scope == "sequencia" and self.n_sequence is None:
            raise ValueError("Seleção 'sequencia' exige n_sequence.")


def _selector(k: int) -> Pipeline:
    """VarianceThreshold → SelectKBest(f_classif, k), como no TCC1."""
    return Pipeline(
        [
            ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
            ("select", SelectKBest(score_func=f_classif, k=k)),
        ]
    )


def build_preprocessor(
    selection: FeatureSelection | None = None, n_features: int | None = None
) -> list[tuple[str, BaseEstimator]]:
    """
    Etapas de pré-processamento.  # ANTI-LEAKAGE: fit exclusivo no treino (dentro do Pipeline).

    Sem ``selection`` (experimento): só o MinMaxScaler. ``n_features`` (total de
    colunas da matriz) só é necessário para a seleção de escopo "sequencia".
    """
    steps: list[tuple[str, BaseEstimator]] = [("scaler", MinMaxScaler())]
    if selection is None:
        return steps
    if selection.scope == "todos" or (n_features is not None and selection.n_sequence >= n_features):
        # Seleção conjunta (TCC1). No Pipeline A, "sequencia" é o mesmo que "todos".
        return steps + [("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
                        ("select", SelectKBest(score_func=f_classif, k=selection.k))]
    if n_features is None:
        raise ValueError("Seleção 'sequencia' exige n_features.")
    seq_cols = list(range(selection.n_sequence))
    onto_cols = list(range(selection.n_sequence, n_features))
    return steps + [
        (
            "select_sequencia",
            ColumnTransformer([("sequencia", _selector(selection.k), seq_cols), ("ontologia", "passthrough", onto_cols)]),
        )
    ]


def build_classifier(model_name: str) -> BaseEstimator:
    """Instancia o classificador com os hiperparâmetros do experimento."""
    params = _model_params()[model_name]
    if model_name == "random_forest":
        return RandomForestClassifier(**params)
    if model_name == "xgboost":
        return BalancedXGBClassifier(**params)
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    if model_name == "svm":
        return SVC(**params)
    raise ValueError(f"Modelo desconhecido: {model_name}")


def build_full_pipeline(
    model_name: str, selection: FeatureSelection | None = None, n_features: int | None = None
) -> Pipeline:
    """
    Pipeline completo (idêntico para os Pipelines A e B; só muda a matriz de entrada).

    No experimento, chame sem ``selection``. ``selection`` e ``n_features`` existem
    só para a análise de sensibilidade.
    """
    return Pipeline(build_preprocessor(selection, n_features) + [("clf", build_classifier(model_name))])


def _json_value(v):
    """Valor de ``get_params()`` em forma gravável (NaN e objetos viram texto)."""
    if isinstance(v, float) and v != v:
        return "nan"
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    return repr(v)


def classifier_params(model_name: str) -> dict:
    """Todos os hiperparâmetros do classificador, lidos com ``get_params()`` (nada transcrito à mão)."""
    return {k: _json_value(v) for k, v in sorted(build_classifier(model_name).get_params().items())}


def save_hyperparameters(output_path: Path = MODELS_DIR / "hyperparameters.json") -> dict:
    """
    Registra pré-processamento e hiperparâmetros em JSON.

    ``models`` traz todos os hiperparâmetros de cada classificador, lidos com
    ``get_params()``; ``definidos_no_codigo`` separa os que foram escolhidos à mão.
    Os valores são os mesmos nos Pipelines A e B.
    """
    import sklearn
    import xgboost

    from src.feature_extraction import SEQUENCE_FEATURE_COLUMNS

    params = {
        "preprocessing": "MinMaxScaler (ajustado só no treino); sem seleção de atributos",
        "n_features": {
            "pipeline_a": len(SEQUENCE_FEATURE_COLUMNS),
            "pipeline_b": "pipeline_a + atributos ontológicos (41 com o CARD 4.0.1: 3 estruturais + 38 flags)",
        },
        "random_state": RANDOM_STATE,
        "iguais_nos_pipelines_a_e_b": True,
        "class_balancing": CLASS_BALANCING,
        "definidos_no_codigo": _model_params(),
        "models": {name: classifier_params(name) for name in MODEL_NAMES},
        "versions": {"scikit-learn": sklearn.__version__, "xgboost": xgboost.__version__},
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Hiperparâmetros salvos em %s", output_path)
    return params


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    for name in MODEL_NAMES:
        logger.info("%s: %s", name, build_full_pipeline(name))
    save_hyperparameters()
