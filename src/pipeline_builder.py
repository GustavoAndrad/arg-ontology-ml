"""
pipeline_builder.py — Pipelines scikit-learn (Seções 4.7, 4.9 e 4.10).

    MinMaxScaler → VarianceThreshold(0.0) → SelectKBest(f_classif, k=100) → classificador

Classificadores (random_state=42):
    * Random Forest — n_estimators=100, class_weight='balanced'
    * XGBoost       — n_estimators=100, eval_metric='mlogloss' (sem class_weight)
    * Regressão Logística — max_iter=1000, class_weight='balanced'
    * SVM (RBF)     — probability=True, class_weight='balanced'

Uso:
    python -m src.pipeline_builder   # grava models/hyperparameters.json
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from src.experiment_config import K_FEATURES, MODEL_NAMES, MODELS_DIR, RANDOM_STATE, VARIANCE_THRESHOLD

logger = logging.getLogger(__name__)

MODEL_LABELS: dict[str, str] = {
    "random_forest": "Random Forest",
    "xgboost": "XGBoost",
    "logistic_regression": "Logistic Regression",
    "svm": "SVM (RBF)",
}


def _model_params() -> dict[str, dict]:
    # SEED FIXA em todos os classificadores
    return {
        "random_forest": {"n_estimators": 100, "class_weight": "balanced", "random_state": RANDOM_STATE, "n_jobs": -1},
        "xgboost": {"n_estimators": 100, "eval_metric": "mlogloss", "random_state": RANDOM_STATE, "n_jobs": -1},
        "logistic_regression": {"max_iter": 1000, "class_weight": "balanced", "random_state": RANDOM_STATE},
        "svm": {"kernel": "rbf", "probability": True, "class_weight": "balanced", "random_state": RANDOM_STATE},
    }


def build_preprocessor(k_features: int = K_FEATURES) -> list[tuple[str, BaseEstimator]]:
    """Etapas de pré-processamento.  # ANTI-LEAKAGE: fit exclusivo no treino (dentro do Pipeline)."""
    return [
        ("scaler", MinMaxScaler()),
        ("variance", VarianceThreshold(threshold=VARIANCE_THRESHOLD)),
        ("select", SelectKBest(score_func=f_classif, k=k_features)),
    ]


def build_classifier(model_name: str) -> BaseEstimator:
    """Instancia o classificador com os hiperparâmetros do experimento."""
    params = _model_params()[model_name]
    if model_name == "random_forest":
        return RandomForestClassifier(**params)
    if model_name == "xgboost":
        from xgboost import XGBClassifier  # rótulos já chegam codificados como inteiros 0..3

        return XGBClassifier(**params)
    if model_name == "logistic_regression":
        return LogisticRegression(**params)
    if model_name == "svm":
        return SVC(**params)
    raise ValueError(f"Modelo desconhecido: {model_name}")


def build_full_pipeline(model_name: str, k_features: int = K_FEATURES) -> Pipeline:
    """Pipeline completo (idêntico para os Pipelines A e B; só muda a matriz de entrada)."""
    return Pipeline(build_preprocessor(k_features) + [("clf", build_classifier(model_name))])


def save_hyperparameters(output_path: Path = MODELS_DIR / "hyperparameters.json") -> dict:
    """Registra pré-processamento e hiperparâmetros em JSON."""
    import sklearn
    import xgboost

    params = {
        "k_features": K_FEATURES,
        "k_selection_criterion": "f_classif (ANOVA F-value)",
        "variance_threshold": VARIANCE_THRESHOLD,
        "random_state": RANDOM_STATE,
        "class_balancing": {
            "random_forest": "class_weight='balanced'",
            "logistic_regression": "class_weight='balanced'",
            "svm": "class_weight='balanced'",
            "xgboost": "nenhum (sem class_weight / sample_weight)",
        },
        "models": _model_params(),
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
