"""
sensitivity_tuning.py — Sensibilidade do resultado ao ajuste de hiperparâmetros.

Pergunta: a conclusão "B > A" depende de os modelos usarem os hiperparâmetros
padrão? Responde ao apontamento D9 (Diego) e completa o ponto 13: o experimento
principal usa hiperparâmetros fixos e iguais para A e B (o objetivo é comparar
conjuntos de atributos nas mesmas condições, não achar o melhor modelo), e o
ajuste entra só aqui, de forma descritiva.

Desenho (decisão 2 do ponto 13; Hastie et al. 2009, seção 7.10 — a escolha de
hiperparâmetros por validação cruzada, feita só dentro do treino):
    * Validação cruzada aninhada. Folds externos = os 5 folds da 1ª repetição da
      validação cruzada repetida do experimento (a decisão do ponto 13 pede uma
      repetição de 5 folds; a busca dentro de cada fold é o que custa). Em cada fold externo, busca em grade (``GridSearchCV``) com 3
      folds internos por grupos (``StratifiedGroupKFold``), só no treino do fold;
      o melhor conjunto é avaliado no fold externo de validação.
    * Teste: a mesma busca no treino inteiro (3 folds internos por grupos); o
      modelo escolhido é avaliado uma vez no conjunto de teste.
    * Mesmo espaço de busca para A e B; peso por classe em todas as combinações
      (o ``BalancedXGBClassifier`` mantém o peso dentro do ``GridSearchCV``).
    * Critério da busca: F1-macro sobre as 5 classes declaradas.

Espaço de busca (``PARAM_GRIDS``): escolhido por nós, pequeno por custo, sempre
**incluindo o valor padrão** usado no experimento (marcado com *):
    Random Forest  n_estimators {100*, 300}; max_features {"sqrt"*, 0.3};
                   min_samples_leaf {1*, 3}                                   (8)
    XGBoost        n_estimators {100*, 300}; learning_rate {0.1, 0.3*};
                   max_depth {3, 6*}                                          (8)
                   (0.3 e 6 são os padrão efetivos, lidos do modelo treinado)
    Reg. Logística C {0.1, 1*, 10, 100}                                       (4)
    SVM (RBF)      C {1*, 10, 100}; gamma {"scale"*, 0.01, 0.1}              (9)
Não há referência da literatura para esses valores; o texto deve dizer que foram
definidos pelos autores em torno do padrão.

É uma análise **descritiva** (sem teste estatístico), como definido no ponto 5.

Saídas:
    results/sensibilidade_ajuste.csv          — modelo × pipeline: padrão vs ajustado
    results/sensibilidade_ajuste_folds.csv    — F1 e melhores parâmetros por fold externo

Uso (cerca de 20 min):
    python -m src.sensitivity_tuning
"""

from __future__ import annotations

import json
import logging
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, make_scorer
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold

from src.experiment_config import (
    MODEL_NAMES,
    RANDOM_STATE,
    RESULTS_DIR,
    choose_cv_seeds,
    load_groups,
    repeated_cv_splits,
    split_indices,
)
from src.pipeline_builder import build_full_pipeline
from src.train_evaluate import compute_metrics, load_feature_matrices

logger = logging.getLogger(__name__)

INNER_SPLITS = 3

PARAM_GRIDS: dict[str, dict[str, list]] = {
    "random_forest": {"clf__n_estimators": [100, 300], "clf__max_features": ["sqrt", 0.3], "clf__min_samples_leaf": [1, 3]},
    "xgboost": {"clf__n_estimators": [100, 300], "clf__learning_rate": [0.1, 0.3], "clf__max_depth": [3, 6]},
    "logistic_regression": {"clf__C": [0.1, 1.0, 10.0, 100.0]},
    "svm": {"clf__C": [1.0, 10.0, 100.0], "clf__gamma": ["scale", 0.01, 0.1]},
}

# A combinação da grade que corresponde ao modelo do experimento (para contar em
# quantos folds a busca escolheu o próprio padrão). No XGBoost, o experimento deixa
# learning_rate e max_depth em None, que valem 0,3 e 6 (lidos do modelo treinado).
DEFAULT_IN_GRID: dict[str, dict] = {
    "random_forest": {"clf__max_features": "sqrt", "clf__min_samples_leaf": 1, "clf__n_estimators": 100},
    "xgboost": {"clf__learning_rate": 0.3, "clf__max_depth": 6, "clf__n_estimators": 100},
    "logistic_regression": {"clf__C": 1.0},
    "svm": {"clf__C": 1.0, "clf__gamma": "scale"},
}

# RF e XGBoost já usam todos os núcleos (n_jobs=-1); RL e SVM são de um núcleo só,
# então a busca deles roda as combinações em paralelo.
SEARCH_N_JOBS: dict[str, int | None] = {"random_forest": None, "xgboost": None, "logistic_regression": -1, "svm": -1}


def f1_macro_scorer(n_classes: int):
    """F1-macro sobre todas as classes declaradas (classe ausente conta como 0)."""
    return make_scorer(f1_score, average="macro", labels=list(range(n_classes)), zero_division=0)


def grid_search(model: str, n_classes: int) -> GridSearchCV:
    return GridSearchCV(
        build_full_pipeline(model),
        PARAM_GRIDS[model],
        scoring=f1_macro_scorer(n_classes),
        cv=StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=RANDOM_STATE),  # SEED FIXA
        n_jobs=SEARCH_N_JOBS[model],
        refit=True,
        error_score="raise",
    )


def _short(params: dict) -> str:
    return json.dumps({k.replace("clf__", ""): v for k, v in params.items()}, ensure_ascii=False, sort_keys=True)


def evaluate_model(model, X, y, n_classes, train_idx, test_idx, groups, splits) -> tuple[dict, list[dict]]:
    Xtr, ytr, gtr = X[train_idx], y[train_idx], groups[train_idx]
    folds = []
    for _, i, fit, val in splits:
        default = build_full_pipeline(model).fit(Xtr[fit], ytr[fit])
        search = grid_search(model, n_classes).fit(Xtr[fit], ytr[fit], groups=gtr[fit])
        f1_default = compute_metrics(ytr[val], default.predict(Xtr[val]), default.predict_proba(Xtr[val]), n_classes)["f1_macro"]
        best = search.best_estimator_
        f1_tuned = compute_metrics(ytr[val], best.predict(Xtr[val]), best.predict_proba(Xtr[val]), n_classes)["f1_macro"]
        folds.append(
            {"fold": i, "f1_padrao": f1_default, "f1_ajustado": f1_tuned,
             "f1_interno_melhor": float(search.best_score_), "melhores_parametros": _short(search.best_params_)}
        )
    default = build_full_pipeline(model).fit(Xtr, ytr)
    search = grid_search(model, n_classes).fit(Xtr, ytr, groups=gtr)
    test_default = compute_metrics(y[test_idx], default.predict(X[test_idx]), default.predict_proba(X[test_idx]), n_classes)
    best = search.best_estimator_
    test_tuned = compute_metrics(y[test_idx], best.predict(X[test_idx]), best.predict_proba(X[test_idx]), n_classes)
    params_per_fold = [f["melhores_parametros"] for f in folds]
    summary = {
        "f1_cv_padrao": float(np.mean([f["f1_padrao"] for f in folds])),
        "f1_cv_ajustado": float(np.mean([f["f1_ajustado"] for f in folds])),
        "f1_cv_std_padrao": float(np.std([f["f1_padrao"] for f in folds], ddof=1)),
        "f1_cv_std_ajustado": float(np.std([f["f1_ajustado"] for f in folds], ddof=1)),
        "f1_teste_padrao": float(test_default["f1_macro"]),
        "f1_teste_ajustado": float(test_tuned["f1_macro"]),
        "parametros_teste": _short(search.best_params_),
        "folds_com_o_padrao_escolhido": sum(p == _short(DEFAULT_IN_GRID[model]) for p in params_per_fold),
        "conjuntos_distintos_nos_folds": len(set(params_per_fold)),
    }
    return summary, folds


def run(model_names: list[str] = MODEL_NAMES, splits=None) -> pd.DataFrame:
    X_a, X_b, y, classes = load_feature_matrices()
    n_classes = len(classes)
    groups = load_groups()
    train_idx, test_idx = split_indices(y, groups)  # mesma partição do experimento  # SEED FIXA
    if splits is None:  # 1ª repetição (5 folds) da validação cruzada repetida do experimento
        cv_seeds, _ = choose_cv_seeds(y[train_idx], groups[train_idx])
        splits = repeated_cv_splits(y[train_idx], groups[train_idx], cv_seeds[:1])
    rows, fold_rows = [], []
    for model in model_names:
        for key, X_df in (("a", X_a), ("b", X_b)):
            t0 = time.time()
            summary, folds = evaluate_model(model, X_df.to_numpy(dtype=float), y, n_classes, train_idx, test_idx, groups, splits)
            rows.append({"modelo": model, "pipeline": key, **summary})
            fold_rows += [{"modelo": model, "pipeline": key, **f} for f in folds]
            logger.info(
                "%-20s %s | CV padrão=%.4f ajustado=%.4f | teste padrão=%.4f ajustado=%.4f | %s | %.1f min",
                model, key.upper(), summary["f1_cv_padrao"], summary["f1_cv_ajustado"],
                summary["f1_teste_padrao"], summary["f1_teste_ajustado"], summary["parametros_teste"], (time.time() - t0) / 60,
            )
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "sensibilidade_ajuste.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "sensibilidade_ajuste_folds.csv", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
