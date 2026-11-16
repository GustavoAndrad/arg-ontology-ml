"""
sensitivity_balancing.py — Sensibilidade do resultado à estratégia de balanceamento.

Pergunta: a conclusão "B > A" depende de como o desbalanceamento das classes é
tratado? Responde aos apontamentos J5 (Jéssica: undersampling, oversampling,
híbridos) e D6 (Diego: abordagem hierárquica) — TCC2, ponto 6.

O experimento principal usa **peso por classe** nos 4 modelos ("peso" abaixo).
As outras estratégias são comparadas a ele, com tudo o mais igual: mesmas 5
classes, mesma partição por grupos, mesmos folds, mesmos hiperparâmetros.

Estratégias:
    nenhum        sem peso e sem reamostragem (referência: o que o peso corrige)
    peso          experimento principal: class_weight / sample_weight 'balanced'
    under         undersampling aleatório só da classe majoritária (inativação),
                  até o tamanho da segunda maior classe do treino do fold
    over          oversampling aleatório de todas as classes até o tamanho da maior
    smote         SMOTE até o tamanho da maior (Pipeline B: SMOTENC; ver abaixo)
    smote_enn     SMOTE (SMOTENC no B) seguido de limpeza por Edited Nearest Neighbours
    hierarquica   nível 1: inativação × resto; nível 2: qual das outras 4 classes.
                  Os dois níveis usam o mesmo classificador, com peso por classe.

Decisões (registradas em TCC2_resultados_para_o_texto.md, pontos 6 e 13):
    * Nas estratégias de reamostragem, os modelos ficam **sem** peso: a
      reamostragem substitui o peso, em vez de somar a ele.
    * A reamostragem roda **só no treino de cada fold** (Pipeline do
      imbalanced-learn: o amostrador só age no ``fit``). Genes sintéticos ou
      repetidos nunca entram na validação nem no teste.
    * Ordem: MinMaxScaler → amostrador → classificador. O SMOTE usa distâncias, e
      sem a escala o comprimento da proteína dominaria a vizinhança.
    * Undersampling "até a segunda maior classe": regra simples e explicável que
      tira o desequilíbrio dominante (inativação ≈ 19 × a segunda maior) sem
      descartar genes das classes pequenas. A prévia usou 600, número sem
      justificativa registrada; por isso não foi mantido.
    * Pipeline B com SMOTENC: as 41 colunas ontológicas (38 flags 0/1 e 3
      estruturais inteiras) são tratadas como categóricas, para que um gene
      sintético só tenha valores que existem na ontologia (o SMOTE comum criaria
      flags como 0,4). No Pipeline A, todas as colunas são contínuas: SMOTE comum.
    * Hierárquica: prevê inativação se o nível 1 der probabilidade ≥ 0,5; senão, a
      classe mais provável do nível 2. A probabilidade conjunta é
      P(inativação) = p1 e P(outra c) = (1 − p1) · p2(c), usada só na ROC-AUC.

É uma análise **descritiva** (sem teste estatístico), como definido no ponto 5.
Os folds são passados de fora (``splits``) para que a versão final use a
validação cruzada repetida do ponto 5 sem mudar este módulo.

Saídas:
    results/sensibilidade_balanceamento.csv            — estratégia × modelo
    results/sensibilidade_balanceamento_folds.csv      — F1-macro de cada fold
    results/sensibilidade_balanceamento_por_classe.csv — F1 por classe no teste

Uso (cerca de 25 min):
    python -m src.sensitivity_balancing
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd
from imblearn.combine import SMOTEENN
from imblearn.over_sampling import SMOTE, SMOTENC, RandomOverSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.under_sampling import RandomUnderSampler
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.metrics import f1_score
from sklearn.preprocessing import MinMaxScaler

from src.experiment_config import (
    MODEL_NAMES,
    RANDOM_STATE,
    RESULTS_DIR,
    choose_cv_seed,
    cv_splitter,
    load_groups,
    split_indices,
)
from src.pipeline_builder import _model_params, build_classifier
from src.train_evaluate import compute_metrics, load_feature_matrices

logger = logging.getLogger(__name__)

STRATEGIES: list[str] = ["nenhum", "peso", "under", "over", "smote", "smote_enn", "hierarquica"]
MAJORITY_CLASS = "antibiotic inactivation"


def unweighted_classifier(model_name: str) -> BaseEstimator:
    """Classificador com os mesmos hiperparâmetros do experimento, mas sem peso por classe."""
    if model_name == "xgboost":
        from xgboost import XGBClassifier

        return XGBClassifier(**_model_params()["xgboost"])
    return build_classifier(model_name).set_params(class_weight=None)


def undersample_to_second_largest(y: np.ndarray) -> dict[int, int]:
    """sampling_strategy do RandomUnderSampler: só a maior classe desce ao tamanho da segunda maior."""
    counts = np.bincount(y)
    order = np.argsort(counts)[::-1]
    return {int(order[0]): int(counts[order[1]])}


class HierarchicalClassifier(ClassifierMixin, BaseEstimator):
    """
    Abordagem hierárquica (apontamento D6): inativação × resto, depois as outras classes.

    ``base`` é um classificador já configurado (com peso por classe); cada nível
    recebe um clone. Os rótulos do nível 2 são renumerados 0..k-1 (o XGBoost exige).
    """

    def __init__(self, base=None, majority_label: int = 1):
        self.base = base
        self.majority_label = majority_label

    def fit(self, X, y):
        y = np.asarray(y)
        self.classes_ = np.unique(y)
        self.level1_ = clone(self.base).fit(X, (y == self.majority_label).astype(int))
        rest = y != self.majority_label
        self.rest_labels_ = np.unique(y[rest])
        remap = {c: i for i, c in enumerate(self.rest_labels_)}
        self.level2_ = clone(self.base).fit(X[rest], np.array([remap[c] for c in y[rest]]))
        return self

    def predict_proba(self, X):
        p1 = self.level1_.predict_proba(X)[:, 1]
        p2 = self.level2_.predict_proba(X)
        proba = np.zeros((len(X), len(self.classes_)))
        proba[:, np.searchsorted(self.classes_, self.majority_label)] = p1
        for j, c in enumerate(self.rest_labels_):
            proba[:, np.searchsorted(self.classes_, c)] = (1 - p1) * p2[:, j]
        return proba

    def predict(self, X):
        p1 = self.level1_.predict_proba(X)[:, 1]
        rest = self.rest_labels_[self.level2_.predict_proba(X).argmax(axis=1)]
        return np.where(p1 >= 0.5, self.majority_label, rest)


def build_strategy_pipeline(
    strategy: str, model_name: str, y_fit: np.ndarray, categorical: list[int] | None, majority_label: int
):
    """Pipeline (scaler → [amostrador] → classificador) da estratégia, para um fold."""
    if strategy == "peso":
        clf = build_classifier(model_name)
        return ImbPipeline([("scaler", MinMaxScaler()), ("clf", clf)])
    if strategy == "nenhum":
        return ImbPipeline([("scaler", MinMaxScaler()), ("clf", unweighted_classifier(model_name))])
    if strategy == "hierarquica":
        clf = HierarchicalClassifier(build_classifier(model_name), majority_label)
        return ImbPipeline([("scaler", MinMaxScaler()), ("clf", clf)])

    smote = (
        SMOTENC(categorical_features=categorical, random_state=RANDOM_STATE)
        if categorical
        else SMOTE(random_state=RANDOM_STATE)
    )
    samplers = {
        "under": RandomUnderSampler(sampling_strategy=undersample_to_second_largest(y_fit), random_state=RANDOM_STATE),
        "over": RandomOverSampler(random_state=RANDOM_STATE),
        "smote": smote,
        "smote_enn": SMOTEENN(smote=smote, random_state=RANDOM_STATE),
    }
    if strategy not in samplers:
        raise ValueError(f"Estratégia desconhecida: {strategy}")
    return ImbPipeline(
        [("scaler", MinMaxScaler()), ("sampler", samplers[strategy]), ("clf", unweighted_classifier(model_name))]
    )


def _resampled_size(pipe, X, y) -> int:
    """Tamanho do treino depois da reamostragem (igual ao original sem amostrador)."""
    if "sampler" not in pipe.named_steps:
        return len(y)
    Xs = pipe.named_steps["scaler"].transform(X)
    return len(pipe.named_steps["sampler"].fit_resample(Xs, y)[1])


def evaluate_strategy(
    strategy, model, X, y, classes, train_idx, test_idx, splits, categorical
) -> tuple[dict, list[dict], list[dict]]:
    n_classes = len(classes)
    majority = classes.index(MAJORITY_CLASS)
    Xtr, ytr = X[train_idx], y[train_idx]
    f1s, folds = [], []
    for i, (fit, val) in enumerate(splits, start=1):
        p = build_strategy_pipeline(strategy, model, ytr[fit], categorical, majority).fit(Xtr[fit], ytr[fit])
        f1 = compute_metrics(ytr[val], p.predict(Xtr[val]), p.predict_proba(Xtr[val]), n_classes)["f1_macro"]
        f1s.append(f1)
        folds.append({"fold": i, "f1_macro": f1})
    final = build_strategy_pipeline(strategy, model, ytr, categorical, majority).fit(Xtr, ytr)
    pred = final.predict(X[test_idx])
    test = compute_metrics(y[test_idx], pred, final.predict_proba(X[test_idx]), n_classes)
    per_class = f1_score(y[test_idx], pred, labels=list(range(n_classes)), average=None, zero_division=0)
    summary = {
        "f1_cv_media": float(np.mean(f1s)),
        "f1_cv_std": float(np.std(f1s)),
        "f1_teste": float(test["f1_macro"]),
        "roc_auc_teste": float(test["roc_auc_ovr"]),
        "n_treino_final": _resampled_size(final, Xtr, ytr),
    }
    return summary, folds, [{"classe": c, "f1_teste": float(v)} for c, v in zip(classes, per_class)]


def run(strategies: list[str] = STRATEGIES, model_names: list[str] = MODEL_NAMES, splits=None) -> pd.DataFrame:
    X_a, X_b, y, classes = load_feature_matrices()
    groups = load_groups()
    train_idx, test_idx = split_indices(y, groups)  # mesma partição do experimento  # SEED FIXA
    if splits is None:  # mesmos 5 folds do experimento principal
        cv_seed, _ = choose_cv_seed(y[train_idx], groups[train_idx])
        splits = list(cv_splitter(cv_seed).split(np.zeros(len(train_idx)), y[train_idx], groups[train_idx]))
    n_seq = X_a.shape[1]
    rows, fold_rows, class_rows = [], [], []
    for strategy in strategies:
        t0 = time.time()
        for model in model_names:
            row = {"estrategia": strategy, "modelo": model}
            for key, X_df in (("a", X_a), ("b", X_b)):
                categorical = list(range(n_seq, X_b.shape[1])) if key == "b" else None
                summary, folds, per_class = evaluate_strategy(
                    strategy, model, X_df.to_numpy(dtype=float), y, classes, train_idx, test_idx, splits, categorical
                )
                row.update({f"{k}_{key}": v for k, v in summary.items()})
                fold_rows += [{"estrategia": strategy, "modelo": model, "pipeline": key, **f} for f in folds]
                class_rows += [{"estrategia": strategy, "modelo": model, "pipeline": key, **c} for c in per_class]
            row["ganho_teste"] = row["f1_teste_b"] - row["f1_teste_a"]
            row["ganho_cv"] = row["f1_cv_media_b"] - row["f1_cv_media_a"]
            rows.append(row)
            logger.info(
                "%-11s %-20s teste A=%.4f B=%.4f | CV A=%.4f B=%.4f | treino A=%d B=%d",
                strategy, model, row["f1_teste_a"], row["f1_teste_b"], row["f1_cv_media_a"], row["f1_cv_media_b"],
                row["n_treino_final_a"], row["n_treino_final_b"],
            )
        logger.info("Estratégia %s concluída em %.1f min.", strategy, (time.time() - t0) / 60)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "sensibilidade_balanceamento.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "sensibilidade_balanceamento_folds.csv", index=False)
    pd.DataFrame(class_rows).to_csv(RESULTS_DIR / "sensibilidade_balanceamento_por_classe.csv", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
