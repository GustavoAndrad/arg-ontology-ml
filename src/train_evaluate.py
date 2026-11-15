"""
train_evaluate.py — Treinamento, validação cruzada e avaliação (Seções 4.8–4.11, 5.2–5.5).

Para cada modelo e cada pipeline (A: 73 features de sequência; B: A + 41 ontológicas = 114):
    * validação cruzada por grupos (StratifiedGroupKFold, 5 folds) no treino;
    * treino final no treino (~80%) e avaliação no teste (~20%).

A partição é feita por grupos de similaridade de sequência (``load_groups``,
limiar de 70%): nenhum grupo aparece no treino e no teste, nem em dois folds.
A escolha da partição de teste (semente e proporções) fica em
``results/particao_teste.json``.

Métricas: acurácia, precisão/revocação/F1 macro, acurácia balanceada e ROC-AUC
macro One-vs-Rest.

Classes declaradas (TCC2, ponto 2): precisão, revocação e F1 macro recebem
``labels`` com todas as classes. Sem isso, o scikit-learn faz a média só sobre as
classes que aparecem no fold, e uma classe ausente simplesmente some da média (era
o que acontecia no TCC1, ao contrário do que diz a Seção 5.5). Para que a média
seja sempre sobre as 5 classes, o código também confere, antes de treinar, que
cada fold de validação tem genes de todas elas (``check_class_counts_per_fold``) e
salva essas contagens em ``results/cv_contagem_classes.csv``.

Uso:
    python -m src.train_evaluate
"""

from __future__ import annotations

import json
import logging
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from src.experiment_config import (
    MIN_GENES_PER_FOLD,
    MODEL_NAMES,
    MODELS_DIR,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET,
    RANDOM_STATE,
    SIMILARITY_THRESHOLD,
    choose_cv_seed,
    choose_test_partition,
    cv_splitter,
    dataset_composition,
    encode_target,
    load_experiment_dataset,
    load_groups,
    load_integrated_dataset,
)
from src.pipeline_builder import build_full_pipeline

logger = logging.getLogger(__name__)
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy", "roc_auc_ovr"]
PREDICTIONS_DIR = RESULTS_DIR / "predicoes"


def load_feature_matrices() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Retorna (X_a, X_b, y, classes) alinhados com o dataset do experimento."""
    data = load_experiment_dataset()
    seq = pd.read_csv(PROCESSED_DIR / "features_sequencia.csv")
    onto = pd.read_csv(PROCESSED_DIR / "features_ontologicas.csv")
    for name, f in (("features_sequencia", seq), ("features_ontologicas", onto)):
        if not (f["ARO_Accession"].values == data["ARO_Accession"].values).all():
            raise ValueError(f"{name}.csv não está alinhado ao dataset; reexecute a extração.")
    X_a = seq.drop(columns=["ARO_Accession"])
    X_b = pd.concat([X_a, onto.drop(columns=["ARO_Accession"])], axis=1)  # concatenação direta (Seção 4.7)
    y, classes = encode_target(data[TARGET])
    logger.info("Pipeline A: %d features | Pipeline B: %d features | %d instâncias", X_a.shape[1], X_b.shape[1], len(y))
    return X_a, X_b, y, classes


def split_dataset(y: np.ndarray, groups: np.ndarray, save: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Partição treino/teste por grupos; índices e escolha salvos.  # SEED FIXA  # ANTI-LEAKAGE

    Salva ``data/processed/{train,test}_indices.npy`` e ``results/particao_teste.json``
    (semente escolhida, fração de cada classe no teste, número de grupos).
    """
    train_idx, test_idx, info = choose_test_partition(y, groups)
    if save:
        np.save(PROCESSED_DIR / "train_indices.npy", train_idx)
        np.save(PROCESSED_DIR / "test_indices.npy", test_idx)
        info["limiar_similaridade"] = SIMILARITY_THRESHOLD
        (RESULTS_DIR / "particao_teste.json").write_text(json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Partição por grupos: treino %d / teste %d genes (semente %d, fold %d; desvio máximo da fração por classe %.3f).",
        len(train_idx), len(test_idx), info["semente"], info["fold"], info["desvio_maximo_da_fracao_por_classe"],
    )
    return train_idx, test_idx


def compute_metrics(y_true, y_pred, proba, n_classes: int) -> dict[str, float]:
    """Métricas macro sobre as ``n_classes`` classes declaradas (ver docstring do módulo)."""
    labels = list(range(n_classes))
    try:
        auc = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
    except ValueError:  # classe ausente no fold de validação (impedido por check_class_counts_per_fold)
        auc = float("nan")
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "recall_macro": recall_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y_true, y_pred),
        "roc_auc_ovr": auc,
    }


def check_class_counts_per_fold(
    y_train: np.ndarray, classes: list[str], groups_train: np.ndarray | None = None, cv_seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """
    Conta os genes de cada classe em cada fold de validação e aplica o critério do ponto 2.

    Os folds dependem só de ``y_train``, dos grupos e da semente, então são os mesmos
    para todos os modelos e para os Pipelines A e B. Erro se alguma classe faltar num
    fold (o F1 dela não existiria); aviso se ficar abaixo de ``MIN_GENES_PER_FOLD``.
    """
    groups_train = np.arange(len(y_train)) if groups_train is None else groups_train
    rows = []
    for i, (_, val_idx) in enumerate(cv_splitter(cv_seed).split(np.zeros(len(y_train)), y_train, groups_train), start=1):
        counts = np.bincount(y_train[val_idx], minlength=len(classes))
        rows.append({"fold": i, **dict(zip(classes, map(int, counts)))})
    table = pd.DataFrame(rows)
    minimum = table[classes].min()
    if (minimum == 0).any():
        raise ValueError(f"Classe ausente em algum fold de validação: {minimum[minimum == 0].index.tolist()}")
    for c, m in minimum.items():
        log = logger.warning if m < MIN_GENES_PER_FOLD else logger.info
        log("Validação cruzada: classe %-32s mínimo de %d genes por fold.", c, m)
    return table


def run_cross_validation(
    model_name: str,
    X_train: np.ndarray,
    y_train: np.ndarray,
    n_classes: int,
    groups_train: np.ndarray,
    cv_seed: int = RANDOM_STATE,
) -> list[dict]:
    """CV por grupos (5 folds); transformadores ajustados só no treino de cada fold.  # ANTI-LEAKAGE  # SEED FIXA"""
    folds = []
    for i, (fit_idx, val_idx) in enumerate(cv_splitter(cv_seed).split(X_train, y_train, groups_train), start=1):
        pipe = build_full_pipeline(model_name).fit(X_train[fit_idx], y_train[fit_idx])
        m = compute_metrics(
            y_train[val_idx], pipe.predict(X_train[val_idx]), pipe.predict_proba(X_train[val_idx]), n_classes
        )
        m["fold"] = i
        folds.append(m)
    return folds


def train_final_model(model_name: str, pipeline_key: str, X_train: np.ndarray, y_train: np.ndarray):
    pipe = build_full_pipeline(model_name).fit(X_train, y_train)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipe, MODELS_DIR / f"pipeline_{pipeline_key}_{model_name}.joblib")
    return pipe


def evaluate_on_test(pipe, X_test: np.ndarray, y_test: np.ndarray, n_classes: int) -> tuple[dict, dict]:
    pred = pipe.predict(X_test)
    proba = pipe.predict_proba(X_test)
    metrics = compute_metrics(y_test, pred, proba, n_classes)
    extra = {
        "y_pred": pred,
        "proba": proba,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=list(range(n_classes))),
        "f1_per_class": f1_score(y_test, pred, average=None, labels=list(range(n_classes)), zero_division=0),
    }
    return metrics, extra


def save_metrics(rows: list[dict], pipeline_key: str) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    df.to_csv(RESULTS_DIR / f"metrics_pipeline_{pipeline_key}.csv", index=False)
    return df


def run_experiment(model_names: list[str] = MODEL_NAMES) -> None:
    start = time.time()
    X_a, X_b, y, classes = load_feature_matrices()
    groups = load_groups()
    train_idx, test_idx = split_dataset(y, groups)
    cv_seed, cv_info = choose_cv_seed(y[train_idx], groups[train_idx])
    part_path = RESULTS_DIR / "particao_teste.json"
    part = json.loads(part_path.read_text(encoding="utf-8"))
    part.update(cv_info)
    part_path.write_text(json.dumps(part, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Semente da validação cruzada: %d (mínimo de %d genes de uma classe num fold).", cv_seed, cv_info["minimo_genes_classe_fold"])
    n_classes = len(classes)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    check_class_counts_per_fold(y[train_idx], classes, groups[train_idx], cv_seed).to_csv(RESULTS_DIR / "cv_contagem_classes.csv", index=False)
    # Tabela 4.1 / Seção 4.3.3: todos os mecanismos do dataset integrado e o destino de cada um.
    dataset_composition(load_integrated_dataset()).to_csv(RESULTS_DIR / "composicao_dataset.csv", index=False)

    cv_rows, per_class_rows = [], []
    for key, X_df in (("a", X_a), ("b", X_b)):
        X = X_df.to_numpy(dtype=float)
        metric_rows = []
        for name in model_names:
            t0 = time.time()
            folds = run_cross_validation(name, X[train_idx], y[train_idx], n_classes, groups[train_idx], cv_seed)
            pipe = train_final_model(name, key, X[train_idx], y[train_idx])
            test_metrics, extra = evaluate_on_test(pipe, X[test_idx], y[test_idx], n_classes)
            np.savez(
                PREDICTIONS_DIR / f"pipeline_{key}_{name}.npz",
                y_true=y[test_idx],
                y_pred=extra["y_pred"],
                proba=extra["proba"],
                confusion_matrix=extra["confusion_matrix"],
            )
            for f in folds:
                cv_rows.append({"pipeline": key, "modelo": name, **f})
            for c, v in zip(classes, extra["f1_per_class"]):
                per_class_rows.append(
                    {
                        "pipeline": key,
                        "modelo": name,
                        "classe": c,
                        "f1_teste": v,
                        "n_teste": int((y[test_idx] == classes.index(c)).sum()),
                    }
                )
            for metric in METRICS:
                vals = np.array([f[metric] for f in folds], dtype=float)
                metric_rows.append(
                    {
                        "modelo": name,
                        "metrica": metric,
                        "valor_teste": test_metrics[metric],
                        "media_cv": np.nanmean(vals),
                        "std_cv": np.nanstd(vals),
                        "folds_validos_cv": int(np.isfinite(vals).sum()),
                    }
                )
            logger.info(
                "Pipeline %s | %-20s F1 teste=%.4f | F1 CV=%.4f ± %.4f | %.1fs",
                key.upper(),
                name,
                test_metrics["f1_macro"],
                np.mean([f["f1_macro"] for f in folds]),
                np.std([f["f1_macro"] for f in folds]),
                time.time() - t0,
            )
        save_metrics(metric_rows, key)
    pd.DataFrame(cv_rows).to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    pd.DataFrame(per_class_rows).to_csv(RESULTS_DIR / "f1_por_classe_teste.csv", index=False)
    logger.info("Experimento concluído em %.1f min.", (time.time() - start) / 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_experiment()
