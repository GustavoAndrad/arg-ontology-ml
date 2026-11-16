"""
train_evaluate.py — Treinamento, validação cruzada e avaliação (Seções 4.8–4.11, 5.2–5.5).

Para cada modelo e cada pipeline (A: 73 features de sequência; B: A + 41 ontológicas = 114):
    * validação cruzada por grupos repetida 10 × 5 (StratifiedGroupKFold) no treino:
      50 estimativas por modelo, pareadas entre A e B (mesmas divisões);
    * treino final no treino (~80%) e avaliação no teste (~20%).

Validação cruzada repetida (TCC2, ponto 5, decisão 2): base do teste t corrigido
(``src.statistical_tests``). As 10 sementes vêm de ``choose_cv_seeds`` (as
primeiras que deixam >= 5 genes de cada classe em cada fold) e ficam em
``results/particao_teste.json``.

A partição é feita por grupos de similaridade de sequência (``load_groups``,
limiar de 70%): nenhum grupo aparece no treino e no teste, nem em dois folds.
A escolha da partição de teste (semente e proporções) fica em
``results/particao_teste.json``.

Métricas: acurácia, precisão/revocação/F1 macro, acurácia balanceada, ROC-AUC
macro One-vs-Rest e PR-AUC macro (TCC2, ponto 10). Também por classe: precisão,
revocação, F1, ROC-AUC e PR-AUC (uma classe contra as outras), no teste e em cada
fold da validação cruzada.

PR-AUC = precisão média (``average_precision_score``): soma das precisões em cada
limiar, ponderada pelo ganho de revocação, sem interpolar a curva. É o cálculo
padrão do scikit-learn. Ao lado da ROC-AUC, mostra o desempenho nas classes
pequenas: com uma classe muito grande, a taxa de falsos positivos da ROC fica baixa
mesmo quando muitos alarmes são falsos.

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
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
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
    SIMILARITY_THRESHOLD,
    choose_cv_seeds,
    choose_test_partition,
    repeated_cv_splits,
    dataset_composition,
    encode_target,
    load_experiment_dataset,
    load_groups,
    load_integrated_dataset,
)
from src.pipeline_builder import build_full_pipeline

logger = logging.getLogger(__name__)
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy", "roc_auc_ovr", "pr_auc_macro"]
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


def pr_auc_per_class(y_true, proba, n_classes: int) -> np.ndarray:
    """PR-AUC (precisão média) de cada classe contra as outras; NaN se a classe não aparece."""
    y_true = np.asarray(y_true)
    out = np.full(n_classes, np.nan)
    for k in range(n_classes):
        if (y_true == k).any():
            out[k] = average_precision_score(y_true == k, proba[:, k])
    return out


def roc_auc_per_class(y_true, proba, n_classes: int) -> np.ndarray:
    """ROC-AUC de cada classe contra as outras; NaN se a classe não aparece (ou é a única)."""
    y_true = np.asarray(y_true)
    out = np.full(n_classes, np.nan)
    for k in range(n_classes):
        if (y_true == k).any() and not (y_true == k).all():
            out[k] = roc_auc_score(y_true == k, proba[:, k])
    return out


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
        "pr_auc_macro": float(np.mean(pr_auc_per_class(y_true, proba, n_classes))),
    }


def per_class_metrics(y_true, y_pred, proba, classes: list[str]) -> list[dict]:
    """Precisão, revocação, F1, ROC-AUC e PR-AUC de cada classe (uma linha por classe)."""
    n = len(classes)
    prec, rec, f1, support = precision_recall_fscore_support(y_true, y_pred, labels=list(range(n)), zero_division=0)
    roc, pr = roc_auc_per_class(y_true, proba, n), pr_auc_per_class(y_true, proba, n)
    return [
        {"classe": c, "precisao": prec[k], "revocacao": rec[k], "f1": f1[k], "roc_auc": roc[k], "pr_auc": pr[k], "n": int(support[k])}
        for k, c in enumerate(classes)
    ]


def check_class_counts_per_fold(y_train: np.ndarray, classes: list[str], splits: list[tuple]) -> pd.DataFrame:
    """
    Conta os genes de cada classe em cada fold de validação e aplica o critério do ponto 2.

    ``splits`` vem de ``repeated_cv_splits``: as mesmas divisões valem para todos os
    modelos e para os Pipelines A e B. Erro se alguma classe faltar num fold (o F1 dela
    não existiria); aviso se ficar abaixo de ``MIN_GENES_PER_FOLD``.
    """
    rows = []
    for rep, fold, _, val_idx in splits:
        counts = np.bincount(y_train[val_idx], minlength=len(classes))
        rows.append({"repeticao": rep, "fold": fold, **dict(zip(classes, map(int, counts)))})
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
    classes: list[str],
    splits: list[tuple],
) -> tuple[list[dict], list[dict]]:
    """
    CV por grupos repetida; transformadores ajustados só no treino de cada fold.  # ANTI-LEAKAGE  # SEED FIXA

    Retorna (métricas macro por fold, métricas por classe por fold), com a repetição,
    o fold e os tamanhos de ajuste e de validação (usados no teste t corrigido).
    """
    n_classes = len(classes)
    folds, per_class = [], []
    for rep, fold, fit_idx, val_idx in splits:
        pipe = build_full_pipeline(model_name).fit(X_train[fit_idx], y_train[fit_idx])
        pred, proba = pipe.predict(X_train[val_idx]), pipe.predict_proba(X_train[val_idx])
        m = compute_metrics(y_train[val_idx], pred, proba, n_classes)
        folds.append({"repeticao": rep, "fold": fold, "n_ajuste": len(fit_idx), "n_validacao": len(val_idx), **m})
        per_class += [{"repeticao": rep, "fold": fold, **r} for r in per_class_metrics(y_train[val_idx], pred, proba, classes)]
    return folds, per_class


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
    cv_seeds, cv_info = choose_cv_seeds(y[train_idx], groups[train_idx])
    part_path = RESULTS_DIR / "particao_teste.json"
    part = json.loads(part_path.read_text(encoding="utf-8"))
    part.update(cv_info)
    part_path.write_text(json.dumps(part, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(
        "Sementes da validação cruzada repetida: %s (mínimo de genes de uma classe num fold, por semente: %s).",
        cv_seeds, cv_info["minimo_genes_classe_fold_por_semente"],
    )
    splits = repeated_cv_splits(y[train_idx], groups[train_idx], cv_seeds)
    n_classes = len(classes)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTIONS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")
    check_class_counts_per_fold(y[train_idx], classes, splits).to_csv(RESULTS_DIR / "cv_contagem_classes.csv", index=False)
    # Tabela 4.1 / Seção 4.3.3: todos os mecanismos do dataset integrado e o destino de cada um.
    dataset_composition(load_integrated_dataset()).to_csv(RESULTS_DIR / "composicao_dataset.csv", index=False)

    cv_rows, cv_class_rows, test_class_rows = [], [], []
    for key, X_df in (("a", X_a), ("b", X_b)):
        X = X_df.to_numpy(dtype=float)
        metric_rows = []
        for name in model_names:
            t0 = time.time()
            folds, folds_per_class = run_cross_validation(name, X[train_idx], y[train_idx], classes, splits)
            pipe = train_final_model(name, key, X[train_idx], y[train_idx])
            test_metrics, extra = evaluate_on_test(pipe, X[test_idx], y[test_idx], n_classes)
            np.savez(
                PREDICTIONS_DIR / f"pipeline_{key}_{name}.npz",
                y_true=y[test_idx],
                y_pred=extra["y_pred"],
                proba=extra["proba"],
                confusion_matrix=extra["confusion_matrix"],
            )
            cv_rows += [{"pipeline": key, "modelo": name, **f} for f in folds]
            cv_class_rows += [{"pipeline": key, "modelo": name, **r} for r in folds_per_class]
            test_class_rows += [
                {"pipeline": key, "modelo": name, **r}
                for r in per_class_metrics(y[test_idx], extra["y_pred"], extra["proba"], classes)
            ]
            for metric in METRICS:
                vals = np.array([f[metric] for f in folds], dtype=float)
                metric_rows.append(
                    {
                        "modelo": name,
                        "metrica": metric,
                        "valor_teste": test_metrics[metric],
                        "media_cv": np.nanmean(vals),
                        # Desvio amostral (n − 1), o mesmo usado no teste t corrigido.
                        "std_cv": np.nanstd(vals, ddof=1),
                        "folds_validos_cv": int(np.isfinite(vals).sum()),
                    }
                )
            logger.info(
                "Pipeline %s | %-20s F1 teste=%.4f | F1 CV=%.4f ± %.4f (%d folds) | %.1fs",
                key.upper(),
                name,
                test_metrics["f1_macro"],
                np.mean([f["f1_macro"] for f in folds]),
                np.std([f["f1_macro"] for f in folds], ddof=1),
                len(folds),
                time.time() - t0,
            )
        save_metrics(metric_rows, key)
    pd.DataFrame(cv_rows).to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    pd.DataFrame(cv_class_rows).to_csv(RESULTS_DIR / "cv_por_classe.csv", index=False)
    pd.DataFrame(test_class_rows).to_csv(RESULTS_DIR / "metricas_por_classe_teste.csv", index=False)
    logger.info("Experimento concluído em %.1f min.", (time.time() - start) / 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_experiment()
