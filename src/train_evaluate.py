"""
train_evaluate.py — Treinamento, validação cruzada e avaliação (Seções 4.8–4.11, 5.2–5.5).

Para cada modelo e cada pipeline (A: 73 features de sequência; B: A + 41 ontológicas = 114):
    * validação cruzada por grupos repetida 10 × 5 (StratifiedGroupKFold) no treino:
      50 estimativas por modelo, pareadas entre A e B (mesmas divisões);
    * treino final no treino (~80%) e avaliação no teste (~20%).

Cenários de avaliação do Pipeline B (TCC2, ponto 4; ver ``src.nearest_neighbor``):
o modelo B é treinado uma vez, como sempre; o que muda é a origem das 41 features
ontológicas na hora de prever.
    * ``b`` (anotação curada): da ficha do próprio gene — o melhor caso;
    * ``b_gene_novo``: da ficha do gene mais parecido do treino daquela divisão
      (fold ou teste final), como seria com um gene ainda não anotado; gene sem
      nenhum parecido recebe a predição do Pipeline A (regra P4-1);
    * ``b_gene_novo_zeros``: igual, mas sem parecido = ontologia zerada (regra
      original de 11/11; só sensibilidade descritiva);
    * ``homologia``: referência sem ML, o mecanismo do gene mais parecido;
    * ``b_gene_novo_piso40`` e ``homologia_piso40``: os mesmos, com o vizinho achado na
      busca do ponto 3 (piso de 40% de identidade) — sensibilidade ao piso da busca.
Os modelos A e B, seus dados e sementes não mudam: as linhas ``a`` e ``b`` são as do
experimento anterior (controle). O gene mais parecido de cada gene de teste fica em
``results/vizinho_mais_parecido_teste.csv``.

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
from src.nearest_neighbor import FLOOR_40_HITS, NO_NEIGHBOR, NeighborIndex, homology_prediction, predict_new_gene
from src.pipeline_builder import build_full_pipeline

logger = logging.getLogger(__name__)
METRICS = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "balanced_accuracy", "roc_auc_ovr", "pr_auc_macro"]
PREDICTIONS_DIR = RESULTS_DIR / "predicoes"
# Cenários avaliados (ver scenario_predictions). "homologia" não depende do modelo.
SCENARIO_KEYS: list[str] = [
    "a", "b", "b_gene_novo", "b_gene_novo_zeros", "homologia", "b_gene_novo_piso40", "homologia_piso40",
]
HOMOLOGY_KEYS: set[str] = {"homologia", "homologia_piso40"}  # sem modelo: guardadas uma vez só


def ontology_columns(X_a: pd.DataFrame, X_b: pd.DataFrame) -> np.ndarray:
    """Posições das 41 features ontológicas em X_b (vêm depois das de sequência)."""
    if list(X_b.columns[: X_a.shape[1]]) != list(X_a.columns):
        raise ValueError("X_b deve começar pelas colunas de X_a.")
    return np.arange(X_a.shape[1], X_b.shape[1])


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


def scenario_predictions(
    pipe_a, pipe_b, X_a: np.ndarray, X_b: np.ndarray, y: np.ndarray,
    fit_idx: np.ndarray, eval_idx: np.ndarray, neighbors: np.ndarray, onto_cols: np.ndarray,
    neighbors_floor40: np.ndarray | None = None,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Predições de uma divisão (fold ou teste final) em todos os cenários (TCC2, ponto 4).

    Índices globais (posições no dataset). ``neighbors[i]`` = gene mais parecido de
    ``eval_idx[i]`` entre ``fit_idx`` (ver ``src.nearest_neighbor``). Chaves:
        a                   Pipeline A
        b                   Pipeline B, anotação curada (ficha do próprio gene)
        b_gene_novo         Pipeline B, ontologia do gene mais parecido do treino;
                            sem vizinho, predição do A (regra adotada, P4-1)
        b_gene_novo_zeros   idem, mas sem vizinho = ontologia zerada (sensibilidade)
        homologia           mecanismo do gene mais parecido, sem ML (referência)
    Com ``neighbors_floor40`` (vizinho achado na busca do ponto 3, com piso de 40% de
    identidade), também ``b_gene_novo_piso40`` e ``homologia_piso40`` (sensibilidade ao
    piso da busca; mesmos modelos).
    Os modelos ``pipe_a`` e ``pipe_b`` são os mesmos nos cenários: muda só a origem
    das features ontológicas na predição.
    """
    out = {
        "a": (pipe_a.predict(X_a[eval_idx]), pipe_a.predict_proba(X_a[eval_idx])),
        "b": (pipe_b.predict(X_b[eval_idx]), pipe_b.predict_proba(X_b[eval_idx])),
    }
    out.update(predict_new_gene(pipe_b, pipe_a, X_a, X_b, eval_idx, neighbors, onto_cols))
    out["homologia"] = homology_prediction(y, fit_idx, neighbors, len(pipe_a.classes_))
    if neighbors_floor40 is not None:
        floor = predict_new_gene(pipe_b, pipe_a, X_a, X_b, eval_idx, neighbors_floor40, onto_cols)
        out["b_gene_novo_piso40"] = floor["b_gene_novo"]
        out["homologia_piso40"] = homology_prediction(y, fit_idx, neighbors_floor40, len(pipe_a.classes_))
    return out


def run_cross_validation_scenarios(
    model_name: str,
    X_a: np.ndarray,
    X_b: np.ndarray,
    y: np.ndarray,
    train_idx: np.ndarray,
    classes: list[str],
    splits: list[tuple],
    nn: NeighborIndex,
    onto_cols: np.ndarray,
    nn_floor40: NeighborIndex | None = None,
) -> tuple[list[dict], list[dict]]:
    """
    CV por grupos repetida com os Pipelines A e B e os cenários do ponto 4.  # ANTI-LEAKAGE  # SEED FIXA

    ``splits`` indexa ``train_idx`` (como em ``run_cross_validation``). Em cada fold, A e B
    são treinados só com os genes de ajuste; o gene mais parecido de cada gene de
    validação é procurado só entre esses genes de ajuste. O treino de A e B é idêntico
    ao de ``run_cross_validation`` (mesmos dados e semente), então as linhas ``a`` e ``b``
    reproduzem o experimento anterior. Retorna (métricas por fold, métricas por classe
    por fold), com a coluna ``pipeline`` e o número de genes sem vizinho no fold.
    """
    n_classes = len(classes)
    folds, per_class = [], []
    for rep, fold, fit_local, val_local in splits:
        fit_idx, val_idx = train_idx[fit_local], train_idx[val_local]
        pipe_a = build_full_pipeline(model_name).fit(X_a[fit_idx], y[fit_idx])
        pipe_b = build_full_pipeline(model_name).fit(X_b[fit_idx], y[fit_idx])
        neighbors, _ = nn.nearest(val_idx, fit_idx)
        n_without = int((neighbors == NO_NEIGHBOR).sum())
        floor = nn_floor40.nearest(val_idx, fit_idx)[0] if nn_floor40 is not None else None
        n_without_floor = int((floor == NO_NEIGHBOR).sum()) if floor is not None else None
        preds = scenario_predictions(pipe_a, pipe_b, X_a, X_b, y, fit_idx, val_idx, neighbors, onto_cols, floor)
        for key, (pred, proba) in preds.items():
            m = compute_metrics(y[val_idx], pred, proba, n_classes)
            base = {"pipeline": key, "repeticao": rep, "fold": fold}
            folds.append({**base, "n_ajuste": len(fit_idx), "n_validacao": len(val_idx), "n_sem_vizinho": n_without,
                          "n_sem_vizinho_piso40": n_without_floor, **m})
            per_class += [{**base, **r} for r in per_class_metrics(y[val_idx], pred, proba, classes)]
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


def _summary_rows(model: str, folds: list[dict], test_metrics: dict) -> list[dict]:
    """Uma linha por métrica: valor no teste, média e desvio amostral (n − 1) da CV."""
    rows = []
    for metric in METRICS:
        vals = np.array([f[metric] for f in folds], dtype=float)
        rows.append(
            {
                "modelo": model,
                "metrica": metric,
                "valor_teste": test_metrics[metric],
                "media_cv": np.nanmean(vals),
                # Desvio amostral (n − 1), o mesmo usado no teste t corrigido.
                "std_cv": np.nanstd(vals, ddof=1),
                "folds_validos_cv": int(np.isfinite(vals).sum()),
            }
        )
    return rows


def save_test_neighbors(
    data: pd.DataFrame, y: np.ndarray, classes: list[str], test_idx, neighbors, identity, neighbors_floor40=None
) -> pd.DataFrame:
    """Gene mais parecido do treino para cada gene do teste (cenário "gene novo")."""
    ids = data["ARO_Accession"].to_numpy()
    has = neighbors != NO_NEIGHBOR
    nb = np.where(has, neighbors, 0)
    table = pd.DataFrame(
        {
            "ARO_Accession": ids[test_idx],
            "mecanismo": np.array(classes)[y[test_idx]],
            "vizinho_treino": np.where(has, ids[nb], ""),
            "mecanismo_vizinho": np.where(has, np.array(classes)[y[nb]], ""),
            "identidade": np.where(has, identity, np.nan),
            "tem_vizinho": has,
        }
    )
    table["mesmo_mecanismo"] = table["tem_vizinho"] & (table["mecanismo"] == table["mecanismo_vizinho"])
    if neighbors_floor40 is not None:
        # Tinha vizinho também na busca do ponto 3 (identidade ≥ 40%)? Define o recorte
        # "genes distantes do treino" do bootstrap.
        table["tem_vizinho_piso40"] = neighbors_floor40 != NO_NEIGHBOR
    table.to_csv(RESULTS_DIR / "vizinho_mais_parecido_teste.csv", index=False)
    return table


def run_experiment(model_names: list[str] = MODEL_NAMES) -> None:
    start = time.time()
    X_a_df, X_b_df, y, classes = load_feature_matrices()
    onto_cols = ontology_columns(X_a_df, X_b_df)
    X_a, X_b = X_a_df.to_numpy(dtype=float), X_b_df.to_numpy(dtype=float)
    data = load_experiment_dataset()
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

    # Ponto 4: gene mais parecido de cada gene do teste, procurado só no treino.  # ANTI-LEAKAGE
    ids = data["ARO_Accession"].to_numpy()
    nn = NeighborIndex.from_files(ids)  # busca sem piso de identidade (principal)
    nn_floor40 = NeighborIndex.from_files(ids, FLOOR_40_HITS)  # busca do ponto 3 (sensibilidade)
    test_neighbors, test_identity = nn.nearest(test_idx, train_idx)
    test_floor40, _ = nn_floor40.nearest(test_idx, train_idx)
    nb_table = save_test_neighbors(data, y, classes, test_idx, test_neighbors, test_identity, test_floor40)
    logger.info(
        "Teste: %d de %d genes têm um parecido no treino (%d com o piso de 40%%); %.1f%% deles com o mesmo mecanismo.",
        nb_table["tem_vizinho"].sum(), len(nb_table), nb_table["tem_vizinho_piso40"].sum(),
        100 * nb_table.loc[nb_table["tem_vizinho"], "mesmo_mecanismo"].mean(),
    )

    cv_rows, cv_class_rows, test_class_rows = [], [], []
    summary: dict[str, list[dict]] = {k: [] for k in SCENARIO_KEYS}
    for i, name in enumerate(model_names):
        t0 = time.time()
        folds, folds_per_class = run_cross_validation_scenarios(
            name, X_a, X_b, y, train_idx, classes, splits, nn, onto_cols, nn_floor40
        )
        pipe_a = train_final_model(name, "a", X_a[train_idx], y[train_idx])
        pipe_b = train_final_model(name, "b", X_b[train_idx], y[train_idx])
        preds = scenario_predictions(
            pipe_a, pipe_b, X_a, X_b, y, train_idx, test_idx, test_neighbors, onto_cols, test_floor40
        )
        for key, (pred, proba) in preds.items():
            # A referência por homologia não depende do modelo: guardada uma vez só.
            if key in HOMOLOGY_KEYS and i > 0:
                continue
            label = "homologia" if key in HOMOLOGY_KEYS else name
            fname = "referencia_" + key if key in HOMOLOGY_KEYS else f"pipeline_{key}_{name}"
            test_metrics = compute_metrics(y[test_idx], pred, proba, n_classes)
            np.savez(
                PREDICTIONS_DIR / f"{fname}.npz",
                y_true=y[test_idx],
                y_pred=pred,
                proba=proba,
                confusion_matrix=confusion_matrix(y[test_idx], pred, labels=list(range(n_classes))),
            )
            key_folds = [f for f in folds if f["pipeline"] == key]
            cv_rows += [{"modelo": label, **f} for f in key_folds]
            cv_class_rows += [{"modelo": label, **r} for r in folds_per_class if r["pipeline"] == key]
            test_class_rows += [
                {"pipeline": key, "modelo": label, **r} for r in per_class_metrics(y[test_idx], pred, proba, classes)
            ]
            summary[key] += _summary_rows(label, key_folds, test_metrics)
            logger.info(
                "%-17s | %-20s F1 teste=%.4f | F1 CV=%.4f ± %.4f (%d folds)",
                key, label, test_metrics["f1_macro"],
                np.mean([f["f1_macro"] for f in key_folds]), np.std([f["f1_macro"] for f in key_folds], ddof=1),
                len(key_folds),
            )
        logger.info("%s concluído em %.1f s.", name, time.time() - t0)
    for key, rows in summary.items():
        save_metrics(rows, "referencia_" + key if key in HOMOLOGY_KEYS else key)
    cols = ["pipeline", "modelo"]
    pd.DataFrame(cv_rows)[lambda d: cols + [c for c in d.columns if c not in cols]].to_csv(RESULTS_DIR / "cv_folds.csv", index=False)
    pd.DataFrame(cv_class_rows)[lambda d: cols + [c for c in d.columns if c not in cols]].to_csv(
        RESULTS_DIR / "cv_por_classe.csv", index=False
    )
    pd.DataFrame(test_class_rows).to_csv(RESULTS_DIR / "metricas_por_classe_teste.csv", index=False)
    logger.info("Experimento concluído em %.1f min.", (time.time() - start) / 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run_experiment()
