"""
leakage_check.py — Dependência do Pipeline B em relação à anotação do CARD (Seção 4.6; TCC2, ponto 4).

Problema (apontamentos D2, D3 e D4 do Diego; K2 e K3 da Kele): o rótulo (mecanismo) e
as 41 features ontológicas de um gene vêm da mesma ficha do CARD. Nenhuma feature
determina o mecanismo (não há vazamento direto), mas o Pipeline B depende da
anotação curada do próprio gene, que um gene novo não teria.

O que este módulo faz (decisões 4 e 5 do ponto 4):

1. **Informação mútua** entre cada feature ontológica e o mecanismo, **só no treino**
   (substitui o Spearman, que mede só relação monotônica com um rótulo codificado
   em números arbitrários). As 41 features são inteiras (profundidade, ancestrais,
   filhos) ou binárias (classes de antibiótico), então se usa o estimador discreto
   (``discrete_features=True``): a conta é exata, feita pela tabela de contagens, sem
   semente nem parâmetro de vizinhos. Também se dá a fração da entropia do rótulo
   que cada feature explica (IM / H(mecanismo)), para a leitura ficar em 0 a 1.
   É **descritiva**: nenhuma feature é removida (decisão 5).
   As 73 features de sequência ficam de fora: são contínuas e exigiriam outro
   estimador (por vizinhos), e os números não seriam comparáveis.

2. **Ablação por grupo de features**, com os mesmos 50 folds da validação cruzada e o
   mesmo teste final do experimento, nos dois cenários do ponto 4 (anotação curada e
   gene novo; ver ``src.nearest_neighbor``):

       A                     só sequência (controle: igual ao experimento)
       A + estruturais       sequência + 3 estruturais (profundidade, ancestrais, filhos)
       A + classes           sequência + 38 flags de classe de antibiótico
       B                     sequência + 41 (controle: igual ao experimento)
       só estruturais        as 3, sem sequência
       só classes            as 38, sem sequência
       só ontologia          as 41, sem sequência (substitui o experimento de sanidade)

   No cenário gene novo, um gene sem nenhum parecido no treino não tem anotação: as
   configurações com sequência usam a predição do Pipeline A (regra P4-1) e as sem
   sequência, a classe mais comum do treino (a mesma regra da referência por
   homologia: sem sequência e sem anotação, não há com que prever).
   Os resultados são **descritivos** (decisão do ponto 5): média da CV com o
   intervalo corrigido de Nadeau e Bengio e a diferença para o A, sem teste.

Saem o Spearman (|ρ| ≥ 0,95) e a regra "F1 ≤ 0,85 = sem vazamento" do experimento de
sanidade (apontamento D3: limiar arbitrário). O módulo **não altera mais**
``data/processed/features_ontologicas.csv``: as 41 features ficam todas no Pipeline B.
Os arquivos antigos (``spearman_features_ontologicas.csv``, ``sanidade_leakage.json``,
``leakage_check.log``) não são mais gerados.

Saídas:
    results/informacao_mutua_ontologia.csv — IM de cada feature ontológica (treino)
    results/ablacao_ontologia.csv          — resumo por configuração × cenário × modelo
    results/ablacao_ontologia_folds.csv    — F1-macro de cada estimativa da CV

Uso (depois de ``src.train_evaluate``, que salva a partição; cerca de 40 min):
    python -m src.leakage_check
"""

from __future__ import annotations

import json
import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from src.experiment_config import (
    MODEL_NAMES,
    PROCESSED_DIR,
    RESULTS_DIR,
    choose_cv_seeds,
    load_experiment_dataset,
    load_groups,
    repeated_cv_splits,
)
from src.nearest_neighbor import NO_NEIGHBOR, NeighborIndex
from src.pipeline_builder import build_full_pipeline
from src.statistical_tests import corrected_interval
from src.train_evaluate import compute_metrics, load_feature_matrices

logger = logging.getLogger(__name__)
STRUCTURAL: list[str] = ["depth_isa", "ancestor_count", "children_count"]


@dataclass(frozen=True)
class AblationConfig:
    name: str
    use_sequence: bool
    ontology: str  # "nenhuma", "estruturais", "classes" ou "todas"


ABLATION_CONFIGS: list[AblationConfig] = [
    AblationConfig("A", True, "nenhuma"),
    AblationConfig("A + estruturais", True, "estruturais"),
    AblationConfig("A + classes", True, "classes"),
    AblationConfig("B", True, "todas"),
    AblationConfig("só estruturais", False, "estruturais"),
    AblationConfig("só classes", False, "classes"),
    AblationConfig("só ontologia", False, "todas"),
]


def entropy_nats(y: np.ndarray) -> float:
    p = np.bincount(y) / len(y)
    p = p[p > 0]
    return float(-(p * np.log(p)).sum())


def mutual_information_table(onto_train: pd.DataFrame, y_train: np.ndarray) -> pd.DataFrame:
    """IM (nats) de cada feature ontológica com o mecanismo, estimador discreto (exato)."""
    mi = mutual_info_classif(onto_train.to_numpy(), y_train, discrete_features=True)
    h = entropy_nats(y_train)
    table = pd.DataFrame(
        {
            "feature": onto_train.columns,
            "grupo": ["estrutural" if c in STRUCTURAL else "classe_de_antibiotico" for c in onto_train.columns],
            "informacao_mutua_nats": mi,
            "fracao_da_entropia_do_rotulo": mi / h,
            "valores_distintos": onto_train.nunique().to_numpy(),
            "genes_com_valor_nao_zero": (onto_train != 0).sum().to_numpy(),
        }
    )
    return table.sort_values("informacao_mutua_nats", ascending=False).reset_index(drop=True)


def config_columns(cfg: AblationConfig, n_seq: int, onto_names: list[str]) -> np.ndarray:
    """Posições das colunas de uma configuração em X_b (sequência primeiro, depois ontologia)."""
    pos = {c: n_seq + i for i, c in enumerate(onto_names)}
    chosen = {
        "nenhuma": [],
        "estruturais": [c for c in onto_names if c in STRUCTURAL],
        "classes": [c for c in onto_names if c not in STRUCTURAL],
        "todas": list(onto_names),
    }[cfg.ontology]
    seq = list(range(n_seq)) if cfg.use_sequence else []
    return np.array(seq + [pos[c] for c in chosen], dtype=int)


def evaluate_split(
    model: str, X_b: np.ndarray, y: np.ndarray, fit_idx, eval_idx, neighbors, columns: dict, n_seq: int, n_classes: int
) -> dict[tuple[str, str], np.ndarray]:
    """
    Treina cada configuração em ``fit_idx`` e prevê ``eval_idx`` nos dois cenários.
    Devolve {(config, cenario): (pred, proba)}.
    """
    has = neighbors != NO_NEIGHBOR
    X_new = X_b[eval_idx].copy()
    X_new[np.ix_(has, np.arange(n_seq, X_b.shape[1]))] = X_b[np.ix_(neighbors[has], np.arange(n_seq, X_b.shape[1]))]
    majority = int(np.bincount(y[fit_idx], minlength=n_classes).argmax())
    out = {}
    for cfg in ABLATION_CONFIGS:
        cols = columns[cfg.name]
        pipe = build_full_pipeline(model).fit(X_b[np.ix_(fit_idx, cols)], y[fit_idx])
        cur = (pipe.predict(X_b[np.ix_(eval_idx, cols)]), pipe.predict_proba(X_b[np.ix_(eval_idx, cols)]))
        out[(cfg.name, "anotacao_curada")] = cur
        if cfg.ontology == "nenhuma":
            out[(cfg.name, "gene_novo")] = cur  # sem ontologia, o cenário não muda nada
            continue
        pred, proba = np.empty_like(cur[0]), np.empty_like(cur[1])
        if has.any():
            pred[has], proba[has] = pipe.predict(X_new[np.ix_(has, cols)]), pipe.predict_proba(X_new[np.ix_(has, cols)])
        if (~has).any():
            if cfg.use_sequence:
                pa, pp = out[("A", "anotacao_curada")]
                pred[~has], proba[~has] = pa[~has], pp[~has]
            else:
                pred[~has], proba[~has] = majority, np.eye(n_classes)[majority]
        out[(cfg.name, "gene_novo")] = (pred, proba)
    return out


def run_ablation(model_names=MODEL_NAMES) -> tuple[pd.DataFrame, pd.DataFrame]:
    X_a_df, X_b_df, y, classes = load_feature_matrices()
    n_seq, onto_names = X_a_df.shape[1], list(X_b_df.columns[X_a_df.shape[1]:])
    X_b = X_b_df.to_numpy(dtype=float)
    data = load_experiment_dataset()
    groups = load_groups(data=data)
    train_idx = np.load(PROCESSED_DIR / "train_indices.npy")
    test_idx = np.load(PROCESSED_DIR / "test_indices.npy")
    cv_seeds, _ = choose_cv_seeds(y[train_idx], groups[train_idx])
    saved = json.loads((RESULTS_DIR / "particao_teste.json").read_text(encoding="utf-8"))
    if saved.get("sementes_cv", cv_seeds) != cv_seeds:
        raise ValueError("Sementes da CV diferentes das do experimento: rode src.train_evaluate de novo.")
    splits = repeated_cv_splits(y[train_idx], groups[train_idx], cv_seeds)
    ratio = float(np.mean([len(v) / len(f) for _, _, f, v in splits]))
    nn = NeighborIndex.from_files(data["ARO_Accession"].to_numpy())
    columns = {cfg.name: config_columns(cfg, n_seq, onto_names) for cfg in ABLATION_CONFIGS}
    n_classes = len(classes)
    test_nb, _ = nn.nearest(test_idx, train_idx)

    fold_rows, rows = [], []
    for model in model_names:
        t0 = time.time()
        for rep, fold, fit_local, val_local in splits:
            fit_idx, val_idx = train_idx[fit_local], train_idx[val_local]
            nb, _ = nn.nearest(val_idx, fit_idx)
            preds = evaluate_split(model, X_b, y, fit_idx, val_idx, nb, columns, n_seq, n_classes)
            for (cfg, scen), (pred, proba) in preds.items():
                f1 = compute_metrics(y[val_idx], pred, proba, n_classes)["f1_macro"]
                fold_rows.append({"config": cfg, "cenario": scen, "modelo": model, "repeticao": rep, "fold": fold, "f1_macro": f1})
        test_preds = evaluate_split(model, X_b, y, train_idx, test_idx, test_nb, columns, n_seq, n_classes)
        folds = pd.DataFrame([r for r in fold_rows if r["modelo"] == model])
        base = folds[(folds["config"] == "A") & (folds["cenario"] == "anotacao_curada")].sort_values(["repeticao", "fold"])
        f1_a_test = compute_metrics(y[test_idx], *test_preds[("A", "anotacao_curada")], n_classes)["f1_macro"]
        for (cfg, scen), (pred, proba) in test_preds.items():
            sub = folds[(folds["config"] == cfg) & (folds["cenario"] == scen)].sort_values(["repeticao", "fold"])
            vals = sub["f1_macro"].to_numpy()
            diff = vals - base["f1_macro"].to_numpy()
            ic = corrected_interval(vals, ratio)
            ic_d = corrected_interval(diff, ratio) if np.any(diff != 0) else (0.0, 0.0)
            f1_test = compute_metrics(y[test_idx], pred, proba, n_classes)["f1_macro"]
            rows.append(
                {
                    "config": cfg, "cenario": scen, "modelo": model,
                    "f1_cv_media": vals.mean(), "f1_cv_ic95_inf": ic[0], "f1_cv_ic95_sup": ic[1],
                    "dif_a_cv_media": diff.mean(), "dif_a_cv_ic95_inf": ic_d[0], "dif_a_cv_ic95_sup": ic_d[1],
                    "f1_teste": f1_test, "dif_a_teste": f1_test - f1_a_test,
                }
            )
        logger.info("Ablação %-20s concluída em %.1f min.", model, (time.time() - t0) / 60)
    summary = pd.DataFrame(rows)
    for _, r in summary[summary["cenario"] == "anotacao_curada"].iterrows():
        logger.info("%-20s %-16s CV=%.4f (dif. A %+.4f) teste=%.4f", r["modelo"], r["config"], r["f1_cv_media"], r["dif_a_cv_media"], r["f1_teste"])
    return summary, pd.DataFrame(fold_rows)


def run() -> None:
    X_a_df, X_b_df, y, _ = load_feature_matrices()
    train_idx = np.load(PROCESSED_DIR / "train_indices.npy")
    onto = X_b_df.iloc[:, X_a_df.shape[1]:]
    mi = mutual_information_table(onto.iloc[train_idx], y[train_idx])
    mi.to_csv(RESULTS_DIR / "informacao_mutua_ontologia.csv", index=False)
    logger.info("Informação mútua (treino): maiores = %s", mi.head(5)[["feature", "fracao_da_entropia_do_rotulo"]].to_dict("records"))
    summary, folds = run_ablation()
    summary.to_csv(RESULTS_DIR / "ablacao_ontologia.csv", index=False)
    folds.to_csv(RESULTS_DIR / "ablacao_ontologia_folds.csv", index=False)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
