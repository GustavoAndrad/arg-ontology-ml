"""
threshold_curve.py — Desempenho dos Pipelines A e B por limiar de similaridade.

Pergunta: a ontologia ajuda mais quando o gene do teste é menos
parecido com os genes que o modelo viu no treino? Para responder, o experimento é
repetido com quatro divisões, das mais fáceis às mais difíceis:

    aleatoria  cada gene é um grupo (equivale à divisão aleatória estratificada)
    90         grupos de identidade ≥ 90% (e cobertura ≥ 50%)
    70         grupos de identidade ≥ 70% (avaliação principal do experimento)
    50         grupos de identidade ≥ 50%

O limiar de 40% fica de fora porque não atende o critério de entrada das classes
(menos de 5 genes de alguma classe por fold). Em cada cenário, a partição de teste e a semente dos
folds são escolhidas pelas mesmas regras do experimento (``choose_test_partition`` e
``choose_cv_seeds``, só pela distribuição das classes), e a validação cruzada é por
grupos, repetida 10 × 5, como no experimento. Um cenário em que alguma classe fica com menos
de 5 genes num fold é marcado em ``atende_criterio`` e não deve ser usado para
conclusões. Atributos, modelos e hiperparâmetros são os do experimento.

O ganho do B na CV vem com o intervalo de 95% corrigido (Nadeau e Bengio, o mesmo do
teste t do experimento), mas **sem teste**: a curva é descritiva.

Em cada cenário entram também o Pipeline B no cenário "gene novo"
(ontologia do gene mais parecido do treino daquela divisão; sem parecido, predição do
A) e a referência por homologia (mecanismo do gene mais parecido), com os mesmos
folds (``src.train_evaluate.run_cross_validation_scenarios``). O ganho do B gene novo
sobre o A também vem com o intervalo corrigido. O número de genes de teste sem
nenhum parecido no treino fica em ``teste_sem_vizinho_no_treino``.

Para cada cenário também é medido o parentesco entre teste e treino: a identidade
de cada gene do teste com o gene mais parecido do treino (``nearest_identity``),
com e sem a exigência de cobertura de 50%.

É uma análise **descritiva** (sem teste estatístico). O cenário "70" reproduz
exatamente o experimento principal (mesma partição, mesmos folds).

Saídas:
    results/curva_limiar.csv            — uma linha por cenário × modelo (F1 de teste e CV: A, B, B gene novo, homologia)
    results/curva_limiar_particoes.csv  — uma linha por cenário (partição, critério por fold, parentesco)
    results/curva_limiar_folds.csv      — F1-macro de cada estimativa da CV (cenário, modelo, pipeline)

Uso (cerca de 1 h):
    python -m src.threshold_curve
"""

from __future__ import annotations

import logging
import sys
import time

import numpy as np
import pandas as pd

from src.experiment_config import (
    MIN_GENES_PER_FOLD,
    MODEL_NAMES,
    RESULTS_DIR,
    choose_cv_seeds,
    choose_test_partition,
    load_experiment_dataset,
    load_groups,
    repeated_cv_splits,
)
from src.homology_groups import WORK_DIR, homology_profile, load_hits, nearest_identity
from src.pipeline_builder import build_full_pipeline
from src.statistical_tests import corrected_interval
from src.nearest_neighbor import NO_NEIGHBOR, NeighborIndex
from src.train_evaluate import (
    check_class_counts_per_fold,
    compute_metrics,
    load_feature_matrices,
    ontology_columns,
    run_cross_validation_scenarios,
    scenario_predictions,
)

logger = logging.getLogger(__name__)
SCENARIOS: list[tuple[str, int | None]] = [("aleatoria", None), ("90", 90), ("70", 70), ("50", 50)]
# A curva inclui o B "gene novo" e a referência por homologia.
CURVE_KEYS: list[str] = ["a", "b", "b_gene_novo", "homologia"]


def run_scenario(name, threshold, X_a, X_b, y, classes, ids, hits, nn, onto_cols) -> tuple[list[dict], dict, list[dict]]:
    groups = load_groups(threshold)
    train_idx, test_idx, info = choose_test_partition(y, groups)
    cv_seeds, cv_info = choose_cv_seeds(y[train_idx], groups[train_idx])
    splits = repeated_cv_splits(y[train_idx], groups[train_idx], cv_seeds)
    counts = check_class_counts_per_fold(y[train_idx], classes, splits)
    part = {"cenario": name, **{k: v for k, v in info.items() if k != "fracao_teste_por_classe"}}
    part.update({k: (str(v) if isinstance(v, list) else v) for k, v in cv_info.items()})
    part["atende_criterio"] = bool(counts[classes].to_numpy().min() >= MIN_GENES_PER_FOLD)
    for c, f in zip(classes, info["fracao_teste_por_classe"]):
        part[f"fracao_teste_{c.replace('antibiotic ', '')}"] = f
    for c in classes:
        part[f"min_por_fold_{c.replace('antibiotic ', '')}"] = int(counts[c].min())
    test_ids, train_ids = ids[test_idx], ids[train_idx]
    for label, cov in (("cobertura50", 0.5), ("qualquer_cobertura", None)):
        prof = homology_profile(nearest_identity(hits, test_ids, train_ids, min_coverage=cov))
        part.update({f"{k}_{label}": v for k, v in prof.items()})

    n_classes = len(classes)
    ratio = float(np.mean([len(val) / len(fit) for _, _, fit, val in splits]))
    test_nb, _ = nn.nearest(test_idx, train_idx)
    part["teste_sem_vizinho_no_treino"] = int((test_nb == NO_NEIGHBOR).sum())
    rows, fold_rows = [], []
    for model in MODEL_NAMES:
        row = {"cenario": name, "modelo": model}
        folds, _ = run_cross_validation_scenarios(model, X_a, X_b, y, train_idx, classes, splits, nn, onto_cols)
        pipe_a = build_full_pipeline(model).fit(X_a[train_idx], y[train_idx])
        pipe_b = build_full_pipeline(model).fit(X_b[train_idx], y[train_idx])
        preds = scenario_predictions(pipe_a, pipe_b, X_a, X_b, y, train_idx, test_idx, test_nb, onto_cols)
        f1 = {}
        for key in CURVE_KEYS:
            kf = [f for f in folds if f["pipeline"] == key]
            f1[key] = np.array([f["f1_macro"] for f in kf])
            fold_rows += [
                {"cenario": name, "modelo": model, "pipeline": key, "repeticao": f["repeticao"], "fold": f["fold"],
                 "n_sem_vizinho": f["n_sem_vizinho"], "f1_macro": f["f1_macro"]}
                for f in kf
            ]
            test = compute_metrics(y[test_idx], *preds[key], n_classes)
            row.update(
                {
                    f"f1_teste_{key}": float(test["f1_macro"]),
                    f"f1_cv_media_{key}": float(f1[key].mean()),
                    f"f1_cv_std_{key}": float(f1[key].std(ddof=1)),
                }
            )
        row["ganho_teste"] = row["f1_teste_b"] - row["f1_teste_a"]
        row["ganho_cv"] = row["f1_cv_media_b"] - row["f1_cv_media_a"]
        row["ganho_cv_ic95_inf"], row["ganho_cv_ic95_sup"] = corrected_interval(f1["b"] - f1["a"], ratio)
        row["ganho_teste_gene_novo"] = row["f1_teste_b_gene_novo"] - row["f1_teste_a"]
        row["ganho_cv_gene_novo"] = row["f1_cv_media_b_gene_novo"] - row["f1_cv_media_a"]
        row["ganho_cv_gene_novo_ic95_inf"], row["ganho_cv_gene_novo_ic95_sup"] = corrected_interval(
            f1["b_gene_novo"] - f1["a"], ratio
        )
        rows.append(row)
        logger.info(
            "%-9s %-20s teste A=%.4f B=%.4f B-novo=%.4f homologia=%.4f | CV A=%.4f B=%.4f B-novo=%.4f homologia=%.4f",
            name, model, row["f1_teste_a"], row["f1_teste_b"], row["f1_teste_b_gene_novo"], row["f1_teste_homologia"],
            row["f1_cv_media_a"], row["f1_cv_media_b"], row["f1_cv_media_b_gene_novo"], row["f1_cv_media_homologia"],
        )
    return rows, part, fold_rows


def run(scenarios=SCENARIOS) -> pd.DataFrame:
    X_a_df, X_b_df, y, classes = load_feature_matrices()
    onto_cols = ontology_columns(X_a_df, X_b_df)
    X_a, X_b = X_a_df.to_numpy(dtype=float), X_b_df.to_numpy(dtype=float)
    ids = load_experiment_dataset()["ARO_Accession"].to_numpy()
    hits = load_hits(WORK_DIR / "hits.m8")  # busca dos grupos: só para o perfil de parentesco teste–treino
    nn = NeighborIndex.from_files(ids)  # gene mais parecido: busca sem piso de identidade
    rows, parts, folds = [], [], []
    for name, threshold in scenarios:
        t0 = time.time()
        r, p, f = run_scenario(name, threshold, X_a, X_b, y, classes, ids, hits, nn, onto_cols)
        rows += r
        parts.append(p)
        folds += f
        logger.info("Cenário %s concluído em %.1f min.", name, (time.time() - t0) / 60)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "curva_limiar.csv", index=False)
    pd.DataFrame(parts).to_csv(RESULTS_DIR / "curva_limiar_particoes.csv", index=False)
    pd.DataFrame(folds).to_csv(RESULTS_DIR / "curva_limiar_folds.csv", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
