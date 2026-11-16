"""
threshold_curve.py — Desempenho dos Pipelines A e B por limiar de similaridade.

Pergunta (TCC2, ponto 3): a ontologia ajuda mais quando o gene do teste é menos
parecido com os genes que o modelo viu no treino? Para responder, o experimento é
repetido com quatro divisões, das mais fáceis às mais difíceis:

    aleatoria  cada gene é um grupo (equivale à divisão aleatória estratificada)
    90         grupos de identidade ≥ 90% (e cobertura ≥ 50%)
    70         grupos de identidade ≥ 70% (avaliação principal do experimento)
    50         grupos de identidade ≥ 50%

O limiar de 40% fica de fora porque não atende o critério de entrada das classes
(documento de decisões, ponto 3). Em cada cenário, a partição de teste e a semente dos
folds são escolhidas pelas mesmas regras do experimento (``choose_test_partition`` e
``choose_cv_seeds``, só pela distribuição das classes), e a validação cruzada é por
grupos, repetida 10 × 5, como no experimento (decisão do ponto 5: "a curva por
limiar usa o mesmo desenho 10 × 5"). Um cenário em que alguma classe fica com menos
de 5 genes num fold é marcado em ``atende_criterio`` e não deve ser usado para
conclusões. Atributos, modelos e hiperparâmetros são os do experimento.

O ganho do B na CV vem com o intervalo de 95% corrigido (Nadeau e Bengio, o mesmo do
teste t do experimento), mas **sem teste**: a curva é descritiva.

Para cada cenário também é medido o parentesco entre teste e treino: a identidade
de cada gene do teste com o gene mais parecido do treino (``nearest_identity``),
com e sem a exigência de cobertura de 50%.

É uma análise **descritiva** (sem teste estatístico), como definido no ponto 5.

Saídas:
    results/curva_limiar.csv            — uma linha por cenário × modelo (F1 de teste e CV, A e B)
    results/curva_limiar_particoes.csv  — uma linha por cenário (partição, critério por fold, parentesco)
    results/curva_limiar_folds.csv      — F1-macro de cada estimativa da CV (cenário, modelo, pipeline)

Uso (cerca de 3 h):
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
from src.train_evaluate import check_class_counts_per_fold, compute_metrics, load_feature_matrices, run_cross_validation

logger = logging.getLogger(__name__)
SCENARIOS: list[tuple[str, int | None]] = [("aleatoria", None), ("90", 90), ("70", 70), ("50", 50)]


def run_scenario(name, threshold, X_a, X_b, y, classes, ids, hits) -> tuple[list[dict], dict, list[dict]]:
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
    rows, fold_rows = [], []
    for model in MODEL_NAMES:
        row = {"cenario": name, "modelo": model}
        f1 = {}
        for key, X_df in (("a", X_a), ("b", X_b)):
            X = X_df.to_numpy(dtype=float)
            folds, _ = run_cross_validation(model, X[train_idx], y[train_idx], classes, splits)
            pipe = build_full_pipeline(model).fit(X[train_idx], y[train_idx])
            test = compute_metrics(y[test_idx], pipe.predict(X[test_idx]), pipe.predict_proba(X[test_idx]), n_classes)
            f1[key] = np.array([f["f1_macro"] for f in folds])
            fold_rows += [
                {"cenario": name, "modelo": model, "pipeline": key, "repeticao": f["repeticao"], "fold": f["fold"], "f1_macro": f["f1_macro"]}
                for f in folds
            ]
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
        rows.append(row)
        logger.info(
            "%-9s %-20s teste A=%.4f B=%.4f | CV A=%.4f B=%.4f",
            name, model, row["f1_teste_a"], row["f1_teste_b"], row["f1_cv_media_a"], row["f1_cv_media_b"],
        )
    return rows, part, fold_rows


def run(scenarios=SCENARIOS) -> pd.DataFrame:
    X_a, X_b, y, classes = load_feature_matrices()
    ids = load_experiment_dataset()["ARO_Accession"].to_numpy()
    hits = load_hits(WORK_DIR / "hits.m8")
    rows, parts, folds = [], [], []
    for name, threshold in scenarios:
        t0 = time.time()
        r, p, f = run_scenario(name, threshold, X_a, X_b, y, classes, ids, hits)
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
