"""
learning_curve.py — Curva de aprendizado e overfitting por classe (capítulo 5).

Pergunta: os modelos "decoram" o treino? Generalizam bem
só para a classe majoritária? E mais dados de treino ajudariam?

**Overfitting**: F1-macro e F1 de cada classe em três lugares:
    * treino: o modelo avaliado nos próprios genes com que foi ajustado
      (nos folds, ``f1_treino_cv``; no modelo final, ``f1_treino_final``);
    * validação: os folds de validação da CV 10 × 5;
    * teste: o conjunto de teste final.
100% no treino é normal em RF e XGBoost (árvores profundas separam os genes de
treino); o que importa é a **queda** do treino para os dados novos e se ela é maior
nas classes pequenas.

**Curva de aprendizado**: em cada um dos 50 folds, o modelo é treinado com 20, 40,
60, 80 e 100% **dos grupos** de cada classe do conjunto de ajuste e avaliado no fold
de validação (inteiro, sempre o mesmo). Em genes, a média acompanha a fração de
grupos (20% dos grupos ≈ 22% dos genes), mas varia de fold para fold por causa dos
grupos grandes (com 20%, de 6% a 54% dos genes). Decisões:
    * **Redução por grupos inteiros**, estratificada por classe: em cada classe, os
      grupos de similaridade (70%) são sorteados, e a fração f mantém os primeiros
      ceil(f × número de grupos da classe). Sortear genes soltos manteria
      representantes de quase todas as famílias e mediria pouco; sorteando grupos,
      a curva mede quanto o modelo melhora quando vê **mais famílias de genes**.
      Nenhum grupo mistura classes (conferido no código), então a estratificação
      por grupo é exata.
    * **Frações encaixadas** (os 20% estão dentro dos 40%, e assim por diante; uma
      única ordem de sorteio por fold): reduz o ruído entre pontos da curva.
    * **As mesmas frações para A, B e os 4 modelos** (mesma semente por fold):
      comparações pareadas.
    * **Frações de 20 a 100% em passos iguais de 20%:** 5 pontos. Em cada fração
      registra-se o menor número de genes de cada classe no treino; a fração em que
      alguma classe fica com menos de 5 genes em algum fold (critério de entrada das
      classes) é **marcada** (``atende_criterio``), como na curva por limiar. Medido: com 20%, a substituição do alvo chega a 4.
    * A fração de 100% é o próprio experimento (controle: reproduz ``cv_folds.csv``).
O F1 médio de cada ponto vem com o IC 95% corrigido de Nadeau e Bengio sobre os 50
folds. Descritivo, sem teste.

Hiperparâmetros fixos e iguais nos dois pipelines; a
sensibilidade ao ajuste está em ``src.sensitivity_tuning``.

Saídas:
    results/curva_aprendizado_folds.csv — F1 (macro e por classe) por fold, fração, modelo e pipeline
    results/curva_aprendizado.csv       — média e IC por fração, com o critério de genes por classe
    results/overfitting.csv             — treino × validação × teste, macro e por classe

Uso (depois de ``src.train_evaluate``; cerca de 1 h):
    python -m src.learning_curve
"""

from __future__ import annotations

import json
import logging
import math
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.experiment_config import (
    MIN_GENES_PER_FOLD,
    MODEL_NAMES,
    MODELS_DIR,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    choose_cv_seeds,
    load_experiment_dataset,
    load_groups,
    repeated_cv_splits,
)
from src.pipeline_builder import build_full_pipeline
from src.statistical_tests import corrected_interval
from src.train_evaluate import load_feature_matrices

logger = logging.getLogger(__name__)
FRACTIONS: tuple[float, ...] = (0.2, 0.4, 0.6, 0.8, 1.0)


def nested_group_subsets(y: np.ndarray, groups: np.ndarray, fractions, rng: np.random.Generator) -> dict[float, np.ndarray]:
    """
    Posições (em ``y``) mantidas em cada fração: por classe, os grupos são embaralhados
    uma vez, e a fração f fica com os primeiros ceil(f × grupos da classe). Encaixadas.
    """
    order = {}
    for k in np.unique(y):
        g = np.unique(groups[y == k])
        if np.intersect1d(g, np.unique(groups[y != k])).size:
            raise ValueError("Há grupos com mais de uma classe; a estratificação por grupo não se aplica.")
        order[k] = rng.permutation(g)
    out = {}
    for f in fractions:
        keep = np.concatenate([o[: math.ceil(f * len(o))] for o in order.values()])
        out[f] = np.flatnonzero(np.isin(groups, keep))
    return out


def f1_scores(y_true, y_pred, n_classes: int) -> dict[str, float]:
    per = f1_score(y_true, y_pred, labels=list(range(n_classes)), average=None, zero_division=0)
    return {"f1_macro": float(per.mean()), **{f"f1_classe_{k}": float(v) for k, v in enumerate(per)}}


def summarize_curve(folds: pd.DataFrame, ratio: float, classes: list[str]) -> pd.DataFrame:
    rows = []
    for (model, key, frac), sub in folds.groupby(["modelo", "pipeline", "fracao"], sort=False):
        row = {"modelo": model, "pipeline": key, "fracao": frac}
        for col, label in [("f1_macro", "macro")] + [(f"f1_classe_{k}", c) for k, c in enumerate(classes)]:
            v = sub[col].to_numpy(dtype=float)
            low, high = corrected_interval(v, ratio)
            row |= {f"media_{label}": float(v.mean()), f"ic95_inf_{label}": low, f"ic95_sup_{label}": high}
        mins = sub[[f"n_treino_classe_{k}" for k in range(len(classes))]].min()
        row |= {f"min_treino_{c}": int(m) for c, m in zip(classes, mins)}
        row["atende_criterio"] = bool(mins.min() >= MIN_GENES_PER_FOLD)
        rows.append(row)
    return pd.DataFrame(rows)


def overfitting_table(folds: pd.DataFrame, final_rows: list[dict], classes: list[str]) -> pd.DataFrame:
    """Treino (folds e modelo final), validação (folds) e teste, macro e por classe."""
    full = folds[folds["fracao"] == 1.0]
    rows = []
    for (model, key), sub in full.groupby(["modelo", "pipeline"], sort=False):
        fin = next(r for r in final_rows if r["modelo"] == model and r["pipeline"] == key)
        for col, label in [("f1_macro", "macro")] + [(f"f1_classe_{k}", c) for k, c in enumerate(classes)]:
            rows.append({
                "modelo": model, "pipeline": key, "classe": label,
                "f1_treino_cv": float(sub[f"treino_{col}"].mean()),
                "f1_validacao_cv": float(sub[col].mean()),
                "f1_treino_final": fin[f"treino_{col}"],
                "f1_teste": fin[f"teste_{col}"],
            })
    out = pd.DataFrame(rows)
    out["queda_treino_validacao"] = out["f1_treino_cv"] - out["f1_validacao_cv"]
    out["queda_treino_teste"] = out["f1_treino_final"] - out["f1_teste"]
    return out


def run(model_names=MODEL_NAMES) -> None:
    start = time.time()
    X_a_df, X_b_df, y, classes = load_feature_matrices()
    X = {"a": X_a_df.to_numpy(dtype=float), "b": X_b_df.to_numpy(dtype=float)}
    n_classes = len(classes)
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
    # Subconjuntos de cada fold: os mesmos para A, B e os 4 modelos.  # SEED FIXA
    subsets = {
        (rep, fold): nested_group_subsets(
            y[train_idx][fit], groups[train_idx][fit], FRACTIONS,
            np.random.default_rng(np.random.SeedSequence([RANDOM_STATE, rep, fold])),
        )
        for rep, fold, fit, _ in splits
    }

    fold_rows, final_rows = [], []
    for model in model_names:
        t0 = time.time()
        for key in ("a", "b"):
            Xk = X[key]
            for rep, fold, fit_local, val_local in splits:
                fit_idx, val_idx = train_idx[fit_local], train_idx[val_local]
                for frac, pos in subsets[(rep, fold)].items():
                    sub_idx = fit_idx[pos]
                    pipe = build_full_pipeline(model).fit(Xk[sub_idx], y[sub_idx])
                    row = {"modelo": model, "pipeline": key, "repeticao": rep, "fold": fold, "fracao": frac,
                           "n_treino": len(sub_idx),
                           **{f"n_treino_classe_{k}": int((y[sub_idx] == k).sum()) for k in range(n_classes)},
                           **f1_scores(y[val_idx], pipe.predict(Xk[val_idx]), n_classes)}
                    if frac == 1.0:
                        row |= {f"treino_{c}": v for c, v in f1_scores(y[sub_idx], pipe.predict(Xk[sub_idx]), n_classes).items()}
                    fold_rows.append(row)
            pipe = joblib.load(MODELS_DIR / f"pipeline_{key}_{model}.joblib")
            final_rows.append({
                "modelo": model, "pipeline": key,
                **{f"treino_{c}": v for c, v in f1_scores(y[train_idx], pipe.predict(Xk[train_idx]), n_classes).items()},
                **{f"teste_{c}": v for c, v in f1_scores(y[test_idx], pipe.predict(Xk[test_idx]), n_classes).items()},
            })
        logger.info("%-20s concluído em %.1f min.", model, (time.time() - t0) / 60)

    folds = pd.DataFrame(fold_rows)
    folds.to_csv(RESULTS_DIR / "curva_aprendizado_folds.csv", index=False)
    curve = summarize_curve(folds, ratio, classes)
    curve.to_csv(RESULTS_DIR / "curva_aprendizado.csv", index=False)
    overfitting_table(folds, final_rows, classes).to_csv(RESULTS_DIR / "overfitting.csv", index=False)
    for _, r in curve[curve["fracao"].isin([0.2, 1.0])].iterrows():
        logger.info("%-20s %s %.0f%%: F1 = %.4f [%.4f; %.4f]%s", r["modelo"], r["pipeline"].upper(), 100 * r["fracao"],
                    r["media_macro"], r["ic95_inf_macro"], r["ic95_sup_macro"], "" if r["atende_criterio"] else " (não atende o critério)")
    logger.info("Curva de aprendizado concluída em %.1f min.", (time.time() - start) / 60)


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
