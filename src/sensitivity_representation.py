"""
sensitivity_representation.py — Sensibilidade do resultado à representação da sequência.

Pergunta: a conclusão "B > A" depende de como a sequência é representada ou do
filtro de atributos? Responde ao apontamento J2/J8 (TCC2, ponto 7) e mede quanto do
ganho do Pipeline B no TCC1 vinha da seleção conjunta de atributos.

Configurações (todas com as mesmas 5 classes, a mesma partição treino/teste, os
mesmos folds e os mesmos hiperparâmetros do experimento; só muda a representação):

    reduzida_73       experimento: 73 atributos de sequência, sem seleção (B = 73 + 41)
    tcc1_k100_todos   TCC1: 426 atributos, SelectKBest k=100 sobre todos (no B, a seleção
                      escolhe entre sequência e ontologia juntas)
    k100_sequencia    426 atributos, k=100 só na sequência; as 41 ontológicas passam inteiras
    completa_426      426 atributos, sem seleção (B = 426 + 41)
    basicas_26        26 básicas do TCC1 (6 físico-químicas + 20 aminoácidos), sem seleção
    k50_sequencia     426 atributos, k=50 só na sequência
    k20_sequencia     426 atributos, k=20 só na sequência

Leituras previstas:
    * tcc1_k100_todos reproduz os números do experimento antes do ponto 7 (controle);
    * tcc1_k100_todos vs k100_sequencia: efeito da seleção conjunta sobre a ontologia;
    * reduzida_73 vs completa_426 vs basicas_26: efeito dos dipeptídeos;
    * k20 / k50 / k100: efeito do k.

É uma análise **descritiva** (sem teste estatístico), como definido no ponto 5.
Usa a mesma partição por grupos de similaridade do experimento (``load_groups``) e a
mesma validação cruzada por grupos. A versão final é refeita com o desenho definitivo
(validação cruzada repetida). ``run(threshold=None)`` roda com a divisão aleatória.

Saídas:
    results/sensibilidade_representacao.csv        — uma linha por configuração × modelo
    results/sensibilidade_representacao_folds.csv  — F1-macro de cada fold

Uso (cerca de 30 min):
    python -m src.sensitivity_representation
"""

from __future__ import annotations

import logging
import sys
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd

from src.config import K_FEATURES
from src.experiment_config import (
    MODEL_NAMES,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET,
    cv_splitter,
    SIMILARITY_THRESHOLD,
    choose_cv_seed,
    encode_target,
    load_experiment_dataset,
    load_groups,
    split_indices,
)
from src.feature_extraction import ALL_PHYSICOCHEMICAL_COLUMNS, AA_COMPOSITION_COLUMNS, REPRESENTATIONS
from src.interpretability import feature_type, get_feature_names_after_pipeline
from src.pipeline_builder import FeatureSelection, build_full_pipeline
from src.train_evaluate import compute_metrics

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RepresentationConfig:
    name: str
    description: str
    columns: list[str]  # colunas de sequência usadas
    k: int | None = None  # None = sem seleção
    scope: str = "todos"


N_426 = len(REPRESENTATIONS["completa_426"])
CONFIGS: list[RepresentationConfig] = [
    RepresentationConfig("reduzida_73", "73 atributos (alfabeto reduzido), sem seleção", REPRESENTATIONS["reduzida_73"]),
    RepresentationConfig("tcc1_k100_todos", "426 atributos, k=100 sobre todos (TCC1)", REPRESENTATIONS["completa_426"], K_FEATURES),
    RepresentationConfig(
        "k100_sequencia", "426 atributos, k=100 só na sequência", REPRESENTATIONS["completa_426"], K_FEATURES, "sequencia"
    ),
    RepresentationConfig("completa_426", "426 atributos, sem seleção", REPRESENTATIONS["completa_426"]),
    RepresentationConfig(
        "basicas_26", "26 básicas (6 físico-químicas + 20 aminoácidos), sem seleção",
        ALL_PHYSICOCHEMICAL_COLUMNS + AA_COMPOSITION_COLUMNS,
    ),
    RepresentationConfig("k50_sequencia", "426 atributos, k=50 só na sequência", REPRESENTATIONS["completa_426"], 50, "sequencia"),
    RepresentationConfig("k20_sequencia", "426 atributos, k=20 só na sequência", REPRESENTATIONS["completa_426"], 20, "sequencia"),
]


def load_matrices() -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray, list[str]]:
    """Retorna (sequência com 426 + 49 pares, ontologia, y, classes), alinhados ao experimento."""
    data = load_experiment_dataset()
    full = pd.read_csv(PROCESSED_DIR / "features_sequencia_completa_426.csv")
    reduced = pd.read_csv(PROCESSED_DIR / "features_sequencia.csv")
    onto = pd.read_csv(PROCESSED_DIR / "features_ontologicas.csv")
    for name, f in (("completa_426", full), ("reduzida_73", reduced), ("ontologia", onto)):
        if not (f["ARO_Accession"].values == data["ARO_Accession"].values).all():
            raise ValueError(f"{name} desalinhado ao dataset; reexecute a extração.")
    seq = pd.concat([full.drop(columns="ARO_Accession"), reduced[[c for c in reduced if c.startswith("pair_")]]], axis=1)
    y, classes = encode_target(data[TARGET])
    return seq, onto.drop(columns="ARO_Accession"), y, classes


def _pipeline(model: str, cfg: RepresentationConfig, n_seq: int, n_total: int):
    selection = None if cfg.k is None else FeatureSelection(k=cfg.k, scope=cfg.scope, n_sequence=n_seq)
    return build_full_pipeline(model, selection, n_features=n_total)


def _n_ontological_in_model(pipe, names: list[str], cfg: RepresentationConfig, n_onto: int) -> int:
    """Quantos atributos ontológicos chegam ao classificador no Pipeline B."""
    if cfg.k is not None and cfg.scope == "todos":
        return sum(feature_type(n) == "ontologica" for n in get_feature_names_after_pipeline(pipe, names))
    return n_onto  # sem seleção, ou seleção só na sequência: todos passam


def evaluate_config(cfg, seq, onto, y, classes, train_idx, test_idx, groups, cv_seed) -> tuple[list[dict], list[dict]]:
    n_classes = len(classes)
    rows, fold_rows = [], []
    for model in MODEL_NAMES:
        result = {"config": cfg.name, "descricao": cfg.description, "modelo": model}
        for key in ("a", "b"):
            X_df = seq[cfg.columns] if key == "a" else pd.concat([seq[cfg.columns], onto], axis=1)
            X, names = X_df.to_numpy(dtype=float), list(X_df.columns)
            Xtr, ytr = X[train_idx], y[train_idx]
            f1s = []
            for i, (fit, val) in enumerate(cv_splitter(cv_seed).split(Xtr, ytr, groups[train_idx]), start=1):
                p = _pipeline(model, cfg, len(cfg.columns), X.shape[1]).fit(Xtr[fit], ytr[fit])
                f1 = compute_metrics(ytr[val], p.predict(Xtr[val]), p.predict_proba(Xtr[val]), n_classes)["f1_macro"]
                f1s.append(f1)
                fold_rows.append({"config": cfg.name, "modelo": model, "pipeline": key, "fold": i, "f1_macro": f1})
            final = _pipeline(model, cfg, len(cfg.columns), X.shape[1]).fit(Xtr, ytr)
            test = compute_metrics(y[test_idx], final.predict(X[test_idx]), final.predict_proba(X[test_idx]), n_classes)
            result.update(
                {
                    f"n_atributos_{key}": int(final.named_steps["clf"].n_features_in_),
                    f"f1_cv_media_{key}": float(np.mean(f1s)),
                    f"f1_cv_std_{key}": float(np.std(f1s)),
                    f"f1_teste_{key}": float(test["f1_macro"]),
                }
            )
            if key == "b":
                result["n_ontologicos_no_modelo_b"] = _n_ontological_in_model(final, names, cfg, onto.shape[1])
        result["ganho_teste"] = result["f1_teste_b"] - result["f1_teste_a"]
        result["ganho_cv"] = result["f1_cv_media_b"] - result["f1_cv_media_a"]
        rows.append(result)
        logger.info(
            "%-16s %-20s A=%.4f B=%.4f (teste) | CV A=%.4f B=%.4f | ontológicos no B: %d",
            cfg.name, model, result["f1_teste_a"], result["f1_teste_b"],
            result["f1_cv_media_a"], result["f1_cv_media_b"], result["n_ontologicos_no_modelo_b"],
        )
    return rows, fold_rows


def run(configs: list[RepresentationConfig] = CONFIGS, threshold: int | None = SIMILARITY_THRESHOLD) -> pd.DataFrame:
    seq, onto, y, classes = load_matrices()
    groups = load_groups(threshold)
    train_idx, test_idx = split_indices(y, groups)  # mesma partição do experimento  # SEED FIXA
    cv_seed, _ = choose_cv_seed(y[train_idx], groups[train_idx])  # mesmos folds do experimento
    rows, fold_rows = [], []
    for cfg in configs:
        t0 = time.time()
        r, f = evaluate_config(cfg, seq, onto, y, classes, train_idx, test_idx, groups, cv_seed)
        rows += r
        fold_rows += f
        logger.info("Configuração %s concluída em %.1f min.", cfg.name, (time.time() - t0) / 60)
    out = pd.DataFrame(rows)
    out.to_csv(RESULTS_DIR / "sensibilidade_representacao.csv", index=False)
    pd.DataFrame(fold_rows).to_csv(RESULTS_DIR / "sensibilidade_representacao_folds.csv", index=False)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
