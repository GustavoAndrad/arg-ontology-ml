"""
case_study.py — Estudo de caso qualitativo dos mecanismos raros (TCC2, ponto 2).

Os mecanismos com poucos genes para a avaliação quantitativa (``CASE_STUDY_CLASSES``:
*reduced permeability to antibiotic*, 8 genes, e *resistance by host-dependent
nutrient acquisition*, 1 gene) ficam fora do treino. Aqui, os 8 modelos finais
(4 algoritmos × Pipelines A e B), treinados com as 5 classes do experimento, são
aplicados a esses genes.

O que se observa, para cada gene: a que mecanismo conhecido ele é atribuído, com
que probabilidade, e se o Pipeline B responde diferente do A.

**Não é avaliação de desempenho.** O mecanismo verdadeiro não está entre as 5
classes que o modelo conhece, então ele sempre "erra" por construção e não há F1.
Com 9 genes, as conclusões são ilustrativas (o texto do TCC deve dizer isso).

Os modelos usados são os mesmos avaliados no conjunto de teste (treinados nos 80%
de treino), para o estudo de caso refletir exatamente os modelos reportados.

Entradas: ``models/pipeline_{a,b}_{modelo}.joblib`` e ``results/classes.json``
(gerados por ``src.train_evaluate``) e os atributos do estudo de caso (gerados por
``src.feature_extraction`` e ``src.ontology_features``).

Saídas:
    results/estudo_caso_predicoes.csv — uma linha por gene × pipeline × modelo, com
        a classe prevista e a probabilidade de cada uma das 5 classes;
    results/estudo_caso_resumo.csv    — uma linha por gene, com a classe prevista
        por cada pipeline/modelo.

Uso:
    python -m src.case_study
"""

from __future__ import annotations

import json
import logging
import sys

import joblib
import numpy as np
import pandas as pd

from src.experiment_config import MODEL_NAMES, MODELS_DIR, PROCESSED_DIR, RESULTS_DIR, TARGET, load_case_study_dataset

logger = logging.getLogger(__name__)
INFO_COLUMNS = ["ARO_Accession", TARGET, "AMR_Gene_Family", "Drug_Class"]


def short_name(mechanism: str) -> str:
    """'antibiotic target protection' -> 'target protection' (para tabelas compactas)."""
    return mechanism.replace("antibiotic ", "").replace(" to antibiotic", "")


def load_case_study_matrices() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Retorna (info, X_a, X_b) dos genes do estudo de caso, com as colunas na mesma
    ordem das matrizes de treino.

    As colunas ontológicas seguem ``features_ontologicas.csv`` do experimento (já
    depois do ``leakage_check``), para o caso de alguma ter sido removida lá.
    """
    info = load_case_study_dataset()
    seq = pd.read_csv(PROCESSED_DIR / "features_sequencia_estudo_caso.csv")
    onto = pd.read_csv(PROCESSED_DIR / "features_ontologicas_estudo_caso.csv")
    for name, f in (("sequência", seq), ("ontologia", onto)):
        if not (f["ARO_Accession"].values == info["ARO_Accession"].values).all():
            raise ValueError(f"Atributos de {name} do estudo de caso desalinhados; reexecute a extração.")
    seq_cols = pd.read_csv(PROCESSED_DIR / "features_sequencia.csv", nrows=0).columns.drop("ARO_Accession")
    onto_cols = pd.read_csv(PROCESSED_DIR / "features_ontologicas.csv", nrows=0).columns.drop("ARO_Accession")
    X_a = seq[seq_cols]
    X_b = pd.concat([X_a, onto[onto_cols]], axis=1)  # mesma concatenação de train_evaluate
    return info[INFO_COLUMNS], X_a, X_b


def predict_case_study(info: pd.DataFrame, X_a: pd.DataFrame, X_b: pd.DataFrame, classes: list[str]) -> pd.DataFrame:
    """Aplica os 8 modelos finais e devolve a tabela longa de previsões."""
    rows = []
    for key, X in (("a", X_a), ("b", X_b)):
        Xn = X.to_numpy(dtype=float)
        for model in MODEL_NAMES:
            pipe = joblib.load(MODELS_DIR / f"pipeline_{key}_{model}.joblib")
            proba = pipe.predict_proba(Xn)
            pred = proba.argmax(axis=1)
            for i in range(len(info)):
                rows.append(
                    {
                        **info.iloc[i].to_dict(),
                        "pipeline": key,
                        "modelo": model,
                        "classe_prevista": classes[pred[i]],
                        "prob_prevista": float(proba[i, pred[i]]),
                        **{f"prob_{short_name(c).replace(' ', '_')}": float(p) for c, p in zip(classes, proba[i])},
                    }
                )
    out = pd.DataFrame(rows)
    prob_cols = [c for c in out.columns if c.startswith("prob_") and c != "prob_prevista"]
    assert np.allclose(out[prob_cols].sum(axis=1), 1.0), "Probabilidades não somam 1"
    return out


def summarize(pred: pd.DataFrame) -> pd.DataFrame:
    """Uma linha por gene; uma coluna por pipeline/modelo com a classe prevista (nome curto)."""
    wide = pred.assign(col=pred["pipeline"].str.upper() + "_" + pred["modelo"]).pivot(
        index="ARO_Accession", columns="col", values="classe_prevista"
    )
    wide = wide.map(short_name)[[f"{k.upper()}_{m}" for k in ("a", "b") for m in MODEL_NAMES]]
    info = pred.drop_duplicates("ARO_Accession").set_index("ARO_Accession")[[TARGET, "AMR_Gene_Family"]]
    return info.join(wide).reset_index()


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    classes = json.loads((RESULTS_DIR / "classes.json").read_text(encoding="utf-8"))
    info, X_a, X_b = load_case_study_matrices()
    pred = predict_case_study(info, X_a, X_b, classes)
    pred.to_csv(RESULTS_DIR / "estudo_caso_predicoes.csv", index=False)
    summary = summarize(pred)
    summary.to_csv(RESULTS_DIR / "estudo_caso_resumo.csv", index=False)
    logger.info("Estudo de caso: %d genes × %d modelos.", len(info), len(pred) // len(info))
    pd.set_option("display.width", 250, "display.max_columns", 20)
    logger.info("\n%s", summary.drop(columns="AMR_Gene_Family").to_string(index=False))
