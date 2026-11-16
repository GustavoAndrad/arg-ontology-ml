"""
bootstrap_test.py — Intervalos de confiança no conjunto de teste por bootstrap de grupos.

Por que (TCC2, ponto 5, decisão 5; apontamento D8 do Diego): o teste final é uma
única partição. Um número sozinho (ex.: F1-macro = 0,97) não diz quanto ele
mudaria com outros genes. O bootstrap reamostra o conjunto de teste e mostra essa
variação, sem treinar de novo (as previsões dos modelos finais são reaproveitadas).

Como (decisões registradas em TCC2_resultados_para_o_texto.md, pontos 5 e 10):
    * **Grupos inteiros** (grupos de similaridade de 70%): genes do mesmo grupo são
      quase cópias, e sorteá-los um a um fingiria ter mais informação do que há
      (Hastie et al. 2009, seção 7.11, para o bootstrap).
    * **Estratificado por classe**: em cada reamostragem, sorteiam-se com reposição os
      grupos de cada classe, no mesmo número de grupos que a classe tem no teste.
      Nenhum grupo mistura mecanismos (0 de 894), então cada grupo pertence a uma
      classe. Sem a estratificação, uma reamostragem poderia ficar sem nenhum gene de
      substituição do alvo (11 grupos no teste) ou de proteção (10), e o F1 dessa
      classe não existiria, derrubando a média de forma artificial.
    * **Pareado**: a mesma reamostragem vale para A e B e para os 4 modelos, o que dá
      o intervalo da diferença B − A.
    * **Intervalo de percentil de 95%** (2,5% e 97,5% das reamostragens), coerente
      com α = 0,05.
    * **2.000 reamostragens** (decisão 5 do ponto 5). Para conferir se bastam, as
      reamostragens são divididas em duas metades, e os limites dos intervalos
      calculados em cada metade são comparados: a maior diferença fica em
      ``results/bootstrap_estabilidade.json``. Se intervalos com metade das
      reamostragens já quase não mudam, os de 2.000 são estáveis.
    * Semente: ``RANDOM_STATE``.  # SEED FIXA

Métricas: F1-macro, e precisão, revocação e F1 de cada classe; para cada pipeline e
para a diferença B − A. O valor pontual é o do teste original.

Cenários do ponto 4: além de A e B (anotação curada), entram o B "gene novo", a
regra "ontologia zerada" (sensibilidade) e a referência por homologia, com as
diferenças pareadas de ``DIFFERENCES``. Tudo descritivo (o teste confirmatório é o
teste t corrigido da validação cruzada). Também é feito o recorte dos genes de teste
sem nenhum parecido com identidade ≥ 40% no treino (``without_neighbor_subset``), e a
sensibilidade ao piso da busca (``b_gene_novo_piso40`` e ``homologia_piso40``).

Saídas:
    results/bootstrap_teste.csv           — valor, IC 95% (por modelo, pipeline/diferença, métrica, classe)
    results/bootstrap_teste_sem_vizinho.csv — idem, só nos genes de teste sem parecido ≥ 40% no treino
    results/bootstrap_estabilidade.json   — conferência do número de reamostragens

Uso (depois de ``src.train_evaluate``):
    python -m src.bootstrap_test
"""

from __future__ import annotations

import json
import logging
import sys

import numpy as np
import pandas as pd

from src.experiment_config import (
    ALPHA_STAT,
    MODEL_NAMES,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    load_groups,
)

logger = logging.getLogger(__name__)
PREDICTIONS_DIR = RESULTS_DIR / "predicoes"
N_BOOTSTRAP: int = 2000  # decisão 5 do ponto 5; conferido em bootstrap_estabilidade.json
PERCENTILES = (100 * ALPHA_STAT / 2, 100 * (1 - ALPHA_STAT / 2))  # 2,5 e 97,5
# Cenários do ponto 4 (ver src.train_evaluate.scenario_predictions).
PIPELINE_KEYS: list[str] = ["a", "b", "b_gene_novo", "b_gene_novo_zeros", "b_gene_novo_piso40"]
HOMOLOGY_KEYS: list[str] = ["homologia", "homologia_piso40"]  # sem modelo
# (nome, minuendo, subtraendo): diferenças pareadas calculadas para cada modelo.
DIFFERENCES: list[tuple[str, str, str]] = [
    ("b_menos_a", "b", "a"),                                    # ganho com anotação curada
    ("b_gene_novo_menos_a", "b_gene_novo", "a"),                # ganho no cenário gene novo
    ("b_gene_novo_menos_b", "b_gene_novo", "b"),                # quanto se perde sem a ficha do próprio gene
    ("b_gene_novo_zeros_menos_b_gene_novo", "b_gene_novo_zeros", "b_gene_novo"),  # efeito da regra sem vizinho
    ("homologia_menos_b", "homologia", "b"),                    # referência por homologia contra o B curado
    ("homologia_menos_b_gene_novo", "homologia", "b_gene_novo"),
    ("b_gene_novo_piso40_menos_b_gene_novo", "b_gene_novo_piso40", "b_gene_novo"),  # efeito do piso de 40% na busca
]


def class_metrics_from_predictions(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """
    Matriz (3, n_classes) com precisão, revocação e F1 de cada classe (divisão por zero = 0,
    como ``zero_division=0`` do scikit-learn). Calculada pela matriz de confusão, para ser rápida.
    """
    cm = np.bincount(y_true * n_classes + y_pred, minlength=n_classes * n_classes).reshape(n_classes, n_classes)
    tp = np.diag(cm).astype(float)
    pred_pos, true_pos = cm.sum(axis=0), cm.sum(axis=1)
    precision = np.divide(tp, pred_pos, out=np.zeros(n_classes), where=pred_pos > 0)
    recall = np.divide(tp, true_pos, out=np.zeros(n_classes), where=true_pos > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(n_classes), where=denom > 0)
    return np.vstack([precision, recall, f1])


def stratified_group_resamples(
    y: np.ndarray, groups: np.ndarray, n_resamples: int, seed: int = RANDOM_STATE
) -> list[np.ndarray]:
    """
    Índices de cada reamostragem: para cada classe, sorteia com reposição tantos grupos
    quanto a classe tem e junta todos os genes dos grupos sorteados.
    """
    rng = np.random.default_rng(seed)
    members: dict = {}
    for i, g in enumerate(groups):
        members.setdefault(g, []).append(i)
    members = {g: np.array(idx) for g, idx in members.items()}
    group_class = {g: y[idx[0]] for g, idx in members.items()}
    if any(len(set(y[idx])) > 1 for idx in members.values()):
        raise ValueError("Há grupos com mais de uma classe; a estratificação por classe não se aplica.")
    by_class = {}
    for g, c in group_class.items():
        by_class.setdefault(c, []).append(g)
    by_class = {c: np.array(gs, dtype=object) for c, gs in sorted(by_class.items())}
    out = []
    for _ in range(n_resamples):
        chosen = []
        for gs in by_class.values():
            for g in rng.choice(gs, size=len(gs), replace=True):
                chosen.append(members[g])
        out.append(np.concatenate(chosen))
    return out


def _percentile_interval(values: np.ndarray) -> tuple[float, float]:
    low, high = np.percentile(values, PERCENTILES)
    return float(low), float(high)


def bootstrap_table(
    predictions: dict[tuple[str, str], tuple[np.ndarray, np.ndarray]],
    classes: list[str],
    resamples: list[np.ndarray],
) -> tuple[pd.DataFrame, dict]:
    """
    ``predictions[(pipeline, modelo)] = (y_true, y_pred)`` no teste, todos na mesma ordem.
    Retorna a tabela de intervalos e as amostras de bootstrap (para a conferência de estabilidade).
    """
    n_classes = len(classes)
    names = ["precisao", "revocacao", "f1"]
    samples, rows = {}, []
    for (key, model), (y_true, y_pred) in predictions.items():
        point = class_metrics_from_predictions(y_true, y_pred, n_classes)
        boot = np.array([class_metrics_from_predictions(y_true[idx], y_pred[idx], n_classes) for idx in resamples])
        samples[(key, model)] = boot
        rows.append({"modelo": model, "comparacao": key, "metrica": "f1_macro", "classe": "macro",
                     "valor": point[2].mean(), "boot": boot[:, 2, :].mean(axis=1)})
        for m, name in enumerate(names):
            for k, c in enumerate(classes):
                rows.append({"modelo": model, "comparacao": key, "metrica": name, "classe": c,
                             "valor": point[m, k], "boot": boot[:, m, k]})
    # Diferenças pareadas (mesmas reamostragens). A referência por homologia não tem
    # modelo: fica guardada com modelo = "homologia" e é comparada com o B de cada modelo.
    by_key = {(r["modelo"], r["comparacao"], r["metrica"], r["classe"]): r for r in rows}
    for model in sorted({r["modelo"] for r in rows} - {"homologia"}):
        for name, minuend, subtrahend in DIFFERENCES:
            for (mdl, comp, metric, cls), r in list(by_key.items()):
                if mdl != model or comp != subtrahend:
                    continue
                rb = by_key.get((model, minuend, metric, cls)) or by_key.get(("homologia", minuend, metric, cls))
                if rb is None:
                    continue
                rows.append({"modelo": model, "comparacao": name, "metrica": metric, "classe": cls,
                             "valor": rb["valor"] - r["valor"], "boot": rb["boot"] - r["boot"]})
    table = []
    for r in rows:
        low, high = _percentile_interval(r["boot"])
        table.append({k: r[k] for k in ("modelo", "comparacao", "metrica", "classe", "valor")} | {
            "ic95_inf": low, "ic95_sup": high, "n_reamostragens": len(r["boot"]),
        })
    return pd.DataFrame(table), {(r["modelo"], r["comparacao"], r["metrica"], r["classe"]): r["boot"] for r in rows}


def stability_check(boot: dict) -> dict:
    """Compara os limites do IC calculados na 1ª e na 2ª metade das reamostragens."""
    diffs = []
    for key, values in boot.items():
        half = len(values) // 2
        a, b = _percentile_interval(values[:half]), _percentile_interval(values[half:])
        diffs.append((max(abs(a[0] - b[0]), abs(a[1] - b[1])), key))
    diffs.sort(key=lambda x: x[0])
    macro = [d for d, k in diffs if k[3] == "macro"]
    return {
        "n_reamostragens": len(next(iter(boot.values()))),
        "comparacao": "limites do IC 95% em duas metades independentes das reamostragens",
        "maior_diferenca_f1_macro": max(macro),
        "mediana_diferenca_f1_macro": float(np.median(macro)),
        "maior_diferenca_qualquer_metrica": diffs[-1][0],
        "onde_maior_diferenca": list(diffs[-1][1]),
        "mediana_diferenca_qualquer_metrica": float(np.median([d for d, _ in diffs])),
    }


def load_test_predictions() -> tuple[dict, np.ndarray]:
    """Previsões no teste de todos os pipelines/cenários e da referência por homologia."""
    predictions, y_ref = {}, None
    files = [((key, model), f"pipeline_{key}_{model}.npz") for key in PIPELINE_KEYS for model in MODEL_NAMES]
    files += [((key, "homologia"), f"referencia_{key}.npz") for key in HOMOLOGY_KEYS]
    for key, fname in files:
        d = np.load(PREDICTIONS_DIR / fname)
        if y_ref is None:
            y_ref = d["y_true"]
        if not np.array_equal(d["y_true"], y_ref):
            raise ValueError("Previsões fora de ordem: rode src.train_evaluate de novo.")
        predictions[key] = (d["y_true"], d["y_pred"])
    return predictions, y_ref


def without_neighbor_subset(predictions: dict, groups: np.ndarray, has_neighbor: np.ndarray):
    """
    Recorte dos genes de teste **distantes do treino** (TCC2, ponto 4, P4-7): os que não
    têm nenhum parecido com identidade ≥ 40% (busca do ponto 3). É onde a homologia
    próxima não ajuda; mostra o que cada abordagem faz com genes pouco parecidos com
    os conhecidos. Descritivo (poucos genes).
    """
    keep = ~has_neighbor
    sub = {k: (yt[keep], yp[keep]) for k, (yt, yp) in predictions.items()}
    return sub, groups[keep]


def run(n_resamples: int = N_BOOTSTRAP) -> pd.DataFrame:
    classes = json.loads((RESULTS_DIR / "classes.json").read_text(encoding="utf-8"))
    test_idx = np.load(PROCESSED_DIR / "test_indices.npy")
    groups = load_groups()[test_idx]
    predictions, y_ref = load_test_predictions()
    resamples = stratified_group_resamples(y_ref, groups, n_resamples)
    table, boot = bootstrap_table(predictions, classes, resamples)
    table.to_csv(RESULTS_DIR / "bootstrap_teste.csv", index=False)
    # Recorte "distantes do treino" (mesma semente; reamostragens próprias do recorte).
    neighbors = pd.read_csv(RESULTS_DIR / "vizinho_mais_parecido_teste.csv")
    sub, sub_groups = without_neighbor_subset(predictions, groups, neighbors["tem_vizinho_piso40"].to_numpy(bool))
    y_sub = next(iter(sub.values()))[0]
    sub_table, _ = bootstrap_table(sub, classes, stratified_group_resamples(y_sub, sub_groups, n_resamples))
    sub_table.insert(0, "n_genes", len(y_sub))
    sub_table.to_csv(RESULTS_DIR / "bootstrap_teste_sem_vizinho.csv", index=False)
    stability = stability_check(boot)
    (RESULTS_DIR / "bootstrap_estabilidade.json").write_text(json.dumps(stability, indent=2, ensure_ascii=False), encoding="utf-8")
    macro = table[(table["metrica"] == "f1_macro")]
    for _, r in macro.iterrows():
        logger.info("%-20s %-9s F1-macro = %.4f [%.4f; %.4f]", r["modelo"], r["comparacao"], r["valor"], r["ic95_inf"], r["ic95_sup"])
    logger.info("Estabilidade (metades de %d): maior diferença nos limites do F1-macro = %.4f",
                n_resamples // 2, stability["maior_diferenca_f1_macro"])
    return table


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
