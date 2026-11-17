"""
interpretability.py — O que cada modelo usa para decidir (Seção 4.11 e capítulo 5; TCC2, ponto 8).

Pergunta (apontamentos J3 e J4 da Jéssica): quais atributos mais pesam nos dois
pipelines, e por quê? Este módulo não muda o experimento: analisa os mesmos modelos,
com as mesmas divisões.

**Importância por permutação** (Hastie et al. 2009, seção 15.3.2): os valores de um
atributo (ou de um grupo de atributos, juntos) são embaralhados entre os genes
avaliados, e mede-se quanto o F1 cai. Se cai muito, o modelo dependia daquele
atributo; se não muda, não o usava. Funciona com qualquer modelo, então é aplicada
aos **4 classificadores** (a decisão de 11/11 previa só RF e XGBoost, "os melhores";
hoje os melhores no B são SVM, XGBoost e Regressão Logística, P6-C1).

* **Por grupo (resultado principal):** atributos parecidos entre si (profundidade e
  número de ancestrais, por exemplo) dividem a importância quando embaralhados um a
  um; embaralhando o grupo inteiro (a mesma permutação de linhas para todas as
  colunas do grupo), mede-se o quanto o modelo depende daquele tipo de informação.
  Grupos: ``fisico_quimicas`` (4), ``aminoacidos`` (20), ``pares_de_grupos`` (49),
  ``estruturais`` (3, só no B) e ``classes_de_antibiotico`` (38, só no B).
* **Por mecanismo:** a mesma permutação mede a queda do F1 **de cada classe**, que
  diz qual grupo o modelo usa para reconhecer cada mecanismo, nos 4 modelos e com a
  mesma métrica do resto do trabalho.
* **Por atributo:** para o top-15 e a estabilidade.

**Onde é medida:** nos **50 folds da validação cruzada** (resultado principal, com o
intervalo de 95% corrigido de Nadeau e Bengio sobre os folds) e no teste final
(complemento). O teste sozinho é dominado pela família PDC (61% do teste; P5-D6),
que distorceria a importância.

**Repetições:** por grupo, 30 permutações em cada fold e no teste (decisão 1 do ponto
8). Por atributo, 30 no teste; nos folds, **1 permutação por fold**: cada fold já
traz outros dados, e a média dos 50 folds equivale a 50 repetições. Fazer 30 em cada
fold custaria ~17 h só para o SVM. A equivalência é **conferida nos dados**
(``check_repeat_stability``: com a Regressão Logística e o XGBoost, que são rápidos,
1 e 30 permutações por fold são comparadas nos 50 folds; resultado em
``results/importancia_estabilidade_repeticoes.json``).

**Estabilidade (decisão 2):** em vez de "aparece no top em X folds" (um X
arbitrário), cada atributo tem a média nos 50 folds com o IC corrigido; é
**consistente** quando o limite inferior do IC é maior que zero.

**Cenário "gene novo" (decisão 4):** a importância por grupo também é medida com os
atributos ontológicos do gene mais parecido do treino (``src.nearest_neighbor``),
entre os genes que têm vizinho (os sem vizinho são classificados pelo Pipeline A e
não usam a ontologia).

**SHAP por mecanismo (decisão 1), como complemento:** só para os modelos de árvore,
onde o cálculo é exato e rápido: Random Forest (``shap.TreeExplainer``; Lundberg e
Lee 2017, referência a adicionar) e XGBoost (cálculo nativo do próprio XGBoost,
``pred_contribs=True``, o mesmo algoritmo TreeSHAP; o ``shap`` 0.45 não lê modelos do
XGBoost 3.2). Para SVM e Regressão Logística o SHAP exigiria uma aproximação lenta;
a permutação por classe cobre esses modelos. O que o SHAP acrescenta à permutação é
a **direção**: para os genes de cada mecanismo, quais atributos empurram o modelo
para esse mecanismo (valor médio positivo) ou para longe dele. Calculado no teste.

**Dicionário de atributos (decisão 3, J4):** os 73 de sequência e os 41 da ontologia,
com grupo e significado (``results/dicionario_atributos.csv``).

**Cuidado no texto (decisão 5):** importância indica associação dentro do CARD, não
causa (Orcales et al. 2024).

A importância Gini do Random Forest deixou de ser calculada: favorece atributos com
muitos valores distintos e foi substituída pela permutação. Os arquivos antigos
(``feature_importance_gini_rf_*.csv`` e a figura) ficam no repositório.

Sementes: as permutações usam geradores derivados de ``RANDOM_STATE`` e da posição
(modelo, repetição, fold, contexto), então são reproduzíveis e independentes.  # SEED FIXA

Saídas:
    results/dicionario_atributos.csv
    results/importancia_grupos.csv, results/importancia_grupos_folds.csv
    results/importancia_atributos.csv, results/importancia_atributos_folds.csv
    results/importancia_estabilidade_repeticoes.json
    results/shap_por_classe.csv

Uso (depois de ``src.train_evaluate``):
    python -m src.interpretability                      # todos os modelos em sequência (~3 h) e junta
    python -m src.interpretability --modelo svm         # um modelo (para rodar os 4 em paralelo)
    python -m src.interpretability --juntar             # junta os parciais dos 4 modelos
"""

from __future__ import annotations

import json
import logging
import sys
import time

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.experiment_config import (
    MODEL_NAMES,
    MODELS_DIR,
    N_PERMUTATION_REPEATS,
    PROCESSED_DIR,
    RANDOM_STATE,
    RESULTS_DIR,
    choose_cv_seeds,
    load_experiment_dataset,
    load_groups,
    repeated_cv_splits,
)
from src.feature_extraction import AMINO_ACIDS
from src.nearest_neighbor import NO_NEIGHBOR, NeighborIndex, new_gene_features
from src.ontology_features import drug_class_column, drug_class_vocabulary
from src.pipeline_builder import build_full_pipeline
from src.statistical_tests import corrected_interval
from src.train_evaluate import load_feature_matrices, ontology_columns

logger = logging.getLogger(__name__)

GROUP_ORDER: list[str] = ["fisico_quimicas", "aminoacidos", "pares_de_grupos", "estruturais", "classes_de_antibiotico"]
PHYSCHEM: dict[str, str] = {
    "length": "comprimento da proteína (número de resíduos)",
    "isoelectric_point": "ponto isoelétrico (pH em que a carga líquida da proteína é zero; Biopython ProtParam)",
    "instability_index": "índice de instabilidade da proteína (Biopython ProtParam; referência a adicionar)",
    "charge_at_pH7": "carga líquida estimada em pH 7 (Biopython ProtParam)",
}
STRUCTURAL: dict[str, str] = {
    "depth_isa": "profundidade do termo do gene na hierarquia is-a da ARO (menor caminho até a raiz)",
    "ancestor_count": "número de termos ancestrais do gene na hierarquia is-a da ARO",
    "children_count": "número de subclasses diretas do termo do gene na ARO (quase sempre 0: os genes são folhas)",
}
AA_NAMES: dict[str, str] = {
    "A": "alanina", "C": "cisteína", "D": "aspartato", "E": "glutamato", "F": "fenilalanina",
    "G": "glicina", "H": "histidina", "I": "isoleucina", "K": "lisina", "L": "leucina",
    "M": "metionina", "N": "asparagina", "P": "prolina", "Q": "glutamina", "R": "arginina",
    "S": "serina", "T": "treonina", "V": "valina", "W": "triptofano", "Y": "tirosina",
}
CONTEXTS: dict[str, str] = {"a": "Pipeline A", "b": "Pipeline B (anotação curada)", "b_gene_novo": "Pipeline B (gene novo)"}


# ── Auxiliares usados pela sensibilidade à representação ─────────────────────────
SEQ_PREFIXES = (
    "length", "molecular_weight", "isoelectric_point", "aromaticity", "instability_index",
    "charge_at_pH7", "aa_comp_", "pair_", "dipep_",
)


def feature_type(name: str) -> str:
    """"sequencia" ou "ontologica" pelo nome (inclui os nomes da representação de 426 atributos)."""
    return "sequencia" if name.startswith(SEQ_PREFIXES) else "ontologica"


def get_feature_names_after_pipeline(pipeline, feature_names_in: list[str]) -> list[str]:
    """
    Nomes dos atributos que chegam ao classificador. No experimento não há seleção, então
    são todos; com a seleção do TCC1 (etapas "variance" e "select", usada só na
    sensibilidade à representação), aplica as máscaras dela.
    """
    names = np.asarray(feature_names_in)
    if "select" in pipeline.named_steps:
        names = names[pipeline.named_steps["variance"].get_support()]
        names = names[pipeline.named_steps["select"].get_support()]
    n_in = pipeline.named_steps["clf"].n_features_in_
    if len(names) != n_in:
        raise ValueError(f"{len(names)} nomes para {n_in} atributos do classificador.")
    return list(names)


# ── Grupos e dicionário ────────────────────────────────────────────────────────
def feature_group(name: str) -> str:
    """Grupo de um atributo pelo nome."""
    if name in PHYSCHEM:
        return "fisico_quimicas"
    if name.startswith("aa_comp_"):
        return "aminoacidos"
    if name.startswith("pair_"):
        return "pares_de_grupos"
    if name in STRUCTURAL:
        return "estruturais"
    if name.startswith("drug_class_"):
        return "classes_de_antibiotico"
    raise ValueError(f"Atributo sem grupo conhecido: {name}")


def group_columns(columns: list[str]) -> dict[str, np.ndarray]:
    """{grupo: posições das colunas}, na ordem de ``GROUP_ORDER``; cada coluna em um grupo só."""
    groups = [feature_group(c) for c in columns]
    return {g: np.array([i for i, x in enumerate(groups) if x == g]) for g in GROUP_ORDER if g in groups}


def feature_dictionary(seq_columns: list[str], onto_columns: list[str], drug_class_values: pd.Series) -> pd.DataFrame:
    """Todos os atributos com grupo, pipelines e significado (apêndice do texto; J4)."""
    vocab = {drug_class_column(c): c for c in drug_class_vocabulary(drug_class_values)}
    rows = []
    for c in seq_columns + onto_columns:
        g = feature_group(c)
        if g == "fisico_quimicas":
            desc = PHYSCHEM[c]
        elif g == "aminoacidos":
            aa = c.removeprefix("aa_comp_")
            desc = f"fração do aminoácido {aa} ({AA_NAMES[aa]}) na sequência"
        elif g == "pares_de_grupos":
            g1, g2 = c.removeprefix("pair_").split("_")
            desc = (f"frequência de pares de resíduos vizinhos em que um resíduo do grupo {{{g1}}} é seguido "
                    f"por um do grupo {{{g2}}} (alfabeto reduzido de 7 grupos, Shen et al. 2007)")
        elif g == "estruturais":
            desc = STRUCTURAL[c]
        else:
            name = vocab.get(c)
            if name is None:
                raise ValueError(f"Classe de antibiótico sem nome original: {c}")
            desc = f"1 se o CARD registra resistência à classe \"{name}\" (coluna Drug Class); 0 caso contrário"
        rows.append({"atributo": c, "grupo": g, "pipelines": "A e B" if c in seq_columns else "B", "significado": desc})
    out = pd.DataFrame(rows)
    flags = drug_class_values.map(lambda v: set(x.strip() for x in str(v).split(";")))
    counts = {drug_class_column(n): int(flags.map(lambda s: n in s).sum()) for n in vocab.values()}
    out["genes_com_valor_1"] = out["atributo"].map(counts)
    return out


# ── Permutação ─────────────────────────────────────────────────────────────────
def _scores(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> tuple[float, np.ndarray]:
    labels = list(range(n_classes))
    per_class = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    return float(per_class.mean()), per_class


def permutation_drops(
    predict, X: np.ndarray, y: np.ndarray, column_sets: dict[str, np.ndarray], n_classes: int,
    n_repeats: int, rng: np.random.Generator,
) -> list[dict]:
    """
    Queda do F1-macro e do F1 de cada classe quando as colunas de cada conjunto são
    embaralhadas **juntas** (a mesma permutação de linhas para todas as colunas do
    conjunto), média de ``n_repeats`` permutações. ``predict`` recebe a matriz inteira.
    """
    base_macro, base_cls = _scores(y, predict(X), n_classes)
    rows = []
    for name, cols in column_sets.items():
        macro, cls = [], []
        for _ in range(n_repeats):
            perm = rng.permutation(len(X))
            Xp = X.copy()
            Xp[:, cols] = X[perm][:, cols]
            m, c = _scores(y, predict(Xp), n_classes)
            macro.append(base_macro - m)
            cls.append(base_cls - c)
        cls = np.array(cls)
        row = {"conjunto": name, "queda_f1_macro": float(np.mean(macro)),
               "queda_f1_macro_dp_repeticoes": float(np.std(macro, ddof=1)) if n_repeats > 1 else np.nan}
        row |= {f"queda_f1_classe_{k}": float(v) for k, v in enumerate(cls.mean(axis=0))}
        rows.append(row)
    return rows


def _rng(*key: int) -> np.random.Generator:
    """Gerador reproduzível e independente para cada (modelo, repetição, fold, contexto).  # SEED FIXA"""
    return np.random.default_rng(np.random.SeedSequence([RANDOM_STATE, *key]))


def split_contexts(pipe_a, pipe_b, X_a, X_b, y, eval_idx, neighbors, onto_cols):
    """
    (X, y, predict) de cada contexto avaliado numa divisão: A, B curado e B gene novo
    (só os genes com vizinho; ver docstring do módulo).
    """
    X_new, has = new_gene_features(X_b, eval_idx, neighbors, onto_cols)
    return {
        "a": (X_a[eval_idx], y[eval_idx], pipe_a.predict),
        "b": (X_b[eval_idx], y[eval_idx], pipe_b.predict),
        "b_gene_novo": (X_new[has], y[eval_idx][has], pipe_b.predict),
    }


def evaluate_split(
    model_idx, key_rep, key_fold, pipe_a, pipe_b, X_a, X_b, y, eval_idx, neighbors, onto_cols,
    groups_a, groups_b, feats_a, feats_b, n_classes, group_repeats, feature_repeats, with_features=True,
):
    """Permutação por grupo (3 contextos) e por atributo (A e B curado) numa divisão."""
    ctx = split_contexts(pipe_a, pipe_b, X_a, X_b, y, eval_idx, neighbors, onto_cols)
    group_rows, feat_rows = [], []
    for c_idx, (name, (X, yy, predict)) in enumerate(ctx.items()):
        sets = groups_a if name == "a" else groups_b
        rows = permutation_drops(predict, X, yy, sets, n_classes, group_repeats, _rng(model_idx, key_rep, key_fold, c_idx))
        group_rows += [{"contexto": name, "n_genes": len(yy), **r} for r in rows]
        if with_features and name in ("a", "b"):
            fsets = feats_a if name == "a" else feats_b
            rows = permutation_drops(predict, X, yy, fsets, n_classes, feature_repeats,
                                     _rng(model_idx, key_rep, key_fold, 10 + c_idx))
            feat_rows += [{"contexto": name, **r} for r in rows]
    return group_rows, feat_rows


# ── Resumo nos folds ───────────────────────────────────────────────────────────
def summarize_folds(folds: pd.DataFrame, test: pd.DataFrame, ratio: float, classes: list[str]) -> pd.DataFrame:
    """Média nos 50 folds com IC 95% corrigido (macro e por classe), ao lado do valor no teste."""
    value_cols = ["queda_f1_macro"] + [f"queda_f1_classe_{k}" for k in range(len(classes))]
    keys = ["contexto", "modelo", "conjunto"]
    rows = []
    for key, sub in folds.groupby(keys, sort=False):
        row = dict(zip(keys, key))
        t = test[(test["contexto"] == key[0]) & (test["modelo"] == key[1]) & (test["conjunto"] == key[2])]
        for col in value_cols:
            v = sub[col].to_numpy(dtype=float)
            low, high = corrected_interval(v, ratio) if np.ptp(v) > 0 else (float(v.mean()), float(v.mean()))
            label = "macro" if col == "queda_f1_macro" else classes[int(col.rsplit("_", 1)[1])]
            row |= {f"cv_media_{label}": float(v.mean()), f"cv_ic95_inf_{label}": low, f"cv_ic95_sup_{label}": high,
                    f"teste_{label}": float(t[col].iloc[0]) if len(t) else np.nan}
        row["consistente"] = bool(row["cv_ic95_inf_macro"] > 0)
        rows.append(row)
    out = pd.DataFrame(rows)
    out["posicao_cv"] = out.groupby(["contexto", "modelo"])["cv_media_macro"].rank(ascending=False, method="first").astype(int)
    return out


def check_repeat_stability(fold_feats_1: pd.DataFrame, rerun_30) -> dict:
    """
    Confere se 1 permutação por fold basta: para a Regressão Logística e o XGBoost
    (rápidos), refaz a permutação por atributo com 30 repetições em cada fold e compara
    as médias nos 50 folds (correlação de postos e sobreposição do top-15).
    """
    from scipy.stats import spearmanr

    out = {"comparacao": "média nos 50 folds da queda do F1-macro por atributo: 1 × 30 permutações por fold"}
    for model, ctx, df30 in rerun_30():
        m1 = fold_feats_1[(fold_feats_1["modelo"] == model) & (fold_feats_1["contexto"] == ctx)]
        a = m1.groupby("conjunto")["queda_f1_macro"].mean()
        b = df30.groupby("conjunto")["queda_f1_macro"].mean().loc[a.index]
        top1, top30 = set(a.nlargest(15).index), set(b.nlargest(15).index)
        out[f"{model}_{ctx}"] = {
            "correlacao_de_postos": float(spearmanr(a, b).statistic),
            "top15_em_comum": len(top1 & top30),
            "maior_diferenca_absoluta": float((a - b).abs().max()),
        }
    return out


# ── SHAP (árvores) ─────────────────────────────────────────────────────────────
def tree_shap_values(pipe, X: np.ndarray, model: str) -> np.ndarray:
    """Valores SHAP (genes × atributos × classes). RF: shap.TreeExplainer; XGBoost: nativo."""
    Xt = pipe[:-1].transform(X)
    clf = pipe.named_steps["clf"]
    if model == "xgboost":
        import xgboost as xgb

        contrib = clf.get_booster().predict(xgb.DMatrix(Xt), pred_contribs=True)  # genes × classes × (atributos + viés)
        return np.transpose(contrib[:, :, :-1], (0, 2, 1))
    import shap

    values = shap.TreeExplainer(clf).shap_values(Xt)
    values = np.array(values)
    return values if values.shape[0] == len(X) else np.transpose(values, (1, 2, 0))


def shap_per_class(values: np.ndarray, y: np.ndarray, columns: list[str], classes: list[str]) -> pd.DataFrame:
    """
    Para os genes de cada mecanismo: média de |SHAP| na saída daquele mecanismo (quanto
    o atributo pesa) e média do SHAP com sinal (direção: positivo empurra para o
    mecanismo), por atributo e somada por grupo.
    """
    rows = []
    groups = [feature_group(c) for c in columns]
    for k, cls in enumerate(classes):
        sv = values[y == k][:, :, k]
        for j, c in enumerate(columns):
            rows.append({"classe": cls, "nivel": "atributo", "nome": c, "grupo": groups[j],
                         "media_abs_shap": float(np.abs(sv[:, j]).mean()), "media_shap": float(sv[:, j].mean())})
        for g in dict.fromkeys(groups):
            cols = [j for j, x in enumerate(groups) if x == g]
            s = sv[:, cols].sum(axis=1)
            rows.append({"classe": cls, "nivel": "grupo", "nome": g, "grupo": g,
                         "media_abs_shap": float(np.abs(s).mean()), "media_shap": float(s.mean())})
    return pd.DataFrame(rows)


# ── Execução ───────────────────────────────────────────────────────────────────
# Cada modelo pode rodar num processo separado (``--modelo``), em paralelo; os
# resultados parciais ficam em ``data/processed/interpretacao_{modelo}_*.csv`` (fora
# do git) e ``--juntar`` monta os arquivos finais. As sementes de cada permutação
# dependem só de (modelo, repetição, fold, contexto), então o resultado não depende
# da ordem nem da divisão em processos.
PARTIAL = "interpretacao_{model}_{part}.csv"
PARTS = ("grupos_folds", "atributos_folds", "grupos_teste", "atributos_teste", "shap", "atributos_folds_30")


def _setup():
    X_a_df, X_b_df, y, classes = load_feature_matrices()
    data = load_experiment_dataset()
    groups = load_groups(data=data)
    train_idx = np.load(PROCESSED_DIR / "train_indices.npy")
    test_idx = np.load(PROCESSED_DIR / "test_indices.npy")
    cv_seeds, _ = choose_cv_seeds(y[train_idx], groups[train_idx])
    saved = json.loads((RESULTS_DIR / "particao_teste.json").read_text(encoding="utf-8"))
    if saved.get("sementes_cv", cv_seeds) != cv_seeds:
        raise ValueError("Sementes da CV diferentes das do experimento: rode src.train_evaluate de novo.")
    splits = repeated_cv_splits(y[train_idx], groups[train_idx], cv_seeds)
    return X_a_df, X_b_df, y, classes, data, train_idx, test_idx, splits


def run_model(model: str) -> None:
    """Permutação (folds e teste), SHAP (árvores) e, na RL e no XGBoost, a conferência 1 × 30 permutações."""
    t0 = time.time()
    X_a_df, X_b_df, y, classes, data, train_idx, test_idx, splits = _setup()
    onto_cols = ontology_columns(X_a_df, X_b_df)
    X_a, X_b = X_a_df.to_numpy(dtype=float), X_b_df.to_numpy(dtype=float)
    cols_a, cols_b = list(X_a_df.columns), list(X_b_df.columns)
    n_classes = len(classes)
    m_idx = MODEL_NAMES.index(model)
    nn = NeighborIndex.from_files(data["ARO_Accession"].to_numpy())
    groups_a, groups_b = group_columns(cols_a), group_columns(cols_b)
    feats_a = {c: np.array([i]) for i, c in enumerate(cols_a)}
    feats_b = {c: np.array([i]) for i, c in enumerate(cols_b)}

    fold_groups, fold_feats, fold_feats_30 = [], [], []
    check_30 = model in ("logistic_regression", "xgboost")
    for n, (rep, fold, fit_local, val_local) in enumerate(splits, 1):
        fit_idx, val_idx = train_idx[fit_local], train_idx[val_local]
        pipe_a = build_full_pipeline(model).fit(X_a[fit_idx], y[fit_idx])
        pipe_b = build_full_pipeline(model).fit(X_b[fit_idx], y[fit_idx])
        nb, _ = nn.nearest(val_idx, fit_idx)
        g_rows, f_rows = evaluate_split(m_idx, rep, fold, pipe_a, pipe_b, X_a, X_b, y, val_idx, nb, onto_cols,
                                        groups_a, groups_b, feats_a, feats_b, n_classes, N_PERMUTATION_REPEATS, 1)
        base = {"modelo": model, "repeticao": rep, "fold": fold}
        fold_groups += [{**base, **r} for r in g_rows]
        fold_feats += [{**base, **r} for r in f_rows]
        if check_30:
            rows = permutation_drops(pipe_b.predict, X_b[val_idx], y[val_idx], feats_b, n_classes,
                                     N_PERMUTATION_REPEATS, _rng(m_idx, rep, fold, 99))
            fold_feats_30 += [{**base, "contexto": "b", **r} for r in rows]
        if n % 10 == 0:
            logger.info("%s: %d de %d folds (%.1f min).", model, n, len(splits), (time.time() - t0) / 60)

    # Teste final (modelos finais do train_evaluate): 30 permutações por grupo e por atributo.
    pipe_a = joblib.load(MODELS_DIR / f"pipeline_a_{model}.joblib")
    pipe_b = joblib.load(MODELS_DIR / f"pipeline_b_{model}.joblib")
    nb, _ = nn.nearest(test_idx, train_idx)
    g_rows, f_rows = evaluate_split(m_idx, 0, 0, pipe_a, pipe_b, X_a, X_b, y, test_idx, nb, onto_cols, groups_a,
                                    groups_b, feats_a, feats_b, n_classes, N_PERMUTATION_REPEATS, N_PERMUTATION_REPEATS)
    logger.info("%s: teste concluído (%.1f min).", model, (time.time() - t0) / 60)
    shap_rows = []
    if model in ("random_forest", "xgboost"):
        for key, X, cols, pipe in (("a", X_a, cols_a, pipe_a), ("b", X_b, cols_b, pipe_b)):
            sv = tree_shap_values(pipe, X[test_idx], model)
            shap_rows.append(shap_per_class(sv, y[test_idx], cols, classes).assign(modelo=model, pipeline=key))
    out = {
        "grupos_folds": pd.DataFrame(fold_groups), "atributos_folds": pd.DataFrame(fold_feats),
        "grupos_teste": pd.DataFrame([{"modelo": model, **r} for r in g_rows]),
        "atributos_teste": pd.DataFrame([{"modelo": model, **r} for r in f_rows]),
        "shap": pd.concat(shap_rows) if shap_rows else pd.DataFrame(),
        "atributos_folds_30": pd.DataFrame(fold_feats_30),
    }
    for part, df in out.items():
        df.to_csv(PROCESSED_DIR / PARTIAL.format(model=model, part=part), index=False)
    logger.info("%s concluído em %.1f min.", model, (time.time() - t0) / 60)


def join_results(model_names=MODEL_NAMES) -> None:
    """Junta os parciais dos modelos e grava os arquivos finais em ``results/``."""
    X_a_df, X_b_df, y, classes, data, train_idx, test_idx, splits = _setup()
    ratio = float(np.mean([len(v) / len(f) for _, _, f, v in splits]))
    feature_dictionary(list(X_a_df.columns), list(X_b_df.columns[X_a_df.shape[1]:]), data["Drug_Class"]).to_csv(
        RESULTS_DIR / "dicionario_atributos.csv", index=False)

    def load(part):
        frames = []
        for m in model_names:
            path = PROCESSED_DIR / PARTIAL.format(model=m, part=part)
            if not path.exists():
                raise FileNotFoundError(f"Falta {path}: rode python -m src.interpretability --modelo {m}.")
            if path.stat().st_size > 2:
                try:
                    frames.append(pd.read_csv(path))
                except pd.errors.EmptyDataError:
                    pass
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    fg, ff, tg, tf = load("grupos_folds"), load("atributos_folds"), load("grupos_teste"), load("atributos_teste")
    fg.to_csv(RESULTS_DIR / "importancia_grupos_folds.csv", index=False)
    ff.to_csv(RESULTS_DIR / "importancia_atributos_folds.csv", index=False)
    summarize_folds(fg, tg, ratio, classes).to_csv(RESULTS_DIR / "importancia_grupos.csv", index=False)
    summarize_folds(ff, tf, ratio, classes).to_csv(RESULTS_DIR / "importancia_atributos.csv", index=False)
    shap = load("shap")
    if len(shap):
        shap.to_csv(RESULTS_DIR / "shap_por_classe.csv", index=False)
    f30 = load("atributos_folds_30")

    def rerun_30():
        for m in ("logistic_regression", "xgboost"):
            sub = f30[f30["modelo"] == m]
            if len(sub):
                yield m, "b", sub

    stability = check_repeat_stability(ff, rerun_30)
    (RESULTS_DIR / "importancia_estabilidade_repeticoes.json").write_text(
        json.dumps(stability, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Estabilidade 1 × 30 permutações: %s", stability)


if __name__ == "__main__":
    import argparse

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    parser = argparse.ArgumentParser(description="Importância por permutação, SHAP e dicionário de atributos.")
    parser.add_argument("--modelo", choices=MODEL_NAMES, help="roda só este modelo (para rodar em paralelo)")
    parser.add_argument("--juntar", action="store_true", help="junta os parciais e grava os arquivos finais")
    args = parser.parse_args()
    if args.modelo:
        run_model(args.modelo)
    elif args.juntar:
        join_results()
    else:
        for m in MODEL_NAMES:
            run_model(m)
        join_results()
