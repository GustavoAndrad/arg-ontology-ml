"""Testes dos módulos de extração, pipelines, estatística e relatório."""

import numpy as np
import pandas as pd
import pytest

from src.experiment_config import (
    CASE_STUDY_CLASSES,
    EFFECTIVE_CLASSES,
    MODEL_NAMES,
    N_CV_SPLITS,
    choose_test_partition,
    SPLIT_CASE_STUDY,
    SPLIT_EXPERIMENT,
    SPLIT_MULTI_LABEL,
    TARGET,
    assign_split,
    cv_splitter,
    dataset_composition,
    partition_dataset,
    split_indices,
)
from src.feature_extraction import (
    AMINO_ACIDS,
    REDUCED_ALPHABET,
    REPRESENTATIONS,
    SEQUENCE_FEATURE_COLUMNS,
    compute_amino_acid_composition,
    compute_dipeptide_composition,
    compute_group_pair_composition,
    compute_sequence_features,
    count_ambiguous_residues,
)
from src.leakage_check import (
    ABLATION_CONFIGS,
    AblationConfig,
    config_columns,
    entropy_nats,
    mutual_information_table,
)
from src.nearest_neighbor import (
    NO_NEIGHBOR,
    NeighborIndex,
    homology_prediction,
    new_gene_features,
    predict_new_gene,
)
from src.ontology_features import (
    IsAGraph,
    compute_ancestor_count,
    compute_children_count,
    compute_hierarchy_depth,
    extract_drug_class_flags,
    iri_to_accession,
)
from src.pipeline_builder import build_full_pipeline
from src.statistical_tests import corrected_resampled_ttest, holm_adjust

SEQ = "MKTAYIAKQRQISFVKSHFSRQ"


def test_sequence_features_count_and_order():
    # Experimento: 4 físico-químicas + 20 aminoácidos + 49 pares de grupos = 73.
    feats = compute_sequence_features(SEQ)
    assert len(SEQUENCE_FEATURE_COLUMNS) == 73
    assert list(feats) == SEQUENCE_FEATURE_COLUMNS
    assert "molecular_weight" not in feats and "aromaticity" not in feats
    # Representação do TCC1, mantida para a análise de sensibilidade: 6 + 20 + 400 = 426.
    legacy = compute_sequence_features(SEQ, representation="completa_426")
    assert list(legacy) == REPRESENTATIONS["completa_426"] and len(legacy) == 426
    # As 24 colunas em comum têm os mesmos valores nas duas representações.
    assert all(legacy[c] == feats[c] for c in feats if c in legacy)


def test_reduced_alphabet_is_a_partition():
    # Cada um dos 20 aminoácidos está em exatamente um dos 7 grupos.
    assert len(REDUCED_ALPHABET) == 7
    assert sorted("".join(REDUCED_ALPHABET)) == sorted(AMINO_ACIDS)


def test_group_pairs_hand_example():
    # MKTA -> grupos YMTS, RK, YMTS, AGV -> pares (YMTS,RK), (RK,YMTS), (YMTS,AGV), 1/3 cada.
    p = compute_group_pair_composition("MKTA")
    assert len(p) == 49
    for key in ("pair_YMTS_RK", "pair_RK_YMTS", "pair_YMTS_AGV"):
        assert p[key] == pytest.approx(1 / 3)
    assert sum(p.values()) == pytest.approx(1.0)


def test_group_pairs_skip_ambiguous_residue():
    # Mesma regra dos dipeptídeos: AX e XG descartados, sem juntar A..G através do X.
    p = compute_group_pair_composition("AXGC")
    assert p["pair_AGV_C"] == pytest.approx(1.0)
    assert p["pair_AGV_AGV"] == 0.0


def test_aromaticity_is_sum_of_fwy():
    # Motivo da remoção da aromaticidade: é exatamente F + W + Y.
    from src.feature_extraction import compute_physicochemical_properties

    comp = compute_amino_acid_composition(SEQ)
    arom = compute_physicochemical_properties(SEQ)["aromaticity"]
    assert arom == pytest.approx(comp["aa_comp_F"] + comp["aa_comp_W"] + comp["aa_comp_Y"])


def test_compositions_sum_to_one():
    assert sum(compute_amino_acid_composition(SEQ).values()) == pytest.approx(1.0)
    assert sum(compute_dipeptide_composition(SEQ).values()) == pytest.approx(1.0)
    assert sum(compute_group_pair_composition(SEQ).values()) == pytest.approx(1.0)


def test_dipeptides_skip_pairs_with_ambiguous_residue():
    dip = compute_dipeptide_composition("AXGA")
    # pares AX e XG descartados; único par válido é GA (não junta A..G através do X)
    assert dip["dipep_GA"] == pytest.approx(1.0)
    assert dip["dipep_AG"] == 0.0
    assert sum(dip.values()) == pytest.approx(1.0)


def test_ambiguous_residues_counted_and_tolerated():
    assert count_ambiguous_residues("MKXAB") == 2
    feats = compute_sequence_features("MKTXAYIAK")
    assert feats["length"] == 9
    assert np.isfinite(feats["isoelectric_point"])


def _toy_graph():
    g = IsAGraph()
    for child, parent in [("ARO:2", "ARO:1"), ("ARO:3", "ARO:2"), ("ARO:4", "ARO:2"), ("ARO:5", "ARO:3")]:
        g.add_edge(child, parent)
    return g


def test_hierarchy_invariants():
    g = _toy_graph()
    assert compute_hierarchy_depth("ARO:1", g) == 0
    assert compute_ancestor_count("ARO:1", g) == 0
    assert compute_hierarchy_depth("ARO:5", g) == 3
    assert compute_ancestor_count("ARO:5", g) == 3
    assert compute_children_count("ARO:2", g) == 2
    assert compute_children_count("ARO:5", g) == 0
    assert compute_hierarchy_depth("ARO:999", g) == -1


def test_iri_mapping():
    assert iri_to_accession("http://purl.obolibrary.org/obo/ARO_3000001") == "ARO:3000001"
    assert iri_to_accession("http://example.org/other") is None


def test_drug_class_flags():
    flags = extract_drug_class_flags(
        pd.Series(["tetracycline antibiotic; peptide antibiotic", "peptide antibiotic", None])
    )
    assert list(flags.columns) == ["drug_class_peptide_antibiotic", "drug_class_tetracycline_antibiotic"]
    assert flags.values.tolist() == [[1, 1], [1, 0], [0, 0]]


def test_drug_class_flags_with_fixed_vocabulary():
    # Estudo de caso: mesmas colunas do experimento; classe desconhecida é ignorada.
    vocab = ["peptide antibiotic", "tetracycline antibiotic"]
    flags = extract_drug_class_flags(pd.Series(["peptide antibiotic; novel antibiotic", None]), vocabulary=vocab)
    assert list(flags.columns) == ["drug_class_peptide_antibiotic", "drug_class_tetracycline_antibiotic"]
    assert flags.values.tolist() == [[1, 0], [0, 0]]


# ── Ponto 4: gene mais parecido, cenário "gene novo", homologia, IM e ablação ──

def _toy_hits():
    """Acertos de brinquedo: g0..g5; os de teste são g0 e g1."""
    rows = [
        # g0: o melhor é g3 (menor e-value); g2 é pior.
        ("g0", "g2", 0.80, 1e-50, 200.0),
        ("g0", "g3", 0.70, 1e-90, 300.0),
        # g1: dois com o mesmo e-value; desempata o maior bit score (g5).
        ("g1", "g4", 0.60, 1e-20, 90.0),
        ("g1", "g5", 0.55, 1e-20, 95.0),
        # acerto de g0 com outro gene de teste (g1): nunca pode ser escolhido.
        ("g0", "g1", 0.99, 1e-200, 900.0),
    ]
    df = pd.DataFrame(rows, columns=["query", "target", "fident", "evalue", "bits"])
    return df


def test_nearest_neighbor_lowest_evalue_then_bits_only_reference():
    ids = [f"g{i}" for i in range(6)]
    nn = NeighborIndex(_toy_hits(), ids)
    nb, ident = nn.nearest([0, 1], [2, 3, 4, 5])
    assert nb.tolist() == [3, 5]  # g1 (teste) nunca é vizinho, mesmo com e-value menor
    assert ident.tolist() == [0.70, 0.55]
    # Tirar o melhor do treino faz o próximo da lista ser escolhido (equivale a buscar só no treino).
    nb2, _ = nn.nearest([0, 1], [2, 4])
    assert nb2.tolist() == [2, 4]
    # Sem nenhum acerto no treino: NO_NEIGHBOR e identidade 0.
    nb3, ident3 = nn.nearest([0], [4, 5])
    assert nb3.tolist() == [NO_NEIGHBOR] and ident3.tolist() == [0.0]


def test_nearest_neighbor_rejects_overlap():
    nn = NeighborIndex(_toy_hits(), [f"g{i}" for i in range(6)])
    with pytest.raises(ValueError):
        nn.nearest([0, 1], [1, 2])


def test_new_gene_features_copy_ontology_keep_sequence():
    # 2 colunas de sequência + 2 ontológicas; gene 0 consulta, vizinho = gene 2; gene 1 sem vizinho.
    X_b = np.array([[1.0, 2.0, 10.0, 11.0], [3.0, 4.0, 12.0, 13.0], [5.0, 6.0, 20.0, 21.0]])
    onto = np.array([2, 3])
    X, has = new_gene_features(X_b, [0, 1], np.array([2, NO_NEIGHBOR]), onto)
    assert X[0].tolist() == [1.0, 2.0, 20.0, 21.0]  # sequência própria, ontologia do vizinho
    assert has.tolist() == [True, False] and np.isnan(X[1, 2:]).all()
    Xz, _ = new_gene_features(X_b, [0, 1], np.array([2, NO_NEIGHBOR]), onto, zero_fallback=True)
    assert Xz[1].tolist() == [3.0, 4.0, 0.0, 0.0]


class _ConstModel:
    """Modelo de brinquedo que sempre prevê a mesma classe (para testar quem prevê o quê)."""

    def __init__(self, label, n_classes=3):
        self.label, self.classes_ = label, np.arange(n_classes)

    def predict(self, X):
        assert not np.isnan(X).any(), "linha sem vizinho chegou ao modelo B"
        return np.full(len(X), self.label)

    def predict_proba(self, X):
        return np.eye(len(self.classes_))[self.predict(X)]


def test_predict_new_gene_uses_pipeline_a_without_neighbor():
    X_a = np.zeros((4, 2))
    X_b = np.zeros((4, 3))
    out = predict_new_gene(_ConstModel(1), _ConstModel(2), X_a, X_b, np.array([0, 1]), np.array([3, NO_NEIGHBOR]), np.array([2]))
    pred, proba = out["b_gene_novo"]
    assert pred.tolist() == [1, 2]  # com vizinho: B; sem vizinho: A
    assert proba.shape == (2, 3)
    assert out["b_gene_novo_zeros"][0].tolist() == [1, 1]  # regra antiga: tudo pelo B


def test_scenario_predictions_keys_and_floor_sensitivity():
    from src.train_evaluate import scenario_predictions

    X_a, X_b = np.zeros((6, 2)), np.zeros((6, 3))
    y = np.array([0, 1, 2, 2, 2, 1])
    out = scenario_predictions(_ConstModel(1), _ConstModel(2), X_a, X_b, y, np.array([2, 3, 4, 5]), np.array([0, 1]),
                               np.array([5, 4]), np.array([2]), neighbors_floor40=np.array([5, NO_NEIGHBOR]))
    assert set(out) == {"a", "b", "b_gene_novo", "b_gene_novo_zeros", "homologia", "b_gene_novo_piso40", "homologia_piso40"}
    assert out["homologia"][0].tolist() == [1, 2]
    assert out["homologia_piso40"][0].tolist() == [1, 2]  # sem vizinho no piso: classe mais comum (2)
    assert out["b_gene_novo_piso40"][0].tolist() == [2, 1]  # com vizinho: B (prevê 2); sem vizinho no piso: A (prevê 1)


def test_homology_prediction_copies_neighbor_or_majority():
    y = np.array([0, 1, 2, 2, 2, 1])
    pred, proba = homology_prediction(y, reference_idx=[2, 3, 4, 5], neighbors=np.array([5, NO_NEIGHBOR]), n_classes=3)
    assert pred.tolist() == [1, 2]  # mecanismo do vizinho; sem vizinho, a classe mais comum do treino
    assert proba.tolist() == [[0, 1, 0], [0, 0, 1]]


def test_mutual_information_matches_hand_calculation():
    # f determina y por completo -> IM = H(y); g é independente de y -> IM = 0.
    y = np.array([0, 0, 1, 1])
    feats = pd.DataFrame({"depth_isa": [3, 3, 5, 5], "drug_class_x": [0, 1, 0, 1]})
    mi = mutual_information_table(feats, y).set_index("feature")
    assert mi.loc["depth_isa", "informacao_mutua_nats"] == pytest.approx(np.log(2))
    assert mi.loc["depth_isa", "fracao_da_entropia_do_rotulo"] == pytest.approx(1.0)
    assert mi.loc["drug_class_x", "informacao_mutua_nats"] == pytest.approx(0.0)
    assert mi.loc["depth_isa", "grupo"] == "estrutural"
    assert entropy_nats(np.array([0, 1, 2, 3])) == pytest.approx(np.log(4))


def test_ablation_configs_columns():
    onto = ["depth_isa", "ancestor_count", "children_count", "drug_class_a", "drug_class_b"]
    cols = {c.name: config_columns(c, 2, onto).tolist() for c in ABLATION_CONFIGS}
    assert cols["A"] == [0, 1]
    assert cols["A + estruturais"] == [0, 1, 2, 3, 4]
    assert cols["A + classes"] == [0, 1, 5, 6]
    assert cols["B"] == [0, 1, 2, 3, 4, 5, 6]
    assert cols["só estruturais"] == [2, 3, 4]
    assert cols["só classes"] == [5, 6]
    assert cols["só ontologia"] == [2, 3, 4, 5, 6]
    assert config_columns(AblationConfig("x", False, "todas"), 2, onto).tolist() == [2, 3, 4, 5, 6]


def test_holm_family_has_eight_confirmatory_tests():
    from src.statistical_tests import run_all_tests

    rng = np.random.default_rng(0)
    rows = []
    for model in MODEL_NAMES:
        for rep in range(10):
            for fold in range(5):
                base = rng.uniform(0.5, 0.6)
                for key, gain in (("a", 0.0), ("b", 0.2), ("b_gene_novo", 0.15), ("b_gene_novo_zeros", 0.1)):
                    rows.append({"pipeline": key, "modelo": model, "repeticao": rep, "fold": fold,
                                 "n_ajuste": 400, "n_validacao": 100, "f1_macro": base + gain + rng.normal(0, 0.01)})
    out = run_all_tests(pd.DataFrame(rows))
    assert len(out) == 8 and (out["testes_na_familia"] == 8).all()
    assert set(out["cenario"]) == {"anotacao_curada", "gene_novo"}
    assert (out["p_holm"] >= out["p_value"]).all()


def test_split_disjoint_and_stratified():
    # Sem grupos (cada gene é um grupo): divisão aleatória estratificada, ~20% de cada classe.
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    tr, te = split_indices(y, None)
    assert not set(tr) & set(te) and len(tr) + len(te) == len(y)
    assert abs((y[te] == 1).mean() - (y == 1).mean()) <= 0.02


def test_group_split_keeps_groups_whole():
    rng = np.random.default_rng(0)
    groups = np.repeat(np.arange(60), 5)  # 60 grupos de 5 genes
    y = np.repeat(rng.integers(0, 3, 60), 5)  # todos os genes de um grupo têm a mesma classe
    tr, te, info = choose_test_partition(y, groups, seeds=range(10))
    assert not set(groups[tr]) & set(groups[te])  # nenhum grupo dos dois lados
    assert info["n_teste"] == len(te) and 0 <= info["desvio_maximo_da_fracao_por_classe"] < 0.2
    # A regra é determinística: mesma entrada, mesma partição.
    tr2, te2, _ = choose_test_partition(y, groups, seeds=range(10))
    assert np.array_equal(te, te2)


def _min_per_fold(y, groups, seed, n_classes=3):
    return min(np.bincount(y[v], minlength=n_classes).min() for _, v in cv_splitter(seed).split(np.zeros(len(y)), y, groups))


def test_cv_seeds_rule_first_valid_in_order():
    # Ponto 5: as primeiras sementes (em ordem) com >= min_genes de cada classe em cada fold.
    from src.experiment_config import choose_cv_seeds

    rng = np.random.default_rng(1)
    sizes = rng.integers(1, 8, 80)  # 80 grupos de tamanhos variados
    groups = np.repeat(np.arange(80), sizes)
    y = np.repeat(np.arange(80) % 3, sizes)
    minimum = {s: _min_per_fold(y, groups, s) for s in range(30)}
    threshold = sorted(minimum.values())[len(minimum) // 2]  # cerca de metade das sementes atende
    seeds, info = choose_cv_seeds(y, groups, n_repeats=4, seeds=range(30), min_genes=threshold)
    expected = [s for s in range(30) if minimum[s] >= threshold][:4]
    assert seeds == expected and info["todas_atendem_o_criterio"]
    assert choose_cv_seeds(y, groups, n_repeats=4, seeds=range(30), min_genes=threshold)[0] == seeds  # determinística


def test_cv_seeds_complete_with_best_invalid_when_too_few_valid():
    from src.experiment_config import choose_cv_seeds

    rng = np.random.default_rng(2)
    sizes = rng.integers(1, 8, 60)
    groups = np.repeat(np.arange(60), sizes)
    y = np.repeat(np.arange(60) % 3, sizes)
    minimum = {s: _min_per_fold(y, groups, s) for s in range(10)}
    top = max(minimum.values())
    seeds, info = choose_cv_seeds(y, groups, n_repeats=10, seeds=range(10), min_genes=top + 1)  # nenhuma atende
    assert not info["todas_atendem_o_criterio"] and len(seeds) == 10
    assert [minimum[s] for s in seeds] == sorted(minimum.values(), reverse=True)


def test_repeated_splits_pair_every_fold():
    from src.experiment_config import repeated_cv_splits

    groups = np.repeat(np.arange(60), 2)
    y = np.repeat(np.arange(60) % 3, 2)
    splits = repeated_cv_splits(y, groups, [0, 1, 2])
    assert len(splits) == 3 * N_CV_SPLITS
    assert [(r, f) for r, f, _, _ in splits][:6] == [(1, 1), (1, 2), (1, 3), (1, 4), (1, 5), (2, 1)]
    for r in (1, 2, 3):  # em cada repetição, cada gene é validado exatamente uma vez
        val = np.concatenate([v for rr, _, _, v in splits if rr == r])
        assert sorted(val) == list(range(len(y)))


def test_cv_has_five_group_folds():
    # 5 folds por grupos (com 10, a substituição do alvo ficaria com ~3 genes por fold).
    assert cv_splitter().get_n_splits() == N_CV_SPLITS == 5
    groups = np.repeat(np.arange(40), 3)
    y = np.repeat(np.arange(40) % 2, 3)
    for fit, val in cv_splitter().split(np.zeros(len(y)), y, groups):
        assert not set(groups[fit]) & set(groups[val])


def test_connected_components_single_linkage():
    from src.homology_groups import connected_components, count_cross_group_pairs, qualifying_pairs

    hits = pd.DataFrame(
        {
            "query": ["A", "B", "D", "E"],
            "target": ["B", "C", "E", "F"],
            "fident": [0.95, 0.75, 0.72, 0.65],
            "qcov": [0.9, 0.9, 0.9, 0.9],
            "tcov": [0.9, 0.9, 0.3, 0.9],  # D-E tem cobertura baixa: não liga
        }
    )
    pairs = qualifying_pairs(hits, 70)
    assert pairs.values.tolist() == [["A", "B"], ["B", "C"]]
    g = connected_components(list("ABCDEF"), pairs)
    # A, B e C juntos por transitividade (A~C só via B); D, E, F sozinhos.
    assert g["A"] == g["B"] == g["C"] and len({g["D"], g["E"], g["F"]}) == 3
    table = pd.DataFrame({"ARO_Accession": list("ABCDEF"), "grupo_70": g.values})
    assert count_cross_group_pairs(table, hits, 70) == 0


# ── Classes do experimento (TCC2, ponto 2) ─────────────────────────────────────


def test_five_experiment_classes():
    assert sorted(EFFECTIVE_CLASSES) == [
        "antibiotic efflux",
        "antibiotic inactivation",
        "antibiotic target alteration",
        "antibiotic target protection",
        "antibiotic target replacement",
    ]
    # Nenhum mecanismo pode estar no experimento e no estudo de caso ao mesmo tempo.
    assert not set(EFFECTIVE_CLASSES) & set(CASE_STUDY_CLASSES)


def test_assign_split_destinations():
    assert assign_split("antibiotic target replacement") == SPLIT_EXPERIMENT
    assert assign_split(" antibiotic efflux ") == SPLIT_EXPERIMENT  # espaços nas bordas não importam
    assert assign_split("reduced permeability to antibiotic") == SPLIT_CASE_STUDY
    assert assign_split("resistance by host-dependent nutrient acquisition") == SPLIT_CASE_STUDY
    assert assign_split("antibiotic efflux;antibiotic target alteration") == SPLIT_MULTI_LABEL


def test_unknown_mechanism_raises_instead_of_being_dropped():
    # No TCC1, classes fora da lista sumiam em silêncio; agora exigem decisão.
    with pytest.raises(ValueError, match="Mecanismo desconhecido"):
        assign_split("resistance by absence")


def _toy_integrated():
    labels = (
        ["antibiotic inactivation"] * 4
        + ["antibiotic target protection"] * 2
        + ["reduced permeability to antibiotic"]
        + ["antibiotic efflux;reduced permeability to antibiotic"]
        + ["antibiotic target replacement"]
    )
    return pd.DataFrame({"ARO_Accession": [f"ARO:{i}" for i in range(len(labels))], TARGET: labels})


def test_partition_closes_and_preserves_order():
    df = _toy_integrated()
    parts = partition_dataset(df)
    sizes = {k: len(v) for k, v in parts.items()}
    assert sizes == {SPLIT_EXPERIMENT: 7, SPLIT_CASE_STUDY: 1, SPLIT_MULTI_LABEL: 1}
    assert sum(sizes.values()) == len(df)
    # A ordem original é mantida (a divisão treino/teste depende dela).
    assert parts[SPLIT_EXPERIMENT]["ARO_Accession"].tolist() == ["ARO:0", "ARO:1", "ARO:2", "ARO:3", "ARO:4", "ARO:5", "ARO:8"]


def test_dataset_composition_table():
    comp = dataset_composition(_toy_integrated())
    assert comp["genes"].sum() == 9
    assert comp.iloc[0].tolist() == ["antibiotic inactivation", 4, SPLIT_EXPERIMENT]
    assert comp["destino"].tolist() == [SPLIT_EXPERIMENT] * 3 + [SPLIT_CASE_STUDY, SPLIT_MULTI_LABEL]


def test_macro_metrics_average_over_all_declared_classes():
    from src.train_evaluate import compute_metrics

    # 3 classes declaradas, mas a classe 2 não aparece no fold (nem no real nem no previsto).
    y_true = np.array([0, 0, 1, 1])
    y_pred = np.array([0, 0, 1, 1])
    proba = np.eye(3)[y_pred]
    m = compute_metrics(y_true, y_pred, proba, n_classes=3)
    # Com labels declarados, a classe ausente entra na média com F1 = 0: (1 + 1 + 0) / 3.
    # Sem labels (TCC1), daria 1,0, como se a classe não existisse.
    assert m["f1_macro"] == pytest.approx(2 / 3)


def test_class_counts_per_fold_criterion():
    from src.experiment_config import repeated_cv_splits
    from src.train_evaluate import check_class_counts_per_fold

    classes = ["a", "b", "c"]
    y = np.repeat([0, 1, 2], [50, 25, 25])  # 5 por fold nas classes menores, com 5 folds
    groups = np.arange(len(y))
    table = check_class_counts_per_fold(y, classes, repeated_cv_splits(y, groups, [0, 1]))
    assert len(table) == 10 and table[classes].sum().tolist() == [100, 50, 50]  # 2 repetições
    assert table[["b", "c"]].min().min() == 5
    y_small = np.repeat([0, 1, 2], [100, 50, 3])  # 3 genes em 5 folds
    with pytest.raises(ValueError, match="Classe ausente"):
        check_class_counts_per_fold(y_small, classes, repeated_cv_splits(y_small, np.arange(len(y_small)), [0]))


def test_per_class_metrics_and_pr_auc():
    from sklearn.metrics import average_precision_score

    from src.train_evaluate import compute_metrics, per_class_metrics

    y_true = np.array([0, 0, 1, 1, 2, 2])
    proba = np.array([[0.8, 0.1, 0.1], [0.4, 0.5, 0.1], [0.2, 0.7, 0.1], [0.3, 0.6, 0.1], [0.1, 0.2, 0.7], [0.5, 0.1, 0.4]])
    y_pred = proba.argmax(axis=1)
    rows = per_class_metrics(y_true, y_pred, proba, ["x", "y", "z"])
    expected = [average_precision_score(y_true == k, proba[:, k]) for k in range(3)]
    assert [r["pr_auc"] for r in rows] == pytest.approx(expected)
    assert compute_metrics(y_true, y_pred, proba, 3)["pr_auc_macro"] == pytest.approx(np.mean(expected))
    assert rows[0]["precisao"] == pytest.approx(0.5) and rows[0]["revocacao"] == pytest.approx(0.5)


def test_case_study_summary_one_row_per_gene():
    from src.case_study import short_name, summarize
    from src.experiment_config import MODEL_NAMES

    assert short_name("reduced permeability to antibiotic") == "reduced permeability"
    rows = [
        {"ARO_Accession": g, TARGET: "reduced permeability to antibiotic", "AMR_Gene_Family": "porin",
         "pipeline": k, "modelo": m, "classe_prevista": "antibiotic efflux"}
        for g in ("ARO:1", "ARO:2") for k in ("a", "b") for m in MODEL_NAMES
    ]
    s = summarize(pd.DataFrame(rows))
    assert len(s) == 2 and s.loc[0, "B_xgboost"] == "efflux"
    assert list(s.columns[3:]) == [f"{k}_{m}" for k in ("A", "B") for m in MODEL_NAMES]


@pytest.mark.parametrize("model", ["random_forest", "xgboost", "logistic_regression", "svm"])
def test_pipeline_structure_and_fit(model):
    # Experimento: sem seleção de atributos; todas as colunas chegam ao classificador.
    pipe = build_full_pipeline(model)
    assert [s for s, _ in pipe.steps] == ["scaler", "clf"]
    assert pipe.named_steps["clf"].get_params()["random_state"] == 42
    rng = np.random.default_rng(0)
    X, y = rng.random((60, 8)), np.repeat([0, 1, 2], 20)
    pipe.fit(X, y)
    assert pipe.predict_proba(X).shape == (60, 3)
    assert pipe.named_steps["clf"].n_features_in_ == 8


def test_xgboost_uses_balanced_sample_weight():
    # Ponto 6: o XGBoost usa a mesma regra de peso dos outros modelos, calculada em cada fit.
    from sklearn.base import clone
    from sklearn.utils.class_weight import compute_sample_weight
    from xgboost import XGBClassifier

    from src.pipeline_builder import BalancedXGBClassifier, _model_params, build_classifier

    rng = np.random.default_rng(0)
    X, y = rng.random((120, 5)), np.repeat([0, 1, 2], [90, 20, 10])
    model = build_classifier("xgboost")
    assert isinstance(model, BalancedXGBClassifier)
    assert isinstance(clone(model), BalancedXGBClassifier)  # clone (CV, GridSearchCV) mantém o peso
    manual = XGBClassifier(**_model_params()["xgboost"]).fit(X, y, sample_weight=compute_sample_weight("balanced", y))
    unweighted = XGBClassifier(**_model_params()["xgboost"]).fit(X, y)
    np.testing.assert_allclose(model.fit(X, y).predict_proba(X), manual.predict_proba(X))
    assert not np.allclose(model.predict_proba(X), unweighted.predict_proba(X))


def test_all_models_balance_classes():
    from src.pipeline_builder import CLASS_BALANCING, build_classifier

    assert set(CLASS_BALANCING) == set(MODEL_NAMES)
    for name in ("random_forest", "logistic_regression", "svm"):
        assert build_classifier(name).get_params()["class_weight"] == "balanced"


def test_hyperparameters_record_is_complete(tmp_path):
    # Ponto 13: o registro vem do get_params(), não de uma lista escrita à mão.
    import json

    from src.pipeline_builder import build_classifier, save_hyperparameters

    out = tmp_path / "hp.json"
    save_hyperparameters(out)
    rec = json.loads(out.read_text(encoding="utf-8"))
    for name in MODEL_NAMES:
        assert set(rec["models"][name]) == set(build_classifier(name).get_params())
        for k, v in rec["definidos_no_codigo"][name].items():
            assert rec["models"][name][k] == v
    assert "sample_weight" in rec["class_balancing"]["xgboost"]


def test_xgboost_effective_params_from_trained_model(tmp_path):
    # Ponto 13: os padrão do XGBoost (None no get_params) são lidos do modelo treinado.
    import joblib

    from src.experiment_record import MAIN_PARAMS, xgboost_effective_params
    from src.pipeline_builder import classifier_params

    rng = np.random.default_rng(0)
    pipe = build_full_pipeline("xgboost").fit(rng.random((60, 4)), np.repeat([0, 1, 2], 20))
    joblib.dump(pipe, tmp_path / "xgb.joblib")
    eff = xgboost_effective_params(tmp_path / "xgb.joblib")
    assert eff["learning_rate"] == "0.3" and eff["max_depth"] == "6"
    assert eff["objective"] == "multi:softprob"
    for name, params in MAIN_PARAMS.items():
        assert set(params) <= set(classifier_params(name))


def test_sensitivity_selection_variants():
    from src.pipeline_builder import FeatureSelection

    rng = np.random.default_rng(0)
    X, y = rng.random((60, 10)), np.repeat([0, 1, 2], 20)
    # "todos": exatamente o pipeline do TCC1.
    joint = build_full_pipeline("random_forest", FeatureSelection(k=4)).fit(X, y)
    assert [s for s, _ in joint.steps] == ["scaler", "variance", "select", "clf"]
    assert joint.named_steps["clf"].n_features_in_ == 4
    # "sequencia": 4 das 7 primeiras colunas (sequência) + as 3 últimas (ontologia) inteiras.
    seq = build_full_pipeline("random_forest", FeatureSelection(k=4, scope="sequencia", n_sequence=7), n_features=10)
    seq.fit(X, y)
    assert seq.named_steps["clf"].n_features_in_ == 4 + 3
    # No Pipeline A (só sequência), "sequencia" equivale a "todos".
    a_only = build_full_pipeline("random_forest", FeatureSelection(k=4, scope="sequencia", n_sequence=7), n_features=7)
    assert [s for s, _ in a_only.steps] == ["scaler", "variance", "select", "clf"]


def test_corrected_ttest_matches_hand_calculation():
    # Ponto 5: t = média(d) / sqrt((1/J + n_val/n_ajuste) · s²), gl = J − 1.
    from scipy import stats

    rng = np.random.default_rng(0)
    a = rng.uniform(0.6, 0.7, 50)
    d = rng.normal(0.05, 0.03, 50)
    res = corrected_resampled_ttest(a, a + d, test_train_ratio=0.25)
    se = np.sqrt((1 / 50 + 0.25) * d.var(ddof=1))
    assert res["t"] == pytest.approx(d.mean() / se)
    assert res["p_value"] == pytest.approx(2 * stats.t.sf(abs(d.mean() / se), 49))
    half = stats.t.ppf(0.975, 49) * se
    assert (res["ic95_inf"], res["ic95_sup"]) == pytest.approx((d.mean() - half, d.mean() + half))
    # A correção deixa o teste mais conservador que o t pareado comum.
    assert res["p_value"] > stats.ttest_rel(a + d, a).pvalue


def test_holm_adjustment():
    p = np.array([0.01, 0.04, 0.03, 0.005])
    # Ordenados: 0,005 x 4 = 0,02; 0,01 x 3 = 0,03; 0,03 x 2 = 0,06; 0,04 x 1 = 0,04 -> máximo acumulado 0,06.
    assert holm_adjust(p).tolist() == pytest.approx([0.03, 0.06, 0.06, 0.02])
    assert holm_adjust([0.9, 0.8]).max() <= 1.0


def test_stratified_group_bootstrap_keeps_groups_and_classes():
    from sklearn.metrics import precision_recall_fscore_support

    from src.bootstrap_test import class_metrics_from_predictions, stratified_group_resamples

    sizes = [3, 2, 4, 1, 1, 2]
    groups = np.repeat(np.arange(6), sizes)
    y = np.repeat([0, 0, 1, 2, 2, 2], sizes)
    for idx in stratified_group_resamples(y, groups, 50, seed=0):
        assert set(y[idx]) == {0, 1, 2}  # nenhuma classe some
        n_groups = {0: 0.0, 1: 0.0, 2: 0.0}
        for g in set(groups[idx]):
            copies = (groups[idx] == g).sum() / (groups == g).sum()
            assert copies == int(copies)  # grupos entram inteiros
            n_groups[y[groups == g][0]] += copies
        assert n_groups == {0: 2, 1: 1, 2: 3}  # mesmo número de grupos por classe
    y_pred = np.array([0, 0, 1, 1, 1, 1, 1, 1, 1, 2, 2, 0, 2])  # 13 genes
    p, r, f, _ = precision_recall_fscore_support(y, y_pred, labels=[0, 1, 2], zero_division=0)
    assert class_metrics_from_predictions(y, y_pred, 3) == pytest.approx(np.vstack([p, r, f]))


# ── Ponto 6: sensibilidade ao balanceamento ──────────────────────────────────


def test_undersampling_reduces_only_majority_to_second_largest():
    from src.sensitivity_balancing import undersample_to_second_largest

    y = np.repeat([0, 1, 2, 3], [30, 400, 50, 10])
    assert undersample_to_second_largest(y) == {1: 50}


@pytest.mark.parametrize("model", ["random_forest", "xgboost", "logistic_regression", "svm"])
def test_unweighted_classifier_keeps_other_params(model):
    from src.pipeline_builder import build_classifier
    from src.sensitivity_balancing import unweighted_classifier

    base, plain = build_classifier(model).get_params(), unweighted_classifier(model).get_params()
    diff = {k for k in base if repr(base[k]) != repr(plain.get(k))}  # repr: NaN != NaN
    assert diff <= {"class_weight"}
    if model != "xgboost":
        assert plain["class_weight"] is None


def test_hierarchical_classifier_consistent():
    from src.pipeline_builder import build_classifier
    from src.sensitivity_balancing import HierarchicalClassifier

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2, 3], [20, 80, 20, 20])
    X = rng.random((len(y), 4)) + np.eye(4)[y] * 2  # classes separáveis
    for model in ("random_forest", "xgboost"):
        h = HierarchicalClassifier(build_classifier(model), majority_label=1).fit(X, y)
        proba = h.predict_proba(X)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0)
        assert (h.predict(X) == y).mean() > 0.95
        assert set(h.predict(X)) <= {0, 1, 2, 3}


def test_smotenc_keeps_ontology_values_and_resampling_only_in_fit():
    from src.sensitivity_balancing import build_strategy_pipeline

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], [20, 100, 15])
    X = np.c_[rng.random((len(y), 3)), rng.integers(0, 2, (len(y), 2)), rng.integers(1, 4, len(y))]
    cat = [3, 4, 5]
    pipe = build_strategy_pipeline("smote", "random_forest", y, cat, majority_label=1).fit(X, y)
    Xs = pipe.named_steps["scaler"].transform(X)
    Xr, yr = pipe.named_steps["sampler"].fit_resample(Xs, y)
    assert np.bincount(yr).tolist() == [100, 100, 100]
    for j in cat:  # gene sintético só tem valores que existem nos dados
        assert set(np.round(Xr[:, j], 9)) <= set(np.round(Xs[:, j], 9))
    assert len(pipe.predict(X)) == len(y)  # predição não reamostra


def test_weight_strategy_equals_main_pipeline():
    from src.sensitivity_balancing import build_strategy_pipeline

    rng = np.random.default_rng(1)
    y = np.repeat([0, 1, 2], [20, 60, 20])
    X = rng.random((len(y), 5))
    for model in ("random_forest", "xgboost"):
        a = build_strategy_pipeline("peso", model, y, None, majority_label=1).fit(X, y).predict_proba(X)
        b = build_full_pipeline(model).fit(X, y).predict_proba(X)
        np.testing.assert_allclose(a, b)


# ── Ponto 13: sensibilidade ao ajuste de hiperparâmetros ─────────────────────


@pytest.mark.parametrize("model", ["random_forest", "xgboost", "logistic_regression", "svm"])
def test_tuning_grid_contains_experiment_defaults(model):
    from src.pipeline_builder import build_full_pipeline
    from src.sensitivity_tuning import DEFAULT_IN_GRID, PARAM_GRIDS

    grid, default = PARAM_GRIDS[model], DEFAULT_IN_GRID[model]
    assert set(grid) == set(default)
    params = build_full_pipeline(model).get_params()
    for k, v in default.items():
        assert v in grid[k]
        if params[k] is not None:  # no XGBoost, None = padrão efetivo (0,3 e 6), conferido em outro teste
            assert params[k] == v


def test_nested_tuning_runs_with_groups():
    from src.sensitivity_tuning import evaluate_model

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], [40, 60, 40])
    X = rng.random((len(y), 4)) + np.eye(3)[y][:, [0, 1, 2, 0]]
    groups = np.arange(len(y)) // 2
    idx = np.arange(len(y))
    train_idx, test_idx = idx[idx % 5 != 0], idx[idx % 5 == 0]
    from sklearn.model_selection import StratifiedGroupKFold

    folds = StratifiedGroupKFold(3, shuffle=True, random_state=0).split(train_idx, y[train_idx], groups[train_idx])
    splits = [(1, i, fit, val) for i, (fit, val) in enumerate(folds, start=1)]
    summary, folds = evaluate_model("logistic_regression", X, y, 3, train_idx, test_idx, groups, splits)
    assert len(folds) == 3 and 0 <= summary["folds_com_o_padrao_escolhido"] <= 3
    assert 0 <= summary["f1_teste_ajustado"] <= 1


def test_macro_pr_curve_shapes():
    # Ponto 10: classificador perfeito -> precisão 1 em toda a curva; sem informação ->
    # precisão perto da prevalência da classe em revocação alta.
    from src.visualization import macro_pr_curve

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1], [900, 100])
    perfect = np.eye(2)[y]
    _, prec, ap = macro_pr_curve(y, perfect)
    assert np.allclose(prec, 1.0) and ap == pytest.approx(1.0)
    noise = rng.random(len(y))
    _, prec, _ = macro_pr_curve(y, np.c_[1 - noise, noise])
    assert prec[-1] < 0.8  # revocação 1: média das prevalências (0,9 e 0,1) = 0,5
