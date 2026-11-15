"""Testes dos módulos de extração, pipelines, estatística e relatório."""

import numpy as np
import pandas as pd
import pytest

from src.experiment_config import (
    CASE_STUDY_CLASSES,
    EFFECTIVE_CLASSES,
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
from src.leakage_check import compute_spearman_correlations, identify_leaky_features
from src.ontology_features import (
    IsAGraph,
    compute_ancestor_count,
    compute_children_count,
    compute_hierarchy_depth,
    extract_drug_class_flags,
    iri_to_accession,
)
from src.pipeline_builder import build_full_pipeline
from src.statistical_tests import run_wilcoxon_test

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


def test_spearman_leakage_detection():
    y = np.array([0, 0, 1, 1, 2, 2])
    feats = pd.DataFrame({"leaky": y * 10, "noise": [1, 0, 0, 1, 1, 0], "const": [1] * 6})
    corr = compute_spearman_correlations(feats, y)
    assert identify_leaky_features(corr) == ["leaky"]
    assert corr["const"] == 0.0


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


def test_cv_seed_rule_maximizes_minimum_per_fold():
    from src.experiment_config import choose_cv_seed

    rng = np.random.default_rng(1)
    sizes = rng.integers(1, 8, 80)  # 80 grupos de tamanhos variados
    groups = np.repeat(np.arange(80), sizes)
    y = np.repeat(np.arange(80) % 3, sizes)
    seed, info = choose_cv_seed(y, groups, seeds=range(15))
    # A semente escolhida tem o maior mínimo entre as testadas.
    for s in range(15):
        folds = cv_splitter(s).split(np.zeros(len(y)), y, groups)
        m = min(np.bincount(y[v], minlength=3).min() for _, v in folds)
        assert m <= info["minimo_genes_classe_fold"]
    assert choose_cv_seed(y, groups, seeds=range(15))[0] == seed  # determinística


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
    from src.train_evaluate import check_class_counts_per_fold

    classes = ["a", "b", "c"]
    y = np.repeat([0, 1, 2], [50, 25, 25])  # 5 por fold nas classes menores, com 5 folds
    table = check_class_counts_per_fold(y, classes)
    assert len(table) == 5 and table[classes].sum().tolist() == [50, 25, 25]
    assert table[["b", "c"]].min().min() == 5
    with pytest.raises(ValueError, match="Classe ausente"):
        check_class_counts_per_fold(np.repeat([0, 1, 2], [100, 50, 3]), classes)  # 3 genes em 5 folds


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


def test_wilcoxon_decision_consistent_with_p():
    a = np.linspace(0.6, 0.7, 10)
    res = run_wilcoxon_test(a, a + 0.05)
    assert res["p_value"] == pytest.approx(0.001953125) and res["decision"] == "reject_H0"
    same = run_wilcoxon_test(a, a)
    assert same["decision"] == "fail_to_reject"
