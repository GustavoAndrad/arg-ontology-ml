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
    # Representação da primeira versão do experimento, mantida para a análise de sensibilidade: 6 + 20 + 400 = 426.
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


# ── Gene mais parecido, cenário "gene novo", homologia, IM e ablação ─────────

def test_ontological_features_from_owl_xml_ignore_restrictions(tmp_path):
    # Leitor de OWL/XML dos atributos: só SubClassOf entre classes nomeadas. A ontologia
    # mínima (a mesma da EDA) tem ARO_3000001 sob ARO_0001004 por is-a e também uma
    # restrição apontando para ARO_0010000, que não pode virar ancestral.  # ANTI-LEAKAGE
    from src.ontology_features import extract_ontological_features
    from tests.test_eda import _owl_xml

    owl = _owl_xml(tmp_path / "aro.owl")
    df = pd.DataFrame({
        "ARO_Accession": ["ARO:3000001", "ARO:0010000", "ARO:9999999"],
        "Drug_Class": ["peptide antibiotic", "tetracycline antibiotic; peptide antibiotic", "peptide antibiotic"],
    })
    f = extract_ontological_features(df, owl).set_index("ARO_Accession")
    assert f.loc["ARO:3000001", ["depth_isa", "ancestor_count", "children_count"]].tolist() == [3, 3, 0]
    assert f.loc["ARO:0010000", ["depth_isa", "ancestor_count", "children_count"]].tolist() == [2, 2, 0]
    # Accession fora da ontologia: estruturais = 0 (com aviso), flags normais.
    assert f.loc["ARO:9999999", ["depth_isa", "ancestor_count", "children_count"]].tolist() == [0, 0, 0]
    assert f.loc["ARO:0010000", "drug_class_tetracycline_antibiotic"] == 1


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
    # As primeiras sementes (em ordem) com >= min_genes de cada classe em cada fold.
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
    # 5 folds por grupos (com 10, a substituição do alvo teria 5,3 genes por fold em média,
    # mas 0 a 2 no pior fold, conforme a semente).
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


# ── Classes do experimento ─────────────────────────────────────────────────────


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
    # Na primeira versão do experimento, classes fora da lista sumiam em silêncio; agora geram erro.
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
    # Sem labels (como na primeira versão), daria 1,0, como se a classe não existisse.
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


def test_case_study_uses_predict_not_highest_probability(monkeypatch):
    # No SVM, a probabilidade (calibração de Platt) pode discordar da classe prevista; o
    # estudo de caso deve mostrar a classe prevista, como o resto do experimento.
    import src.case_study as cs

    class _Disagree:
        def predict(self, X):
            return np.zeros(len(X), dtype=int)  # prevê a classe 0...

        def predict_proba(self, X):
            return np.tile([0.3, 0.7], (len(X), 1))  # ...mas a maior probabilidade é a da classe 1

    monkeypatch.setattr(cs.joblib, "load", lambda path: _Disagree())
    info = pd.DataFrame({"ARO_Accession": ["ARO:1"], TARGET: ["reduced permeability to antibiotic"]})
    X = pd.DataFrame({"x": [0.0]})
    out = cs.predict_case_study(info, X, X, ["antibiotic efflux", "antibiotic inactivation"])
    assert (out["classe_prevista"] == "antibiotic efflux").all()
    assert out["prob_prevista"].tolist() == pytest.approx([0.3] * len(out))


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
    # O XGBoost usa a mesma regra de peso dos outros modelos, calculada em cada fit.
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
    # O registro vem do get_params(), não de uma lista escrita à mão.
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
    # Os valores padrão do XGBoost (None no get_params) são lidos do modelo treinado.
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
    # "todos": exatamente o pipeline da primeira versão do experimento.
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
    # t = média(d) / sqrt((1/J + n_val/n_ajuste) · s²), gl = J − 1.
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


# ── Sensibilidade ao balanceamento ───────────────────────────────────────────


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


# ── Sensibilidade ao ajuste de hiperparâmetros ───────────────────────────────


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
    # Classificador perfeito -> precisão 1 em toda a curva; sem informação ->
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


# ── Interpretação, curva de aprendizado e overfitting ─────────────────────────

def test_feature_groups_partition_all_columns():
    from src.feature_extraction import SEQUENCE_FEATURE_COLUMNS
    from src.interpretability import group_columns

    onto = ["depth_isa", "ancestor_count", "children_count", "drug_class_a", "drug_class_b"]
    cols = list(SEQUENCE_FEATURE_COLUMNS) + onto
    groups = group_columns(cols)
    sizes = {g: len(v) for g, v in groups.items()}
    assert sizes == {"fisico_quimicas": 4, "aminoacidos": 20, "pares_de_grupos": 49, "estruturais": 3, "classes_de_antibiotico": 2}
    allpos = np.concatenate(list(groups.values()))
    assert sorted(allpos.tolist()) == list(range(len(cols)))  # cada coluna em exatamente um grupo


def test_permutation_detects_used_feature_and_ignores_unused():
    from src.interpretability import permutation_drops

    rng = np.random.default_rng(0)
    y = np.repeat([0, 1, 2], 50)
    X = np.column_stack([y.astype(float), rng.normal(size=len(y))])
    predict = lambda M: M[:, 0].round().astype(int)
    rows = {r["conjunto"]: r for r in permutation_drops(predict, X, y, {"usado": np.array([0]), "ruido": np.array([1])}, 3, 10, rng)}
    assert rows["usado"]["queda_f1_macro"] > 0.4
    assert rows["ruido"]["queda_f1_macro"] == 0.0
    assert {"queda_f1_classe_0", "queda_f1_classe_2"} <= set(rows["usado"])


def test_group_permutation_moves_columns_together():
    from src.interpretability import permutation_drops

    rng = np.random.default_rng(1)
    base = rng.normal(size=120)
    y = np.repeat([0, 1], 60)
    X = np.column_stack([base, base + y])  # a classe está na diferença entre as duas colunas
    predict = lambda M: (M[:, 1] - M[:, 0] > 0.5).astype(int)
    joint = permutation_drops(predict, X, y, {"grupo": np.array([0, 1])}, 2, 5, rng)[0]
    alone = permutation_drops(predict, X, y, {"so_0": np.array([0])}, 2, 5, rng)[0]
    assert joint["queda_f1_macro"] > 0.3  # embaralhar as linhas juntas quebra a relação com y
    assert alone["queda_f1_macro"] > 0.3
    # a diferença dentro de cada linha é preservada quando o grupo é embaralhado junto
    perm = np.random.default_rng(2).permutation(len(X))
    Xp = X.copy()
    Xp[:, [0, 1]] = X[perm][:, [0, 1]]
    assert np.allclose(Xp[:, 1] - Xp[:, 0], (X[:, 1] - X[:, 0])[perm])


def test_feature_dictionary_describes_every_feature():
    from src.interpretability import feature_dictionary

    seq = ["length", "aa_comp_A", "pair_AGV_RK"]
    onto = ["depth_isa", "drug_class_peptide_antibiotic"]
    drug = pd.Series(["peptide antibiotic; tetracycline antibiotic", "tetracycline antibiotic"])
    d = feature_dictionary(seq, onto, drug).set_index("atributo")
    assert len(d) == 5 and d["significado"].str.len().gt(10).all()
    assert d.loc["aa_comp_A", "significado"].startswith("fração do aminoácido A (alanina)")
    assert '"peptide antibiotic"' in d.loc["drug_class_peptide_antibiotic", "significado"]
    assert d.loc["drug_class_peptide_antibiotic", "genes_com_valor_1"] == 1
    assert d.loc["length", "pipelines"] == "A e B" and d.loc["depth_isa", "pipelines"] == "B"


def test_nested_group_subsets_whole_groups_per_class():
    from src.learning_curve import nested_group_subsets

    y = np.array([0] * 10 + [1] * 6)
    groups = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7])  # 5 grupos na classe 0, 3 na classe 1
    sub = nested_group_subsets(y, groups, (0.2, 0.6, 1.0), np.random.default_rng(0))
    assert len(sub[1.0]) == len(y)
    assert set(sub[0.2]) <= set(sub[0.6]) <= set(sub[1.0])  # encaixadas
    for idx in sub.values():  # grupos inteiros
        kept = set(groups[idx])
        assert all(set(np.flatnonzero(groups == g)) <= set(idx) for g in kept)
    # ceil(0,2 × 5) = 1 grupo da classe 0 e ceil(0,2 × 3) = 1 da classe 1
    assert len(set(groups[sub[0.2]])) == 2 and set(y[sub[0.2]]) == {0, 1}
    with pytest.raises(ValueError):
        nested_group_subsets(np.array([0, 1]), np.array([9, 9]), (0.5,), np.random.default_rng(0))


def test_shap_per_class_sums_groups():
    from src.interpretability import shap_per_class

    values = np.zeros((4, 3, 2))  # 4 genes, 3 atributos, 2 classes
    values[:, 0, 0] = 1.0
    values[:, 1, 0] = -0.5
    y = np.array([0, 0, 1, 1])
    out = shap_per_class(values, y, ["aa_comp_A", "aa_comp_C", "length"], ["c0", "c1"])
    g = out[(out["classe"] == "c0") & (out["nivel"] == "grupo") & (out["nome"] == "aminoacidos")].iloc[0]
    assert g["media_shap"] == pytest.approx(0.5) and g["media_abs_shap"] == pytest.approx(0.5)
    a = out[(out["classe"] == "c0") & (out["nome"] == "aa_comp_C")].iloc[0]
    assert a["media_shap"] == pytest.approx(-0.5) and a["media_abs_shap"] == pytest.approx(0.5)


def test_neighbor_index_from_files_cache_matches_constructor(tmp_path):
    from src.homology_groups import HIT_COLUMNS

    ids = [f"ARO:{i}" for i in range(6)]
    hits = _toy_hits().assign(alnlen=100, qlen=100, tlen=100, qcov=1.0, tcov=1.0)
    raw = hits.assign(query=hits["query"].str.replace("g", "ARO_"), target=hits["target"].str.replace("g", "ARO_"))
    path = tmp_path / "hits.m8"
    raw[HIT_COLUMNS].to_csv(path, sep="\t", header=False, index=False)
    from src.homology_groups import load_hits

    ref = NeighborIndex(load_hits(path), ids)  # caminho antigo (tabela de acertos lida do mesmo arquivo)
    first = NeighborIndex.from_files(ids, path)  # monta e grava o cache
    second = NeighborIndex.from_files(ids, path)  # lê do cache
    assert (tmp_path / "hits_indice.npz").exists()
    for idx in (first, second):
        assert np.array_equal(idx.q, ref.q) and np.array_equal(idx.t, ref.t) and np.array_equal(idx.evalue, ref.evalue)
        assert idx.nearest([0, 1], [2, 3, 4, 5])[0].tolist() == [3, 5]


# ── Caracterização dos dados e dos resultados (src.data_characterization) ─────────

def _toy_char():
    data = pd.DataFrame({
        "ARO_Accession": [f"ARO:{i}" for i in range(6)],
        "AMR_Gene_Family": ["fam1", "fam1", "fam2; fam3", "fam2", "fam4", "fam4"],
        TARGET: ["inat", "inat", "efl", "efl", "alt", "alt"],
    })
    return data


def test_family_table_and_overlap():
    from src.data_characterization import family_overlap_by_class, family_table, split_families

    assert split_families("a; b ;") == {"a", "b"} and split_families(None) == set()
    data = _toy_char()
    t = family_table(data, groups=np.array([0, 1, 2, 2, 3, 3])).set_index("familia")
    assert t.loc["fam1", "genes"] == 2 and t.loc["fam1", "n_grupos_70"] == 2  # fam1 em dois grupos
    assert (t["n_mecanismos"] == 1).all() and t.loc["fam2", "genes"] == 2
    # teste = genes 1, 3, 5; treino = 0, 2, 4. Gene 3 (fam2) tem fam2 no treino (gene 2, "fam2; fam3").
    ov = family_overlap_by_class(data, np.array([0, 2, 4]), np.array([1, 3, 5])).set_index("mecanismo")
    assert ov["genes_teste_familia_no_treino"].to_dict() == {"alt": 1, "efl": 1, "inat": 1}


def test_families_mixing_only_by_multilabel():
    from src.data_characterization import families_mixing_mechanisms

    d = pd.DataFrame({"AMR_Gene_Family": ["f", "f", "g", "g"], TARGET: ["x", "x;y", "x", "y"]})
    r = families_mixing_mechanisms(d)
    assert r["familias_com_mais_de_um_rotulo"] == 2
    assert r["familias_com_mais_de_um_mecanismo_entre_genes_de_rotulo_unico"] == 1  # só g mistura de verdade
    assert r["familias_que_misturam_so_por_genes_multirrotulo"] == ["f"]


def test_ontology_vectors_purity():
    from src.data_characterization import ontology_vectors

    onto = pd.DataFrame({"depth_isa": [6, 6, 6, 4], "drug_class_x": [1, 1, 1, 0]})
    y = np.array(["inat", "inat", "efl", "efl"])
    table, s = ontology_vectors(onto, y)
    assert s["vetores_distintos"] == 2 and s["pureza_global"] == pytest.approx(3 / 4)
    assert s["vetores_com_mais_de_uma_classe"] == 1 and s["genes_em_vetores_com_mais_de_uma_classe"] == 3


def test_lookup_rule_uses_only_training():
    from src.data_characterization import lookup_rule

    onto = pd.DataFrame({"depth_isa": [6, 6, 4, 4, 6, 9]})
    y = np.array(["inat", "inat", "efl", "efl", "inat", "efl"])
    r = lookup_rule(onto, y, np.arange(4), np.array([4, 5]), ["depth_isa"])
    # 6 visto no treino -> inat (certo); 9 nunca visto -> classe mais comum do treino (empate resolvido pelo value_counts)
    assert r["combinacoes_no_treino"] == 2 and r["genes_teste_com_combinacao_nao_vista"] == 1
    assert 0.5 <= r["acuracia_teste"] <= 1.0


def test_gain_per_class_cv_pairs_folds_and_uses_corrected_interval():
    from src.data_characterization import gain_per_class_cv
    from src.statistical_tests import corrected_interval

    rows = []
    rng = np.random.default_rng(0)
    for model in MODEL_NAMES:
        for rep in range(1, 3):
            for fold in range(1, 6):
                for c in ("x", "y"):
                    base = rng.uniform(0.3, 0.6)
                    for key, gain in (("a", 0.0), ("b", 0.2), ("b_gene_novo", 0.1)):
                        rows.append({"pipeline": key, "modelo": model, "repeticao": rep, "fold": fold, "classe": c, "f1": base + gain})
    out = gain_per_class_cv(pd.DataFrame(rows), 0.25)
    r = out[(out.cenario == "anotacao_curada") & (out.modelo == "svm") & (out.classe == "x")].iloc[0]
    assert r.dif_cv == pytest.approx(0.2) and r.folds_b_maior == 10 and r.ic_acima_de_zero
    g = out[(out.cenario == "gene_novo") & (out.modelo == "svm") & (out.classe == "x")].iloc[0]
    assert g.dif_cv == pytest.approx(0.1)


def test_main_errors_counts_both_directions():
    from src.data_characterization import ALTERATION, INACTIVATION, confusion_long, main_errors

    classes = [INACTIVATION, ALTERATION, "antibiotic efflux"]
    cm = np.array([[10, 3, 0], [2, 5, 1], [0, 0, 4]])
    conf = confusion_long({("a", "svm"): {"confusion_matrix": cm}}, classes)
    e = main_errors(conf).iloc[0]
    assert e.erros_inativacao_alteracao_dois_sentidos == 5 and e.erros_totais == 6
    assert e.maior_erro == f"{INACTIVATION} -> {ALTERATION}" and e.genes_no_maior_erro == 3


def test_mechanism_category_from_participates_in(tmp_path):
    # ARO_10 (categoria) participates_in ARO_20 (mecanismo, filho de ARO:1000002); o gene ARO_30 está sob ARO_10.
    from src.data_characterization import isa_category_by_mechanism

    obo = "http://purl.obolibrary.org/obo/"
    cls = lambda n: f'<Class IRI="{obo}ARO_{n}"/>'
    lab = lambda n, t: (f'<AnnotationAssertion><AnnotationProperty abbreviatedIRI="rdfs:label"/><IRI>{obo}ARO_{n}</IRI>'
                        f"<Literal>{t}</Literal></AnnotationAssertion>")
    body = "".join(f"<SubClassOf>{cls(c)}{cls(p)}</SubClassOf>" for c, p in [("1000002", "1000001"), ("20", "1000002"), ("10", "1000001"), ("30", "10"), ("31", "1000001")])
    body += (f'<SubClassOf>{cls("10")}<ObjectSomeValuesFrom><ObjectProperty IRI="{obo}RO_0000056"/>{cls("20")}'
             "</ObjectSomeValuesFrom></SubClassOf>")
    body += lab("20", "antibiotic efflux") + lab("10", "efflux pump")
    owl = tmp_path / "aro.owl"
    owl.write_text(f'<?xml version="1.0"?>\n<Ontology xmlns="http://www.w3.org/2002/07/owl#">{body}</Ontology>', encoding="utf-8")
    data = pd.DataFrame({"ARO_Accession": ["ARO:30", "ARO:31"], TARGET: ["antibiotic efflux", "antibiotic efflux"]})
    table, s = isa_category_by_mechanism(data, owl)
    assert s["genes_com_categoria_entre_os_ancestrais"] == 1
    assert s["genes_em_que_a_categoria_coincide_com_o_rotulo"] == 1
    assert s["termos_ligados_a_mecanismo_por_participates_in"] == {"efflux pump": "antibiotic efflux"}


def test_figure_catalog_entries_are_complete_and_unique():
    # Toda figura gerada precisa dizer o que mostra, de onde vêm os números e por que existe.
    from src.figure_catalog import CATALOG

    files = [e["arquivo"] for e in CATALOG]
    assert len(files) == len(set(files))
    for e in CATALOG:
        assert {"arquivo", "secao", "fonte", "o_que_mostra", "por_que", "legenda"} <= set(e)
        assert all(str(e[k]).strip() for k in ("o_que_mostra", "por_que", "legenda", "fonte"))


def test_figure_number_format():
    from src.visualization_analyses import _fmt

    assert _fmt(0.2361) == "0,236" and _fmt(-0.05, 2) == "-0,05" and _fmt(0.13, 2, sign=True) == "+0,13"


def test_axis_decimal_comma():
    """Os eixos numéricos saem com vírgula decimal; eixos com rótulos fixos não são tocados."""
    import matplotlib.pyplot as plt

    from src.plot_style import decimal_comma

    fig, (ax, bx) = plt.subplots(1, 2)
    ax.plot([0, 0.5, 1], [0, 0.25, 1])
    inset = ax.inset_axes([0.5, 0.5, 0.4, 0.4])  # detalhe ampliado: não está em fig.axes
    inset.plot([0, 0.05, 0.1], [0.8, 0.9, 1.0])
    bx.set_xticks([0, 1], ["um", "dois"])
    decimal_comma(fig)
    fig.canvas.draw()
    for axes in (ax, inset):
        labels = [t.get_text() for t in axes.get_xticklabels() if t.get_text()]
        assert any("," in s for s in labels) and not any("." in s for s in labels)
    assert [t.get_text() for t in bx.get_xticklabels()] == ["um", "dois"]
    plt.close(fig)
