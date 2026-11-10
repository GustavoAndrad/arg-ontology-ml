"""Testes dos módulos de extração, pipelines, estatística e relatório."""

import numpy as np
import pandas as pd
import pytest

from src.experiment_config import EFFECTIVE_CLASSES, N_CV_SPLITS_TCC, cv_splitter, split_indices
from src.feature_extraction import (
    SEQUENCE_FEATURE_COLUMNS,
    compute_amino_acid_composition,
    compute_dipeptide_composition,
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
    feats = compute_sequence_features(SEQ)
    assert len(SEQUENCE_FEATURE_COLUMNS) == 426
    assert list(feats) == SEQUENCE_FEATURE_COLUMNS


def test_compositions_sum_to_one():
    assert sum(compute_amino_acid_composition(SEQ).values()) == pytest.approx(1.0)
    assert sum(compute_dipeptide_composition(SEQ).values()) == pytest.approx(1.0)


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
    assert np.isfinite(feats["molecular_weight"])


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


def test_spearman_leakage_detection():
    y = np.array([0, 0, 1, 1, 2, 2])
    feats = pd.DataFrame({"leaky": y * 10, "noise": [1, 0, 0, 1, 1, 0], "const": [1] * 6})
    corr = compute_spearman_correlations(feats, y)
    assert identify_leaky_features(corr) == ["leaky"]
    assert corr["const"] == 0.0


def test_split_disjoint_and_stratified():
    y = np.array([0] * 80 + [1] * 15 + [2] * 5)
    tr, te = split_indices(y)
    assert not set(tr) & set(te) and len(tr) + len(te) == len(y)
    assert abs((y[te] == 1).mean() - (y == 1).mean()) <= 0.02


def test_cv_has_ten_folds():
    assert cv_splitter().get_n_splits() == N_CV_SPLITS_TCC == 10
    assert len(EFFECTIVE_CLASSES) == 4


@pytest.mark.parametrize("model", ["random_forest", "xgboost", "logistic_regression", "svm"])
def test_pipeline_structure_and_fit(model):
    pipe = build_full_pipeline(model, k_features=5)
    assert [s for s, _ in pipe.steps] == ["scaler", "variance", "select", "clf"]
    assert pipe.named_steps["clf"].get_params()["random_state"] == 42
    rng = np.random.default_rng(0)
    X, y = rng.random((60, 8)), np.repeat([0, 1, 2], 20)
    assert pipe.fit(X, y).predict_proba(X).shape == (60, 3)


def test_wilcoxon_decision_consistent_with_p():
    a = np.linspace(0.6, 0.7, 10)
    res = run_wilcoxon_test(a, a + 0.05)
    assert res["p_value"] == pytest.approx(0.001953125) and res["decision"] == "reject_H0"
    same = run_wilcoxon_test(a, a)
    assert same["decision"] == "fail_to_reject"
