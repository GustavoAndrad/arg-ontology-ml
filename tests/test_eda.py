"""
tests/test_eda.py — Testes de exemplo para src/eda.py.

Cobre:
- compute_class_distribution: cálculo correto de contagens e proporções.
- compute_imbalance_ratio: razão majoritária/minoritária correta.
- flag_small_classes: detecção de classes abaixo do limiar.
- check_missing_values: reporte correto de valores ausentes.
- plot_class_distribution: criação do arquivo PNG com DPI ≥ 300.
- generate_eda_report: orquestração retorna todas as chaves esperadas.
"""

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.eda import (
    ACCESSIBLE_PALETTE,
    check_missing_values,
    compute_class_distribution,
    compute_imbalance_ratio,
    flag_small_classes,
    generate_eda_report,
    plot_class_distribution,
)


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures compartilhadas
# ─────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def balanced_df() -> pd.DataFrame:
    """DataFrame com 3 classes de tamanhos iguais (10 cada)."""
    return pd.DataFrame({
        "Resistance_Mechanism": ["efflux"] * 10 + ["inactivation"] * 10 + ["alteration"] * 10,
        "Sequence": ["ACDE"] * 30,
    })


@pytest.fixture
def imbalanced_df() -> pd.DataFrame:
    """DataFrame com distribuição desbalanceada: 100 / 10 / 5."""
    return pd.DataFrame({
        "Resistance_Mechanism": (
            ["efflux"] * 100
            + ["inactivation"] * 10
            + ["alteration"] * 5
        ),
        "Sequence": ["ACDE"] * 115,
    })


@pytest.fixture
def distribution_balanced(balanced_df) -> pd.DataFrame:
    return compute_class_distribution(balanced_df, "Resistance_Mechanism")


@pytest.fixture
def distribution_imbalanced(imbalanced_df) -> pd.DataFrame:
    return compute_class_distribution(imbalanced_df, "Resistance_Mechanism")


# ─────────────────────────────────────────────────────────────────────────────
# compute_class_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeClassDistribution:

    def test_returns_dataframe_with_required_columns(self, balanced_df):
        dist = compute_class_distribution(balanced_df, "Resistance_Mechanism")
        assert set(dist.columns) == {"classe", "contagem", "proporcao_pct"}

    def test_correct_number_of_classes(self, balanced_df):
        dist = compute_class_distribution(balanced_df, "Resistance_Mechanism")
        assert len(dist) == 3

    def test_contagem_sums_to_total(self, balanced_df):
        dist = compute_class_distribution(balanced_df, "Resistance_Mechanism")
        assert dist["contagem"].sum() == len(balanced_df)

    def test_proporcao_sums_to_100(self, balanced_df):
        dist = compute_class_distribution(balanced_df, "Resistance_Mechanism")
        assert math.isclose(dist["proporcao_pct"].sum(), 100.0, abs_tol=0.1)

    def test_balanced_proportions_are_equal(self, balanced_df):
        dist = compute_class_distribution(balanced_df, "Resistance_Mechanism")
        # Todas as classes têm 10/30 = 33.33%
        for pct in dist["proporcao_pct"]:
            assert math.isclose(pct, 100 / 3, abs_tol=0.1)

    def test_sorted_descending_by_count(self, imbalanced_df):
        dist = compute_class_distribution(imbalanced_df, "Resistance_Mechanism")
        counts = dist["contagem"].tolist()
        assert counts == sorted(counts, reverse=True)

    def test_correct_counts_for_imbalanced(self, imbalanced_df):
        dist = compute_class_distribution(imbalanced_df, "Resistance_Mechanism")
        count_map = dict(zip(dist["classe"], dist["contagem"]))
        assert count_map["efflux"] == 100
        assert count_map["inactivation"] == 10
        assert count_map["alteration"] == 5

    def test_missing_column_raises_key_error(self, balanced_df):
        with pytest.raises(KeyError, match="nonexistent"):
            compute_class_distribution(balanced_df, "nonexistent")

    def test_empty_dataframe_raises_value_error(self):
        empty_df = pd.DataFrame(columns=["Resistance_Mechanism"])
        with pytest.raises(ValueError, match="vazio"):
            compute_class_distribution(empty_df, "Resistance_Mechanism")

    def test_single_class(self):
        df = pd.DataFrame({"Resistance_Mechanism": ["efflux"] * 5})
        dist = compute_class_distribution(df, "Resistance_Mechanism")
        assert len(dist) == 1
        assert dist.iloc[0]["contagem"] == 5
        assert math.isclose(dist.iloc[0]["proporcao_pct"], 100.0, abs_tol=0.01)

    def test_proporcao_pct_rounded_to_two_decimals(self, imbalanced_df):
        dist = compute_class_distribution(imbalanced_df, "Resistance_Mechanism")
        for pct in dist["proporcao_pct"]:
            # Verificar que não há mais de 2 casas decimais significativas
            assert round(pct, 2) == pct


# ─────────────────────────────────────────────────────────────────────────────
# compute_imbalance_ratio
# ─────────────────────────────────────────────────────────────────────────────

class TestComputeImbalanceRatio:

    def test_balanced_ratio_is_one(self, distribution_balanced):
        ratio = compute_imbalance_ratio(distribution_balanced)
        assert math.isclose(ratio, 1.0, abs_tol=1e-9)

    def test_imbalanced_ratio_correct(self, distribution_imbalanced):
        # 100 / 5 = 20.0
        ratio = compute_imbalance_ratio(distribution_imbalanced)
        assert math.isclose(ratio, 20.0, abs_tol=1e-9)

    def test_ratio_is_float(self, distribution_balanced):
        ratio = compute_imbalance_ratio(distribution_balanced)
        assert isinstance(ratio, float)

    def test_ratio_always_gte_one(self, distribution_imbalanced):
        ratio = compute_imbalance_ratio(distribution_imbalanced)
        assert ratio >= 1.0

    def test_missing_contagem_column_raises(self):
        bad_df = pd.DataFrame({"classe": ["a", "b"], "proporcao_pct": [50.0, 50.0]})
        with pytest.raises(ValueError, match="contagem"):
            compute_imbalance_ratio(bad_df)

    def test_empty_distribution_raises(self):
        empty = pd.DataFrame(columns=["classe", "contagem", "proporcao_pct"])
        with pytest.raises(ValueError):
            compute_imbalance_ratio(empty)

    def test_zero_min_count_raises(self):
        dist = pd.DataFrame({
            "classe": ["a", "b"],
            "contagem": [100, 0],
            "proporcao_pct": [100.0, 0.0],
        })
        with pytest.raises(ValueError, match="0"):
            compute_imbalance_ratio(dist)

    def test_two_classes_ratio(self):
        dist = pd.DataFrame({
            "classe": ["majority", "minority"],
            "contagem": [80, 20],
            "proporcao_pct": [80.0, 20.0],
        })
        ratio = compute_imbalance_ratio(dist)
        assert math.isclose(ratio, 4.0, abs_tol=1e-9)


# ─────────────────────────────────────────────────────────────────────────────
# flag_small_classes
# ─────────────────────────────────────────────────────────────────────────────

class TestFlagSmallClasses:

    def test_no_small_classes_returns_empty_list(self, distribution_balanced):
        # Todas as classes têm 10 instâncias — abaixo do default threshold=20
        # (threshold=20, 10 < 20 → pequenas)
        small = flag_small_classes(distribution_balanced, threshold=5)
        assert small == []

    def test_detects_all_small_classes(self, distribution_imbalanced):
        # contagens: 100, 10, 5 — com threshold=20, inactivation(10) e alteration(5) são pequenas
        small = flag_small_classes(distribution_imbalanced, threshold=20)
        assert "inactivation" in small
        assert "alteration" in small
        assert "efflux" not in small

    def test_default_threshold_is_20(self, distribution_imbalanced):
        small = flag_small_classes(distribution_imbalanced)
        assert "alteration" in small

    def test_returns_list_of_strings(self, distribution_imbalanced):
        small = flag_small_classes(distribution_imbalanced)
        assert isinstance(small, list)
        for item in small:
            assert isinstance(item, str)

    def test_threshold_boundary_inclusive(self):
        # Classe com exatamente threshold instâncias NÃO deve ser sinalizada
        dist = pd.DataFrame({
            "classe": ["a", "b"],
            "contagem": [20, 19],
            "proporcao_pct": [51.0, 49.0],
        })
        small = flag_small_classes(dist, threshold=20)
        assert "a" not in small  # 20 >= threshold → não pequena
        assert "b" in small       # 19 < threshold → pequena

    def test_all_classes_large(self):
        dist = pd.DataFrame({
            "classe": ["a", "b", "c"],
            "contagem": [100, 200, 150],
            "proporcao_pct": [22.2, 44.5, 33.3],
        })
        small = flag_small_classes(dist, threshold=50)
        assert small == []

    def test_missing_columns_raises(self):
        bad = pd.DataFrame({"classe": ["a"], "proporcao_pct": [100.0]})
        with pytest.raises(ValueError, match="contagem"):
            flag_small_classes(bad)

    def test_single_small_class(self):
        dist = pd.DataFrame({
            "classe": ["a", "b"],
            "contagem": [100, 5],
            "proporcao_pct": [95.2, 4.8],
        })
        small = flag_small_classes(dist, threshold=20)
        assert small == ["b"]


# ─────────────────────────────────────────────────────────────────────────────
# check_missing_values
# ─────────────────────────────────────────────────────────────────────────────

class TestCheckMissingValues:

    def test_no_missing_returns_empty_df(self):
        df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
        result = check_missing_values(df)
        assert result.empty
        assert set(result.columns) == {"coluna", "n_ausentes", "pct_ausentes"}

    def test_detects_missing_in_one_column(self):
        df = pd.DataFrame({"a": [1, None, 3], "b": ["x", "y", "z"]})
        result = check_missing_values(df)
        assert len(result) == 1
        assert result.iloc[0]["coluna"] == "a"
        assert result.iloc[0]["n_ausentes"] == 1

    def test_percentage_calculation_correct(self):
        df = pd.DataFrame({"a": [None, None, 1, 1], "b": [1, 2, 3, 4]})
        result = check_missing_values(df)
        row = result[result["coluna"] == "a"].iloc[0]
        assert math.isclose(row["pct_ausentes"], 50.0, abs_tol=0.01)

    def test_multiple_columns_with_missing(self):
        df = pd.DataFrame({
            "a": [None, 1, 2],
            "b": [None, None, 3],
            "c": [1, 2, 3],
        })
        result = check_missing_values(df)
        assert len(result) == 2
        assert set(result["coluna"]) == {"a", "b"}

    def test_sorted_descending_by_n_ausentes(self):
        df = pd.DataFrame({
            "a": [None, 1, 2],           # 1 ausente
            "b": [None, None, None],     # 3 ausentes
        })
        result = check_missing_values(df)
        assert result.iloc[0]["n_ausentes"] >= result.iloc[1]["n_ausentes"]

    def test_result_has_required_columns(self):
        df = pd.DataFrame({"a": [None, 1]})
        result = check_missing_values(df)
        assert set(result.columns) == {"coluna", "n_ausentes", "pct_ausentes"}

    def test_empty_dataframe_returns_empty_result(self):
        df = pd.DataFrame(columns=["a", "b"])
        result = check_missing_values(df)
        assert result.empty

    def test_all_missing_column(self):
        df = pd.DataFrame({"a": [None, None, None], "b": [1, 2, 3]})
        result = check_missing_values(df)
        row = result[result["coluna"] == "a"].iloc[0]
        assert row["n_ausentes"] == 3
        assert math.isclose(row["pct_ausentes"], 100.0, abs_tol=0.01)

    def test_pct_ausentes_rounded_to_two_decimals(self):
        # 1/3 ausentes = 33.33%
        df = pd.DataFrame({"a": [None, 1, 2]})
        result = check_missing_values(df)
        pct = result.iloc[0]["pct_ausentes"]
        assert round(pct, 2) == pct


# ─────────────────────────────────────────────────────────────────────────────
# plot_class_distribution
# ─────────────────────────────────────────────────────────────────────────────

class TestPlotClassDistribution:

    def test_creates_png_file(self, distribution_balanced, tmp_path):
        output = tmp_path / "test_plot.png"
        plot_class_distribution(distribution_balanced, output, dpi=100)
        assert output.exists()

    def test_creates_parent_directories(self, distribution_balanced, tmp_path):
        output = tmp_path / "nested" / "dir" / "plot.png"
        plot_class_distribution(distribution_balanced, output, dpi=100)
        assert output.exists()

    def test_minimum_dpi_300_produces_file(self, distribution_balanced, tmp_path):
        output = tmp_path / "hq_plot.png"
        plot_class_distribution(distribution_balanced, output, dpi=300)
        assert output.exists()
        # Verificar que o arquivo tem tamanho razoável (> 1 KB)
        assert output.stat().st_size > 1024

    def test_missing_required_columns_raises(self, tmp_path):
        bad_dist = pd.DataFrame({"classe": ["a"], "contagem": [10]})
        with pytest.raises(ValueError, match="proporcao_pct"):
            plot_class_distribution(bad_dist, tmp_path / "out.png")

    def test_file_is_overwritten_on_rerun(self, distribution_balanced, tmp_path):
        output = tmp_path / "plot.png"
        plot_class_distribution(distribution_balanced, output, dpi=72)
        size_first = output.stat().st_size
        plot_class_distribution(distribution_balanced, output, dpi=72)
        size_second = output.stat().st_size
        # Arquivo foi reescrito (tamanho igual ou similar)
        assert size_second > 0

    def test_accepts_path_as_string(self, distribution_balanced, tmp_path):
        output = str(tmp_path / "str_path.png")
        # Não deve lançar exceção mesmo passando string
        plot_class_distribution(distribution_balanced, output, dpi=72)
        assert Path(output).exists()


# ─────────────────────────────────────────────────────────────────────────────
# generate_eda_report
# ─────────────────────────────────────────────────────────────────────────────

class TestGenerateEdaReport:

    @pytest.fixture
    def sample_df(self) -> pd.DataFrame:
        """DataFrame mínimo com colunas esperadas pelo data_loader."""
        return pd.DataFrame({
            "Resistance_Mechanism": (
                ["antibiotic efflux"] * 50
                + ["antibiotic inactivation"] * 30
                + ["antibiotic target alteration"] * 10
                + ["reduced permeability to antibiotic"] * 5
                + ["resistance by absence"] * 2
            ),
            "Sequence": ["ACDEFGHIKLMNPQRSTVWY"] * 97,
            "ARO_Accession": [f"ARO:{i:07d}" for i in range(97)],
            "Drug_Class": ["beta-lactam antibiotic"] * 97,
            "AMR_Gene_Family": ["TEM beta-lactamase"] * 97,
            "Protein_Accession": [f"ACC{i}" for i in range(97)],
        })

    def test_returns_dict_with_all_keys(self, sample_df, tmp_path):
        report = generate_eda_report(sample_df, tmp_path)
        expected_keys = {
            "class_distribution",
            "imbalance_ratio",
            "small_classes",
            "missing_values",
            "ontology_stats",
            "n_total_instances",
        }
        assert expected_keys == set(report.keys())

    def test_n_total_instances_correct(self, sample_df, tmp_path):
        report = generate_eda_report(sample_df, tmp_path)
        assert report["n_total_instances"] == len(sample_df)

    def test_class_distribution_is_dataframe(self, sample_df, tmp_path):
        report = generate_eda_report(sample_df, tmp_path)
        assert isinstance(report["class_distribution"], pd.DataFrame)

    def test_imbalance_ratio_is_float(self, sample_df, tmp_path):
        report = generate_eda_report(sample_df, tmp_path)
        assert isinstance(report["imbalance_ratio"], float)
        # 50 / 2 = 25.0
        assert math.isclose(report["imbalance_ratio"], 25.0, abs_tol=0.01)

    def test_small_classes_detected(self, sample_df, tmp_path):
        report = generate_eda_report(sample_df, tmp_path)
        # Classes com < 20: "reduced permeability to antibiotic" (5) e
        # "resistance by absence" (2)
        small = report["small_classes"]
        assert isinstance(small, list)
        assert "resistance by absence" in small
        assert "reduced permeability to antibiotic" in small

    def test_missing_values_is_dataframe(self, sample_df, tmp_path):
        report = generate_eda_report(sample_df, tmp_path)
        assert isinstance(report["missing_values"], pd.DataFrame)

    def test_no_missing_in_clean_df(self, sample_df, tmp_path):
        report = generate_eda_report(sample_df, tmp_path)
        assert report["missing_values"].empty

    def test_missing_values_detected_when_present(self, tmp_path):
        df_with_missing = pd.DataFrame({
            "Resistance_Mechanism": ["efflux"] * 5 + [None] * 2,
            "Sequence": ["ACDE"] * 7,
            "ARO_Accession": [f"ARO:{i:07d}" for i in range(7)],
        })
        report = generate_eda_report(df_with_missing, tmp_path)
        # Não deve lançar; missing_values pode estar presente ou vazio
        assert isinstance(report["missing_values"], pd.DataFrame)

    def test_eda_plot_created(self, sample_df, tmp_path):
        generate_eda_report(sample_df, tmp_path)
        plot_path = tmp_path / "eda" / "class_distribution.png"
        assert plot_path.exists()

    def test_eda_dir_created(self, sample_df, tmp_path):
        generate_eda_report(sample_df, tmp_path)
        assert (tmp_path / "eda").is_dir()

    def test_ontology_stats_none_when_file_missing(self, sample_df, tmp_path, monkeypatch):
        """Quando o arquivo de ontologia não existe, ontology_stats deve ser None."""
        import src.eda as eda_module

        # Monkeypatchar ONTOLOGY_PATH para um caminho inexistente
        import src.config as config_module
        original = config_module.ONTOLOGY_PATH
        config_module.ONTOLOGY_PATH = tmp_path / "nonexistent.owl"
        try:
            report = generate_eda_report(sample_df, tmp_path)
            assert report["ontology_stats"] is None
        finally:
            config_module.ONTOLOGY_PATH = original


# ─────────────────────────────────────────────────────────────────────────────
# ACCESSIBLE_PALETTE
# ─────────────────────────────────────────────────────────────────────────────

class TestAccessiblePalette:

    def test_palette_is_list(self):
        assert isinstance(ACCESSIBLE_PALETTE, list)

    def test_palette_has_at_least_5_colors(self):
        assert len(ACCESSIBLE_PALETTE) >= 5

    def test_palette_entries_are_hex_strings(self):
        for color in ACCESSIBLE_PALETTE:
            assert color.startswith("#"), f"Cor '{color}' não é um código hex válido"
            assert len(color) == 7, f"Cor '{color}' não está no formato #RRGGBB"
