"""
test_data_loader.py — Testes de exemplo para src/data_loader.py.

Testa:
- Parsing de headers FASTA com diferentes formatos
- Integração de DataFrames sintéticos
- Rejeição de instâncias com campos ausentes
- Validação do dataset com violações propositais
"""

import io
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from src.data_loader import (
    _normalize_columns,
    integrate_dataset,
    load_aro_categories_index,
    load_aro_index,
    parse_fasta_header,
    validate_dataset,
    load_protein_fasta,
)


# ──────────────────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────────────────

@pytest.fixture
def sample_aro_index_df() -> pd.DataFrame:
    """DataFrame mínimo com as colunas padronizadas de aro_index."""
    return pd.DataFrame({
        "ARO_Accession": ["ARO:0000001", "ARO:0000002", "ARO:0000003"],
        "Model_Name": ["GeneA", "GeneB", "GeneC"],
        "ARO_Name": ["Gene A full", "Gene B full", "Gene C full"],
        "Protein_Accession": ["AAA00001.1", "AAA00002.1", "AAA00003.1"],
        "AMR_Gene_Family": ["FamilyX", "FamilyY", "FamilyZ"],
        "Drug_Class": ["penicillin antibiotic", "tetracycline antibiotic", ""],
        "Resistance_Mechanism": [
            "antibiotic inactivation",
            "antibiotic efflux",
            "antibiotic target alteration",
        ],
        "CARD_Short_Name": ["short_a", "short_b", "short_c"],
    })


@pytest.fixture
def sample_fasta_sequences() -> dict[str, str]:
    """Dict mínimo mapeando ARO_Accession -> sequência proteica."""
    return {
        "ARO:0000001": "MKAYFIAILTLFTCIATVVRA",
        "ARO:0000002": "ACDEFGHIKLMNPQRSTVWY",
        # ARO:0000003 propositalmente ausente para testar inner join
    }


# ──────────────────────────────────────────────────────────────────────────────
# Testes de parse_fasta_header
# ──────────────────────────────────────────────────────────────────────────────

class TestParseFastaHeader:
    """Testes para parse_fasta_header com diferentes formatos de header."""

    def test_format_homolog_model(self):
        """Formato real do protein_fasta_protein_homolog_model.fasta."""
        header = "gb|ACT97415.1|ARO:3002999|CblA-1 [mixed culture bacterium AX_gF3SD01_15]"
        aro_acc, prot_acc = parse_fasta_header(header)
        assert aro_acc == "ARO:3002999"
        assert prot_acc == "ACT97415.1"

    def test_format_spec_described(self):
        """Formato descrito na especificação: ARO:XXXXXXX|gb|ACCESSION|..."""
        header = "ARO:3005099|gb|AAB60941.1| blaTEM-15 [Homo sapiens]"
        aro_acc, prot_acc = parse_fasta_header(header)
        assert aro_acc == "ARO:3005099"
        assert prot_acc == "AAB60941.1"

    def test_format_klebsiella(self):
        """Formato com organismo contendo espaços múltiplos."""
        header = "gb|AEJ08681.1|ARO:3001109|SHV-52 [Klebsiella pneumoniae]"
        aro_acc, prot_acc = parse_fasta_header(header)
        assert aro_acc == "ARO:3001109"
        assert prot_acc == "AEJ08681.1"

    def test_format_without_organism(self):
        """Formato sem descrição de organismo."""
        header = "gb|AAD01868.1|ARO:3002867|dfrF"
        aro_acc, prot_acc = parse_fasta_header(header)
        assert aro_acc == "ARO:3002867"
        assert prot_acc == "AAD01868.1"

    def test_aro_accession_starts_with_prefix(self):
        """O ARO accession retornado deve sempre começar com 'ARO:'."""
        header = "gb|AFJ59957.1|ARO:3001989|CTX-M-130 [Escherichia coli]"
        aro_acc, _ = parse_fasta_header(header)
        assert aro_acc.startswith("ARO:")

    def test_invalid_header_raises_value_error(self):
        """Header sem ARO accession deve levantar ValueError."""
        with pytest.raises(ValueError, match="ARO Accession não encontrado"):
            parse_fasta_header("gb|AAA00001.1|SomeGene [Organism]")

    def test_emb_database_prefix(self):
        """Header com prefixo 'emb' (EMBL) em vez de 'gb'."""
        header = "emb|CAA47209.1|ARO:3000521|OXA-1 [Pseudomonas aeruginosa]"
        aro_acc, prot_acc = parse_fasta_header(header)
        assert aro_acc == "ARO:3000521"
        assert prot_acc == "CAA47209.1"


# ──────────────────────────────────────────────────────────────────────────────
# Testes de _normalize_columns
# ──────────────────────────────────────────────────────────────────────────────

class TestNormalizeColumns:
    """Testes para normalização de nomes de colunas."""

    def test_spaces_to_underscores(self):
        """Colunas com espaços devem ser renomeadas para underscore."""
        df = pd.DataFrame(columns=[
            "ARO Accession", "Model Name", "Resistance Mechanism", "Drug Class"
        ])
        result = _normalize_columns(df)
        assert "ARO_Accession" in result.columns
        assert "Model_Name" in result.columns
        assert "Resistance_Mechanism" in result.columns
        assert "Drug_Class" in result.columns

    def test_already_normalized(self):
        """Colunas já com underscore não devem ser alteradas."""
        df = pd.DataFrame(columns=["ARO_Accession", "Model_Name"])
        result = _normalize_columns(df)
        assert "ARO_Accession" in result.columns
        assert "Model_Name" in result.columns

    def test_unknown_columns_preserved(self):
        """Colunas não mapeadas devem ser preservadas como estão."""
        df = pd.DataFrame(columns=["ARO_Accession", "Custom_Column_XYZ"])
        result = _normalize_columns(df)
        assert "Custom_Column_XYZ" in result.columns


# ──────────────────────────────────────────────────────────────────────────────
# Testes de load_aro_index (com arquivo temporário)
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadAroIndex:
    """Testes para load_aro_index usando arquivos TSV temporários."""

    def _write_tsv(self, path: Path, content: str) -> None:
        path.write_text(content, encoding="utf-8")

    def test_loads_with_space_headers(self, tmp_path):
        """Deve carregar TSV com colunas com espaços e renomear corretamente."""
        tsv_content = (
            "ARO Accession\tModel Name\tARO Name\tProtein Accession\t"
            "AMR Gene Family\tDrug Class\tResistance Mechanism\tCARD Short Name\n"
            "ARO:0000001\tGeneA\tGene A\tAAA00001.1\tFamilyX\tpenicillin antibiotic\t"
            "antibiotic inactivation\tshort_a\n"
        )
        tsv_file = tmp_path / "aro_index.tsv"
        self._write_tsv(tsv_file, tsv_content)

        df = load_aro_index(tsv_file)
        assert "ARO_Accession" in df.columns
        assert "Resistance_Mechanism" in df.columns
        assert len(df) == 1
        assert df["ARO_Accession"].iloc[0] == "ARO:0000001"

    def test_loads_with_underscore_headers(self, tmp_path):
        """Deve carregar TSV com colunas já com underscore."""
        tsv_content = (
            "ARO_Accession\tModel_Name\tARO_Name\tProtein_Accession\t"
            "AMR_Gene_Family\tDrug_Class\tResistance_Mechanism\tCARD_Short_Name\n"
            "ARO:0000002\tGeneB\tGene B\tAAA00002.1\tFamilyY\ttetracycline antibiotic\t"
            "antibiotic efflux\tshort_b\n"
        )
        tsv_file = tmp_path / "aro_index.tsv"
        self._write_tsv(tsv_file, tsv_content)

        df = load_aro_index(tsv_file)
        assert "ARO_Accession" in df.columns
        assert df["ARO_Accession"].iloc[0] == "ARO:0000002"

    def test_missing_required_column_raises_error(self, tmp_path):
        """TSV sem coluna obrigatória deve levantar ValueError."""
        # Falta 'CARD Short Name'
        tsv_content = (
            "ARO Accession\tModel Name\tARO Name\tProtein Accession\t"
            "AMR Gene Family\tDrug Class\tResistance Mechanism\n"
            "ARO:0000001\tGeneA\tGene A\tAAA00001.1\tFamilyX\tpenicillin antibiotic\t"
            "antibiotic inactivation\n"
        )
        tsv_file = tmp_path / "aro_index.tsv"
        self._write_tsv(tsv_file, tsv_content)

        with pytest.raises(ValueError, match="Colunas obrigatórias ausentes"):
            load_aro_index(tsv_file)

    def test_file_not_found_raises_error(self, tmp_path):
        """Arquivo inexistente deve levantar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_aro_index(tmp_path / "nonexistent.tsv")


# ──────────────────────────────────────────────────────────────────────────────
# Testes de load_aro_categories_index
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadAroCategoriesIndex:
    """Testes para load_aro_categories_index."""

    def test_loads_basic_categories(self, tmp_path):
        """Deve carregar TSV com colunas de categorias."""
        tsv_content = (
            "Protein Accession\tDNA Accession\tAMR Gene Family\t"
            "Drug Class\tResistance Mechanism\n"
            "AAB60941.1\tAF002716.1\tErm 23S ribosomal RNA methyltransferase\t"
            "lincosamide antibiotic\tantibiotic target alteration\n"
        )
        tsv_file = tmp_path / "aro_categories_index.tsv"
        tsv_file.write_text(tsv_content, encoding="utf-8")

        df = load_aro_categories_index(tsv_file)
        assert "Protein_Accession" in df.columns
        assert "Resistance_Mechanism" in df.columns
        assert len(df) == 1

    def test_file_not_found(self, tmp_path):
        """Arquivo inexistente deve levantar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_aro_categories_index(tmp_path / "nonexistent.tsv")


# ──────────────────────────────────────────────────────────────────────────────
# Testes de load_protein_fasta (com arquivo temporário)
# ──────────────────────────────────────────────────────────────────────────────

class TestLoadProteinFasta:
    """Testes para load_protein_fasta com arquivos FASTA temporários."""

    def test_loads_sequences_correctly(self, tmp_path):
        """Deve mapear ARO accession para sequência proteica corretamente."""
        fasta_content = (
            ">gb|ACT97415.1|ARO:3002999|CblA-1 [mixed culture]\n"
            "MKAYFIAILTLFTCIATVVRA\n"
            ">gb|AEJ08681.1|ARO:3001109|SHV-52 [Klebsiella pneumoniae]\n"
            "ACDEFGHIKLMNPQRSTVWY\n"
        )
        fasta_file = tmp_path / "test.fasta"
        fasta_file.write_text(fasta_content, encoding="utf-8")

        result = load_protein_fasta(fasta_file)
        assert "ARO:3002999" in result
        assert result["ARO:3002999"] == "MKAYFIAILTLFTCIATVVRA"
        assert "ARO:3001109" in result
        assert len(result) == 2

    def test_file_not_found(self, tmp_path):
        """Arquivo inexistente deve levantar FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            load_protein_fasta(tmp_path / "nonexistent.fasta")


# ──────────────────────────────────────────────────────────────────────────────
# Testes de integrate_dataset
# ──────────────────────────────────────────────────────────────────────────────

class TestIntegrateDataset:
    """Testes para integrate_dataset com DataFrames sintéticos."""

    def test_inner_join_keeps_only_matching(
        self, sample_aro_index_df, sample_fasta_sequences
    ):
        """Inner join deve manter apenas ARO Accessions presentes em ambas as fontes."""
        result = integrate_dataset(sample_aro_index_df, sample_fasta_sequences)
        # ARO:0000001 e ARO:0000002 existem em ambas as fontes
        # ARO:0000003 não tem sequência FASTA → descartado
        assert set(result["ARO_Accession"]) == {"ARO:0000001", "ARO:0000002"}
        assert len(result) == 2

    def test_output_columns(self, sample_aro_index_df, sample_fasta_sequences):
        """O DataFrame resultante deve ter exatamente as 6 colunas esperadas."""
        result = integrate_dataset(sample_aro_index_df, sample_fasta_sequences)
        expected_cols = {
            "ARO_Accession", "Sequence", "Resistance_Mechanism",
            "Drug_Class", "AMR_Gene_Family", "Protein_Accession",
        }
        assert set(result.columns) == expected_cols

    def test_sequence_values_correct(
        self, sample_aro_index_df, sample_fasta_sequences
    ):
        """As sequências integradas devem corresponder ao dict FASTA."""
        result = integrate_dataset(sample_aro_index_df, sample_fasta_sequences)
        row = result[result["ARO_Accession"] == "ARO:0000001"].iloc[0]
        assert row["Sequence"] == sample_fasta_sequences["ARO:0000001"]

    def test_missing_resistance_mechanism_discarded(self, sample_fasta_sequences):
        """Instâncias com Resistance_Mechanism vazio/NaN devem ser descartadas."""
        aro_df = pd.DataFrame({
            "ARO_Accession": ["ARO:0000001", "ARO:0000002"],
            "Model_Name": ["GeneA", "GeneB"],
            "ARO_Name": ["Gene A", "Gene B"],
            "Protein_Accession": ["AAA00001.1", "AAA00002.1"],
            "AMR_Gene_Family": ["FamilyX", "FamilyY"],
            "Drug_Class": ["penicillin antibiotic", "tetracycline antibiotic"],
            "Resistance_Mechanism": ["antibiotic inactivation", ""],  # ARO:0000002 vazio
            "CARD_Short_Name": ["short_a", "short_b"],
        })
        result = integrate_dataset(aro_df, sample_fasta_sequences)
        assert len(result) == 1
        assert result["ARO_Accession"].iloc[0] == "ARO:0000001"

    def test_nan_resistance_mechanism_discarded(self, sample_fasta_sequences):
        """Instâncias com Resistance_Mechanism=NaN devem ser descartadas."""
        aro_df = pd.DataFrame({
            "ARO_Accession": ["ARO:0000001", "ARO:0000002"],
            "Model_Name": ["GeneA", "GeneB"],
            "ARO_Name": ["Gene A", "Gene B"],
            "Protein_Accession": ["AAA00001.1", "AAA00002.1"],
            "AMR_Gene_Family": ["FamilyX", "FamilyY"],
            "Drug_Class": ["penicillin antibiotic", "tetracycline antibiotic"],
            "Resistance_Mechanism": ["antibiotic inactivation", None],  # None → NaN
            "CARD_Short_Name": ["short_a", "short_b"],
        })
        result = integrate_dataset(aro_df, sample_fasta_sequences)
        assert len(result) == 1
        assert result["ARO_Accession"].iloc[0] == "ARO:0000001"

    def test_empty_fasta_returns_empty_df(self, sample_aro_index_df):
        """FASTA vazio deve retornar DataFrame vazio."""
        result = integrate_dataset(sample_aro_index_df, {})
        assert len(result) == 0
        assert set(result.columns) == {
            "ARO_Accession", "Sequence", "Resistance_Mechanism",
            "Drug_Class", "AMR_Gene_Family", "Protein_Accession",
        }

    def test_empty_aro_index_returns_empty_df(self, sample_fasta_sequences):
        """ARO index vazio deve retornar DataFrame vazio."""
        empty_aro = pd.DataFrame(columns=[
            "ARO_Accession", "Model_Name", "ARO_Name", "Protein_Accession",
            "AMR_Gene_Family", "Drug_Class", "Resistance_Mechanism", "CARD_Short_Name",
        ])
        result = integrate_dataset(empty_aro, sample_fasta_sequences)
        assert len(result) == 0


# ──────────────────────────────────────────────────────────────────────────────
# Testes de validate_dataset
# ──────────────────────────────────────────────────────────────────────────────

class TestValidateDataset:
    """Testes para validate_dataset com violações propositais."""

    def _make_valid_df(self) -> pd.DataFrame:
        """Cria um DataFrame válido para testes."""
        return pd.DataFrame({
            "ARO_Accession": ["ARO:0000001", "ARO:0000002"],
            "Sequence": ["MKAYFIAILTLFT", "ACDEFGHIKLMNP"],
            "Resistance_Mechanism": ["antibiotic inactivation", "antibiotic efflux"],
            "Drug_Class": ["penicillin antibiotic", "tetracycline antibiotic"],
            "AMR_Gene_Family": ["FamilyX", "FamilyY"],
            "Protein_Accession": ["AAA00001.1", "AAA00002.1"],
        })

    def test_valid_dataset_passes(self):
        """Dataset válido não deve levantar exceção."""
        df = self._make_valid_df()
        validate_dataset(df)  # não deve lançar

    def test_duplicate_aro_accession_raises(self):
        """ARO_Accession duplicado deve levantar ValueError."""
        df = self._make_valid_df()
        df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # duplicata
        with pytest.raises(ValueError, match="ARO_Accession duplicados"):
            validate_dataset(df)

    def test_empty_sequence_raises(self):
        """Sequence vazia deve levantar ValueError."""
        df = self._make_valid_df()
        df.loc[0, "Sequence"] = ""
        with pytest.raises(ValueError, match="Sequence vazia"):
            validate_dataset(df)

    def test_nan_sequence_raises(self):
        """Sequence NaN deve levantar ValueError."""
        df = self._make_valid_df()
        df.loc[0, "Sequence"] = None
        with pytest.raises(ValueError, match="Sequence vazia"):
            validate_dataset(df)

    def test_missing_resistance_mechanism_raises(self):
        """Resistance_Mechanism ausente deve levantar ValueError."""
        df = self._make_valid_df()
        df.loc[1, "Resistance_Mechanism"] = None
        with pytest.raises(ValueError, match="Resistance_Mechanism ausente"):
            validate_dataset(df)

    def test_empty_resistance_mechanism_raises(self):
        """Resistance_Mechanism vazio deve levantar ValueError."""
        df = self._make_valid_df()
        df.loc[0, "Resistance_Mechanism"] = ""
        with pytest.raises(ValueError, match="Resistance_Mechanism ausente"):
            validate_dataset(df)

    def test_multiple_violations_in_one_message(self):
        """Múltiplas violações devem estar presentes na mensagem de erro."""
        df = self._make_valid_df()
        df.loc[0, "Sequence"] = ""
        df.loc[1, "Resistance_Mechanism"] = None
        with pytest.raises(ValueError) as exc_info:
            validate_dataset(df)
        msg = str(exc_info.value)
        assert "Sequence" in msg
        assert "Resistance_Mechanism" in msg
