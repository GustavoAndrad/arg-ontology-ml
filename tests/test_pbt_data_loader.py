"""
test_pbt_data_loader.py — Teste de Propriedade P9: Integridade do Dataset Final.

Valida: Requirements 2.3, 2.6

Para qualquer par (aro_index, fasta_sequences):
  df = integrate_dataset(aro_index, fasta_sequences)
  assert df['ARO_Accession'].notna().all()
  assert df['Sequence'].str.len().gt(0).all()
  assert df['Resistance_Mechanism'].notna().all()
  assert df['Resistance_Mechanism'].str.strip().str.len().gt(0).all()

Usa hypothesis com @settings(max_examples=100, deadline=None).
"""

import string

import pandas as pd
import pytest
from hypothesis import given, settings, assume
from hypothesis import strategies as st

from src.data_loader import integrate_dataset

# ──────────────────────────────────────────────────────────────────────────────
# Estratégias de geração
# ──────────────────────────────────────────────────────────────────────────────

# Mecanismos de resistência válidos (não vazios, sem espaços só)
_VALID_MECHANISMS = [
    "antibiotic inactivation",
    "antibiotic efflux",
    "antibiotic target alteration",
    "reduced permeability to antibiotic",
    "resistance by absence",
]

# Aminoácidos canônicos
_AMINO_ACIDS = "ACDEFGHIKLMNPQRSTVWY"


def aro_accession_strategy():
    """Gera ARO Accessions válidos no formato 'ARO:NNNNNNN'."""
    return st.integers(min_value=1, max_value=9999999).map(
        lambda n: f"ARO:{n:07d}"
    )


def protein_sequence_strategy():
    """Gera sequências proteicas não-vazias com aminoácidos canônicos."""
    return st.text(
        alphabet=_AMINO_ACIDS,
        min_size=1,
        max_size=200,
    )


def resistance_mechanism_strategy():
    """Gera mecanismos de resistência válidos (não nulos, não vazios)."""
    return st.sampled_from(_VALID_MECHANISMS)


def drug_class_strategy():
    """Gera Drug Class como string simples."""
    return st.text(
        alphabet=string.ascii_letters + string.digits + " ",
        min_size=0,
        max_size=50,
    )


def aro_index_strategy(max_rows: int = 10):
    """
    Gera DataFrames para aro_index com:
    - Mistura de entradas com Resistance_Mechanism válido e inválido (vazio/None)
    - ARO Accessions únicos
    """
    @st.composite
    def _build(draw):
        n = draw(st.integers(min_value=1, max_value=max_rows))
        # ARO accessions únicos
        accessions = draw(
            st.lists(
                aro_accession_strategy(),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
        mechanisms = draw(
            st.lists(
                st.one_of(
                    resistance_mechanism_strategy(),
                    st.just(""),      # vazio — deve ser filtrado
                    st.none(),        # None → NaN — deve ser filtrado
                ),
                min_size=n,
                max_size=n,
            )
        )
        drug_classes = draw(
            st.lists(drug_class_strategy(), min_size=n, max_size=n)
        )
        amr_families = draw(
            st.lists(
                st.text(min_size=0, max_size=30),
                min_size=n,
                max_size=n,
            )
        )
        protein_accs = draw(
            st.lists(
                st.text(
                    alphabet=string.ascii_letters + string.digits + ".",
                    min_size=1,
                    max_size=15,
                ),
                min_size=n,
                max_size=n,
            )
        )
        short_names = draw(
            st.lists(
                st.text(min_size=0, max_size=20),
                min_size=n,
                max_size=n,
            )
        )
        model_names = draw(
            st.lists(
                st.text(min_size=0, max_size=30),
                min_size=n,
                max_size=n,
            )
        )
        aro_names = draw(
            st.lists(
                st.text(min_size=0, max_size=50),
                min_size=n,
                max_size=n,
            )
        )

        return pd.DataFrame({
            "ARO_Accession": accessions,
            "Model_Name": model_names,
            "ARO_Name": aro_names,
            "Protein_Accession": protein_accs,
            "AMR_Gene_Family": amr_families,
            "Drug_Class": drug_classes,
            "Resistance_Mechanism": mechanisms,
            "CARD_Short_Name": short_names,
        })

    return _build()


def fasta_sequences_strategy(aro_accessions_from_df=None, max_extra: int = 5):
    """
    Gera dicionários ARO_Accession -> sequência.
    Pode incluir accessions extras não presentes em aro_index (para testar inner join).
    """
    @st.composite
    def _build(draw):
        # Tamanho do dict independente (pode ter sobreposição parcial com aro_index)
        n = draw(st.integers(min_value=0, max_value=15))
        accessions = draw(
            st.lists(
                aro_accession_strategy(),
                min_size=n,
                max_size=n,
                unique=True,
            )
        )
        sequences = draw(
            st.lists(
                protein_sequence_strategy(),
                min_size=n,
                max_size=n,
            )
        )
        return dict(zip(accessions, sequences))

    return _build()


# ──────────────────────────────────────────────────────────────────────────────
# P9 — Integridade do Dataset Final
#
# Validates: Requirements 2.3, 2.6
# ──────────────────────────────────────────────────────────────────────────────

@given(
    aro_index=aro_index_strategy(max_rows=10),
    fasta_sequences=fasta_sequences_strategy(),
)
@settings(max_examples=100, deadline=None)
def test_p9_dataset_integrity(
    aro_index: pd.DataFrame,
    fasta_sequences: dict[str, str],
):
    """
    P9 — Integridade do Dataset Final.

    **Validates: Requirements 2.3, 2.6**

    Para qualquer par (aro_index, fasta_sequences):
    - df['ARO_Accession'] nunca contém valores nulos
    - df['Sequence'] nunca contém strings de comprimento zero
    - df['Resistance_Mechanism'] nunca contém valores nulos
    - df['Resistance_Mechanism'] após strip nunca contém strings vazias

    Estas propriedades devem valer independentemente do conteúdo das entradas,
    pois integrate_dataset filtra instâncias inválidas antes de retornar.
    """
    df = integrate_dataset(aro_index, fasta_sequences)

    # O dataset pode estar vazio (sem matches no inner join, ou tudo filtrado)
    # Mas se não estiver vazio, as invariantes devem valer
    if df.empty:
        return  # propriedade vacuamente verdadeira

    # P9.1: ARO_Accession sem nulos
    assert df["ARO_Accession"].notna().all(), (
        f"ARO_Accession contém nulos.\n{df['ARO_Accession'].isna().sum()} nulos encontrados."
    )

    # P9.2: Sequence com comprimento > 0
    assert df["Sequence"].str.len().gt(0).all(), (
        f"Sequence contém strings de comprimento zero ou nulas.\n"
        f"Mínimo: {df['Sequence'].str.len().min()}"
    )

    # P9.3: Resistance_Mechanism sem nulos
    assert df["Resistance_Mechanism"].notna().all(), (
        f"Resistance_Mechanism contém nulos.\n"
        f"{df['Resistance_Mechanism'].isna().sum()} nulos encontrados."
    )

    # P9.4: Resistance_Mechanism sem strings vazias após strip
    assert df["Resistance_Mechanism"].str.strip().str.len().gt(0).all(), (
        f"Resistance_Mechanism contém strings vazias após strip.\n"
        f"Valores problemáticos: {df.loc[df['Resistance_Mechanism'].str.strip().str.len() == 0, 'Resistance_Mechanism'].tolist()}"
    )


@given(
    aro_index=aro_index_strategy(max_rows=15),
    fasta_sequences=fasta_sequences_strategy(),
)
@settings(max_examples=50, deadline=None)
def test_p9_output_columns_always_present(
    aro_index: pd.DataFrame,
    fasta_sequences: dict[str, str],
):
    """
    P9 (complementar) — As colunas de saída obrigatórias devem estar sempre presentes.

    **Validates: Requirements 2.6**
    """
    df = integrate_dataset(aro_index, fasta_sequences)

    expected_cols = {
        "ARO_Accession",
        "Sequence",
        "Resistance_Mechanism",
        "Drug_Class",
        "AMR_Gene_Family",
        "Protein_Accession",
    }
    assert expected_cols.issubset(set(df.columns)), (
        f"Colunas obrigatórias ausentes: {expected_cols - set(df.columns)}"
    )


@given(
    aro_index=aro_index_strategy(max_rows=10),
    fasta_sequences=fasta_sequences_strategy(),
)
@settings(max_examples=50, deadline=None)
def test_p9_result_is_subset_of_inputs(
    aro_index: pd.DataFrame,
    fasta_sequences: dict[str, str],
):
    """
    P9 (complementar) — O conjunto de ARO_Accession no resultado é subconjunto
    da interseção entre aro_index e fasta_sequences (inner join).

    **Validates: Requirements 2.3, 2.6**
    """
    df = integrate_dataset(aro_index, fasta_sequences)

    if df.empty:
        return

    result_accessions = set(df["ARO_Accession"])
    index_accessions = set(aro_index["ARO_Accession"])
    fasta_accessions = set(fasta_sequences.keys())

    # Resultado deve ser subconjunto da interseção
    assert result_accessions.issubset(index_accessions & fasta_accessions), (
        f"Resultado contém ARO Accessions não presentes em ambas as fontes."
    )
