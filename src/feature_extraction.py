"""
feature_extraction.py — Features de sequência proteica (Pipeline A, Seção 4.4).

426 atributos por sequência:
    * 6 propriedades físico-químicas (BioPython ProtParam): comprimento, peso
      molecular, ponto isoelétrico, aromaticidade, índice de instabilidade e
      carga em pH 7;
    * 20 frações de aminoácidos canônicos;
    * 400 frequências de dipeptídeos consecutivos.

Resíduos ambíguos (X, B, Z, U, ...) são excluídos do denominador das composições
e do cálculo físico-químico; o comprimento usa a sequência original. Nos
dipeptídeos, pares que contêm resíduo ambíguo são descartados (sem juntar os
vizinhos do resíduo removido).

Saídas: ``data/processed/features_sequencia.csv`` (genes do experimento) e
``data/processed/features_sequencia_estudo_caso.csv`` (genes do estudo de caso,
TCC2 ponto 2). Os atributos dependem só da sequência de cada gene, então os
genes que já estavam no TCC1 têm exatamente os mesmos valores.

Uso:
    python -m src.feature_extraction
"""

from __future__ import annotations

import itertools
import logging
import sys
from collections import Counter

import pandas as pd
from Bio.SeqUtils.ProtParam import ProteinAnalysis

from src.experiment_config import PROCESSED_DIR, load_case_study_dataset, load_experiment_dataset

logger = logging.getLogger(__name__)

AMINO_ACIDS: str = "ACDEFGHIKLMNPQRSTVWY"
DIPEPTIDES: list[str] = ["".join(p) for p in itertools.product(AMINO_ACIDS, repeat=2)]
PHYSICOCHEMICAL_COLUMNS: list[str] = [
    "length",
    "molecular_weight",
    "isoelectric_point",
    "aromaticity",
    "instability_index",
    "charge_at_pH7",
]
SEQUENCE_FEATURE_COLUMNS: list[str] = (
    PHYSICOCHEMICAL_COLUMNS + [f"aa_comp_{a}" for a in AMINO_ACIDS] + [f"dipep_{d}" for d in DIPEPTIDES]
)


def canonical_residues(sequence: str) -> str:
    """Retorna a sequência apenas com os 20 aminoácidos canônicos (maiúsculos)."""
    return "".join(ch for ch in sequence.upper() if ch in AMINO_ACIDS)


def count_ambiguous_residues(sequence: str) -> int:
    """Conta resíduos que não pertencem aos 20 aminoácidos canônicos."""
    return len(sequence) - len(canonical_residues(sequence))


def compute_physicochemical_properties(sequence: str) -> dict[str, float]:
    """Calcula as 6 propriedades físico-químicas via ``ProteinAnalysis``."""
    clean = canonical_residues(sequence)
    if not clean:
        raise ValueError("Sequência sem aminoácidos canônicos.")
    pa = ProteinAnalysis(clean)
    return {
        "length": float(len(sequence)),
        "molecular_weight": pa.molecular_weight(),
        "isoelectric_point": pa.isoelectric_point(),
        "aromaticity": pa.aromaticity(),
        "instability_index": pa.instability_index(),
        "charge_at_pH7": pa.charge_at_pH(7.0),
    }


def compute_amino_acid_composition(sequence: str) -> dict[str, float]:
    """Fração de cada aminoácido canônico (denominador sem resíduos ambíguos)."""
    clean = canonical_residues(sequence)
    n = len(clean) or 1
    counts = Counter(clean)
    return {f"aa_comp_{a}": counts[a] / n for a in AMINO_ACIDS}


def compute_dipeptide_composition(sequence: str) -> dict[str, float]:
    """
    Frequência fracional dos 400 dipeptídeos consecutivos.

    Os pares são lidos na sequência original; pares que contêm resíduo ambíguo
    (p.ex. ``AX``, ``XG``) não são contados e o denominador é o total de pares
    válidos, de modo que as 400 frequências somam 1. Para sequências sem
    resíduos ambíguos, equivale a contagem / (comprimento - 1).
    """
    seq = sequence.upper()
    counts = Counter(seq[i : i + 2] for i in range(len(seq) - 1))
    valid = {d: counts[d] for d in DIPEPTIDES}
    total = sum(valid.values()) or 1
    return {f"dipep_{d}": valid[d] / total for d in DIPEPTIDES}


def compute_sequence_features(sequence: str) -> dict[str, float]:
    """Vetor completo de 426 features para uma sequência."""
    feats: dict[str, float] = {}
    feats.update(compute_physicochemical_properties(sequence))
    feats.update(compute_amino_acid_composition(sequence))
    feats.update(compute_dipeptide_composition(sequence))
    return feats


def extract_sequence_features(df: pd.DataFrame, sequence_col: str = "Sequence") -> pd.DataFrame:
    """
    Aplica a extração a todo o DataFrame.

    Retorno: DataFrame com ``ARO_Accession`` + 426 colunas, na mesma ordem de ``df``.
    """
    rows = [compute_sequence_features(str(s)) for s in df[sequence_col]]
    out = pd.DataFrame(rows, columns=SEQUENCE_FEATURE_COLUMNS)
    n_amb = sum(count_ambiguous_residues(str(s)) > 0 for s in df[sequence_col])
    logger.info(
        "Features de sequência: %d instâncias x %d features (%d sequências com resíduos ambíguos).",
        len(out),
        out.shape[1],
        n_amb,
    )
    assert out.shape[1] == 426, "Esperadas 426 features de sequência"
    out.insert(0, "ARO_Accession", df["ARO_Accession"].values)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    data = load_experiment_dataset()
    features = extract_sequence_features(data)
    output = PROCESSED_DIR / "features_sequencia.csv"
    features.to_csv(output, index=False)
    logger.info("Salvo em %s", output)

    case_output = PROCESSED_DIR / "features_sequencia_estudo_caso.csv"
    extract_sequence_features(load_case_study_dataset()).to_csv(case_output, index=False)
    logger.info("Estudo de caso salvo em %s", case_output)
