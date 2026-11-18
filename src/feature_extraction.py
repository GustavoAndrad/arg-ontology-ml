"""
feature_extraction.py — Atributos de sequência proteica (Pipeline A, Seção 4.4).

Representação do experimento: **73 atributos por sequência**
    * 4 propriedades físico-químicas (Biopython ProtParam): comprimento, ponto
      isoelétrico, índice de instabilidade e carga em pH 7;
    * 20 frações de aminoácidos canônicos;
    * 49 frequências de pares de grupos de aminoácidos vizinhos (alfabeto
      reduzido de 7 grupos, ``REDUCED_ALPHABET``).

Por que esta representação:
    * Na primeira versão do experimento, os 400 dipeptídeos eram 94% dos atributos,
      acrescentavam pouco ao desempenho e 19 deles valiam zero em mais de 95% dos
      genes. Agrupando os 20 aminoácidos em 7 grupos de propriedades parecidas, a
      informação "que tipo de resíduo vem ao lado de qual" é mantida com 49 colunas;
      nenhuma passa de 95% de zeros (a mais rara, (C, C), é zero em 93,6% dos genes,
      por causa da baixa frequência da cisteína).
      Agrupamento: Shen J. et al. (2007), "Predicting protein–protein interactions
      based only on sequences information", PNAS 104(11):4337–4341,
      doi:10.1073/pnas.0607879104. Os autores agrupam os 20 aminoácidos em 7
      classes pelo dipolo e pelo volume da cadeia lateral, para reduzir a dimensão
      e tratar como equivalentes substituições entre resíduos parecidos, e contam
      trios de resíduos vizinhos ("conjoint triad", 7×7×7 = 343 atributos). Aqui o
      mesmo agrupamento é aplicado a **pares** (7×7 = 49), e a frequência é
      normalizada pelo total de pares (não pela regra min/max do artigo). A lista
      das 7 classes foi conferida na SI Fig. 3 do material suplementar do artigo.
    * Sai o **peso molecular**: duplica o comprimento (correlação 0,997).
    * Sai a **aromaticidade**: no Biopython ela é exatamente a soma das frações de
      F, W e Y (diferença máxima 2e-16), ou seja, um cálculo direto de outros três
      atributos. Mesmo critério do peso molecular.

Resíduos ambíguos (X, B, Z, U, ...) são excluídos do denominador das composições
e do cálculo físico-químico; o comprimento usa a sequência original. Nos pares
(de aminoácidos ou de grupos), pares que contêm resíduo ambíguo são descartados,
sem juntar os vizinhos do resíduo removido, e as frequências são divididas pelo
total de pares válidos (somam 1). Exceção: o índice de instabilidade do ProtParam
é calculado sobre a sequência já sem os ambíguos, então ali os vizinhos do resíduo
removido ficam juntos. No CARD 4.0.1 isso atinge 5 sequências (14 resíduos X).

A representação da primeira versão do experimento (426 atributos: 6
físico-químicas + 20 + 400 dipeptídeos) continua disponível
(``representation="completa_426"``) só para a análise de sensibilidade à
representação (``src.sensitivity_representation``).

Saídas:
    data/processed/features_sequencia.csv                — 73 atributos, genes do experimento
    data/processed/features_sequencia_estudo_caso.csv    — 73 atributos, genes do estudo de caso
    data/processed/features_sequencia_completa_426.csv   — 426 atributos, para a sensibilidade

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

# Alfabeto reduzido de 7 grupos (Shen et al. 2007). Cada aminoácido canônico está em
# exatamente um grupo (conferido em teste).
REDUCED_ALPHABET: list[str] = ["AGV", "ILFP", "YMTS", "HNQW", "RK", "DE", "C"]
GROUP_OF: dict[str, str] = {aa: g for g in REDUCED_ALPHABET for aa in g}
GROUP_PAIRS: list[tuple[str, str]] = list(itertools.product(REDUCED_ALPHABET, repeat=2))

ALL_PHYSICOCHEMICAL_COLUMNS: list[str] = [
    "length",
    "molecular_weight",
    "isoelectric_point",
    "aromaticity",
    "instability_index",
    "charge_at_pH7",
]
# Removidas na representação do experimento, com o motivo (ver docstring do módulo).
REMOVED_PHYSICOCHEMICAL: dict[str, str] = {
    "molecular_weight": "duplica o comprimento (correlação 0,997)",
    "aromaticity": "é exatamente aa_comp_F + aa_comp_W + aa_comp_Y",
}
PHYSICOCHEMICAL_COLUMNS: list[str] = [c for c in ALL_PHYSICOCHEMICAL_COLUMNS if c not in REMOVED_PHYSICOCHEMICAL]
AA_COMPOSITION_COLUMNS: list[str] = [f"aa_comp_{a}" for a in AMINO_ACIDS]
GROUP_PAIR_COLUMNS: list[str] = [f"pair_{g1}_{g2}" for g1, g2 in GROUP_PAIRS]
DIPEPTIDE_COLUMNS: list[str] = [f"dipep_{d}" for d in DIPEPTIDES]

REPRESENTATIONS: dict[str, list[str]] = {
    # Experimento: 4 + 20 + 49 = 73.
    "reduzida_73": PHYSICOCHEMICAL_COLUMNS + AA_COMPOSITION_COLUMNS + GROUP_PAIR_COLUMNS,
    # Primeira versão do experimento: 6 + 20 + 400 = 426 (só para a análise de sensibilidade).
    "completa_426": ALL_PHYSICOCHEMICAL_COLUMNS + AA_COMPOSITION_COLUMNS + DIPEPTIDE_COLUMNS,
}
SEQUENCE_FEATURE_COLUMNS: list[str] = REPRESENTATIONS["reduzida_73"]


def canonical_residues(sequence: str) -> str:
    """Retorna a sequência apenas com os 20 aminoácidos canônicos (maiúsculos)."""
    return "".join(ch for ch in sequence.upper() if ch in AMINO_ACIDS)


def count_ambiguous_residues(sequence: str) -> int:
    """Conta resíduos que não pertencem aos 20 aminoácidos canônicos."""
    return len(sequence) - len(canonical_residues(sequence))


def compute_physicochemical_properties(sequence: str) -> dict[str, float]:
    """Calcula as 6 propriedades físico-químicas via ``ProteinAnalysis`` (a seleção vem depois)."""
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


def _valid_pairs(sequence: str) -> list[str]:
    """Pares de resíduos vizinhos na sequência original, sem os que têm resíduo ambíguo."""
    seq = sequence.upper()
    pairs = (seq[i : i + 2] for i in range(len(seq) - 1))
    return [p for p in pairs if p[0] in AMINO_ACIDS and p[1] in AMINO_ACIDS]


def compute_dipeptide_composition(sequence: str) -> dict[str, float]:
    """
    Frequência fracional dos 400 dipeptídeos consecutivos (representação da primeira versão do experimento).

    Os pares são lidos na sequência original; pares que contêm resíduo ambíguo
    (p.ex. ``AX``, ``XG``) não são contados e o denominador é o total de pares
    válidos, de modo que as 400 frequências somam 1. Para sequências sem
    resíduos ambíguos, equivale a contagem / (comprimento - 1).
    """
    counts = Counter(_valid_pairs(sequence))
    total = sum(counts.values()) or 1
    return {f"dipep_{d}": counts[d] / total for d in DIPEPTIDES}


def compute_group_pair_composition(sequence: str) -> dict[str, float]:
    """
    Frequência fracional dos 49 pares de grupos do alfabeto reduzido.

    Mesma regra dos dipeptídeos: pares lidos na sequência original, pares com
    resíduo ambíguo descartados, divisão pelo total de pares válidos (somam 1).
    Exemplo: ``MKTA`` → grupos ``YMTS RK YMTS AGV`` → pares (YMTS,RK), (RK,YMTS),
    (YMTS,AGV), cada um com frequência 1/3.
    """
    counts = Counter((GROUP_OF[p[0]], GROUP_OF[p[1]]) for p in _valid_pairs(sequence))
    total = sum(counts.values()) or 1
    return {f"pair_{g1}_{g2}": counts[(g1, g2)] / total for g1, g2 in GROUP_PAIRS}


def compute_sequence_features(sequence: str, representation: str = "reduzida_73") -> dict[str, float]:
    """Vetor de atributos de uma sequência, na ordem de ``REPRESENTATIONS[representation]``."""
    feats: dict[str, float] = {}
    feats.update(compute_physicochemical_properties(sequence))
    feats.update(compute_amino_acid_composition(sequence))
    if representation == "reduzida_73":
        feats.update(compute_group_pair_composition(sequence))
    elif representation == "completa_426":
        feats.update(compute_dipeptide_composition(sequence))
    else:
        raise ValueError(f"Representação desconhecida: {representation}")
    return {c: feats[c] for c in REPRESENTATIONS[representation]}


def extract_sequence_features(
    df: pd.DataFrame, sequence_col: str = "Sequence", representation: str = "reduzida_73"
) -> pd.DataFrame:
    """
    Aplica a extração a todo o DataFrame.

    Retorno: DataFrame com ``ARO_Accession`` + as colunas da representação, na mesma ordem de ``df``.
    """
    columns = REPRESENTATIONS[representation]
    rows = [compute_sequence_features(str(s), representation) for s in df[sequence_col]]
    out = pd.DataFrame(rows, columns=columns)
    n_amb = sum(count_ambiguous_residues(str(s)) > 0 for s in df[sequence_col])
    logger.info(
        "Features de sequência (%s): %d instâncias x %d features (%d sequências com resíduos ambíguos).",
        representation,
        len(out),
        out.shape[1],
        n_amb,
    )
    out.insert(0, "ARO_Accession", df["ARO_Accession"].values)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    data = load_experiment_dataset()
    outputs = [
        (data, "reduzida_73", "features_sequencia.csv"),
        (load_case_study_dataset(), "reduzida_73", "features_sequencia_estudo_caso.csv"),
        (data, "completa_426", "features_sequencia_completa_426.csv"),
    ]
    for frame, representation, filename in outputs:
        extract_sequence_features(frame, representation=representation).to_csv(PROCESSED_DIR / filename, index=False)
        logger.info("Salvo em %s", PROCESSED_DIR / filename)
