"""
config.py — Constantes globais do experimento arg-ml-semantic-classification.

Centraliza todos os parâmetros e caminhos para garantir reprodutibilidade
e facilitar alterações sem modificar múltiplos módulos.
"""

from pathlib import Path

# ── Raiz do projeto ────────────────────────────────────────────────────────────
_PROJECT_ROOT: Path = Path(__file__).parent.parent

# ── Caminhos de dados ──────────────────────────────────────────────────────────
DATA_DIR: Path = _PROJECT_ROOT / "data"
RAW_CARD_DIR: Path = _PROJECT_ROOT / "data" / "dataset" / "card-data"
ONTOLOGY_PATH: Path = _PROJECT_ROOT / "data" / "ontologia" / "aro.owl"
PROCESSED_DIR: Path = _PROJECT_ROOT / "data" / "processed"
MODELS_DIR: Path = _PROJECT_ROOT / "models"
RESULTS_DIR: Path = _PROJECT_ROOT / "results"
PLOTS_DIR: Path = _PROJECT_ROOT / "plots"
REPORTS_DIR: Path = _PROJECT_ROOT / "reports"

# ── Parâmetros do experimento ──────────────────────────────────────────────────
RANDOM_STATE: int = 42          # SEED FIXA — divisão treino/teste e modelos
TEST_SIZE: float = 0.2
N_CV_SPLITS: int = 5
ALPHA_STAT: float = 0.05

# ── Seleção de features ────────────────────────────────────────────────────────
# O experimento não usa seleção de atributos (TCC2, ponto 7). Estes valores são os
# do TCC1 (SelectKBest k=100 após VarianceThreshold 0.0) e só são usados pela
# análise de sensibilidade à representação (src/sensitivity_representation.py).
K_FEATURES: int = 100
VARIANCE_THRESHOLD: float = 0.0

# ── Parâmetros de avaliação ────────────────────────────────────────────────────
# Ponto 8, decisão 1: 30 permutações (Hastie et al. 2009, seção 15.3.2).
N_PERMUTATION_REPEATS: int = 30
DPI_FIGURES: int = 300
# O Spearman (|ρ| ≥ 0,95) e a regra de sanidade (F1 ≤ 0,85) saíram no TCC2 (ponto 4):
# a dependência da anotação é medida por informação mútua e ablação (src/leakage_check.py).

# ── Alvo de classificação ──────────────────────────────────────────────────────
TARGET_COLUMN: str = "Resistance Mechanism"

# Classes do experimento (TCC2, ponto 2 de docs/TCC2_decisoes_banca.md).
#
# Por que estas 5: são os mecanismos de rótulo único do dataset integrado
# (CARD 4.0.1, protein homolog model, 6.052 genes) com genes suficientes para
# ter cerca de 5 por fold de validação. São filhos diretos de "mechanism of
# antibiotic resistance" na ARO; nenhum é subtipo de outro, então não se juntam.
#
# Histórico do erro corrigido aqui: a lista anterior (copiada da especificação
# do planejamento, .kiro/specs/.../requirements.md) tinha "reduced permeability"
# e "resistance by absence", mas não "target protection" (170 genes) nem
# "target replacement" (70 genes), que existem nos dados. O TCC1 acabou usando
# 4 classes por causa disso, e não por escolha de método.
TARGET_CLASSES: list[str] = [
    "antibiotic inactivation",       # 5.239 genes
    "antibiotic target alteration",  # 285
    "antibiotic efflux",             # 269
    "antibiotic target protection",  # 170
    "antibiotic target replacement", # 70
]

# Mecanismos raros demais para a avaliação quantitativa (8 e 1 genes: 0 ou 1 por
# fold). Ficam fora do treino e viram um estudo de caso qualitativo: os modelos
# treinados com as 5 classes são aplicados a esses genes.
CASE_STUDY_CLASSES: list[str] = [
    "reduced permeability to antibiotic",                 # 8 genes
    "resistance by host-dependent nutrient acquisition",  # 1 gene
]

# Genes com mais de um mecanismo aparecem com os rótulos separados por ";"
# (10 genes). Ficam fora: a tarefa é multiclasse de rótulo único.
MULTI_LABEL_SEPARATOR: str = ";"

# "resistance by absence" (definida na ARO) não aparece no dataset: é perda de
# gene, sem proteína no FASTA do protein homolog model.
