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
# SEED FIXA — modelos, amostradores, permutações, frações da curva de aprendizado e
# bootstrap. A partição de teste e as repetições da validação cruzada têm sementes
# próprias, escolhidas em src/experiment_config.py.
RANDOM_STATE: int = 42
TEST_SIZE: float = 0.2          # fração-alvo de cada classe no conjunto de teste
N_CV_SPLITS: int = 5
ALPHA_STAT: float = 0.05

# ── Seleção de features ────────────────────────────────────────────────────────
# O experimento não usa seleção de atributos. Estes valores (SelectKBest k=100 após
# VarianceThreshold 0.0, a configuração da primeira versão do experimento) só são
# usados pela análise de sensibilidade à representação (src/sensitivity_representation.py).
K_FEATURES: int = 100
VARIANCE_THRESHOLD: float = 0.0

# ── Parâmetros de avaliação ────────────────────────────────────────────────────
# 30 permutações por atributo ou grupo (Hastie et al. 2009, seção 15.3.2).
N_PERMUTATION_REPEATS: int = 30
DPI_FIGURES: int = 300
# A dependência da anotação é medida por informação mútua e ablação (src/leakage_check.py).

# ── Alvo de classificação ──────────────────────────────────────────────────────
# Nome da coluna no aro_index.tsv original; depois da integração ela se chama
# "Resistance_Mechanism" (src.experiment_config.TARGET).
TARGET_COLUMN: str = "Resistance Mechanism"

# Classes do experimento.
#
# Por que estas 5: são os mecanismos de rótulo único do dataset integrado
# (CARD 4.0.1, protein homolog model, 6.052 genes) com genes suficientes para
# ter cerca de 5 por fold de validação. São filhos diretos de "mechanism of
# antibiotic resistance" na ARO; nenhum é subtipo de outro, então não se juntam.
#
# Na primeira versão do experimento, a lista tinha "reduced permeability" e
# "resistance by absence", mas não "target protection" (170 genes) nem
# "target replacement" (70 genes), que existem nos dados; por isso aquela versão
# usou 4 classes. A lista é conferida contra os dados em src.experiment_config.
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
