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
K_FEATURES: int = 100           # Determinado empiricamente; registrado em hyperparameters.json
VARIANCE_THRESHOLD: float = 0.0

# ── Parâmetros de avaliação ────────────────────────────────────────────────────
N_PERMUTATION_REPEATS: int = 10
DPI_FIGURES: int = 300
SPEARMAN_LEAKAGE_THRESHOLD: float = 0.95

# ── Alvo de classificação ──────────────────────────────────────────────────────
TARGET_COLUMN: str = "Resistance Mechanism"
TARGET_CLASSES: list[str] = [
    "antibiotic inactivation",
    "antibiotic efflux",
    "antibiotic target alteration",
    "reduced permeability to antibiotic",
    "resistance by absence",
]
