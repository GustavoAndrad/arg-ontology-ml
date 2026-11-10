"""
eda.py — Análise Exploratória dos Dados (EDA).

Responsabilidade: Gerar o relatório EDA completo, visualizações e registrar
decisões metodológicas antes de qualquer modelagem.

Dependências externas: pandas, matplotlib, seaborn, rdflib
Dependências internas: config
"""

import logging
from collections import deque
from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # backend não-interativo para ambientes sem display
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Paleta acessível (ColorBrewer Set2 — 8 cores, segura para daltonismo) ─────
# Referência: https://colorbrewer2.org/#type=qualitative&scheme=Set2
ACCESSIBLE_PALETTE: list[str] = [
    "#66c2a5",
    "#fc8d62",
    "#8da0cb",
    "#e78ac3",
    "#a6d854",
    "#ffd92f",
    "#e5c494",
    "#b3b3b3",
]


# ──────────────────────────────────────────────────────────────────────────────
# 5.1 — Distribuição de classes
# ──────────────────────────────────────────────────────────────────────────────

def compute_class_distribution(df: pd.DataFrame, target_col: str) -> pd.DataFrame:
    """
    Calcula contagem e proporção de cada classe da variável-alvo.

    Parâmetros:
        df: DataFrame com coluna de rótulos.
        target_col: Nome da coluna-alvo (ex.: 'Resistance_Mechanism').

    Retorno:
        DataFrame com colunas: 'classe', 'contagem', 'proporcao_pct'.
        Ordenado de forma decrescente por contagem.

    Levanta:
        KeyError: Se target_col não existir no DataFrame.
        ValueError: Se o DataFrame estiver vazio.
    """
    if target_col not in df.columns:
        raise KeyError(
            f"Coluna '{target_col}' não encontrada no DataFrame. "
            f"Colunas disponíveis: {list(df.columns)}"
        )
    if df.empty:
        raise ValueError("DataFrame vazio: impossível calcular distribuição de classes.")

    counts = df[target_col].value_counts(dropna=True)
    total = counts.sum()

    distribution = pd.DataFrame({
        "classe": counts.index,
        "contagem": counts.values,
        "proporcao_pct": (counts.values / total * 100).round(2),
    }).reset_index(drop=True)

    logger.info(
        "Distribuição de '%s': %d classes, %d instâncias totais.",
        target_col,
        len(distribution),
        total,
    )
    return distribution


# ──────────────────────────────────────────────────────────────────────────────
# 5.2 — Razão de desbalanceamento
# ──────────────────────────────────────────────────────────────────────────────

def compute_imbalance_ratio(distribution: pd.DataFrame) -> float:
    """
    Calcula a razão entre a classe majoritária e a minoritária.

    Parâmetros:
        distribution: DataFrame retornado por compute_class_distribution(),
                      com coluna 'contagem'.

    Retorno:
        float: Razão de desbalanceamento (majoritária / minoritária).
               Retorna 1.0 se houver apenas uma classe.

    Levanta:
        ValueError: Se 'contagem' não estiver presente ou se a menor contagem
                    for zero (evita divisão por zero).
    """
    if "contagem" not in distribution.columns:
        raise ValueError(
            "Coluna 'contagem' ausente no DataFrame de distribuição. "
            "Use compute_class_distribution() para gerar a entrada correta."
        )
    if distribution.empty:
        raise ValueError("DataFrame de distribuição vazio.")

    max_count = int(distribution["contagem"].max())
    min_count = int(distribution["contagem"].min())

    if min_count == 0:
        raise ValueError(
            "A menor classe tem contagem 0. Verifique os dados antes de calcular a razão."
        )

    ratio = max_count / min_count
    logger.info(
        "Razão de desbalanceamento: %.2f (majoritária=%d, minoritária=%d).",
        ratio,
        max_count,
        min_count,
    )
    return ratio


# ──────────────────────────────────────────────────────────────────────────────
# 5.3 — Identificação de classes pequenas
# ──────────────────────────────────────────────────────────────────────────────

def flag_small_classes(
    distribution: pd.DataFrame, threshold: int = 20
) -> list[str]:
    """
    Identifica classes com menos de `threshold` instâncias.

    Parâmetros:
        distribution: DataFrame retornado por compute_class_distribution(),
                      com colunas 'classe' e 'contagem'.
        threshold: Número mínimo de instâncias para uma classe não ser
                   sinalizada. Padrão: 20.

    Retorno:
        Lista de nomes de classes candidatas a agrupamento ou exclusão.
        Lista vazia se nenhuma classe for pequena.

    Levanta:
        ValueError: Se 'classe' ou 'contagem' não estiverem presentes.
    """
    required = {"classe", "contagem"}
    missing = required - set(distribution.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no DataFrame de distribuição: {missing}."
        )

    small = distribution[distribution["contagem"] < threshold]["classe"].tolist()

    if small:
        logger.warning(
            "%d classe(s) com menos de %d instâncias (candidatas a agrupamento/exclusão): %s",
            len(small),
            threshold,
            small,
        )
    else:
        logger.info(
            "Nenhuma classe abaixo do limiar de %d instâncias.", threshold
        )

    return small


# ──────────────────────────────────────────────────────────────────────────────
# 5.4 — Valores ausentes
# ──────────────────────────────────────────────────────────────────────────────

def check_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Verifica e reporta a presença de valores ausentes em todas as colunas.

    Parâmetros:
        df: DataFrame a ser inspecionado.

    Retorno:
        DataFrame com colunas: 'coluna', 'n_ausentes', 'pct_ausentes'.
        Inclui apenas colunas com pelo menos um valor ausente, ordenadas
        de forma decrescente por n_ausentes.
        Retorna DataFrame vazio (mas com as colunas corretas) se não houver ausentes.
    """
    if df.empty:
        logger.info("DataFrame vazio: nenhum valor ausente a verificar.")
        return pd.DataFrame(columns=["coluna", "n_ausentes", "pct_ausentes"])

    n_total = len(df)
    missing_counts = df.isnull().sum()
    missing_counts = missing_counts[missing_counts > 0].sort_values(ascending=False)

    if missing_counts.empty:
        logger.info("Nenhum valor ausente encontrado em %d colunas.", len(df.columns))
        return pd.DataFrame(columns=["coluna", "n_ausentes", "pct_ausentes"])

    result = pd.DataFrame({
        "coluna": missing_counts.index,
        "n_ausentes": missing_counts.values,
        "pct_ausentes": (missing_counts.values / n_total * 100).round(2),
    }).reset_index(drop=True)

    logger.warning(
        "%d coluna(s) com valores ausentes:\n%s",
        len(result),
        result.to_string(index=False),
    )
    return result


# ──────────────────────────────────────────────────────────────────────────────
# 5.5 — Análise da estrutura da ontologia ARO
# ──────────────────────────────────────────────────────────────────────────────

def analyze_ontology_structure(ontology_path: Path) -> dict:
    """
    Analisa a estrutura hierárquica da ontologia ARO (OWL/RDF) e reporta
    estatísticas topológicas.

    Utiliza rdflib para carregar a ontologia sem ferramentas externas de
    raciocínio ontológico.

    Parâmetros:
        ontology_path: Caminho para o arquivo aro.owl.

    Retorno:
        dict com campos:
            - n_classes (int): Número total de classes OWL na ontologia.
            - n_is_a_relations (int): Número de relações rdfs:subClassOf.
            - max_depth (int): Profundidade máxima da hierarquia is-a (raiz = 0).
            - resistance_mechanisms_found (list[str]): Mecanismos de resistência
              identificados via labels das classes.

    Levanta:
        FileNotFoundError: Se o arquivo OWL não existir.
        RuntimeError: Se a ontologia não puder ser carregada.
    """
    import rdflib
    from rdflib.namespace import OWL, RDFS, RDF

    ontology_path = Path(ontology_path)
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontologia não encontrada: {ontology_path}")

    logger.info("Carregando ontologia de: %s", ontology_path)
    graph = rdflib.Graph()
    try:
        graph.parse(str(ontology_path))
    except Exception as exc:
        raise RuntimeError(f"Falha ao carregar a ontologia: {exc}") from exc

    # ── Contar classes OWL ────────────────────────────────────────────────────
    owl_classes = set(graph.subjects(RDF.type, OWL.Class))
    n_classes = len(owl_classes)
    logger.info("Número de classes OWL: %d", n_classes)

    # ── Contar relações is-a (rdfs:subClassOf) ────────────────────────────────
    is_a_triples = list(graph.triples((None, RDFS.subClassOf, None)))
    n_is_a = len(is_a_triples)
    logger.info("Número de relações rdfs:subClassOf: %d", n_is_a)

    # ── Construir grafo is-a para cálculo de profundidade ────────────────────
    # filho -> lista de pais diretos
    children_to_parents: dict[str, list[str]] = {}
    for subj, _, obj in is_a_triples:
        subj_str = str(subj)
        obj_str = str(obj)
        if subj_str not in children_to_parents:
            children_to_parents[subj_str] = []
        children_to_parents[subj_str].append(obj_str)

    # ── Calcular profundidade máxima via BFS a partir de cada raiz ────────────
    # Nós raiz: classes sem superclasse (não aparecem como filho)
    all_nodes = set(children_to_parents.keys())
    all_parents = {p for parents in children_to_parents.values() for p in parents}
    roots = all_parents - all_nodes

    max_depth = 0
    if roots:
        # BFS do conjunto de raízes (pais -> filhos): inverter o grafo
        parents_to_children: dict[str, list[str]] = {}
        for child, parents in children_to_parents.items():
            for parent in parents:
                if parent not in parents_to_children:
                    parents_to_children[parent] = []
                parents_to_children[parent].append(child)

        # BFS para calcular profundidade máxima
        queue: deque[tuple[str, int]] = deque()
        for root in roots:
            queue.append((root, 0))

        visited_depth: dict[str, int] = {}
        while queue:
            node, depth = queue.popleft()
            if node in visited_depth and visited_depth[node] >= depth:
                continue
            visited_depth[node] = depth
            max_depth = max(max_depth, depth)
            for child in parents_to_children.get(node, []):
                queue.append((child, depth + 1))

    logger.info("Profundidade máxima da hierarquia is-a: %d", max_depth)

    # ── Identificar mecanismos de resistência encontrados ────────────────────
    # Buscamos labels que contenham termos de mecanismo de resistência
    _MECHANISM_KEYWORDS = [
        "antibiotic inactivation",
        "antibiotic efflux",
        "antibiotic target alteration",
        "reduced permeability",
        "resistance by absence",
        "antibiotic target protection",
        "antibiotic resistance",
    ]

    resistance_mechanisms_found: list[str] = []
    for _, _, label_obj in graph.triples((None, RDFS.label, None)):
        label_str = str(label_obj).lower().strip()
        for keyword in _MECHANISM_KEYWORDS:
            if keyword in label_str and label_str not in resistance_mechanisms_found:
                resistance_mechanisms_found.append(label_str)
                break

    logger.info(
        "Mecanismos de resistência encontrados na ontologia: %d",
        len(resistance_mechanisms_found),
    )

    return {
        "n_classes": n_classes,
        "n_is_a_relations": n_is_a,
        "max_depth": max_depth,
        "resistance_mechanisms_found": sorted(resistance_mechanisms_found),
    }


# ──────────────────────────────────────────────────────────────────────────────
# 5.6 — Visualização da distribuição de classes
# ──────────────────────────────────────────────────────────────────────────────

def plot_class_distribution(
    distribution: pd.DataFrame,
    output_path: Path,
    dpi: int = 300,
) -> None:
    """
    Gera gráfico de barras horizontal da distribuição de classes da variável-alvo.

    Salva o gráfico em formato PNG com resolução `dpi` em `output_path`.
    Usa paleta acessível ColorBrewer Set2.

    Parâmetros:
        distribution: DataFrame retornado por compute_class_distribution(),
                      com colunas 'classe', 'contagem' e 'proporcao_pct'.
        output_path: Caminho completo do arquivo PNG de saída
                     (ex.: plots/eda/class_distribution.png).
        dpi: Resolução da figura em pontos por polegada. Padrão: 300.

    Levanta:
        ValueError: Se as colunas necessárias não estiverem presentes.
    """
    required = {"classe", "contagem", "proporcao_pct"}
    missing = required - set(distribution.columns)
    if missing:
        raise ValueError(
            f"Colunas ausentes no DataFrame de distribuição: {missing}."
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    n_classes = len(distribution)
    palette = (ACCESSIBLE_PALETTE * ((n_classes // len(ACCESSIBLE_PALETTE)) + 1))[:n_classes]

    fig, ax = plt.subplots(figsize=(10, max(4, n_classes * 0.9)))

    bars = ax.barh(
        y=distribution["classe"],
        width=distribution["contagem"],
        color=palette,
        edgecolor="white",
        linewidth=0.8,
    )

    # Adicionar rótulos com contagem e proporção
    for bar, (_, row) in zip(bars, distribution.iterrows()):
        width = bar.get_width()
        ax.text(
            width + max(distribution["contagem"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['contagem'])} ({row['proporcao_pct']:.1f}%)",
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlabel("Número de Instâncias", fontsize=11)
    ax.set_ylabel("Mecanismo de Resistência", fontsize=11)
    ax.set_title("Distribuição de Classes — Mecanismo de Resistência", fontsize=13, pad=12)
    ax.invert_yaxis()  # classe com mais instâncias no topo
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Margem à direita para os rótulos de texto
    ax.set_xlim(0, distribution["contagem"].max() * 1.25)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info("Gráfico de distribuição de classes salvo em: %s (dpi=%d)", output_path, dpi)


# ──────────────────────────────────────────────────────────────────────────────
# 5.7 — Orquestrador do relatório EDA
# ──────────────────────────────────────────────────────────────────────────────

def generate_eda_report(df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Orquestra todas as análises EDA e salva visualizações em `output_dir/eda/`.

    Executa em sequência:
    1. Distribuição de classes do Alvo_Classificacao.
    2. Razão de desbalanceamento.
    3. Identificação de classes pequenas (< 20 instâncias).
    4. Verificação de valores ausentes.
    5. Análise da estrutura da ontologia ARO (se ONTOLOGY_PATH existir).
    6. Geração do gráfico de distribuição de classes.

    Parâmetros:
        df: DataFrame integrado retornado por data_loader.integrate_dataset().
        output_dir: Diretório raiz de plots (ex.: Path('plots')).
                    As figuras EDA serão salvas em output_dir/eda/.

    Retorno:
        dict com campos:
            - class_distribution (pd.DataFrame)
            - imbalance_ratio (float)
            - small_classes (list[str])
            - missing_values (pd.DataFrame)
            - ontology_stats (dict | None)  — None se a ontologia não existir
            - n_total_instances (int)

    Levanta:
        ValueError: Se o DataFrame estiver vazio ou TARGET_COLUMN ausente.
    """
    try:
        from src.config import TARGET_COLUMN, ONTOLOGY_PATH
    except ImportError:
        from config import TARGET_COLUMN, ONTOLOGY_PATH  # type: ignore[no-redef]

    output_dir = Path(output_dir)
    eda_dir = output_dir / "eda"
    eda_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=== Iniciando EDA ===")
    logger.info("Total de instâncias no dataset: %d", len(df))

    # 1. Distribuição de classes
    target_col = TARGET_COLUMN
    # Aceitar também 'Resistance_Mechanism' como nome normalizado
    if target_col not in df.columns and "Resistance_Mechanism" in df.columns:
        target_col = "Resistance_Mechanism"

    class_distribution = compute_class_distribution(df, target_col)

    # 2. Razão de desbalanceamento
    imbalance_ratio = compute_imbalance_ratio(class_distribution)
    logger.info("Razão de desbalanceamento: %.2f", imbalance_ratio)

    # 3. Classes pequenas
    small_classes = flag_small_classes(class_distribution, threshold=20)
    if small_classes:
        logger.warning(
            "Classes com < 20 instâncias (candidatas a agrupamento/exclusão): %s",
            small_classes,
        )

    # 4. Valores ausentes
    missing_values = check_missing_values(df)

    # 5. Estrutura da ontologia
    ontology_stats: dict | None = None
    if Path(ONTOLOGY_PATH).exists():
        try:
            ontology_stats = analyze_ontology_structure(Path(ONTOLOGY_PATH))
        except Exception as exc:
            logger.warning("Falha ao analisar ontologia: %s", exc)
    else:
        logger.warning(
            "Arquivo de ontologia não encontrado em %s. "
            "Análise ontológica pulada.",
            ONTOLOGY_PATH,
        )

    # 6. Gráfico de distribuição de classes
    plot_path = eda_dir / "class_distribution.png"
    plot_class_distribution(class_distribution, plot_path, dpi=300)

    report = {
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "small_classes": small_classes,
        "missing_values": missing_values,
        "ontology_stats": ontology_stats,
        "n_total_instances": len(df),
    }

    logger.info("=== EDA concluída. Relatório gerado. ===")
    return report


# ──────────────────────────────────────────────────────────────────────────────
# Bloco principal — execução direta com dados reais do CARD
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    try:
        from config import PROCESSED_DIR, PLOTS_DIR
    except ImportError:
        from src.config import PROCESSED_DIR, PLOTS_DIR  # type: ignore[no-redef]

    dataset_path = PROCESSED_DIR / "dataset_integrado.csv"
    if not dataset_path.exists():
        logger.error(
            "Dataset integrado não encontrado em %s. "
            "Execute src/data_loader.py primeiro.",
            dataset_path,
        )
        sys.exit(1)

    df_loaded = pd.read_csv(dataset_path)
    logger.info("Dataset carregado: %d instâncias.", len(df_loaded))

    report = generate_eda_report(df_loaded, PLOTS_DIR)

    logger.info("Distribuição de classes:")
    logger.info("\n%s", report["class_distribution"].to_string(index=False))
    logger.info("Razão de desbalanceamento: %.2f", report["imbalance_ratio"])
    if report["small_classes"]:
        logger.warning("Classes pequenas: %s", report["small_classes"])
    if not report["missing_values"].empty:
        logger.warning("Valores ausentes:\n%s", report["missing_values"].to_string(index=False))
    if report["ontology_stats"]:
        logger.info("Estatísticas da ontologia: %s", report["ontology_stats"])
