"""
eda.py — Análise Exploratória dos Dados (EDA).

Responsabilidade: Gerar o relatório EDA completo, visualizações e registrar
decisões metodológicas antes de qualquer modelagem.

A EDA descreve o dataset integrado (todos os rótulos do CARD) e marca o destino de
cada mecanismo no TCC; a razão de desbalanceamento é dada nas duas visões. A
estrutura da ontologia é lida com o mesmo leitor dos atributos do Pipeline B.

Saídas: plots/eda/class_distribution.png e results/eda_resumo.json.

Dependências externas: pandas, matplotlib, seaborn
Dependências internas: config, experiment_config, ontology_features
"""

import logging
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


def add_destination(distribution: pd.DataFrame) -> pd.DataFrame:
    """
    Acrescenta a coluna 'destino' com o uso de cada mecanismo no TCC.

    A EDA descreve o dataset integrado (todos os rótulos do CARD), mas o
    experimento usa só as 5 classes de ``config.TARGET_CLASSES``. O destino vem de
    ``experiment_config.assign_split``: 'experimento', 'estudo_de_caso' ou
    'multirrotulo'. Um rótulo fora das listas recebe 'desconhecido' (na carga do
    experimento ele gera erro; aqui a EDA só o descreve).

    Parâmetros:
        distribution: DataFrame retornado por compute_class_distribution().

    Retorno:
        Cópia de `distribution` com a coluna 'destino'.
    """
    from src.experiment_config import assign_split

    def _destination(label: str) -> str:
        try:
            return assign_split(label)
        except ValueError:
            return "desconhecido"

    out = distribution.copy()
    out["destino"] = out["classe"].map(_destination)
    return out


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

# Termo da ARO cujas subclasses diretas são os mecanismos de resistência
# ("mechanism of antibiotic resistance").
MECHANISM_ROOT: str = "ARO:1000002"


def analyze_ontology_structure(ontology_path: Path) -> dict:
    """
    Estatísticas da hierarquia is-a da ARO, lidas do ``aro.owl``.

    O ``aro.owl`` do CARD está em sintaxe OWL/XML, que o rdflib não lê. A hierarquia
    vem do mesmo leitor que gera os atributos do Pipeline B
    (``ontology_features.load_ontology``), para que a EDA descreva exatamente o
    grafo usado nos atributos. Uma segunda leitura do arquivo conta as classes
    declaradas, os axiomas com restrição e os rótulos.

    Parâmetros:
        ontology_path: Caminho para o arquivo aro.owl (OWL/XML).

    Retorno:
        dict com campos:
            - n_classes (int): classes nomeadas declaradas na ontologia.
            - n_is_a_relations (int): axiomas SubClassOf entre duas classes
              nomeadas, que são as arestas usadas nos atributos.
            - n_restrictions (int): axiomas SubClassOf com restrição
              (ObjectSomeValuesFrom etc.), que os atributos ignoram.
            - n_roots (int): classes sem superclasse.
            - max_depth (int): maior profundidade is-a, com a mesma definição de
              ``depth_isa`` (menor caminho até uma raiz; raiz = 0).
            - resistance_mechanisms_found (list[str]): rótulos das subclasses
              diretas de "mechanism of antibiotic resistance" (ARO:1000002).

    Levanta:
        FileNotFoundError: Se o arquivo OWL não existir.
    """
    import xml.etree.ElementTree as ET

    from src.ontology_features import OWL_NS, iri_to_accession, load_ontology

    ontology_path = Path(ontology_path)
    if not ontology_path.exists():
        raise FileNotFoundError(f"Ontologia não encontrada: {ontology_path}")

    logger.info("Carregando ontologia de: %s", ontology_path)
    graph = load_ontology(ontology_path)

    # Segunda leitura: declarações, axiomas com restrição e rótulos.
    n_classes = 0
    n_restrictions = 0
    labels: dict[str, str] = {}
    for _, el in ET.iterparse(str(ontology_path), events=("end",)):
        if el.tag == OWL_NS + "Declaration":
            n_classes += sum(1 for k in el if k.tag == OWL_NS + "Class")
            el.clear()
        elif el.tag == OWL_NS + "SubClassOf":
            if any(k.tag != OWL_NS + "Class" for k in el):
                n_restrictions += 1
            el.clear()
        elif el.tag == OWL_NS + "AnnotationAssertion":
            kids = list(el)
            if len(kids) == 3 and kids[0].get("abbreviatedIRI") == "rdfs:label":
                accession = iri_to_accession(kids[1].text)
                if accession and kids[2].text:
                    labels[accession] = kids[2].text.strip()
            el.clear()

    depths = graph.depths()
    mechanisms = sorted(
        labels.get(acc, acc) for acc in graph.children.get(MECHANISM_ROOT, ())
    )
    stats = {
        "n_classes": n_classes,
        "n_is_a_relations": sum(len(p) for p in graph.parents.values()),
        "n_restrictions": n_restrictions,
        "n_roots": len(graph.roots),
        "max_depth": max(depths.values()) if depths else 0,
        "resistance_mechanisms_found": mechanisms,
    }
    logger.info(
        "Ontologia: %d classes declaradas, %d relações is-a entre classes nomeadas, "
        "%d axiomas com restrição, %d raízes, profundidade máxima %d, %d mecanismos.",
        stats["n_classes"],
        stats["n_is_a_relations"],
        stats["n_restrictions"],
        stats["n_roots"],
        stats["max_depth"],
        len(mechanisms),
    )
    return stats


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
    Usa paleta acessível ColorBrewer Set2. Com a coluna 'destino'
    (add_destination), a cor da barra indica o uso do mecanismo no TCC
    (experimento, estudo de caso ou multirrótulo, fora) e o destino vai escrito
    no rótulo, porque as barras dos mecanismos raros somem na escala linear.

    Parâmetros:
        distribution: DataFrame retornado por compute_class_distribution(),
                      com colunas 'classe', 'contagem' e 'proporcao_pct'
                      (e, opcionalmente, 'destino').
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
    has_dest = "destino" in distribution.columns
    dest_colors = {"experimento": ACCESSIBLE_PALETTE[0], "estudo_de_caso": ACCESSIBLE_PALETTE[1],
                   "multirrotulo": ACCESSIBLE_PALETTE[2], "desconhecido": "0.7"}
    dest_labels = {"experimento": "Experimento", "estudo_de_caso": "Estudo de caso",
                   "multirrotulo": "Multirrótulo (fora)", "desconhecido": "Desconhecido"}
    dest_notes = {"estudo_de_caso": " [estudo de caso]", "multirrotulo": " [multirrótulo, fora]", "desconhecido": " [desconhecido]"}
    if has_dest:
        palette = [dest_colors.get(d, "0.7") for d in distribution["destino"]]
    else:
        palette = (ACCESSIBLE_PALETTE * ((n_classes // len(ACCESSIBLE_PALETTE)) + 1))[:n_classes]

    fig, ax = plt.subplots(figsize=(10, max(4, n_classes * 0.9 if not has_dest else n_classes * 0.45 + 1.5)))

    bars = ax.barh(
        y=distribution["classe"],
        width=distribution["contagem"],
        color=palette,
        edgecolor="white",
        linewidth=0.8,
    )

    # Adicionar rótulos com contagem e proporção (e o destino, quando não é o experimento)
    for bar, (_, row) in zip(bars, distribution.iterrows()):
        width = bar.get_width()
        note = dest_notes.get(row["destino"], "") if has_dest else ""
        ax.text(
            width + max(distribution["contagem"]) * 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['contagem'])} ({row['proporcao_pct']:.1f}%){note}".replace(".", ","),
            va="center",
            ha="left",
            fontsize=9,
        )

    ax.set_xlabel("Número de genes", fontsize=11)
    ax.set_ylabel("Mecanismo de resistência (rótulo do CARD)", fontsize=11)
    ax.set_title("Genes por mecanismo de resistência no dataset integrado", fontsize=13, pad=12, fontweight="bold")
    ax.invert_yaxis()  # classe com mais instâncias no topo
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    if has_dest:
        from matplotlib.patches import Patch

        present = [d for d in dest_colors if d in set(distribution["destino"])]
        ax.legend(handles=[Patch(color=dest_colors[d], label=dest_labels[d]) for d in present], loc="lower right", fontsize=9)

    # Margem à direita para os rótulos de texto
    ax.set_xlim(0, distribution["contagem"].max() * (1.4 if has_dest else 1.25))

    plt.tight_layout()
    # PNG e PDF pelo estilo comum (vírgula decimal no eixo; PDF sem data).
    from src.plot_style import save

    save(fig, output_path, dpi)

    logger.info("Gráfico de distribuição de classes salvo em: %s (dpi=%d)", output_path, dpi)


# ──────────────────────────────────────────────────────────────────────────────
# 5.7 — Orquestrador do relatório EDA
# ──────────────────────────────────────────────────────────────────────────────

def generate_eda_report(
    df: pd.DataFrame,
    output_dir: Path,
    ontology_path: Path | None = None,
) -> dict:
    """
    Orquestra todas as análises EDA e salva visualizações em `output_dir/eda/`.

    Executa em sequência:
    1. Distribuição de classes do dataset integrado, com o destino de cada
       mecanismo no TCC (experimento, estudo de caso ou multirrótulo).
    2. Razão de desbalanceamento: no dataset integrado e só nas classes do
       experimento, que é a que vale para os modelos.
    3. Identificação de classes pequenas (< 20 instâncias).
    4. Verificação de valores ausentes.
    5. Análise da estrutura da ontologia ARO (se o arquivo existir).
    6. Geração do gráfico de distribuição de classes, colorido pelo destino.

    Parâmetros:
        df: DataFrame integrado retornado por data_loader.integrate_dataset().
        output_dir: Diretório raiz de plots (ex.: Path('plots')).
                    As figuras EDA serão salvas em output_dir/eda/.
        ontology_path: Caminho do aro.owl. Padrão: config.ONTOLOGY_PATH.

    Retorno:
        dict com campos:
            - class_distribution (pd.DataFrame), com a coluna 'destino'
            - imbalance_ratio (float): dataset integrado
            - imbalance_ratio_experimento (float | None): só as classes do
              experimento; None se nenhuma estiver presente
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
    if ontology_path is None:
        ontology_path = ONTOLOGY_PATH

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

    class_distribution = add_destination(compute_class_distribution(df, target_col))

    # 2. Razão de desbalanceamento: a do dataset integrado é dominada pelos
    # mecanismos raros (1 gene) e não vale para os modelos; a do experimento sim.
    imbalance_ratio = compute_imbalance_ratio(class_distribution)
    logger.info("Razão de desbalanceamento (dataset integrado): %.2f", imbalance_ratio)
    experiment_rows = class_distribution[class_distribution["destino"] == "experimento"]
    imbalance_ratio_experiment: float | None = None
    if not experiment_rows.empty:
        imbalance_ratio_experiment = compute_imbalance_ratio(experiment_rows)
        logger.info("Razão de desbalanceamento (experimento): %.2f", imbalance_ratio_experiment)

    # 3. Classes pequenas
    small_classes = flag_small_classes(class_distribution, threshold=20)  # já registra o aviso

    # 4. Valores ausentes
    missing_values = check_missing_values(df)

    # 5. Estrutura da ontologia
    # Arquivo ausente: a análise é pulada com aviso. Arquivo presente mas ilegível:
    # o erro sobe, para não esconder uma falha (a versão anterior engolia a
    # exceção e a análise nunca rodou sem que ninguém percebesse).
    ontology_stats: dict | None = None
    if Path(ontology_path).exists():
        ontology_stats = analyze_ontology_structure(Path(ontology_path))
    else:
        logger.warning(
            "Arquivo de ontologia não encontrado em %s. "
            "Análise ontológica pulada.",
            ontology_path,
        )

    # 6. Gráfico de distribuição de classes
    plot_path = eda_dir / "class_distribution.png"
    plot_class_distribution(class_distribution, plot_path, dpi=300)

    report = {
        "class_distribution": class_distribution,
        "imbalance_ratio": imbalance_ratio,
        "imbalance_ratio_experimento": imbalance_ratio_experiment,
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

    import json

    from src.config import PLOTS_DIR, PROCESSED_DIR, RESULTS_DIR

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
    if not report["missing_values"].empty:
        logger.warning("Valores ausentes:\n%s", report["missing_values"].to_string(index=False))

    # Resumo em arquivo, para o texto citar os números sem depender do log.
    summary = {
        "sobre": {
            "o_que_mostra": "Distribuição dos mecanismos do dataset integrado com o destino de cada um no TCC, razão de desbalanceamento (integrado e só as 5 classes do experimento) e estatísticas da hierarquia is-a da ARO.",
            "como": "Contagens do dataset_integrado.csv; destino por experiment_config.assign_split; ontologia lida com o mesmo leitor OWL/XML dos atributos (ontology_features).",
            "por_que": "Revisão geral, R1 e R7: a análise da ontologia na EDA nunca tinha funcionado (rdflib não lê OWL/XML) e a razão de 5.239:1 do integrado não vale para os modelos; a do experimento é 74,84:1. Os números ficam em arquivo para o texto citar (seção 4.3).",
            "fontes": ["data/processed/dataset_integrado.csv", "data/ontologia/aro.owl"],
        },
        "n_total_instances": report["n_total_instances"],
        "class_distribution": report["class_distribution"].to_dict(orient="records"),
        "imbalance_ratio": round(report["imbalance_ratio"], 2),
        "imbalance_ratio_experimento": (
            round(report["imbalance_ratio_experimento"], 2)
            if report["imbalance_ratio_experimento"] is not None
            else None
        ),
        "ontology_stats": report["ontology_stats"],
    }
    summary_path = RESULTS_DIR / "eda_resumo.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    logger.info("Resumo da EDA salvo em: %s", summary_path)
