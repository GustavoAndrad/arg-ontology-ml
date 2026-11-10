"""
data_loader.py — Carregamento e Integração dos Dados do CARD.

Responsabilidade: Carregar e integrar os dados do CARD (TSV, FASTA),
retornando um DataFrame unificado pronto para EDA e extração de features.

Dependências externas: pandas, biopython (Bio.SeqIO)
Dependências internas: config
"""

import logging
from pathlib import Path

import pandas as pd
from Bio import SeqIO

logger = logging.getLogger(__name__)

# Mapeamento de variações de nomes de colunas para nomes padronizados (com underscore)
_COLUMN_ALIASES: dict[str, str] = {
    # ARO Accession
    "ARO Accession": "ARO_Accession",
    "ARO_Accession": "ARO_Accession",
    # CVTERM ID
    "CVTERM ID": "CVTERM_ID",
    "CVTERM_ID": "CVTERM_ID",
    # Model Sequence ID
    "Model Sequence ID": "Model_Sequence_ID",
    "Model_Sequence_ID": "Model_Sequence_ID",
    # Model ID
    "Model ID": "Model_ID",
    "Model_ID": "Model_ID",
    # Model Name
    "Model Name": "Model_Name",
    "Model_Name": "Model_Name",
    # ARO Name
    "ARO Name": "ARO_Name",
    "ARO_Name": "ARO_Name",
    # Protein Accession
    "Protein Accession": "Protein_Accession",
    "Protein_Accession": "Protein_Accession",
    # DNA Accession
    "DNA Accession": "DNA_Accession",
    "DNA_Accession": "DNA_Accession",
    # AMR Gene Family
    "AMR Gene Family": "AMR_Gene_Family",
    "AMR_Gene_Family": "AMR_Gene_Family",
    # Drug Class
    "Drug Class": "Drug_Class",
    "Drug_Class": "Drug_Class",
    # Resistance Mechanism
    "Resistance Mechanism": "Resistance_Mechanism",
    "Resistance_Mechanism": "Resistance_Mechanism",
    # CARD Short Name
    "CARD Short Name": "CARD_Short_Name",
    "CARD_Short_Name": "CARD_Short_Name",
}

# Colunas obrigatórias que o aro_index deve ter após renomeação
_ARO_INDEX_REQUIRED_COLS: list[str] = [
    "ARO_Accession",
    "Model_Name",
    "ARO_Name",
    "Protein_Accession",
    "AMR_Gene_Family",
    "Drug_Class",
    "Resistance_Mechanism",
    "CARD_Short_Name",
]


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renomeia colunas do DataFrame usando o mapeamento _COLUMN_ALIASES.
    Colunas não mapeadas são mantidas como estão.

    Parâmetros:
        df: DataFrame com colunas originais (possivelmente com espaços).

    Retorno:
        DataFrame com colunas renomeadas para o padrão com underscore.
    """
    rename_map = {col: _COLUMN_ALIASES[col] for col in df.columns if col in _COLUMN_ALIASES}
    return df.rename(columns=rename_map)


def load_aro_index(path: Path) -> pd.DataFrame:
    """
    Carrega aro_index.tsv detectando automaticamente as colunas reais do arquivo.
    Lida com variações de nomes de colunas (espaços vs underscores).

    Parâmetros:
        path: Caminho para o arquivo aro_index.tsv.

    Retorno:
        DataFrame com colunas padronizadas: ARO_Accession, Model_Name, ARO_Name,
        Protein_Accession, AMR_Gene_Family, Drug_Class, Resistance_Mechanism,
        CARD_Short_Name (além de outras colunas presentes no arquivo).

    Levanta:
        FileNotFoundError: Se o arquivo não existir.
        ValueError: Se colunas obrigatórias estiverem ausentes após renomeação.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    logger.info("Carregando ARO index de: %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = _normalize_columns(df)

    # Verificar colunas obrigatórias
    missing = [c for c in _ARO_INDEX_REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Colunas obrigatórias ausentes em {path.name}: {missing}. "
            f"Colunas disponíveis: {list(df.columns)}"
        )

    logger.info(
        "ARO index carregado com %d entradas e %d colunas.", len(df), len(df.columns)
    )
    return df


def load_aro_categories_index(path: Path) -> pd.DataFrame:
    """
    Carrega aro_categories_index.tsv com detecção automática de colunas.

    Parâmetros:
        path: Caminho para o arquivo aro_categories_index.tsv.

    Retorno:
        DataFrame com colunas padronizadas: Protein_Accession, DNA_Accession,
        AMR_Gene_Family, Drug_Class, Resistance_Mechanism (e demais colunas presentes).

    Levanta:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {path}")

    logger.info("Carregando ARO categories index de: %s", path)
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = _normalize_columns(df)

    logger.info(
        "ARO categories index carregado com %d entradas e %d colunas.",
        len(df),
        len(df.columns),
    )
    return df


def parse_fasta_header(header: str) -> tuple[str, str]:
    """
    Extrai ARO_Accession e Protein_Accession do cabeçalho FASTA do CARD.

    O formato do CARD pode ser:
      - Formato A (homolog model): >gb|PROTEIN_ACC|ARO:XXXXXXX|model_name [organism]
      - Formato B (descrito na spec): >ARO:XXXXXXX|gb|PROTEIN_ACC|...

    Parâmetros:
        header: Linha de cabeçalho FASTA sem o '>' inicial.
                Exemplo: "gb|ACT97415.1|ARO:3002999|CblA-1 [mixed culture bacterium]"

    Retorno:
        tuple (aro_accession, protein_accession) — ambos como strings.
        Exemplo: ("ARO:3002999", "ACT97415.1")

    Levanta:
        ValueError: Se o header não contiver um ARO Accession válido.
    """
    # Remove espaços iniciais/finais
    header = header.strip()

    # Separar a parte dos campos (antes do espaço/descrição) dos campos '|'
    # O header pode ter uma descrição entre colchetes ou após o primeiro espaço
    parts = header.split("|")

    aro_accession: str = ""
    protein_accession: str = ""

    for i, part in enumerate(parts):
        # Limpar espaços e texto de descrição do último campo
        part_clean = part.split()[0] if part.split() else part

        if part_clean.startswith("ARO:"):
            aro_accession = part_clean
        elif part in ("gb", "emb", "dbj", "ref", "sp", "pir", "prf"):
            # O campo seguinte é o accession da proteína
            if i + 1 < len(parts):
                protein_accession = parts[i + 1].split()[0]

    # Se o formato B for usado: ARO:XXXXXXX|gb|PROTEIN_ACC|...
    if not aro_accession:
        # Tentar o primeiro campo como ARO accession
        first = parts[0].split()[0] if parts else ""
        if first.startswith("ARO:"):
            aro_accession = first
            # Protein accession seria o terceiro campo (índice 2)
            if len(parts) >= 3:
                protein_accession = parts[2].split()[0]

    if not aro_accession:
        raise ValueError(
            f"ARO Accession não encontrado no header FASTA: '{header}'"
        )

    return aro_accession, protein_accession


def load_protein_fasta(path: Path) -> dict[str, str]:
    """
    Carrega um arquivo FASTA de proteínas usando Bio.SeqIO e mapeia
    ARO_Accession -> sequência proteica em string.

    O ARO Accession é extraído do cabeçalho FASTA via parse_fasta_header.

    Parâmetros:
        path: Caminho para o arquivo FASTA (ex.: protein_fasta_protein_homolog_model.fasta).

    Retorno:
        dict mapeando ARO_Accession (str, formato "ARO:XXXXXXX") -> sequência proteica (str).
        Em caso de headers duplicados com o mesmo ARO Accession, o último prevalece
        e um aviso é registrado em log.

    Levanta:
        FileNotFoundError: Se o arquivo não existir.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo FASTA não encontrado: {path}")

    logger.info("Carregando FASTA de proteínas de: %s", path)
    sequences: dict[str, str] = {}
    duplicates = 0

    for record in SeqIO.parse(str(path), "fasta"):
        try:
            aro_accession, _ = parse_fasta_header(record.description)
            if aro_accession in sequences:
                logger.warning(
                    "ARO Accession duplicado no FASTA: %s — sequência sobrescrita.",
                    aro_accession,
                )
                duplicates += 1
            sequences[aro_accession] = str(record.seq)
        except ValueError as exc:
            logger.warning("Ignorando entrada FASTA com header inválido: %s — %s", record.id, exc)

    logger.info(
        "FASTA carregado: %d sequências únicas (%d duplicatas ignoradas).",
        len(sequences),
        duplicates,
    )
    return sequences


def integrate_dataset(
    aro_index: pd.DataFrame,
    fasta_sequences: dict[str, str],
) -> pd.DataFrame:
    """
    Integra o índice ARO com as sequências FASTA via ARO_Accession.

    Realiza um inner join entre aro_index e fasta_sequences para garantir
    que apenas entradas com sequência disponível sejam incluídas.
    Remove instâncias com Resistance_Mechanism ausente ou vazio.
    Registra instâncias descartadas em log (logging.warning).

    Parâmetros:
        aro_index: DataFrame retornado por load_aro_index().
        fasta_sequences: dict retornado por load_protein_fasta().

    Retorno:
        DataFrame com colunas: ARO_Accession, Sequence, Resistance_Mechanism,
        Drug_Class, AMR_Gene_Family, Protein_Accession.
        Cada linha é uma instância única com identificador ARO.
    """
    total_aro = len(aro_index)
    logger.info("Iniciando integração: %d entradas no ARO index.", total_aro)

    # Converter FASTA dict para DataFrame para join
    fasta_df = pd.DataFrame(
        list(fasta_sequences.items()),
        columns=["ARO_Accession", "Sequence"],
    )

    # Inner join: apenas entradas com sequência disponível
    merged = aro_index.merge(fasta_df, on="ARO_Accession", how="inner")
    n_without_sequence = total_aro - len(merged)
    if n_without_sequence > 0:
        logger.warning(
            "%d entradas descartadas por ausência de sequência FASTA.",
            n_without_sequence,
        )

    # Remover instâncias sem Resistance_Mechanism
    before_filter = len(merged)
    mask_missing_rm = merged["Resistance_Mechanism"].isna() | (
        merged["Resistance_Mechanism"].str.strip() == ""
    )
    discarded_rm = merged[mask_missing_rm]
    if len(discarded_rm) > 0:
        logger.warning(
            "%d instâncias descartadas por Resistance_Mechanism ausente/vazio:",
            len(discarded_rm),
        )
        for _, row in discarded_rm.iterrows():
            logger.warning(
                "  Descartado: ARO_Accession=%s, Model_Name=%s",
                row.get("ARO_Accession", "N/A"),
                row.get("Model_Name", "N/A"),
            )

    merged = merged[~mask_missing_rm].copy()
    after_filter = len(merged)
    logger.info(
        "Após filtro de Resistance_Mechanism: %d instâncias retidas (descartadas: %d).",
        after_filter,
        before_filter - after_filter,
    )

    # Selecionar e reordenar colunas de saída
    output_cols = [
        "ARO_Accession",
        "Sequence",
        "Resistance_Mechanism",
        "Drug_Class",
        "AMR_Gene_Family",
        "Protein_Accession",
    ]
    # Garantir que todas as colunas de saída existem
    for col in output_cols:
        if col not in merged.columns:
            logger.warning("Coluna '%s' ausente no DataFrame integrado; preenchida com NaN.", col)
            merged[col] = pd.NA

    # Deduplicar por ARO_Accession: o dataset final deve ter um único registro por ARO
    # (o índice ARO pode conter duplicatas de modelos diferentes para o mesmo ARO Accession)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(subset=["ARO_Accession"], keep="first")
    n_deduped = before_dedup - len(merged)
    if n_deduped > 0:
        logger.warning(
            "%d duplicatas de ARO_Accession removidas (mantida a primeira ocorrência).",
            n_deduped,
        )

    result = merged[output_cols].reset_index(drop=True)
    logger.info("Dataset integrado final: %d instâncias.", len(result))
    return result


def validate_dataset(df: pd.DataFrame) -> None:
    """
    Valida a integridade do dataset integrado.

    Verifica:
    - Sem ARO_Accession duplicados
    - Sem Sequence vazia (None, NaN, ou string vazia)
    - Sem Resistance_Mechanism ausente (None, NaN, ou string vazia)

    Parâmetros:
        df: DataFrame retornado por integrate_dataset().

    Levanta:
        ValueError: Com mensagem descritiva em caso de qualquer violação.
    """
    errors: list[str] = []

    # 1. Verificar duplicatas de ARO_Accession
    duplicated = df["ARO_Accession"].duplicated(keep=False)
    if duplicated.any():
        dup_values = df.loc[duplicated, "ARO_Accession"].unique().tolist()
        errors.append(
            f"ARO_Accession duplicados encontrados ({len(dup_values)} valores únicos duplicados): "
            f"{dup_values[:10]}{'...' if len(dup_values) > 10 else ''}"
        )

    # 2. Verificar sequências vazias
    empty_seq = df["Sequence"].isna() | (df["Sequence"].astype(str).str.strip() == "")
    if empty_seq.any():
        n_empty = empty_seq.sum()
        bad_ids = df.loc[empty_seq, "ARO_Accession"].tolist()
        errors.append(
            f"{n_empty} instâncias com Sequence vazia ou ausente: "
            f"{bad_ids[:5]}{'...' if len(bad_ids) > 5 else ''}"
        )

    # 3. Verificar Resistance_Mechanism ausente
    missing_rm = df["Resistance_Mechanism"].isna() | (
        df["Resistance_Mechanism"].astype(str).str.strip() == ""
    )
    if missing_rm.any():
        n_missing = missing_rm.sum()
        bad_ids = df.loc[missing_rm, "ARO_Accession"].tolist()
        errors.append(
            f"{n_missing} instâncias com Resistance_Mechanism ausente ou vazio: "
            f"{bad_ids[:5]}{'...' if len(bad_ids) > 5 else ''}"
        )

    if errors:
        raise ValueError(
            "Validação do dataset falhou com os seguintes problemas:\n"
            + "\n".join(f"  - {e}" for e in errors)
        )

    logger.info(
        "Validação do dataset aprovada: %d instâncias, sem duplicatas, "
        "sem sequências vazias, sem Resistance_Mechanism ausente.",
        len(df),
    )


if __name__ == "__main__":
    """
    Bloco principal: carrega os dados reais do CARD e salva o dataset integrado
    em data/processed/dataset_integrado.csv.

    Uso:
        python src/data_loader.py
    """
    import sys

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    # Importar configurações
    try:
        from config import (
            RAW_CARD_DIR,
            PROCESSED_DIR,
        )
    except ImportError:
        from src.config import (
            RAW_CARD_DIR,
            PROCESSED_DIR,
        )

    aro_index_path = RAW_CARD_DIR / "aro_index.tsv"
    fasta_path = RAW_CARD_DIR / "protein_fasta_protein_homolog_model.fasta"
    output_path = PROCESSED_DIR / "dataset_integrado.csv"

    logger.info("=== Iniciando carregamento dos dados do CARD ===")

    # 1. Carregar ARO index
    aro_index = load_aro_index(aro_index_path)
    logger.info("ARO index carregado: %d entradas", len(aro_index))

    # 2. Carregar sequências FASTA
    fasta_sequences = load_protein_fasta(fasta_path)
    logger.info("Sequências FASTA carregadas: %d entradas", len(fasta_sequences))

    # 3. Integrar dataset
    dataset = integrate_dataset(aro_index, fasta_sequences)
    logger.info("Dataset integrado: %d instâncias", len(dataset))

    # 4. Validar dataset
    validate_dataset(dataset)
    logger.info("Dataset validado com sucesso.")

    # 5. Salvar dataset integrado
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    dataset.to_csv(output_path, index=False)
    logger.info("Dataset integrado salvo em: %s", output_path)

    logger.info("=== Carregamento concluído com sucesso ===")
    logger.info("Distribuição de Resistance_Mechanism:")
    for mech, count in dataset["Resistance_Mechanism"].value_counts().items():
        logger.info("  %s: %d", mech, count)
