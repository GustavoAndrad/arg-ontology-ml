"""
provenance.py — Rastreabilidade Científica do experimento arg-ml-semantic-classification.

Calcula hashes MD5 e SHA-256 dos arquivos de dados brutos e registra a
proveniência do dataset em `results/data_provenance.json`, garantindo que os
dados usados no experimento sejam identificáveis e reproduzíveis.

Por que o SHA-256 (TCC2, ponto 13): o ``aro.owl`` não traz número de versão; a
versão da ARO usada fica identificada pela data de download e pelo checksum.
O MD5 continua, para comparar com o registro do TCC1.

Data de acesso: é a do primeiro registro (14/06/2026, quando os dados do TCC1
foram baixados; é também a data de modificação do ``aro.owl``). Rodar o módulo de
novo **não** troca essa data: ela é lida do JSON existente. Antes do TCC2, cada
execução gravava a data do dia, o que apagaria a informação.

Caminhos: gravados relativos à raiz do projeto, para não depender da pasta (nem
do usuário) de quem rodou. A data de modificação de cada arquivo também é
registrada, como evidência auxiliar (pode mudar se o arquivo for copiado).
"""

import hashlib
import json
import logging
import re
from datetime import date, datetime
from pathlib import Path

from src.config import _PROJECT_ROOT, RAW_CARD_DIR, ONTOLOGY_PATH, RESULTS_DIR

logger = logging.getLogger(__name__)


def compute_sha256(file_path: Path) -> str:
    """SHA-256 de um arquivo, lido em blocos (mesma leitura do ``compute_md5``)."""
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
    hasher = hashlib.sha256()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _relative_path(file_path: Path, base_dir: Path | None) -> str:
    """Caminho relativo a ``base_dir`` (com "/"), ou o próprio caminho se estiver fora dele."""
    if base_dir is not None:
        try:
            return Path(file_path).resolve().relative_to(Path(base_dir).resolve()).as_posix()
        except ValueError:
            pass
    return str(file_path)


def compute_md5(file_path: Path) -> str:
    """
    Calcula o hash MD5 de um arquivo em modo binário.

    Lê o arquivo em blocos de 8 KB para evitar carregar arquivos grandes
    inteiramente na memória.

    Parâmetros:
        file_path: Caminho absoluto ou relativo ao arquivo.

    Retorno:
        str: String hexadecimal do hash MD5 (32 caracteres).

    Levanta:
        FileNotFoundError: Se o arquivo não existir.
    """
    file_path = Path(file_path)
    if not file_path.exists():
        raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")

    hasher = hashlib.md5()
    with file_path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def extract_card_version(readme_path: Path, card_json_path: Path) -> str:
    """
    Extrai a versão do CARD do README ou do campo ``_version`` em ``card.json``.

    Tenta primeiro localizar um padrão de versão semântica (ex.: ``3.2.6``) no
    arquivo README. Se não encontrar, cai para o campo ``_version`` do
    ``card.json``. Retorna ``'unknown'`` se nenhuma fonte fornecer a versão.

    Parâmetros:
        readme_path: Caminho para ``CARD-Download-README.txt``.
        card_json_path: Caminho para ``card.json``.

    Retorno:
        str: String de versão (ex.: ``'3.2.6'``) ou ``'unknown'`` se não
             encontrada.
    """
    # Tenta extrair versão do README (padrão semântico X.Y.Z)
    readme_path = Path(readme_path)
    if readme_path.exists():
        try:
            content = readme_path.read_text(encoding="utf-8", errors="replace")
            match = re.search(r"\bCARD[^\n]*?(\d+\.\d+\.\d+)", content, re.IGNORECASE)
            if match:
                version = match.group(1)
                logger.info("Versão do CARD extraída do README: %s", version)
                return version
        except Exception as exc:
            logger.warning("Falha ao ler README para versão do CARD: %s", exc)

    # Cai para o campo '_version' de card.json
    card_json_path = Path(card_json_path)
    if card_json_path.exists():
        try:
            with card_json_path.open("r", encoding="utf-8") as fh:
                data = json.load(fh)
            version = data.get("_version") or data.get("version")
            if version:
                logger.info("Versão do CARD extraída do card.json: %s", version)
                return str(version)
        except Exception as exc:
            logger.warning("Falha ao ler card.json para versão do CARD: %s", exc)

    logger.warning("Versão do CARD não encontrada; usando 'unknown'.")
    return "unknown"


def generate_provenance(
    data_files: dict[str, Path],
    card_version: str,
    output_path: Path,
    access_date: str | None = None,
    base_dir: Path | None = None,
) -> dict:
    """
    Gera e salva o arquivo ``data_provenance.json``.

    Para cada arquivo listado em ``data_files``, calcula MD5 e SHA-256 e registra
    o caminho (relativo a ``base_dir``, se dado) e a data de modificação.

    Parâmetros:
        data_files: Dicionário ``{nome_logico: caminho}`` dos arquivos de dados
                    brutos a serem rastreados.
        card_version: Versão do CARD identificada (ex.: ``'3.2.6'``).
        output_path: Caminho de saída para o JSON (ex.:
                     ``results/data_provenance.json``).
        access_date: Data de acesso (ISO 8601). Se ``None``, mantém a do JSON já
                     existente em ``output_path``; se não houver, usa a de hoje.
        base_dir: Raiz para gravar caminhos relativos (``None`` = caminho como veio).

    Retorno:
        dict: Conteúdo do provenance gerado, com campos:

              - ``card_version`` (str): versão do CARD.
              - ``access_date`` (str): data de acesso no formato ISO 8601
                (``YYYY-MM-DD``).
              - ``files`` (dict): mapeamento
                ``{nome_logico: {"path", "md5", "sha256", "modificado_em"}}``.

    Levanta:
        FileNotFoundError: Se algum arquivo em ``data_files`` não existir.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if access_date is None and output_path.exists():
        try:
            access_date = json.loads(output_path.read_text(encoding="utf-8")).get("access_date")
        except (OSError, json.JSONDecodeError):
            access_date = None
    if access_date is None:
        access_date = date.today().isoformat()

    files_info: dict[str, dict] = {}
    for logical_name, file_path in data_files.items():
        file_path = Path(file_path)
        logger.info("Calculando MD5 e SHA-256 de '%s' (%s)…", logical_name, file_path)
        files_info[logical_name] = {
            "path": _relative_path(file_path, base_dir),
            "md5": compute_md5(file_path),
            "sha256": compute_sha256(file_path),
            "modificado_em": datetime.fromtimestamp(file_path.stat().st_mtime).date().isoformat(),
        }

    provenance = {
        "card_version": card_version,
        "access_date": access_date,
        "files": files_info,
    }

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(provenance, fh, indent=2, ensure_ascii=False)

    logger.info("Proveniência salva em: %s", output_path)
    return provenance


# ── Ponto de entrada ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    # Arquivos de dados brutos do CARD + ontologia ARO
    data_files: dict[str, Path] = {
        "aro_index":                RAW_CARD_DIR / "aro_index.tsv",
        "aro_categories_index":     RAW_CARD_DIR / "aro_categories_index.tsv",
        "aro_categories":           RAW_CARD_DIR / "aro_categories.tsv",
        "card_json":                RAW_CARD_DIR / "card.json",
        "protein_fasta_homolog":    RAW_CARD_DIR / "protein_fasta_protein_homolog_model.fasta",
        "protein_fasta_knockout":   RAW_CARD_DIR / "protein_fasta_protein_knockout_model.fasta",
        "protein_fasta_overexpression": RAW_CARD_DIR / "protein_fasta_protein_overexpression_model.fasta",
        "protein_fasta_variant":    RAW_CARD_DIR / "protein_fasta_protein_variant_model.fasta",
        "aro_owl":                  ONTOLOGY_PATH,
    }

    readme_path = RAW_CARD_DIR / "CARD-Download-README.txt"
    card_json_path = RAW_CARD_DIR / "card.json"

    card_version = extract_card_version(readme_path, card_json_path)
    output_path = RESULTS_DIR / "data_provenance.json"

    provenance = generate_provenance(data_files, card_version, output_path, base_dir=_PROJECT_ROOT)
    print(f"Proveniência gerada com sucesso. Versão do CARD: {provenance['card_version']}")
    print(f"Data de acesso: {provenance['access_date']}")
    print(f"Arquivos rastreados: {list(provenance['files'].keys())}")
