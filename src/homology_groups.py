"""
homology_groups.py — Grupos de similaridade de sequência para a divisão treino/teste.

Por que: o CARD tem muitas variantes quase idênticas do mesmo gene. Numa divisão aleatória, uma variante cai no teste e a
"irmã" fica no treino, e o modelo acerta por já ter visto quase a mesma proteína.
A divisão passa a ser feita por grupos: genes parecidos ficam no mesmo grupo, e o
grupo inteiro vai para o treino ou para o teste.

Como os grupos são formados:
    1. Busca de todos contra todos com o MMseqs2 (``easy-search``), sobre os 6.052
       genes do dataset integrado. A busca é feita uma vez, com identidade mínima de
       40% e sem filtro de cobertura; os filtros são aplicados depois, em Python,
       para que a mesma busca sirva a todos os limiares e às análises de parentesco.
    2. Para cada limiar t (40, 50, 70 e 90%), dois genes do experimento são ligados
       se o alinhamento tem identidade ≥ t **e** cobre ≥ 50% das duas sequências.
    3. Grupos = componentes conexos desse grafo ("single-linkage"). Por construção,
       nenhum par de genes em grupos diferentes passa do limiar, entre os pares que
       a busca encontrou. O agrupamento padrão do MMseqs2 (``cluster``) liga cada gene
       a um representante e não garante essa propriedade, por isso não é usado.

Identidade: ``fident`` do MMseqs2 = resíduos idênticos / comprimento do alinhamento.
Cobertura: ``qcov`` e ``tcov`` = fração de cada sequência coberta pelo alinhamento.

Limitação: a busca do MMseqs2 é heurística (filtro de k-mers antes do alinhamento).
Para reduzir pares perdidos: sensibilidade máxima (``-s 7.5``) e ``--max-seqs``
acima do número de genes. Conferência: a busca sem piso de identidade de
``nearest_neighbor``, rodada à parte, também não acha nenhum par treino–teste da
partição final com identidade ≥ 70% (o maior é 69,6%, com ou sem a exigência de
cobertura).

O que os grupos separam: sequências parecidas, não famílias de genes. Membros de uma
mesma família AMR com identidade < 70% podem cair em grupos diferentes (e, depois, em
treino e teste), e no CARD a família define o mecanismo.

Saídas:
    results/grupos_similaridade.csv  — grupo de cada gene do experimento, por limiar
    results/grupos_resumo.csv        — por limiar e classe: grupos e maior grupo
    results/mmseqs_parametros.json   — versão, SHA-256, comando e parâmetros
    data/processed/mmseqs/hits.m8    — pares encontrados (não versionado)

Uso:
    python -m src.homology_groups
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.config import _PROJECT_ROOT
from src.experiment_config import (
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET,
    load_experiment_dataset,
    load_integrated_dataset,
)

logger = logging.getLogger(__name__)

THRESHOLDS: tuple[int, ...] = (40, 50, 70, 90)  # identidade mínima (%) de cada agrupamento
MIN_COVERAGE: float = 0.5  # cobertura mínima das duas sequências
SEARCH_MIN_IDENTITY: float = 0.4  # a busca guarda tudo a partir de 40%
SENSITIVITY: float = 7.5  # máxima do MMseqs2
EVALUE: float = 1e-3  # padrão do MMseqs2
MMSEQS_VERSION: str = "18-8cc5c"
MMSEQS_BIN: Path = Path(
    os.environ.get("MMSEQS_BIN", _PROJECT_ROOT / "tools" / f"mmseqs_{MMSEQS_VERSION}" / "mmseqs" / "bin" / "mmseqs.exe")
)
WORK_DIR: Path = PROCESSED_DIR / "mmseqs"
HIT_COLUMNS: list[str] = ["query", "target", "fident", "alnlen", "qlen", "tlen", "qcov", "tcov", "evalue", "bits"]


def group_column(threshold: int) -> str:
    return f"grupo_{threshold}"


# ── Busca ──────────────────────────────────────────────────────────────────────
def write_fasta(df: pd.DataFrame, path: Path) -> None:
    """FASTA com ``ARO_Accession`` como identificador (os ':' viram '_' para o MMseqs2)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="ascii", newline="\n") as fh:
        for acc, seq in zip(df["ARO_Accession"], df["Sequence"]):
            fh.write(f">{acc.replace(':', '_')}\n{str(seq).upper()}\n")


def _sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def run_mmseqs_search(fasta: Path, out: Path, threads: int | None = None) -> dict:
    """
    Roda o ``mmseqs easy-search`` de todos contra todos e devolve os parâmetros usados.

    O MMseqs2 para Windows usa ferramentas do BusyBox que ficam na mesma pasta do
    executável; por isso a pasta entra no PATH. Os caminhos são relativos à pasta de
    trabalho, para evitar problemas de conversão de caminhos do Windows.
    """
    if not MMSEQS_BIN.exists():
        raise FileNotFoundError(f"MMseqs2 não encontrado em {MMSEQS_BIN}. Veja o README (Instalação).")
    threads = threads or max(1, (os.cpu_count() or 2) - 1)
    args = [
        "easy-search", fasta.name, fasta.name, out.name, "tmp",
        "--min-seq-id", str(SEARCH_MIN_IDENTITY),
        "-c", "0", "--cov-mode", "0",
        "-s", str(SENSITIVITY),
        "-e", str(EVALUE),
        "--max-seqs", "10000",
        "--alignment-mode", "3",
        "--format-output", ",".join(HIT_COLUMNS),
        "--threads", str(threads),
    ]
    env = dict(os.environ, PATH=str(MMSEQS_BIN.parent) + os.pathsep + os.environ.get("PATH", ""))
    t0 = time.time()
    proc = subprocess.run([str(MMSEQS_BIN), *args], cwd=fasta.parent, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not out.exists():
        raise RuntimeError(f"MMseqs2 falhou (código {proc.returncode}):\n{proc.stderr[-3000:]}")
    version = subprocess.run([str(MMSEQS_BIN), "version"], env=env, capture_output=True, text=True).stdout.strip()
    return {
        "programa": "MMseqs2",
        "versao_release": MMSEQS_VERSION,
        "versao_commit": version,
        "sha256_executavel": _sha256(MMSEQS_BIN),
        "sha256_zip": _sha256(MMSEQS_BIN.parents[2] / "mmseqs-win64.zip"),
        "comando": "mmseqs " + " ".join(args),
        "identidade": "fident = resíduos idênticos / comprimento do alinhamento",
        "cobertura_minima_nas_duas_sequencias": MIN_COVERAGE,
        "limiares_de_identidade": list(THRESHOLDS),
        "tempo_s": round(time.time() - t0, 1),
    }


def load_hits(path: Path) -> pd.DataFrame:
    """Pares encontrados, com os identificadores de volta ao formato ``ARO:xxxxxxx``, sem auto-pares."""
    hits = pd.read_csv(path, sep="\t", header=None, names=HIT_COLUMNS)
    for c in ("query", "target"):
        hits[c] = hits[c].str.replace("_", ":", n=1, regex=False)
    return hits[hits["query"] != hits["target"]].reset_index(drop=True)


# ── Grupos ─────────────────────────────────────────────────────────────────────
def qualifying_pairs(hits: pd.DataFrame, threshold: int, min_coverage: float = MIN_COVERAGE) -> pd.DataFrame:
    """Pares com identidade ≥ limiar e cobertura ≥ mínimo nas duas sequências."""
    mask = (hits["fident"] >= threshold / 100) & (hits["qcov"] >= min_coverage) & (hits["tcov"] >= min_coverage)
    return hits.loc[mask, ["query", "target"]]


def connected_components(ids: list[str], pairs: pd.DataFrame) -> pd.Series:
    """
    Componentes conexos (union-find). Pares com genes fora de ``ids`` são ignorados.

    Retorna, para cada id, um rótulo de grupo inteiro; os grupos são numerados do maior
    para o menor (empates pela ordem do primeiro gene), para o resultado ser determinístico.
    """
    index = {g: i for i, g in enumerate(ids)}
    parent = list(range(len(ids)))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for q, t in zip(pairs["query"], pairs["target"]):
        if q in index and t in index:
            a, b = find(index[q]), find(index[t])
            if a != b:
                parent[max(a, b)] = min(a, b)
    roots = np.array([find(i) for i in range(len(ids))])
    sizes = pd.Series(roots).value_counts()
    order = sorted(sizes.index, key=lambda r: (-sizes[r], r))
    relabel = {r: k for k, r in enumerate(order)}
    return pd.Series([relabel[r] for r in roots], index=ids, name="grupo")


def build_group_table(exp: pd.DataFrame, hits: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Uma coluna de grupo por limiar, na ordem dos genes do experimento."""
    ids = exp["ARO_Accession"].tolist()
    table = pd.DataFrame({"ARO_Accession": ids, "mecanismo": exp[TARGET].values})
    for t in thresholds:
        table[group_column(t)] = connected_components(ids, qualifying_pairs(hits, t)).values
    return table


def count_cross_group_pairs(table: pd.DataFrame, hits: pd.DataFrame, threshold: int) -> int:
    """Pares acima do limiar entre grupos diferentes (tem de ser 0, por construção)."""
    g = table.set_index("ARO_Accession")[group_column(threshold)]
    pairs = qualifying_pairs(hits, threshold)
    pairs = pairs[pairs["query"].isin(g.index) & pairs["target"].isin(g.index)]
    return int((g.loc[pairs["query"]].values != g.loc[pairs["target"]].values).sum())


def summarize_groups(table: pd.DataFrame, thresholds=THRESHOLDS) -> pd.DataFrame:
    """Por limiar e classe: genes, grupos, tamanho do maior grupo e sua fração na classe."""
    rows = []
    for t in thresholds:
        col = group_column(t)
        for mech, sub in table.groupby("mecanismo"):
            sizes = sub[col].value_counts()
            rows.append(
                {
                    "limiar": t,
                    "mecanismo": mech,
                    "genes": len(sub),
                    "grupos": int(sizes.size),
                    "maior_grupo": int(sizes.max()),
                    "fracao_maior_grupo": float(sizes.max() / len(sub)),
                    "grupos_unitarios": int((sizes == 1).sum()),
                }
            )
        # grupos com genes de mais de um mecanismo (o grupo inteiro vai para um lado só)
        mixed = table.groupby(col)["mecanismo"].nunique()
        rows.append({"limiar": t, "mecanismo": "(grupos com mais de um mecanismo)", "grupos": int((mixed > 1).sum())})
    return pd.DataFrame(rows)


# ── Parentesco entre conjuntos ──────────────────────────────────────────────────
def nearest_identity(
    hits: pd.DataFrame, query_ids, reference_ids, min_coverage: float | None = MIN_COVERAGE
) -> pd.Series:
    """
    Para cada gene de ``query_ids``, a maior identidade com algum gene de ``reference_ids``
    (0 se a busca não achou nenhum par a partir de 40%). ``min_coverage=None`` ignora a cobertura.
    """
    q, ref = set(query_ids), set(reference_ids)
    h = hits[hits["query"].isin(q) & hits["target"].isin(ref)]
    if min_coverage is not None:
        h = h[(h["qcov"] >= min_coverage) & (h["tcov"] >= min_coverage)]
    best = h.groupby("query")["fident"].max()
    return best.reindex(list(query_ids)).fillna(0.0)


def homology_profile(nearest: pd.Series, cutoffs=(0.5, 0.7, 0.9, 0.99)) -> dict[str, float]:
    """Fração dos genes com um parente acima de cada corte."""
    return {f"pct_parente_ge_{int(c * 100)}": float((nearest >= c).mean() * 100) for c in cutoffs}


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    integrated = load_integrated_dataset()
    fasta = WORK_DIR / "dataset_integrado.fasta"
    write_fasta(integrated, fasta)
    params = run_mmseqs_search(fasta, WORK_DIR / "hits.m8")
    hits = load_hits(WORK_DIR / "hits.m8")
    params.update({"genes_buscados": len(integrated), "pares_encontrados_sem_autopares": len(hits)})
    logger.info("MMseqs2: %d pares (sem auto-pares) em %.0f s.", len(hits), params["tempo_s"])

    exp = load_experiment_dataset()
    table = build_group_table(exp, hits)
    for t in THRESHOLDS:
        n_cross = count_cross_group_pairs(table, hits, t)
        assert n_cross == 0, f"{n_cross} pares acima de {t}% entre grupos diferentes"
        logger.info("Limiar %d%%: %d grupos (pares acima do limiar entre grupos: 0).", t, table[group_column(t)].nunique())
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    table.to_csv(RESULTS_DIR / "grupos_similaridade.csv", index=False)
    summarize_groups(table).to_csv(RESULTS_DIR / "grupos_resumo.csv", index=False)
    (RESULTS_DIR / "mmseqs_parametros.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Grupos salvos em %s.", RESULTS_DIR / "grupos_similaridade.csv")
