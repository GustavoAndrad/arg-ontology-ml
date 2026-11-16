"""
nearest_neighbor.py — Gene mais parecido do treino, cenário "gene novo" e referência por homologia.

Por que (TCC2, ponto 4; apontamentos D2 do Diego e K2 da Kele): o rótulo (mecanismo)
e as 41 features ontológicas de um gene saem da mesma ficha do CARD. Um gene novo,
ainda não anotado, não teria essa ficha. Na prática, um gene novo é anotado pela
semelhança com genes conhecidos (é o que faz o RGI do CARD). Este módulo reproduz
isso de forma simples:

* **Gene mais parecido** (``NeighborIndex.nearest``): para cada gene consultado, o
  acerto da busca do MMseqs2 com **menor e-value**, desempatado pelo **maior bit
  score** (e depois pelo índice do gene, para ser determinístico), procurado **só
  entre os genes de referência** (o treino daquela divisão). Menor e-value é o
  critério de Yagimoto et al. (2024) para a referência por BLAST.
* **Cenário "gene novo"** (``new_gene_features``): o gene de teste fica com as
  suas 73 features de sequência e recebe as 41 features ontológicas **da ficha do
  gene mais parecido do treino**. O modelo B é o mesmo, treinado como sempre.
* **Referência por homologia** (``homology_prediction``): copia o mecanismo do
  gene mais parecido, sem aprendizado de máquina.

Busca (decisão 3 do ponto 4, com a correção P4-3b): o mesmo programa e os mesmos
parâmetros da busca do ponto 3 (MMseqs2, sensibilidade 7,5, e-value ≤ 10⁻³, sem
filtro de cobertura), de todos contra todos os genes do experimento, mas **sem piso
de identidade** (``--min-seq-id 0``), em ``data/processed/mmseqs/vizinho_sem_piso/``.
Por que sem piso: o piso de 40% da busca do ponto 3 existe para **formar os grupos**
(o menor limiar de agrupamento é 40%), não para transferir anotação; o critério da
decisão 3 (Yagimoto et al., 2024: mecanismo do acerto de menor e-value) não tem piso
de identidade, só o corte de e-value. Com o piso, 58 dos 1.051 genes de teste ficavam
sem vizinho; sem ele, 13 — e os 45 que passam a ter vizinho (identidade mediana de
34%) têm todos o mesmo mecanismo dele. O piso prejudicava artificialmente a referência
por homologia e o B "gene novo". O piso de 40% (``hits.m8`` do ponto 3) continua
calculado como sensibilidade (``FLOOR_40_HITS``). Os grupos e a partição do ponto 3
não mudam.

Em cada divisão, ficam só os acertos cujo alvo está no treino daquela divisão. Para
um mesmo gene consultado, a ordem dos acertos por e-value não depende de quais outros
genes estão no banco (o e-value é proporcional a 2^(−bit score) vezes o tamanho do
banco, o mesmo para todos os acertos de uma consulta; referência a adicionar: Karlin
e Altschul); por isso filtrar dá o mesmo vizinho que uma busca feita só no treino. A
única diferença possível é na borda do corte de e-value (o banco do treino é menor, e
um acerto com e-value um pouco acima de 10⁻³ no banco todo poderia passar no treino).
Não há filtro de cobertura: a cobertura mínima de 50% é a regra para formar os grupos
(ponto 3), não para transferir anotação (com ela e o piso de 40%, os genes de teste sem
vizinho passavam de 58 para 60).

Gene sem nenhum parecido no treino (nenhum acerto com e-value ≤ 10⁻³): não há de onde
copiar a anotação. Regra adotada (registrada como P4-1, que substitui o cuidado 3 da
decisão de 11/11, "features zeradas"): **o gene é classificado só pela sequência,
com a predição do Pipeline A**. Motivos: nenhum gene do treino tem as 41 features
zeradas (a profundidade mínima é 3 e todos têm ao menos uma classe de antibiótico),
então zerar daria ao modelo uma entrada que ele nunca viu; e, no uso real, um gene
sem parente conhecido não tem anotação, só sequência. A regra antiga (zerar) é
calculada também, como sensibilidade descritiva (``zero_fallback=True``). Na
referência por homologia, que não tem modelo, o gene sem parecido recebe a classe
mais comum do treino daquela divisão.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.homology_groups import WORK_DIR, load_hits

NO_NEIGHBOR: int = -1


class NeighborIndex:
    """
    Acertos da busca em índices inteiros (posições no dataset do experimento), já
    ordenados por consulta, e-value crescente e bit score decrescente.
    """

    def __init__(self, hits: pd.DataFrame, ids) -> None:
        ids = pd.Index(np.asarray(ids))
        if not ids.is_unique:
            raise ValueError("Identificadores repetidos no dataset.")
        q_all, t_all = ids.get_indexer(hits["query"]), ids.get_indexer(hits["target"])
        inside = (q_all >= 0) & (t_all >= 0)  # só genes do experimento
        h = hits[inside]
        q, t = q_all[inside], t_all[inside]
        keep = q != t
        frame = pd.DataFrame(
            {"q": q[keep], "t": t[keep], "evalue": h["evalue"].to_numpy()[keep],
             "bits": h["bits"].to_numpy()[keep], "fident": h["fident"].to_numpy()[keep]}
        ).sort_values(["q", "evalue", "bits", "t"], ascending=[True, True, False, True], kind="mergesort")
        self.n = len(ids)
        self.q = frame["q"].to_numpy()
        self.t = frame["t"].to_numpy()
        self.evalue = frame["evalue"].to_numpy()
        self.bits = frame["bits"].to_numpy()
        self.fident = frame["fident"].to_numpy()

    @classmethod
    def from_files(cls, ids, path: Path | None = None) -> "NeighborIndex":
        """Padrão: a busca sem piso (``NEIGHBOR_HITS``); ``FLOOR_40_HITS`` para a sensibilidade."""
        return cls(load_hits(path or NEIGHBOR_HITS), ids)

    def nearest(self, query_idx, reference_idx) -> tuple[np.ndarray, np.ndarray]:
        """
        Para cada posição em ``query_idx``: o índice do gene mais parecido entre
        ``reference_idx`` (``NO_NEIGHBOR`` se não houver acerto) e a identidade do
        alinhamento (0 se não houver). Consulta e referência devem ser disjuntas.
        """
        query_idx, reference_idx = np.asarray(query_idx), np.asarray(reference_idx)
        if np.intersect1d(query_idx, reference_idx).size:
            raise ValueError("Consulta e referência se sobrepõem: o vizinho tem de vir só do treino.")
        in_q = np.zeros(self.n, bool)
        in_r = np.zeros(self.n, bool)
        in_q[query_idx] = True
        in_r[reference_idx] = True
        sel = np.flatnonzero(in_q[self.q] & in_r[self.t])
        # A lista está ordenada por consulta e, dentro dela, do melhor para o pior acerto:
        # a primeira ocorrência de cada consulta é o gene mais parecido.
        first_q, first_pos = np.unique(self.q[sel], return_index=True)
        best = sel[first_pos]
        nb = np.full(self.n, NO_NEIGHBOR)
        ident = np.zeros(self.n)
        nb[first_q] = self.t[best]
        ident[first_q] = self.fident[best]
        return nb[query_idx], ident[query_idx]


def new_gene_features(
    X_b: np.ndarray, query_idx, neighbors: np.ndarray, onto_cols: np.ndarray, zero_fallback: bool = False
) -> tuple[np.ndarray, np.ndarray]:
    """
    Matriz do Pipeline B no cenário "gene novo" para os genes ``query_idx``.

    Sequência: a do próprio gene. Ontologia: a do gene mais parecido do treino.
    Devolve (matriz, máscara de quem tem vizinho). Sem vizinho: as features
    ontológicas ficam como estão (a linha não deve ser usada; quem chama usa a
    predição do Pipeline A), ou zeradas se ``zero_fallback`` (regra antiga, só
    como sensibilidade).
    """
    query_idx = np.asarray(query_idx)
    X = X_b[query_idx].copy()
    has = neighbors != NO_NEIGHBOR
    X[np.ix_(has, onto_cols)] = X_b[np.ix_(neighbors[has], onto_cols)]
    if zero_fallback:
        X[np.ix_(~has, onto_cols)] = 0.0
    else:
        X[np.ix_(~has, onto_cols)] = np.nan  # garante que essas linhas não sejam usadas por engano
    return X, has


def predict_new_gene(pipe_b, pipe_a, X_a, X_b, query_idx, neighbors, onto_cols) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Predições do Pipeline B no cenário "gene novo", com as duas regras para quem não
    tem vizinho: ``b_gene_novo`` (predição do A; regra adotada) e
    ``b_gene_novo_zeros`` (features zeradas; sensibilidade). Cada valor é (pred, proba).
    """
    query_idx = np.asarray(query_idx)
    X_new, has = new_gene_features(X_b, query_idx, neighbors, onto_cols)
    X_zero, _ = new_gene_features(X_b, query_idx, neighbors, onto_cols, zero_fallback=True)
    n_classes = len(pipe_b.classes_)
    pred = np.empty(len(query_idx), dtype=int)
    proba = np.empty((len(query_idx), n_classes))
    if has.any():
        pred[has], proba[has] = pipe_b.predict(X_new[has]), pipe_b.predict_proba(X_new[has])
    if (~has).any():
        Xa = X_a[query_idx[~has]]
        pred[~has], proba[~has] = pipe_a.predict(Xa), pipe_a.predict_proba(Xa)
    return {
        "b_gene_novo": (pred, proba),
        "b_gene_novo_zeros": (pipe_b.predict(X_zero), pipe_b.predict_proba(X_zero)),
    }


def homology_prediction(y: np.ndarray, reference_idx, neighbors: np.ndarray, n_classes: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Referência por homologia: o mecanismo do gene mais parecido do treino; sem
    vizinho, a classe mais comum do treino. "Probabilidade" = 1 na classe prevista
    (é uma regra, não um modelo; a ROC-AUC e a PR-AUC dela tratam a predição como
    uma pontuação de 0 ou 1).
    """
    majority = int(np.bincount(y[np.asarray(reference_idx)], minlength=n_classes).argmax())
    pred = np.where(neighbors != NO_NEIGHBOR, y[np.where(neighbors != NO_NEIGHBOR, neighbors, 0)], majority)
    proba = np.eye(n_classes)[pred]
    return pred, proba


# ── Busca usada para achar o vizinho ─────────────────────────────────────────────
NEIGHBOR_DIR: Path = WORK_DIR / "vizinho_sem_piso"
NEIGHBOR_HITS: Path = NEIGHBOR_DIR / "hits.m8"  # principal: sem piso de identidade (P4-3b)
FLOOR_40_HITS: Path = WORK_DIR / "hits.m8"  # busca do ponto 3 (piso de 40%): sensibilidade


def run_neighbor_search(data: pd.DataFrame, work_dir: Path = NEIGHBOR_DIR) -> dict:
    """
    ``mmseqs easy-search`` de todos contra todos os genes de ``data``, **sem piso de
    identidade** e com os demais parâmetros da busca do ponto 3. Grava ``hits.m8`` e
    devolve os parâmetros (salvos em ``results/mmseqs_vizinho_parametros.json``).
    """
    import os
    import subprocess
    import time

    from src.homology_groups import EVALUE, HIT_COLUMNS, MMSEQS_BIN, MMSEQS_VERSION, SENSITIVITY, write_fasta

    if not MMSEQS_BIN.exists():
        raise FileNotFoundError(f"MMseqs2 não encontrado em {MMSEQS_BIN}. Veja o README (Instalação).")
    work_dir.mkdir(parents=True, exist_ok=True)
    write_fasta(data, work_dir / "experimento.fasta")
    args = [
        "easy-search", "experimento.fasta", "experimento.fasta", "hits.m8", "tmp",
        "--min-seq-id", "0", "-c", "0", "--cov-mode", "0", "-s", str(SENSITIVITY), "-e", str(EVALUE),
        "--max-seqs", "10000", "--alignment-mode", "3", "--format-output", ",".join(HIT_COLUMNS),
        "--threads", str(max(1, (os.cpu_count() or 2) - 1)),
    ]
    env = dict(os.environ, PATH=str(MMSEQS_BIN.parent) + os.pathsep + os.environ.get("PATH", ""))
    t0 = time.time()
    proc = subprocess.run([str(MMSEQS_BIN), *args], cwd=work_dir, env=env, capture_output=True, text=True)
    if proc.returncode != 0 or not (work_dir / "hits.m8").exists():
        raise RuntimeError(f"MMseqs2 falhou (código {proc.returncode}):\n{proc.stderr[-3000:]}")
    hits = load_hits(work_dir / "hits.m8")
    return {
        "programa": "MMseqs2",
        "versao_release": MMSEQS_VERSION,
        "comando": "mmseqs " + " ".join(args),
        "uso": "gene mais parecido do treino (cenário gene novo e referência por homologia, ponto 4)",
        "diferenca_para_a_busca_do_ponto_3": "sem piso de identidade (--min-seq-id 0); demais parâmetros iguais",
        "genes_buscados": int(len(data)),
        "pares_encontrados_sem_autopares": int(len(hits)),
        "pares_com_identidade_abaixo_de_40": int((hits["fident"] < 0.4).sum()),
        "tempo_s": round(time.time() - t0, 1),
    }


if __name__ == "__main__":
    import json
    import logging
    import sys

    from src.experiment_config import RESULTS_DIR, load_experiment_dataset

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    params = run_neighbor_search(load_experiment_dataset())
    (RESULTS_DIR / "mmseqs_vizinho_parametros.json").write_text(json.dumps(params, indent=2, ensure_ascii=False), encoding="utf-8")
    logging.getLogger(__name__).info("Busca sem piso: %s", params)
