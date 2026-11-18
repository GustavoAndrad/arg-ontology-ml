"""
data_characterization.py — Caracterização dos dados e dos resultados que o texto cita.

Por que este módulo existe: a revisão geral do experimento (17–18/11/2026) encontrou
números importantes para as conclusões que vinham de cálculos avulsos, sem arquivo em
``results/``. Este módulo os gera a partir dos dados e dos resultados já salvos, sem
treinar nenhum modelo, e grava ao lado de cada tabela **o que ela mostra, como foi
calculada, por que existe e de onde vem** (``results/caracterizacao_metadados.json``).

O que é gerado (cada item responde a um achado da revisão, citado entre parênteses):

1. ``caracterizacao_familias.csv`` (R2): cada família AMR do experimento, com genes,
   mecanismos e número de grupos de similaridade. Mostra que no CARD a família define
   o mecanismo.
2. ``familias_treino_teste.csv`` (R2): por classe, quantos genes de teste são de uma
   família que também tem genes no treino. Mostra o que o teste mede em cada classe.
3. ``vetores_ontologicos.csv`` (R3): os vetores distintos dos 41 atributos da ARO, a
   classe de cada um e a pureza. Mostra quanto a ontologia, sozinha, já separa as classes.
4. ``categoria_isa_por_mecanismo.csv`` (R8): a categoria de mecanismo que a ARO liga
   (``participates_in``) a algum ancestral is-a do gene, cruzada com o rótulo.
5. ``regra_simples_ontologia.csv`` (R8): uma regra sem aprendizado de máquina
   ("vetor visto no treino → classe mais comum dele"), treino → teste, com cada
   grupo de atributos da ARO. Mede quanto cada grupo revela da classe.
6. ``ganho_por_classe_cv.csv`` (R19): B − A por classe na CV, pareado nos 50 folds, com
   o IC corrigido de Nadeau e Bengio, nos dois cenários confirmatórios.
7. ``recortes_teste.csv`` (pontos 4 e 5, R12): acurácia no teste por recorte (com
   parente no treino, distantes sem parente ≥ 40%, sem nenhum parente), por pipeline.
8. ``erros_por_classe.csv`` (pontos 5, 9 e 10): a matriz de confusão do teste em formato
   longo (pipeline, modelo, verdadeiro, previsto, genes), e ``erros_principais.csv``,
   com a soma dos erros entre inativação e alteração do alvo e o maior erro de cada modelo.
9. ``atributos_por_mecanismo.csv`` (ponto 8, R8): a fração de genes de cada mecanismo com
   cada classe de antibiótico, a distribuição da profundidade na ARO por mecanismo e a
   mediana da cisteína (os "fatos dos dados" da interpretação).

Tudo é descritivo. Nenhum resultado do experimento é alterado.

Entradas: dados do experimento, atributos (``data/processed``), partição salva,
``results/cv_por_classe.csv``, ``results/cv_folds.csv``, ``results/predicoes/*.npz``,
``results/vizinho_mais_parecido_teste.csv``, ``results/grupos_similaridade.csv`` e o ``aro.owl``.

Uso (depois de ``src.train_evaluate``; segundos):
    python -m src.data_characterization
"""

from __future__ import annotations

import json
import logging
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from src.config import ONTOLOGY_PATH
from src.experiment_config import (
    MODEL_NAMES,
    PROCESSED_DIR,
    RESULTS_DIR,
    TARGET,
    load_experiment_dataset,
    load_integrated_dataset,
)

logger = logging.getLogger(__name__)

PREDICTIONS_DIR = RESULTS_DIR / "predicoes"
STRUCTURAL = ["depth_isa", "ancestor_count", "children_count"]
MECHANISM_ROOT = "ARO:1000002"  # "mechanism of antibiotic resistance"
PARTICIPATES_IN = "http://purl.obolibrary.org/obo/RO_0000056"
INACTIVATION, ALTERATION = "antibiotic inactivation", "antibiotic target alteration"


# ── R2: família → mecanismo ──────────────────────────────────────────────────────
def split_families(value: object) -> set[str]:
    """Famílias de um gene (a coluna ``AMR Gene Family`` pode ter nomes compostos com ';')."""
    if not isinstance(value, str):
        return set()
    return {f.strip() for f in value.split(";") if f.strip()}


def family_table(data: pd.DataFrame, groups: np.ndarray) -> pd.DataFrame:
    """Uma linha por família: genes, mecanismos (e quantos) e grupos de similaridade de 70%."""
    rows: dict[str, dict] = {}
    for fams, mech, g in zip(data["AMR_Gene_Family"].map(split_families), data[TARGET], groups):
        for f in fams:
            r = rows.setdefault(f, {"familia": f, "genes": 0, "mecanismos": set(), "grupos": set()})
            r["genes"] += 1
            r["mecanismos"].add(mech)
            r["grupos"].add(int(g))
    out = pd.DataFrame(rows.values())
    out["n_mecanismos"] = out["mecanismos"].map(len)
    out["n_grupos_70"] = out["grupos"].map(len)
    out["mecanismos"] = out["mecanismos"].map(lambda s: "; ".join(sorted(s)))
    return out.drop(columns="grupos").sort_values(["genes", "familia"], ascending=[False, True]).reset_index(drop=True)


def families_mixing_mechanisms(integrated: pd.DataFrame) -> dict:
    """
    No dataset integrado: famílias com mais de um mecanismo, separando as que só misturam
    por causa dos genes multirrótulo (rótulo com ';', que trazem o mecanismo da família e
    mais um) das que misturam entre genes de rótulo único.
    """
    fams = integrated["AMR_Gene_Family"].map(split_families)
    labels = integrated[TARGET].astype(str)
    single = ~labels.str.contains(";", regex=False)
    mixed_all, mixed_single = [], []
    for f in sorted(set().union(*fams)):
        has = fams.map(lambda s: f in s)
        if labels[has].nunique() > 1:
            mixed_all.append(f)
        if labels[has & single].nunique() > 1:
            mixed_single.append(f)
    return {
        "familias_no_integrado": len(set().union(*fams)),
        "familias_com_mais_de_um_rotulo": len(mixed_all),
        "familias_com_mais_de_um_mecanismo_entre_genes_de_rotulo_unico": len(mixed_single),
        "familias_que_misturam_so_por_genes_multirrotulo": sorted(set(mixed_all) - set(mixed_single)),
    }


def family_overlap_by_class(data: pd.DataFrame, train_idx: np.ndarray, test_idx: np.ndarray) -> pd.DataFrame:
    """
    Por classe: genes de teste e quantos têm **alguma** família com genes no treino.

    "Alguma" porque um gene pode ter mais de uma família; basta uma estar no treino para
    o modelo ter visto parentes de família daquele gene.
    """
    fams = data["AMR_Gene_Family"].map(split_families).to_numpy()
    train_fams = set().union(*fams[train_idx])
    y = data[TARGET].to_numpy()
    rows = []
    for c in sorted(set(y)):
        idx = test_idx[y[test_idx] == c]
        seen = sum(bool(fams[i] & train_fams) for i in idx)
        test_fams = set().union(*fams[idx]) if len(idx) else set()
        rows.append({
            "mecanismo": c, "genes_teste": len(idx), "genes_teste_familia_no_treino": int(seen),
            "fracao_familia_no_treino": seen / len(idx) if len(idx) else np.nan,
            "familias_no_teste": len(test_fams), "familias_no_teste_ausentes_do_treino": len(test_fams - train_fams),
        })
    return pd.DataFrame(rows)


# ── R3: vetores ontológicos ──────────────────────────────────────────────────────
def ontology_vectors(onto: pd.DataFrame, y: np.ndarray) -> tuple[pd.DataFrame, dict]:
    """
    Vetores distintos dos atributos da ARO: genes, classe majoritária e pureza de cada um.

    A pureza global (fração de genes na classe majoritária do seu vetor) é calculada no
    próprio conjunto, portanto é otimista: diz quanto a ontologia separa as classes, não
    quanto um modelo acertaria em genes novos (isso está em ``regra_simples_ontologia.csv``).
    """
    key = onto.astype(str).agg("|".join, axis=1).to_numpy()
    df = pd.DataFrame({"vetor": key, "classe": y})
    counts = df.groupby(["vetor", "classe"]).size().unstack(fill_value=0)
    table = pd.DataFrame({
        "genes": counts.sum(axis=1), "classe_majoritaria": counts.idxmax(axis=1),
        "genes_na_majoritaria": counts.max(axis=1), "n_classes": (counts > 0).sum(axis=1),
    })
    table["pureza"] = table["genes_na_majoritaria"] / table["genes"]
    for c in counts.columns:
        table[f"genes_{c.replace('antibiotic ', '').replace(' ', '_')}"] = counts[c]
    table = table.sort_values("genes", ascending=False).reset_index()
    table.insert(0, "id_vetor", range(len(table)))
    ambiguous = table[table["n_classes"] > 1]
    summary = {
        "vetores_distintos": int(len(table)),
        "pureza_global": float(table["genes_na_majoritaria"].sum() / table["genes"].sum()),
        "vetores_com_mais_de_uma_classe": int(len(ambiguous)),
        "genes_em_vetores_com_mais_de_uma_classe": int(ambiguous["genes"].sum()),
    }
    return table.drop(columns="vetor"), summary


# ── R8: a hierarquia is-a e as categorias de mecanismo ───────────────────────────
def mechanism_category_links(owl_path: Path) -> tuple[dict[str, str], dict[str, str]]:
    """
    Termos da ARO ligados por ``participates_in`` a um mecanismo (subclasse direta de
    ARO:1000002), e os rótulos dos termos. Lê o OWL/XML com o mesmo formato de
    ``ontology_features``.
    """
    from src.ontology_features import OWL_NS, iri_to_accession, load_ontology

    mechanisms = set(load_ontology(owl_path).children.get(MECHANISM_ROOT, ()))
    links, labels = {}, {}
    for _, el in ET.iterparse(str(owl_path), events=("end",)):
        if el.tag == OWL_NS + "SubClassOf":
            kids = list(el)
            if len(kids) == 2 and kids[1].tag == OWL_NS + "ObjectSomeValuesFrom":
                prop, filler = list(kids[1])[0], list(kids[1])[-1]
                target = iri_to_accession(filler.get("IRI"))
                if prop.get("IRI") == PARTICIPATES_IN and target in mechanisms:
                    links[iri_to_accession(kids[0].get("IRI"))] = target
            el.clear()
        elif el.tag == OWL_NS + "AnnotationAssertion":
            kids = list(el)
            if len(kids) == 3 and kids[0].get("abbreviatedIRI") == "rdfs:label":
                acc = iri_to_accession(kids[1].text)
                if acc and kids[2].text:
                    labels[acc] = kids[2].text.strip()
            el.clear()
    return links, labels


def ancestors(graph, accession: str) -> set[str]:
    seen, stack = set(), list(graph.parents.get(accession, ()))
    while stack:
        n = stack.pop()
        if n not in seen:
            seen.add(n)
            stack.extend(graph.parents.get(n, ()))
    return seen


def isa_category_by_mechanism(data: pd.DataFrame, owl_path: Path) -> tuple[pd.DataFrame, dict]:
    """Rótulo × mecanismo da categoria encontrada entre os ancestrais is-a de cada gene."""
    from src.ontology_features import load_ontology

    graph = load_ontology(owl_path)
    links, labels = mechanism_category_links(owl_path)
    cats = []
    for acc in data["ARO_Accession"]:
        found = sorted({labels.get(links[a], links[a]) for a in ancestors(graph, acc) if a in links})
        cats.append("; ".join(found) if found else "(nenhuma)")
    table = pd.crosstab(data[TARGET].rename("mecanismo"), pd.Series(cats, name="categoria_isa", index=data.index))
    with_cat = np.array([c != "(nenhuma)" for c in cats])
    agree = np.array([c == m for c, m in zip(cats, data[TARGET])])
    summary = {
        "termos_ligados_a_mecanismo_por_participates_in": {labels.get(k, k): labels.get(v, v) for k, v in sorted(links.items())},
        "genes_com_categoria_entre_os_ancestrais": int(with_cat.sum()),
        "genes_em_que_a_categoria_coincide_com_o_rotulo": int(agree.sum()),
        "genes_sem_categoria_por_mecanismo": data[TARGET][~with_cat].value_counts().to_dict(),
    }
    return table.reset_index(), summary


def lookup_rule(onto: pd.DataFrame, y: np.ndarray, train_idx, test_idx, columns: list[str]) -> dict:
    """
    Regra sem aprendizado de máquina: cada combinação de valores vista no treino prevê a
    classe mais comum dela; combinação nova prevê a classe mais comum do treino.
    """
    key = onto[columns].astype(str).agg("|".join, axis=1).to_numpy()
    table = pd.Series(y[train_idx]).groupby(key[train_idx]).agg(lambda s: s.value_counts().index[0])
    majority = pd.Series(y[train_idx]).value_counts().index[0]
    pred = np.array([table.get(k, majority) for k in key[test_idx]])
    labels = sorted(set(y))
    return {
        "f1_macro_teste": float(f1_score(y[test_idx], pred, labels=labels, average="macro", zero_division=0)),
        "acuracia_teste": float((pred == y[test_idx]).mean()),
        "combinacoes_no_treino": int(len(table)),
        "genes_teste_com_combinacao_nao_vista": int(sum(k not in table.index for k in key[test_idx])),
    }


# ── R19: ganho por classe na CV ──────────────────────────────────────────────────
def gain_per_class_cv(cv_per_class: pd.DataFrame, ratio: float) -> pd.DataFrame:
    """B − A no F1 de cada classe, pareado por (repetição, fold), com o IC corrigido."""
    from src.statistical_tests import corrected_interval

    rows = []
    for scenario, key in (("anotacao_curada", "b"), ("gene_novo", "b_gene_novo")):
        for model in MODEL_NAMES:
            for c in sorted(cv_per_class["classe"].unique()):
                sel = lambda p: cv_per_class[(cv_per_class.pipeline == p) & (cv_per_class.modelo == model)
                                             & (cv_per_class.classe == c)].sort_values(["repeticao", "fold"])
                a, b = sel("a"), sel(key)
                if not (a[["repeticao", "fold"]].to_numpy() == b[["repeticao", "fold"]].to_numpy()).all():
                    raise ValueError("Folds de A e B não pareados.")
                d = b["f1"].to_numpy() - a["f1"].to_numpy()
                lo, hi = corrected_interval(d, ratio)
                rows.append({"cenario": scenario, "modelo": model, "classe": c, "f1_a_cv": a["f1"].mean(),
                             "f1_b_cv": b["f1"].mean(), "dif_cv": d.mean(), "ic95_inf": lo, "ic95_sup": hi,
                             "folds_b_maior": int((d > 0).sum()), "folds_a_maior": int((d < 0).sum()),
                             "ic_acima_de_zero": bool(lo > 0)})
    return pd.DataFrame(rows)


# ── Recortes do teste e erros ────────────────────────────────────────────────────
def _test_predictions(classes: list[str]) -> dict[tuple[str, str], np.ndarray]:
    preds = {}
    for key in ("a", "b", "b_gene_novo"):
        for model in MODEL_NAMES:
            preds[(key, model)] = np.load(PREDICTIONS_DIR / f"pipeline_{key}_{model}.npz")
    preds[("homologia", "homologia")] = np.load(PREDICTIONS_DIR / "referencia_homologia.npz")
    return preds


def test_subsets(neighbors: pd.DataFrame, preds: dict, classes: list[str]) -> pd.DataFrame:
    """Acurácia por recorte do teste. A acurácia (e não o F1-macro) porque os recortes pequenos têm classes com 1 gene (R12)."""
    masks = {
        "com_parente_no_treino": neighbors["tem_vizinho"].to_numpy(bool),
        "distantes_sem_parente_40": ~neighbors["tem_vizinho_piso40"].to_numpy(bool),
        "sem_nenhum_parente": ~neighbors["tem_vizinho"].to_numpy(bool),
    }
    rows = []
    for name, m in masks.items():
        y_true = next(iter(preds.values()))["y_true"][m]
        comp = "; ".join(f"{classes[k].replace('antibiotic ', '')} {n}" for k, n in enumerate(np.bincount(y_true, minlength=len(classes))) if n)
        for (key, model), z in preds.items():
            rows.append({"recorte": name, "genes": int(m.sum()), "composicao": comp, "pipeline": key, "modelo": model,
                         "acuracia": float((z["y_pred"][m] == z["y_true"][m]).mean()) if m.any() else np.nan})
    return pd.DataFrame(rows)


def confusion_long(preds: dict, classes: list[str]) -> pd.DataFrame:
    rows = []
    for (key, model), z in preds.items():
        cm = z["confusion_matrix"]
        for i, t in enumerate(classes):
            for j, p in enumerate(classes):
                rows.append({"pipeline": key, "modelo": model, "verdadeiro": t, "previsto": p, "genes": int(cm[i, j])})
    return pd.DataFrame(rows)


def main_errors(conf: pd.DataFrame) -> pd.DataFrame:
    """Por pipeline e modelo: erros entre inativação e alteração (os dois sentidos) e o maior erro."""
    rows = []
    for (key, model), sub in conf.groupby(["pipeline", "modelo"], sort=False):
        off = sub[sub.verdadeiro != sub.previsto]
        pair = off[((off.verdadeiro == INACTIVATION) & (off.previsto == ALTERATION))
                   | ((off.verdadeiro == ALTERATION) & (off.previsto == INACTIVATION))]["genes"].sum()
        top = off.sort_values("genes", ascending=False).iloc[0]
        rows.append({"pipeline": key, "modelo": model, "erros_totais": int(off["genes"].sum()),
                     "erros_inativacao_alteracao_dois_sentidos": int(pair),
                     "maior_erro": f"{top.verdadeiro} -> {top.previsto}", "genes_no_maior_erro": int(top.genes)})
    return pd.DataFrame(rows)


# ── Fatos dos atributos por mecanismo ────────────────────────────────────────────
def attributes_by_mechanism(onto: pd.DataFrame, seq: pd.DataFrame, y: np.ndarray) -> pd.DataFrame:
    """Fração com cada classe de antibiótico, distribuição da profundidade e mediana da cisteína, por mecanismo."""
    rows = []
    s = pd.Series(y)
    for col in [c for c in onto.columns if c.startswith("drug_class_")]:
        frac = onto[col].groupby(s.values).mean()
        rows += [{"atributo": col, "estatistica": "fracao_dos_genes_do_mecanismo_com_valor_1", "mecanismo": m, "valor": float(v)}
                 for m, v in frac.items()]
    for d, sub in onto.groupby("depth_isa"):
        counts = s[sub.index].value_counts()
        rows += [{"atributo": f"depth_isa={d}", "estatistica": "genes", "mecanismo": m, "valor": float(counts.get(m, 0))}
                 for m in sorted(set(y))]
    med = seq["aa_comp_C"].groupby(s.values).median()
    rows += [{"atributo": "aa_comp_C", "estatistica": "mediana", "mecanismo": m, "valor": float(v)} for m, v in med.items()]
    return pd.DataFrame(rows)


# ── Parentesco teste → treino em cada divisão (para a figura da motivação da 4.8) ──
SPLIT_SCENARIOS: list[tuple[str, int | None]] = [("aleatoria", None), ("90", 90), ("70", 70), ("50", 50)]


def identity_by_split(data: pd.DataFrame, y_codes: np.ndarray) -> pd.DataFrame:
    """
    Para cada divisão (aleatória e grupos de 90, 70 e 50%), a maior identidade de cada gene
    de teste com algum gene de treino (busca dos grupos, piso de 40%, cobertura ≥ 0,5).

    As partições são as mesmas da curva por limiar (``choose_test_partition`` com os
    grupos de cada limiar), então a de 70% é a partição do experimento. Identidade 0 =
    nenhum par com identidade ≥ 40% e cobertura ≥ 0,5.
    """
    from src.experiment_config import choose_test_partition, load_groups
    from src.homology_groups import WORK_DIR, load_hits, nearest_identity

    hits = load_hits(WORK_DIR / "hits.m8")
    ids = data["ARO_Accession"].to_numpy()
    rows = []
    for name, threshold in SPLIT_SCENARIOS:
        tr, te, _ = choose_test_partition(y_codes, load_groups(threshold, data=data))
        ident = nearest_identity(hits, ids[te], ids[tr], min_coverage=0.5).to_numpy()
        rows.append(pd.DataFrame({"cenario": name, "ARO_Accession": ids[te], "mecanismo": data[TARGET].to_numpy()[te],
                                  "identidade_max_com_treino": ident}))
    return pd.concat(rows, ignore_index=True)


# ── Ramos reais da ARO (para a figura da hierarquia, seção 2.5.1) ─────────────────
EXAMPLE_FAMILIES: dict[str, str] = {
    "antibiotic inactivation": "TEM beta-lactamase",
    "antibiotic target alteration": "glycopeptide resistance gene cluster",
    "antibiotic efflux": "resistance-nodulation-cell division (RND) antibiotic efflux pump",
    "antibiotic target protection": "quinolone resistance protein (qnr)",
    "antibiotic target replacement": "trimethoprim resistant dihydrofolate reductase dfr",
}


# Genes clássicos de cada família, para o exemplo ser reconhecível pelo leitor.
EXAMPLE_GENES: dict[str, str] = {
    "antibiotic inactivation": "TEM-1", "antibiotic target alteration": "vanA", "antibiotic efflux": "acrB",
    "antibiotic target protection": "QnrA1", "antibiotic target replacement": "dfrA1",
}


def aro_branches(data: pd.DataFrame, owl_path: Path) -> pd.DataFrame:
    """
    Caminho is-a mais curto até a raiz de um gene de exemplo por mecanismo (o primeiro
    gene de uma família conhecida), com os rótulos e a marca dos termos que a ARO liga a
    um mecanismo por ``participates_in``. Mostra, com dados reais, a organização da
    hierarquia por categoria de mecanismo (R8).
    """
    from src.ontology_features import load_ontology

    graph = load_ontology(owl_path)
    depth = graph.depths()
    links, labels = mechanism_category_links(owl_path)
    rows = []
    for mech, fam in EXAMPLE_FAMILIES.items():
        sub = data[(data[TARGET] == mech) & data["AMR_Gene_Family"].map(lambda v: fam in split_families(v))]
        if sub.empty:
            continue
        names = sub["ARO_Accession"].map(lambda a: labels.get(a, a))
        preferred = names[names == EXAMPLE_GENES[mech]]
        gene = (preferred.index[0] if len(preferred) else sub.index[0])
        gene = sub.loc[gene, "ARO_Accession"]
        path, node = [gene], gene
        while graph.parents.get(node):
            node = min(graph.parents[node], key=lambda p: (depth.get(p, 10**6), p))
            path.append(node)
        for level, term in enumerate(reversed(path)):
            rows.append({"mecanismo": mech, "familia": fam, "gene": gene, "nivel": level, "termo": term,
                         "rotulo": labels.get(term, term), "categoria_de_mecanismo": term in links,
                         "mecanismo_ligado": labels.get(links[term], links[term]) if term in links else ""})
    return pd.DataFrame(rows)


# ── Execução ─────────────────────────────────────────────────────────────────────
def run(owl_path: Path = ONTOLOGY_PATH) -> dict:
    from src.experiment_config import load_groups
    from src.statistical_tests import test_train_ratio

    data = load_experiment_dataset()
    y = data[TARGET].to_numpy()
    classes = json.loads((RESULTS_DIR / "classes.json").read_text(encoding="utf-8"))
    onto = pd.read_csv(PROCESSED_DIR / "features_ontologicas.csv").drop(columns="ARO_Accession")
    seq = pd.read_csv(PROCESSED_DIR / "features_sequencia.csv").drop(columns="ARO_Accession")
    train_idx, test_idx = np.load(PROCESSED_DIR / "train_indices.npy"), np.load(PROCESSED_DIR / "test_indices.npy")
    groups = load_groups(data=data)
    meta: dict[str, dict] = {}

    def save(name: str, df: pd.DataFrame, entry: dict) -> None:
        df.to_csv(RESULTS_DIR / name, index=False)
        meta[name] = entry
        logger.info("Salvo: %s (%d linhas)", name, len(df))

    fam = family_table(data, groups)
    save("caracterizacao_familias.csv", fam, {
        "o_que_mostra": "Cada família AMR do experimento: genes, mecanismos, número de mecanismos e número de grupos de similaridade de 70% que a família ocupa.",
        "como": "Coluna AMR_Gene_Family do aro_index (nomes compostos separados por ';'); grupos de results/grupos_similaridade.csv.",
        "por_que": "R2 da revisão geral: se cada família tem um único mecanismo, classificar o mecanismo equivale a reconhecer a família, o que explica a força da referência por homologia. E, se uma família ocupa vários grupos, a divisão por grupos pode pôr membros dela no treino e no teste.",
        "fontes": ["data/processed/dataset_integrado.csv", "results/grupos_similaridade.csv"],
        "numeros": {"familias_no_experimento": int(len(fam)),
                    "familias_com_mais_de_um_mecanismo_no_experimento": int((fam.n_mecanismos > 1).sum()),
                    "familias_em_mais_de_um_grupo_de_70": int((fam.n_grupos_70 > 1).sum()),
                    "dataset_integrado": families_mixing_mechanisms(load_integrated_dataset())},
        "observacao": "Contagem com os nomes compostos separados por ';' (um gene pode ter duas famílias). Sem separar, o experimento tem 388 nomes de família distintos; a revisão de 17/11 misturou os dois critérios ('392 famílias', '87 de 388') e foi corrigida para os valores deste arquivo.",
    })
    ov = family_overlap_by_class(data, train_idx, test_idx)
    small = ov[ov.mecanismo != INACTIVATION]
    save("familias_treino_teste.csv", ov, {
        "o_que_mostra": "Por mecanismo: genes de teste, quantos têm alguma família com genes no treino, e famílias do teste ausentes do treino.",
        "como": "Partição final (train/test_indices.npy); família presente = alguma das famílias do gene aparece em algum gene de treino.",
        "por_que": "R2: diz o que o teste mede em cada classe. Nas classes pequenas, quase todo gene de teste é variante nova de família conhecida; o resultado não se estende a famílias desconhecidas (limitação 12 do consolidado).",
        "fontes": ["data/processed/train_indices.npy", "data/processed/test_indices.npy"],
        "numeros": {"classes_pequenas": f"{int(small.genes_teste_familia_no_treino.sum())} de {int(small.genes_teste.sum())}",
                    "inativacao": f"{int(ov[ov.mecanismo == INACTIVATION].genes_teste_familia_no_treino.iloc[0])} de {int(ov[ov.mecanismo == INACTIVATION].genes_teste.iloc[0])}"},
    })
    vec, vsum = ontology_vectors(onto, y)
    save("vetores_ontologicos.csv", vec, {
        "o_que_mostra": "Os vetores distintos formados pelos 41 atributos da ARO, com genes por classe, classe majoritária e pureza.",
        "como": "Agrupa os 6.033 genes pelo vetor completo de atributos ontológicos; pureza = genes na classe majoritária / genes do vetor. Calculado no próprio conjunto (medida otimista).",
        "por_que": "R3: quantifica quanto a ontologia, sozinha, separa as classes; complementa a informação mútua (que olha um atributo por vez) e mostra onde fica a ambiguidade (proteção, efluxo e inativação).",
        "fontes": ["data/processed/features_ontologicas.csv"],
        "numeros": vsum,
    })
    cat, csum = isa_category_by_mechanism(data, owl_path)
    save("categoria_isa_por_mecanismo.csv", cat, {
        "o_que_mostra": "Rótulo × categoria de mecanismo encontrada entre os ancestrais is-a do gene (categoria = termo que a ARO liga a um mecanismo por participates_in).",
        "como": "Hierarquia is-a do aro.owl (o mesmo leitor dos atributos) e os axiomas SubClassOf com ObjectSomeValuesFrom(participates_in, mecanismo).",
        "por_que": "R8: a is-a da ARO coloca os genes sob categorias com nome de mecanismo. Por isso os atributos estruturais são contagens, e não a identidade dos ancestrais; e a importância da profundidade reflete essa organização curada.",
        "fontes": ["data/ontologia/aro.owl"],
        "numeros": csum,
    })
    rule_rows = []
    for name, cols in (("profundidade", ["depth_isa"]), ("3 estruturais", STRUCTURAL),
                       ("38 classes de antibiotico", [c for c in onto.columns if c.startswith("drug_class_")]),
                       ("41 atributos da ARO", list(onto.columns))):
        rule_rows.append({"atributos": name, **lookup_rule(onto, y, train_idx, test_idx, cols)})
    rule = pd.DataFrame(rule_rows)
    save("regra_simples_ontologia.csv", rule, {
        "o_que_mostra": "F1-macro e acurácia no teste de uma regra sem aprendizado de máquina: combinação de valores vista no treino → classe mais comum dela.",
        "como": "Treino e teste da partição final; combinação não vista no treino prevê a classe mais comum do treino.",
        "por_que": "R8: mede quanto cada grupo de atributos da ARO revela da classe por si só. As estruturais separam basicamente a inativação; as classes de antibiótico carregam a maior parte da informação, coerente com a ablação.",
        "fontes": ["data/processed/features_ontologicas.csv", "data/processed/train_indices.npy", "data/processed/test_indices.npy"],
        "numeros": {r.atributos: round(r.f1_macro_teste, 3) for r in rule.itertuples()},
    })
    gain = gain_per_class_cv(pd.read_csv(RESULTS_DIR / "cv_por_classe.csv"), test_train_ratio(pd.read_csv(RESULTS_DIR / "cv_folds.csv")))
    cur = gain[gain.cenario == "anotacao_curada"]
    save("ganho_por_classe_cv.csv", gain, {
        "o_que_mostra": "B − A no F1 de cada classe na validação cruzada (50 folds pareados), com o IC 95% corrigido de Nadeau e Bengio, nos cenários anotação curada e gene novo.",
        "como": "results/cv_por_classe.csv; razão n_validação/n_ajuste média dos folds; mesmo erro padrão corrigido do teste t do experimento.",
        "por_que": "R19: as conclusões se apoiam na CV, mas o ganho por classe só existia no bootstrap do teste. Descritivo (sem correção para comparações múltiplas; R13).",
        "fontes": ["results/cv_por_classe.csv", "results/cv_folds.csv"],
        "numeros": {c.replace("antibiotic ", ""): f"IC acima de zero em {int(cur[cur.classe == c].ic_acima_de_zero.sum())} de 4 modelos (curado)"
                    for c in sorted(cur.classe.unique())},
    })
    neighbors = pd.read_csv(RESULTS_DIR / "vizinho_mais_parecido_teste.csv")
    preds = _test_predictions(classes)
    if not (neighbors["ARO_Accession"].to_numpy() == data["ARO_Accession"].to_numpy()[test_idx]).all():
        raise ValueError("vizinho_mais_parecido_teste.csv fora da ordem do teste.")
    sub = test_subsets(neighbors, preds, classes)
    save("recortes_teste.csv", sub, {
        "o_que_mostra": "Acurácia no teste por recorte (com parente no treino; distantes, sem parente com identidade ≥ 40%; sem nenhum parente), para A, B, B gene novo e a homologia.",
        "como": "results/vizinho_mais_parecido_teste.csv (tem_vizinho, tem_vizinho_piso40) e as predições salvas.",
        "por_que": "Ponto 4: separa onde a homologia resolve (com parente) de onde nenhuma abordagem resolve (sem parente). Usa acurácia porque os recortes pequenos têm classes com 1 gene (R12).",
        "fontes": ["results/vizinho_mais_parecido_teste.csv", "results/predicoes/*.npz"],
        "numeros": {r: int(sub[sub.recorte == r].genes.iloc[0]) for r in sub.recorte.unique()},
    })
    conf = confusion_long(preds, classes)
    save("erros_por_classe.csv", conf, {
        "o_que_mostra": "Matriz de confusão do teste em formato longo (pipeline, modelo, verdadeiro, previsto, genes).",
        "como": "confusion_matrix das predições salvas (classe prevista por predict()).",
        "por_que": "Pontos 5, 9 e 10: tipo de erro de cada modelo (deixar passar × alarme falso) e o erro mais comum.",
        "fontes": ["results/predicoes/*.npz"],
    })
    err = main_errors(conf)
    save("erros_principais.csv", err, {
        "o_que_mostra": "Por pipeline e modelo: erros totais, erros entre inativação e alteração do alvo (somando os dois sentidos) e o maior erro.",
        "como": "Soma das células de erros_por_classe.csv.",
        "por_que": "Ponto 9 (P9-D6): o erro mais comum do A, nos 4 modelos, é entre inativação e alteração, e o B o reduz.",
        "fontes": ["results/erros_por_classe.csv"],
        "numeros": {f"{r.pipeline}_{r.modelo}": int(r.erros_inativacao_alteracao_dois_sentidos) for r in err.itertuples() if r.pipeline in ("a", "b")},
    })
    attr = attributes_by_mechanism(onto, seq, y)
    deep = onto["depth_isa"] >= 6
    save("atributos_por_mecanismo.csv", attr, {
        "o_que_mostra": "Por mecanismo: fração de genes com cada classe de antibiótico, genes por profundidade na ARO e mediana da fração de cisteína.",
        "como": "Atributos do experimento (6.033 genes).",
        "por_que": "Ponto 8 e R8: os 'fatos dos dados' que explicam a importância dos atributos (profundidade → inativação; diaminopirimidina → substituição; glicopeptídeo → alteração; cisteína → proteção).",
        "fontes": ["data/processed/features_ontologicas.csv", "data/processed/features_sequencia.csv"],
        "numeros": {
            "inativacao_entre_genes_com_profundidade_6_ou_mais": float((y[deep] == INACTIVATION).mean()),
            "substituicao_com_diaminopirimidina": float(onto["drug_class_diaminopyrimidine_antibiotic"][y == "antibiotic target replacement"].mean()),
            "alteracao_com_glicopeptideo": float(onto["drug_class_glycopeptide_antibiotic"][y == ALTERATION].mean()),
            "profundidade_igual_a_ancestrais": float((onto["depth_isa"] == onto["ancestor_count"]).mean()),
        },
    })
    from src.experiment_config import encode_target

    idt = identity_by_split(data, encode_target(data[TARGET])[0])
    prof = {c: {f"ge_{k}": round(float((g >= k / 100).mean() * 100), 1) for k in (50, 70, 90)} for c, g in idt.groupby("cenario")["identidade_max_com_treino"]}
    save("identidade_teste_treino.csv", idt, {
        "o_que_mostra": "Para cada divisão (aleatória e grupos de 90, 70 e 50%), a maior identidade de cada gene de teste com algum gene de treino.",
        "como": "Partições da curva por limiar (mesma regra do experimento); busca dos grupos (MMseqs2, piso de 40%), cobertura ≥ 0,5 nas duas sequências. 0 = nenhum par ≥ 40%.",
        "por_que": "Motivação da divisão por grupos (seção 4.8): na divisão aleatória quase todo gene de teste tem um parente quase idêntico no treino; com grupos de 70%, nenhum tem parente ≥ 70%. Confere com curva_limiar_particoes.csv.",
        "fontes": ["data/processed/mmseqs/hits.m8", "results/grupos_similaridade.csv"],
        "numeros": {"percentual_de_genes_de_teste_com_parente": prof},
    })
    br = aro_branches(data, owl_path)
    save("ramos_aro_exemplos.csv", br, {
        "o_que_mostra": "O caminho is-a mais curto até a raiz de um gene clássico por mecanismo (TEM-1, vanA, acrB, QnrA1, dfrA1), com os termos ligados a um mecanismo por participates_in.",
        "como": "Hierarquia is-a do aro.owl (leitor dos atributos); em cada passo, o pai de menor profundidade; categorias pelos axiomas participates_in.",
        "por_que": "R8 e seção 2.5.1: mostrar com dados reais que o nível de cima da is-a agrupa os genes por categoria de mecanismo (substitui a Figura 2.1 desenhada à mão).",
        "fontes": ["data/ontologia/aro.owl"],
        "numeros": {m: int(g["nivel"].max()) for m, g in br.groupby("mecanismo")},
    })
    (RESULTS_DIR / "caracterizacao_metadados.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    logger.info("Metadados: %s", RESULTS_DIR / "caracterizacao_metadados.json")
    return meta


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    run()
