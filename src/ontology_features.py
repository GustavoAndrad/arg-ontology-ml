"""
ontology_features.py — Features ontológicas da ARO (Pipeline B, Seção 4.5).

39 atributos por instância:
    * ``depth_isa``      — profundidade na hierarquia is-a (raiz = 0, menor caminho);
    * ``ancestor_count`` — número de ancestrais (fecho transitivo de is-a);
    * ``children_count`` — número de subclasses diretas;
    * 36 flags binárias ``drug_class_*`` — classes de antibiótico associadas ao gene
      (coluna ``Drug Class`` do CARD, separada por ';').

O ``aro.owl`` do CARD 4.0.1 está em sintaxe OWL/XML (não RDF/XML), que o rdflib não
lê. Por isso a hierarquia é extraída com ``xml.etree`` dos axiomas
``<SubClassOf><Class/><Class/></SubClassOf>``. Restrições (``ObjectSomeValuesFrom``)
são ignoradas.  # ANTI-LEAKAGE: apenas relações is-a

Uso:
    python -m src.ontology_features
"""

from __future__ import annotations

import logging
import re
import sys
import xml.etree.ElementTree as ET
from collections import defaultdict, deque
from pathlib import Path

import pandas as pd

from src.config import ONTOLOGY_PATH
from src.experiment_config import PROCESSED_DIR, load_experiment_dataset

logger = logging.getLogger(__name__)

OWL_NS = "{http://www.w3.org/2002/07/owl#}"
STRUCTURAL_COLUMNS: list[str] = ["depth_isa", "ancestor_count", "children_count"]


class IsAGraph:
    """Hierarquia is-a: ``parents[n]`` e ``children[n]`` com accessions ``ARO:XXXXXXX``."""

    def __init__(self) -> None:
        self.parents: dict[str, set[str]] = defaultdict(set)
        self.children: dict[str, set[str]] = defaultdict(set)
        self.nodes: set[str] = set()
        self._depth: dict[str, int] | None = None

    def add_edge(self, child: str, parent: str) -> None:
        self.parents[child].add(parent)
        self.children[parent].add(child)
        self.nodes.update((child, parent))
        self._depth = None

    @property
    def roots(self) -> list[str]:
        return sorted(n for n in self.nodes if not self.parents.get(n))

    def depths(self) -> dict[str, int]:
        """BFS a partir das raízes; profundidade = menor distância até uma raiz."""
        if self._depth is None:
            depth: dict[str, int] = {}
            queue = deque((r, 0) for r in self.roots)
            while queue:
                node, d = queue.popleft()
                if node in depth:
                    continue
                depth[node] = d
                queue.extend((c, d + 1) for c in self.children.get(node, ()))
            self._depth = depth
        return self._depth


def iri_to_accession(iri: str | None) -> str | None:
    """``http://purl.obolibrary.org/obo/ARO_3000001`` -> ``ARO:3000001``."""
    if not iri:
        return None
    m = re.search(r"ARO_(\d+)$", iri)
    return f"ARO:{m.group(1)}" if m else None


def load_ontology(owl_path: Path) -> IsAGraph:
    """Carrega a hierarquia is-a do ``aro.owl`` (OWL/XML)."""
    owl_path = Path(owl_path)
    if not owl_path.exists():
        raise FileNotFoundError(f"Ontologia não encontrada: {owl_path}")
    return build_is_a_graph(owl_path)


def build_is_a_graph(owl_path: Path) -> IsAGraph:
    """Constrói o grafo a partir dos axiomas SubClassOf entre duas classes nomeadas."""
    # ANTI-LEAKAGE: apenas relações is-a (rdfs:subClassOf entre classes nomeadas)
    graph = IsAGraph()
    for _, el in ET.iterparse(str(owl_path), events=("end",)):
        if el.tag == OWL_NS + "SubClassOf":
            kids = list(el)
            if len(kids) == 2 and all(k.tag == OWL_NS + "Class" for k in kids):
                child, parent = iri_to_accession(kids[0].get("IRI")), iri_to_accession(kids[1].get("IRI"))
                if child and parent:
                    graph.add_edge(child, parent)
            el.clear()
    logger.info(
        "Hierarquia is-a: %d nós, %d arestas, %d raízes.",
        len(graph.nodes),
        sum(len(v) for v in graph.parents.values()),
        len(graph.roots),
    )
    return graph


def compute_hierarchy_depth(aro_accession: str, graph: IsAGraph) -> int:
    """Profundidade is-a (raiz = 0); -1 se o conceito não estiver na ontologia."""
    return graph.depths().get(aro_accession, -1)


def compute_ancestor_count(aro_accession: str, graph: IsAGraph) -> int:
    """Número de ancestrais distintos (fecho transitivo); -1 se ausente."""
    if aro_accession not in graph.nodes:
        return -1
    seen: set[str] = set()
    stack = list(graph.parents.get(aro_accession, ()))
    while stack:
        node = stack.pop()
        if node not in seen:
            seen.add(node)
            stack.extend(graph.parents.get(node, ()))
    return len(seen)


def compute_children_count(aro_accession: str, graph: IsAGraph) -> int:
    """Número de subclasses diretas; -1 se ausente."""
    if aro_accession not in graph.nodes:
        return -1
    return len(graph.children.get(aro_accession, ()))


def split_drug_classes(value: object) -> set[str]:
    if not isinstance(value, str):
        return set()
    return {c.strip() for c in value.split(";") if c.strip()}


def drug_class_column(name: str) -> str:
    return "drug_class_" + re.sub(r"[^0-9a-zA-Z]+", "_", name.strip().lower()).strip("_")


def extract_drug_class_flags(drug_class_values: pd.Series) -> pd.DataFrame:
    """
    Flags 0/1 para cada classe de antibiótico observada (ordem alfabética).

    # ANTI-LEAKAGE: Drug Class NÃO é proxy de Resistance_Mechanism — descreve o
    # antibiótico afetado, não o mecanismo pelo qual o gene confere resistência.
    """
    sets = [split_drug_classes(v) for v in drug_class_values]
    classes = sorted(set().union(*sets))
    return pd.DataFrame({drug_class_column(c): [int(c in s) for s in sets] for c in classes})


def extract_ontological_features(df: pd.DataFrame, owl_path: Path = ONTOLOGY_PATH) -> pd.DataFrame:
    """Retorna ``ARO_Accession`` + 3 features estruturais + flags de Drug Class."""
    graph = load_ontology(owl_path)
    ids = df["ARO_Accession"].tolist()
    missing = [a for a in ids if a not in graph.nodes]
    if missing:
        logger.warning("%d ARO Accessions ausentes na ontologia (preenchidos com 0): %s", len(missing), missing[:10])
    structural = pd.DataFrame(
        {
            "depth_isa": [max(compute_hierarchy_depth(a, graph), 0) for a in ids],
            "ancestor_count": [max(compute_ancestor_count(a, graph), 0) for a in ids],
            "children_count": [max(compute_children_count(a, graph), 0) for a in ids],
        }
    )
    flags = extract_drug_class_flags(df["Drug_Class"])
    out = pd.concat([structural, flags], axis=1)
    logger.info("Features ontológicas: %d estruturais + %d drug_class = %d.", 3, flags.shape[1], out.shape[1])
    out.insert(0, "ARO_Accession", ids)
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    data = load_experiment_dataset()
    features = extract_ontological_features(data)
    output = PROCESSED_DIR / "features_ontologicas.csv"
    features.to_csv(output, index=False)
    logger.info("Salvo em %s (antes da verificação de leakage).", output)
