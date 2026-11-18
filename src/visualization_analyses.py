"""
visualization_analyses.py — Figuras dos dados e das análises complementares (estilo do artigo).

Complementa ``src.visualization`` com as figuras que faltavam para o texto, **um gráfico
por arquivo** (para entrar direto no texto) e no mesmo estilo das figuras do artigo
(``src.plot_style``: paleta Set2, título no topo, valores nas barras):

* dados (``plots/dados/``): parentesco teste → treino em cada divisão, grupos de
  similaridade por limiar, famílias no treino e no teste, ramos reais da ARO, categoria
  de mecanismo na hierarquia is-a, profundidade por mecanismo, informação mútua, regra
  simples com os atributos da ARO, classes de antibiótico por mecanismo e atributos de
  sequência por mecanismo;
* análises (``plots/analises/``): os 50 folds da CV, recortes do teste, ablação (um
  arquivo por cenário), as sensibilidades, erros entre inativação e alteração e o
  estudo de caso.

Critério: cada figura mostra só o necessário para a sua mensagem. Onde há muitas
configurações (ablação, sensibilidades), cada barra é a média dos 4 modelos, com um
marcador por modelo (sem número), e o valor de cada modelo com o intervalo fica na
tabela LaTeX e no CSV. As diferenças entre cenários na CV e os vetores ontológicos ambíguos
viraram tabelas (``tabela_cenarios_cv.tex`` e ``tabela_vetores_ontologicos.tex``): a
figura não acrescentava nada à tabela.

Cada figura lê um arquivo de ``results/`` já gerado e justificado pelo módulo de
origem. O catálogo ``results/figuras_catalogo.json`` diz, para cada figura (desta e de
``src.visualization``), o que mostra, de onde vêm os números, por que existe, a seção do
texto e uma legenda sugerida.

Uso (dentro de ``python -m src.visualization``, ou sozinho):
    python -m src.visualization_analyses
"""

from __future__ import annotations

import json
import logging
import sys
import textwrap

import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Patch

from src.experiment_config import MODEL_NAMES, PLOTS_DIR, PROCESSED_DIR, RESULTS_DIR, TARGET, load_experiment_dataset
from src.pipeline_builder import MODEL_LABELS
from src.plot_style import (
    ACCESSIBLE_PALETTE, COLOR_A, COLOR_B, COLOR_HOMOLOGY, COLOR_NEW_GENE, GROUP_COLORS, GROUP_LABELS, MECHANISM_COLORS,
    MECHANISMS, clean_axes, num, save, short,
)

import matplotlib.pyplot as plt  # noqa: E402  (backend já definido em plot_style)

logger = logging.getLogger(__name__)
DATA_DIR = PLOTS_DIR / "dados"
ANALYSIS_DIR = PLOTS_DIR / "analises"
MODEL_COLORS = dict(zip(MODEL_NAMES, ACCESSIBLE_PALETTE[:4]))
MODEL_SERIES = [(MODEL_LABELS[m], MODEL_COLORS[m]) for m in MODEL_NAMES]


def _need(name: str) -> bool:
    if (RESULTS_DIR / name).exists():
        return True
    logger.warning("Figura não gerada: falta results/%s.", name)
    return False


_fmt = num  # número com vírgula decimal (nome antigo, usado nos testes)


# Marcadores dos modelos sobre as barras de média: forma (não cor), para não confundir com
# as cores das barras (que já dizem o pipeline ou o grupo de atributos).
MODEL_MARKERS = dict(zip(MODEL_NAMES, ["o", "s", "^", "D"]))


def _mean_range_barh(ax, labels, per_model, colors, fmt=lambda v: num(v, 2), alphas=None) -> None:
    """
    Uma barra por configuração com a média dos 4 modelos e um único número (a média); em
    cima, um marcador por modelo (forma = modelo, sem número), que mostra a dispersão e
    qual modelo foge do padrão (por exemplo, o RF na ablação). ``per_model``: lista (uma
    por configuração) com o valor de cada modelo, na ordem de ``MODEL_NAMES``. Substitui as
    4 barras por configuração, com intervalo e número em cada uma (28 números numa figura).
    """
    y = np.arange(len(labels))
    vals = [np.asarray(v, dtype=float) for v in per_model]
    means = np.array([v.mean() for v in vals])
    hi = np.array([v.max() for v in vals])
    for yi, m, c, a in zip(y, means, colors, alphas or [1.0] * len(y)):
        ax.barh(yi, m, 0.62, color=c, alpha=a)
    for k, model in enumerate(MODEL_NAMES):
        ax.plot([v[k] for v in vals], y, MODEL_MARKERS[model], ls="none", color="0.2", mfc="white", mew=1.2, ms=6, zorder=3)
    # A média fica escrita dentro da barra, no começo: à direita, ela se afastava da barra
    # quando um modelo ia bem mais longe e podia ser lida como o valor desse modelo.
    span = max(hi.max(), 0.05)
    for yi, m in zip(y, means):
        ax.text(span * 0.012, yi, fmt(m), va="center", ha="left", fontsize=9, fontweight="bold")
    ax.set_yticks(y, labels)
    ax.invert_yaxis()


def _model_marker_handles():
    from matplotlib.lines import Line2D

    return [Line2D([], [], marker=MODEL_MARKERS[m], ls="none", color="0.2", mfc="white", mew=1.2, ms=6, label=MODEL_LABELS[m])
            for m in MODEL_NAMES]


# ── Dados ────────────────────────────────────────────────────────────────────────
def plot_identity_by_split(output_path) -> None:
    """Fração dos genes de teste com um gene do treino acima de cada identidade, por divisão."""
    df = pd.read_csv(RESULTS_DIR / "identidade_teste_treino.csv", dtype={"cenario": str})
    order = [("aleatoria", "Aleatória"), ("90", "Grupos de 90%"), ("70", "Grupos de 70% (experimento)"), ("50", "Grupos de 50%")]
    thr = np.linspace(0.4, 1.0, 241)
    fig, ax = plt.subplots(figsize=(9, 5))
    for (name, label), color in zip(order, ACCESSIBLE_PALETTE):
        v = df[df.cenario == name]["identidade_max_com_treino"].to_numpy()
        ax.plot(thr * 100, [(v >= t).mean() * 100 for t in thr], color=color, lw=2.5 if name == "70" else 2, label=label)
        at70 = (v >= 0.7).mean() * 100
        ax.plot(70, at70, "o", color=color, ms=7)
        if name in ("aleatoria", "90"):
            ax.text(69, at70 + 2, f"{at70:.1f}%".replace(".", ","), ha="right", fontsize=9)
    ax.text(71, 3, "0% (grupos de 70% e 50%)", fontsize=9)
    ax.axvline(70, color="0.5", lw=0.8, ls="--")
    ax.set_xlim(40, 100)
    ax.set_ylim(0, 100)
    ax.set_xlabel("Identidade com o gene mais parecido do treino, pelo menos (%)")
    ax.set_ylabel("Genes de teste (%)")
    ax.set_title("Parentesco entre teste e treino em cada divisão", fontweight="bold")
    ax.legend(loc="lower left", fontsize=9)
    clean_axes(ax)
    save(fig, output_path)


def _group_lines(output_path, col, ylabel, title, log=False, pct=False):
    g = pd.read_csv(RESULTS_DIR / "grupos_resumo.csv")
    g = g[~g["mecanismo"].astype(str).str.startswith("(")]
    fig, ax = plt.subplots(figsize=(8.5, 5))
    for m in MECHANISMS:
        s = g[g.mecanismo == m].sort_values("limiar")
        yv = s[col].to_numpy() * (100 if pct else 1)
        ax.plot(s["limiar"], yv, "-o", color=MECHANISM_COLORS[m], lw=2, ms=6, label=short(m))
        v70 = float(yv[s.limiar.to_numpy() == 70][0])
        ax.text(71.5, v70, f"{v70:.0f}%" if pct else f"{v70:.0f}", fontsize=8, va="center")
    if log:
        ax.set_yscale("log")
        ax.set_yticks([5, 10, 20, 50, 100, 200, 500], ["5", "10", "20", "50", "100", "200", "500"])
        ax.minorticks_off()
    ax.axvline(70, color="0.5", lw=0.8, ls="--")
    ax.set_xticks([40, 50, 70, 90], ["40%", "50%", "70%", "90%"])
    ax.set_xlabel("Limiar de identidade para ligar dois genes (MMseqs2)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(fontsize=8.5)
    clean_axes(ax)
    save(fig, output_path)


def plot_groups_per_threshold(output_path) -> None:
    """Número de grupos de similaridade de cada classe em cada limiar (o experimento usa 70%)."""
    _group_lines(output_path, "grupos", "Grupos da classe (escala log)", "Grupos de similaridade por classe e limiar", log=True)


def plot_largest_group(output_path) -> None:
    """Fração dos genes de cada classe que cai no maior grupo, em cada limiar."""
    _group_lines(output_path, "fracao_maior_grupo", "Genes da classe no maior grupo (%)", "Concentração de cada classe no maior grupo", pct=True)


def _stacked_share(ax, rows, colors, labels, inside=True):
    """Barras horizontais 100% empilhadas, com a contagem em cada parte que couber (``inside``)."""
    for i, parts in enumerate(rows):
        total = sum(parts)
        left = 0
        for v, color in zip(parts, colors):
            if v <= 0:
                continue
            w = v / total * 100
            ax.barh(i, w, left=left, color=color, edgecolor="white", height=0.6)
            if inside and w >= 6:
                ax.text(left + w / 2, i, str(int(v)), ha="center", va="center", fontsize=8.5)
            left += w
    ax.set_yticks(range(len(rows)), labels)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("Genes de teste da classe (%)")


def plot_families_train_test(output_path) -> None:
    """Genes de teste cuja família AMR também tem genes no treino, por classe."""
    f = pd.read_csv(RESULTS_DIR / "familias_treino_teste.csv").set_index("mecanismo")
    order = [m for m in MECHANISMS if m in f.index]
    rows = [(f.loc[m, "genes_teste_familia_no_treino"], f.loc[m, "genes_teste"] - f.loc[m, "genes_teste_familia_no_treino"]) for m in order]
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    # Só a contagem à direita ("120/900"): escrita também dentro da barra, o número aparecia duas vezes.
    _stacked_share(ax, rows, [ACCESSIBLE_PALETTE[0], "0.75"], [short(m) for m in order], inside=False)
    for i, (seen, new) in enumerate(rows):
        ax.text(101, i, f"{int(seen)}/{int(seen + new)}", va="center", fontsize=9)
    ax.set_title("Genes de teste cuja família AMR também tem genes no treino", fontweight="bold")
    ax.legend(handles=[Patch(color=ACCESSIBLE_PALETTE[0], label="família com genes no treino"), Patch(color="0.75", label="família ausente do treino")],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=9)
    clean_axes(ax, "x")
    save(fig, output_path)


def plot_isa_category(output_path) -> None:
    """Genes sob a categoria do próprio mecanismo, de outro mecanismo ou de nenhuma, na hierarquia is-a."""
    t = pd.read_csv(RESULTS_DIR / "categoria_isa_por_mecanismo.csv").set_index("mecanismo")
    order = [m for m in MECHANISMS if m in t.index]
    rows = []
    for m in order:
        row = t.loc[m]
        none, own = row.get("(nenhuma)", 0), row.get(m, 0)
        rows.append((own, row.sum() - own - none, none))
    fig, ax = plt.subplots(figsize=(9.5, 4.2))
    kinds = [(ACCESSIBLE_PALETTE[2], "categoria do próprio mecanismo"), (ACCESSIBLE_PALETTE[1], "categoria de outro mecanismo"),
             ("0.85", "sem categoria")]
    _stacked_share(ax, rows, [c for c, _ in kinds], [short(m) for m in order])
    ax.set_xlabel("Genes da classe (%)")
    ax.set_title("Genes sob uma categoria de mecanismo na hierarquia is-a da ARO", fontweight="bold")
    # Na legenda, só o que aparece no desenho (hoje nenhum gene cai sob a categoria de outro mecanismo).
    ax.legend(handles=[Patch(color=c, label=lab) for k, (c, lab) in enumerate(kinds) if any(r[k] > 0 for r in rows)],
              loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, frameon=False, fontsize=9)
    clean_axes(ax, "x")
    save(fig, output_path)


def plot_aro_branches(output_path) -> None:
    """Caminho is-a mais curto da raiz da ARO até um gene representativo de cada mecanismo."""
    b = pd.read_csv(RESULTS_DIR / "ramos_aro_exemplos.csv")
    mechs = [m for m in MECHANISMS if m in set(b.mecanismo)]
    depth = int(b.nivel.max())
    fig, ax = plt.subplots(figsize=(15, 8.5))
    ax.set_xlim(-0.55, len(mechs) - 0.45)
    ax.set_ylim(depth + 0.8, -0.55)
    ax.axis("off")
    center = (len(mechs) - 1) / 2
    drawn = set()
    for j, m in enumerate(mechs):
        g = b[b.mecanismo == m].sort_values("nivel")
        levels = g.nivel.to_numpy()
        xs = [center if lv <= 1 else j for lv in levels]
        for x0, y0, x1, y1 in zip(xs[:-1], levels[:-1], xs[1:], levels[1:]):
            ax.plot([x0, x1], [y0 + 0.33, y1 - 0.33], color="0.55", lw=1, zorder=1)
        for x, (_, r) in zip(xs, g.iterrows()):
            if (r.nivel, r.termo) in drawn:
                continue
            drawn.add((r.nivel, r.termo))
            is_cat = str(r.categoria_de_mecanismo) == "True"
            is_gene = r.nivel == levels.max()
            fc = MECHANISM_COLORS[r.mecanismo_ligado] if is_cat and r.mecanismo_ligado in MECHANISM_COLORS else ("0.3" if is_gene else "0.93")
            text = "\n".join(textwrap.wrap(str(r.rotulo), 60 if r.nivel <= 1 else 24))
            w = 2.6 if r.nivel <= 1 else 0.94
            ax.add_patch(FancyBboxPatch((x - w / 2, r.nivel - 0.36), w, 0.72, boxstyle="round,pad=0,rounding_size=0.06",
                                        mutation_aspect=0.6, lw=0.6, ec="0.6", fc=fc, zorder=2))
            ax.text(x, r.nivel, text, ha="center", va="center", fontsize=9, color="white" if is_gene else "black", zorder=3)
        ax.text(j, depth + 0.65, short(m), ha="center", fontsize=11, fontweight="bold")
    # Caixa colorida = categoria ligada ao mecanismo da coluna (relação participates_in), na
    # cor do mecanismo; o nome do mecanismo embaixo da coluna já explica a cor.
    ax.legend(handles=[Patch(fc="0.93", ec="0.6", label="termo da ARO"), Patch(color="0.3", label="gene")],
              loc="upper center", bbox_to_anchor=(0.5, 0.02), ncol=2, frameon=False, fontsize=10)
    ax.set_title("Caminho na ARO até um gene de cada mecanismo", fontweight="bold", fontsize=13)
    save(fig, output_path)


def plot_depth_by_mechanism(output_path) -> None:
    """Genes por profundidade do termo na hierarquia is-a, empilhados por mecanismo."""
    a = pd.read_csv(RESULTS_DIR / "atributos_por_mecanismo.csv")
    d = a[a.atributo.str.startswith("depth_isa=")].copy()
    d["prof"] = d.atributo.str.split("=").str[1].astype(int)
    piv = d.pivot_table(index="prof", columns="mecanismo", values="valor", aggfunc="sum").fillna(0)
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(piv))
    bottom = np.zeros(len(piv))
    for m in MECHANISMS:
        if m in piv:
            ax.bar(x, piv[m], bottom=bottom, color=MECHANISM_COLORS[m], edgecolor="white", width=0.7, label=short(m))
            bottom += piv[m].to_numpy()
    for xi, tot in zip(x, bottom):
        frac = piv.iloc[xi].get("antibiotic inactivation", 0) / tot * 100
        ax.text(xi, tot + bottom.max() * 0.01, f"{int(tot)}\n({frac:.0f}% inativ.)", ha="center", va="bottom", fontsize=8)
    ax.set_xticks(x, [str(p) for p in piv.index])
    ax.set_ylim(0, bottom.max() * 1.18)
    ax.set_xlabel("Profundidade do termo do gene na hierarquia is-a (depth_isa)")
    ax.set_ylabel("Genes")
    ax.set_title("Genes por profundidade na ARO e mecanismo", fontweight="bold")
    ax.legend(fontsize=8.5, loc="upper left")
    clean_axes(ax)
    save(fig, output_path)


MI_TOP = 15  # os 26 restantes têm até 2,5% cada (estão no CSV)


def plot_mutual_information(output_path) -> None:
    """Informação mútua dos 15 atributos da ARO mais informativos com o mecanismo (fração da entropia do rótulo, no treino)."""
    full = pd.read_csv(RESULTS_DIR / "informacao_mutua_ontologia.csv").sort_values("fracao_da_entropia_do_rotulo", ascending=False)
    mi = full.head(MI_TOP)
    colors = [GROUP_COLORS["estruturais"] if g == "estrutural" else GROUP_COLORS["classes_de_antibiotico"] for g in mi.grupo]
    y = np.arange(len(mi))
    vals = mi.fracao_da_entropia_do_rotulo.to_numpy() * 100
    fig, ax = plt.subplots(figsize=(9, 5.5))
    ax.barh(y, vals, color=colors, height=0.7)
    for yi, v in zip(y, vals):
        ax.text(v + 0.5, yi, f"{num(v, 1)}%", va="center", fontsize=8.5)
    ax.set_yticks(y, [f.replace("drug_class_", "").replace("_", " ") for f in mi.feature], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("Informação mútua com o mecanismo (% da entropia do rótulo; treino)")
    ax.set_title(f"Os {MI_TOP} atributos da ARO mais informativos sobre o mecanismo (de {len(full)})", fontweight="bold")
    ax.legend(handles=[Patch(color=GROUP_COLORS["estruturais"], label=GROUP_LABELS["estruturais"]),
                       Patch(color=GROUP_COLORS["classes_de_antibiotico"], label=GROUP_LABELS["classes_de_antibiotico"])], loc="lower right")
    clean_axes(ax, "x")
    save(fig, output_path)


def plot_simple_rule(output_path) -> None:
    """F1-macro no teste de uma regra de consulta (combinação de atributos → classe mais comum no treino)."""
    rule = pd.read_csv(RESULTS_DIR / "regra_simples_ontologia.csv")
    names = {"profundidade": "Só a profundidade", "3 estruturais": "As 3 estruturais", "38 classes de antibiotico": "As 38 classes de antibiótico",
             "41 atributos da ARO": "Os 41 atributos da ARO"}
    colors = [GROUP_COLORS["estruturais"], GROUP_COLORS["estruturais"], GROUP_COLORS["classes_de_antibiotico"], ACCESSIBLE_PALETTE[3]]
    y = np.arange(len(rule))
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    ax.barh(y, rule.f1_macro_teste, color=colors[: len(rule)], height=0.6)
    # Só o F1-macro (a métrica do trabalho); a acurácia fica no CSV.
    for yi, r in zip(y, rule.itertuples()):
        ax.text(r.f1_macro_teste + 0.01, yi, num(r.f1_macro_teste, 2), va="center", fontsize=10)
    ax.set_yticks(y, [names.get(a, a) for a in rule.atributos])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("F1-macro no teste")
    ax.set_title("Regra sem aprendizado de máquina com os atributos da ARO (treino → teste)", fontweight="bold")
    clean_axes(ax, "x")
    save(fig, output_path)


DRUG_CLASS_MIN_PCT = 5  # na figura, só as classes presentes em pelo menos 5% dos genes de algum mecanismo


def plot_drug_classes_by_mechanism(output_path) -> None:
    """
    Fração dos genes de cada mecanismo com cada classe de antibiótico. Só as classes com pelo
    menos 5% em algum mecanismo: das 38, as outras ficavam como linhas quase vazias e
    escondiam o padrão (todas estão em ``atributos_por_mecanismo.csv``).
    """
    a = pd.read_csv(RESULTS_DIR / "atributos_por_mecanismo.csv")
    d = a[a.atributo.str.startswith("drug_class_")]
    piv = d.pivot_table(index="atributo", columns="mecanismo", values="valor")[MECHANISMS]
    onto = pd.read_csv(PROCESSED_DIR / "features_ontologicas.csv")
    piv["genes"] = [int(onto[f].sum()) for f in piv.index]
    total = len(piv)
    piv = piv[(piv[MECHANISMS] * 100).max(axis=1) >= DRUG_CLASS_MIN_PCT].sort_values("genes", ascending=False)
    vals = piv[MECHANISMS].to_numpy() * 100
    fig, ax = plt.subplots(figsize=(8.5, 0.36 * len(piv) + 2))
    im = ax.imshow(vals, cmap="Blues", vmin=0, vmax=100, aspect="auto")
    for i in range(vals.shape[0]):
        for j in range(vals.shape[1]):
            if vals[i, j] >= 1:
                ax.text(j, i, f"{vals[i, j]:.0f}", ha="center", va="center", fontsize=8.5, color="white" if vals[i, j] > 60 else "black")
    ax.set_yticks(range(len(piv)), [f.replace("drug_class_", "").replace("_", " ") for f in piv.index], fontsize=9)
    ax.set_xticks(range(len(MECHANISMS)), [short(m) for m in MECHANISMS], rotation=30, ha="right")
    fig.colorbar(im, ax=ax, fraction=0.04, label="% dos genes do mecanismo")
    ax.set_title(f"Classes de antibiótico por mecanismo ({len(piv)} das {total}, com ≥ {DRUG_CLASS_MIN_PCT}% em algum mecanismo)",
                 fontweight="bold", fontsize=11)
    save(fig, output_path)


SEQ_FEATURES = [("length", "Comprimento da proteína (resíduos)", 1), ("isoelectric_point", "Ponto isoelétrico", 1),
                ("charge_at_pH7", "Carga em pH 7", 1), ("aa_comp_C", "Fração de cisteína (%)", 100)]


def plot_sequence_feature(col, label, scale, output_path) -> None:
    """Distribuição de um atributo de sequência por mecanismo (caixa: quartis; traço: mediana)."""
    data = load_experiment_dataset()
    seq = pd.read_csv(PROCESSED_DIR / "features_sequencia.csv")
    if len(seq) != len(data) and "ARO_Accession" in seq.columns:
        seq = data[["ARO_Accession"]].merge(seq, on="ARO_Accession", how="left")
    y = data[TARGET].to_numpy()
    vals = [seq[col].to_numpy()[y == m] * scale for m in MECHANISMS]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    bp = ax.boxplot(vals, vert=False, patch_artist=True, widths=0.55, showfliers=False,
                    medianprops={"color": "black", "lw": 1.5}, whiskerprops={"color": "0.4"}, capprops={"color": "0.4"})
    for patch, m in zip(bp["boxes"], MECHANISMS):
        patch.set_facecolor(MECHANISM_COLORS[m])
        patch.set_edgecolor("0.4")
    for i, v in enumerate(vals, start=1):
        med = np.median(v)
        ax.text(med, i - 0.34, (f"{med:.0f}" if col == "length" else f"{med:.1f}").replace(".", ","), ha="center", fontsize=8)
    ax.set_yticks(range(1, len(MECHANISMS) + 1), [short(m) for m in MECHANISMS])
    ax.invert_yaxis()
    ax.set_xlabel(label)
    ax.set_title(f"{label} por mecanismo", fontweight="bold")
    clean_axes(ax, "x")
    save(fig, output_path)


# ── Análises ─────────────────────────────────────────────────────────────────────
def plot_cv_folds(output_path) -> None:
    """F1-macro de A e B nos 50 folds da CV, por modelo, com a contagem de folds em que B > A."""
    cv = pd.read_csv(RESULTS_DIR / "cv_folds.csv")
    fig, ax = plt.subplots(figsize=(10, 5))
    pos, data, colors = [], [], []
    for i, m in enumerate(MODEL_NAMES):
        pair = []
        for k, key in enumerate(("a", "b")):
            v = cv[(cv.pipeline == key) & (cv.modelo == m)].sort_values(["repeticao", "fold"]).f1_macro.to_numpy()
            pos.append(i * 3 + k)
            data.append(v)
            colors.append(COLOR_A if key == "a" else COLOR_B)
            pair.append(v)
        ax.text(i * 3 + 0.5, 1.02, f"B > A em {int((pair[1] > pair[0]).sum())}/{len(pair[0])}", ha="center", fontsize=8.5)
    # Só a caixa (quartis, mediana e folds atípicos); os 50 pontos por cima repetiam a mesma informação.
    bp = ax.boxplot(data, positions=pos, widths=0.7, patch_artist=True, medianprops={"color": "black"},
                    flierprops={"marker": "o", "ms": 4, "mfc": "0.4", "mec": "0.4"})
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
    ax.set_xticks([i * 3 + 0.5 for i in range(len(MODEL_NAMES))], [MODEL_LABELS[m] for m in MODEL_NAMES])
    ax.set_ylim(min(0.3, min(v.min() for v in data) - 0.03), 1.06)
    ax.set_ylabel("F1-macro na validação")
    ax.set_title("F1-macro nos 50 folds da validação cruzada (10 × 5, por grupos)", fontweight="bold")
    ax.legend(handles=[Patch(color=COLOR_A, label="Pipeline A (Sequência)"), Patch(color=COLOR_B, label="Pipeline B (Sequência + Ontologia)")],
              loc="lower right", fontsize=9)
    clean_axes(ax)
    save(fig, output_path)


def plot_test_subsets(output_path) -> None:
    """Acurácia no teste por recorte de parentesco com o treino: homologia, A, B curado e B gene novo."""
    rc = pd.read_csv(RESULTS_DIR / "recortes_teste.csv")
    names = [("com_parente_no_treino", "Com parente no treino"), ("distantes_sem_parente_40", "Distantes (sem parente ≥ 40%)"),
             ("sem_nenhum_parente", "Sem nenhum parente")]
    series = [("homologia", COLOR_HOMOLOGY), ("a", COLOR_A), ("b", COLOR_B), ("b_gene_novo", COLOR_NEW_GENE)]
    fig, ax = plt.subplots(figsize=(10.5, 5))
    w = 0.2
    for i, (r, _) in enumerate(names):
        sub = rc[rc.recorte == r]
        for k, (key, color) in enumerate(series):
            v = sub[sub.pipeline == key].acuracia.to_numpy()
            if not len(v):
                continue
            x = i + (k - 1.5) * w
            mean, lo, hi = v.mean(), v.min(), v.max()
            ax.bar(x, mean, w * 0.95, color=color)
            if hi > lo:
                ax.errorbar(x, mean, yerr=[[mean - lo], [hi - mean]], fmt="none", ecolor="0.2", capsize=3, lw=1)
            ax.text(x, hi + 0.02, num(mean, 2), ha="center", fontsize=8.5)
    ax.set_xticks(range(len(names)), [f"{l}\n({int(rc[rc.recorte == r].genes.iloc[0])} genes)" for r, l in names])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Acurácia no teste\n(média dos 4 modelos; traço: pior e melhor)")
    ax.set_title("Acurácia no teste segundo o parentesco com o treino", fontweight="bold")
    ax.legend(handles=[Patch(color=COLOR_HOMOLOGY, label="Homologia (sem aprendizado de máquina)"),
                       Patch(color=COLOR_A, label="Pipeline A"),
                       Patch(color=COLOR_B, label="Pipeline B (anotação curada)"),
                       Patch(color=COLOR_NEW_GENE, label="Pipeline B (gene novo)")],
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=8.5)
    clean_axes(ax)
    save(fig, output_path)


# Nomes das configurações da ablação (``leakage_check.ABLATION_CONFIGS``). "classes" são
# as 38 classes de antibiótico (Drug Class), não as classes de mecanismo.
ABLATION_LABELS = {
    "A": "A (só sequência)", "A + estruturais": "A + estruturais (3)", "A + classes": "A + classes de antibiótico (38)",
    "B": "B (A + os 41 da ARO)", "só estruturais": "Só estruturais (3)", "só classes": "Só classes de antibiótico (38)",
    "só ontologia": "Só os 41 da ARO",
}


def plot_ablation(scenario, output_path) -> None:
    """
    F1-macro na CV de cada configuração da ablação dos atributos da ARO, num cenário: uma
    barra por configuração (média dos 4 modelos) e um marcador por modelo. Os marcadores
    mostram a exceção do RF (P4-D6: nele, as classes de antibiótico sozinhas não chegam
    ao B). Os valores de cada modelo, com o IC corrigido, estão em ``tabela_ablacao.tex``.
    Cor: verde = só sequência; laranja = sequência + ARO; cinza = só ARO (sem sequência).
    """
    ab = pd.read_csv(RESULTS_DIR / "ablacao_ontologia.csv")
    sub = ab[ab.cenario == scenario]
    configs = list(dict.fromkeys(sub.config))
    per_model = [[sub[(sub.modelo == m) & (sub.config == c)].f1_cv_media.iloc[0] for m in MODEL_NAMES] for c in configs]
    no_seq = [not (c == "A" or c.startswith("A +") or c == "B") for c in configs]
    colors = [COLOR_A if c == "A" else ("0.72" if ns else COLOR_B) for c, ns in zip(configs, no_seq)]
    fig, ax = plt.subplots(figsize=(9, 4.8))
    _mean_range_barh(ax, [ABLATION_LABELS.get(c, c) for c in configs], per_model, colors)
    if any(no_seq):
        ax.axhline(no_seq.index(True) - 0.5, color="0.5", lw=0.8, ls="--")
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("F1-macro na validação cruzada (barra: média dos 4 modelos; marcador: cada modelo)")
    ax.set_title(f"Ablação dos atributos da ARO — {'anotação curada' if scenario == 'anotacao_curada' else 'gene novo'}",
                 fontweight="bold")
    ax.legend(handles=[Patch(color=COLOR_A, label="só sequência"), Patch(color=COLOR_B, label="sequência + ARO"),
                       Patch(color="0.72", label="só ARO")] + _model_marker_handles(),
              loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=9)
    clean_axes(ax, "x")
    save(fig, output_path)


def _sensitivity_gain(csv, key_col, title, output_path, labels: dict[str, str], experiment: str) -> None:
    """
    Ganho B − A na CV por configuração: uma barra por configuração (média dos 4 modelos) e
    um marcador por modelo. A configuração do experimento fica em cor cheia, as outras em
    cor clara. Os valores de cada modelo, com o IC corrigido, estão na tabela LaTeX.
    """
    sd = pd.read_csv(RESULTS_DIR / csv)
    cats = [c for c in labels if c in set(sd[key_col])] + [c for c in dict.fromkeys(sd[key_col]) if c not in labels]
    per_model = [[sd[(sd.modelo == m) & (sd[key_col] == c)].ganho_cv.iloc[0] for m in MODEL_NAMES] for c in cats]
    fig, ax = plt.subplots(figsize=(9, 0.55 * len(cats) + 1.8))
    _mean_range_barh(ax, [labels.get(c, c) for c in cats], per_model, [COLOR_B] * len(cats),
                     fmt=lambda v: num(v, 2, sign=True), alphas=[1.0 if c == experiment else 0.5 for c in cats])
    ax.axvline(0, color="0.4", lw=0.8)
    ax.set_xlim(0, None)
    ax.set_xlabel("Ganho B − A no F1-macro na CV (barra: média dos 4 modelos; marcador: cada modelo)")
    ax.set_title(title, fontweight="bold")
    ax.legend(handles=_model_marker_handles(), loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, frameon=False, fontsize=9)
    clean_axes(ax, "x")
    save(fig, output_path)


# Rótulos legíveis no lugar dos códigos de ``sensitivity_representation.REPRESENTATIONS``, na
# ordem de leitura: a do experimento, as básicas, as 426 completas e as seleções com k.
REPRESENTATION_LABELS = {
    "reduzida_73": "73, alfabeto reduzido (experimento)",
    "basicas_26": "26 básicas (6 físico-químicas + 20 aminoácidos)",
    "completa_426": "426, sem seleção",
    "k100_sequencia": "426, k = 100 só na sequência",
    "k50_sequencia": "426, k = 50 só na sequência",
    "k20_sequencia": "426, k = 20 só na sequência",
    "tcc1_k100_todos": "426, k = 100 sobre todos (TCC1)",
}
BALANCING_LABELS = {
    "peso": "Peso por classe (experimento)",
    "nenhum": "Nenhum",
    "under": "Subamostragem da maior classe",
    "over": "Sobreamostragem aleatória",
    "smote": "SMOTE",
    "smote_enn": "SMOTE-ENN",
    "hierarquica": "Hierárquica (inativação × resto)",
}


def plot_sensitivity_representation(output_path) -> None:
    """Ganho B − A na CV com cada representação da sequência (a do experimento é a reduzida de 73)."""
    _sensitivity_gain("sensibilidade_representacao.csv", "config", "Ganho da ontologia com outras representações da sequência",
                      output_path, REPRESENTATION_LABELS, "reduzida_73")


def plot_sensitivity_balancing(output_path) -> None:
    """Ganho B − A na CV com cada estratégia de balanceamento (a do experimento é o peso por classe)."""
    _sensitivity_gain("sensibilidade_balanceamento.csv", "estrategia", "Ganho da ontologia com outras estratégias de balanceamento",
                      output_path, BALANCING_LABELS, "peso")


def plot_sensitivity_tuning(where, output_path) -> None:
    """
    F1-macro de A e B com hiperparâmetros padrão e ajustados, na CV aninhada (``where="cv"``,
    a figura do texto) ou no teste (``where="teste"``; só na tabela, por repetir a mensagem).
    """
    st = pd.read_csv(RESULTS_DIR / "sensibilidade_ajuste.csv")
    cols = [("a", "padrao", "A padrão"), ("a", "ajustado", "A ajustado"), ("b", "padrao", "B padrão"), ("b", "ajustado", "B ajustado")]
    colors = [COLOR_A, "#2e8b6e", COLOR_B, "#d9622b"]
    fig, ax = plt.subplots(figsize=(10, 5))
    w = 0.2
    for k, ((key, kind, label), color) in enumerate(zip(cols, colors)):
        vals = [st[(st.modelo == m) & (st.pipeline == key)][f"f1_{where}_{kind}"].iloc[0] for m in MODEL_NAMES]
        x = np.arange(len(MODEL_NAMES)) + (k - 1.5) * w
        ax.bar(x, vals, w * 0.95, color=color, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 0.01, num(v, 2), ha="center", fontsize=8)
    ax.set_xticks(range(len(MODEL_NAMES)), [MODEL_LABELS[m] for m in MODEL_NAMES])
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("F1-macro " + ("no teste" if where == "teste" else "na validação cruzada"))
    ax.set_title("F1-macro com hiperparâmetros padrão e ajustados" + (" (teste)" if where == "teste" else " (CV aninhada)"), fontweight="bold")
    ax.legend(fontsize=8.5, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08), frameon=False)
    clean_axes(ax)
    save(fig, output_path)


def plot_main_errors(output_path) -> None:
    """Genes de teste trocados entre inativação e alteração do alvo (os dois sentidos), por modelo."""
    e = pd.read_csv(RESULTS_DIR / "erros_principais.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.38
    x = np.arange(len(MODEL_NAMES))
    top = 0
    for off, key, color, label in ((-w / 2, "a", COLOR_A, "Pipeline A (Sequência)"), (w / 2, "b", COLOR_B, "Pipeline B (Sequência + Ontologia)")):
        vals = [e[(e.pipeline == key) & (e.modelo == m)].erros_inativacao_alteracao_dois_sentidos.iloc[0] for m in MODEL_NAMES]
        top = max(top, max(vals))
        ax.bar(x + off, vals, w, color=color, edgecolor="black", linewidth=0.5, label=label)
        for xi, v in zip(x + off, vals):
            ax.text(xi, v + 0.5, str(int(v)), ha="center", fontsize=10)
    ax.set_xticks(x, [MODEL_LABELS[m] for m in MODEL_NAMES])
    ax.set_ylim(0, top * 1.15 + 1)
    ax.set_ylabel("Genes de teste")
    ax.set_title("Erros entre inativação e alteração do alvo no teste (os dois sentidos)", fontweight="bold")
    ax.legend(fontsize=9)
    clean_axes(ax)
    save(fig, output_path)


def plot_case_study(output_path) -> None:
    """
    Mecanismo previsto pelos 8 modelos finais para cada gene dos mecanismos raros (fora do
    experimento). A cor da célula diz o mecanismo previsto (sem texto dentro: a cor e a
    legenda bastam). Linha = família AMR e acesso ARO; o mecanismo verdadeiro (do CARD) é
    o mesmo em quase todos os genes, então só o gene de mecanismo diferente é marcado (*).
    """
    ec = pd.read_csv(RESULTS_DIR / "estudo_caso_resumo.csv")
    cols = [f"{k}_{m}" for k in ("A", "B") for m in MODEL_NAMES]
    full = {short(m): m for m in MECHANISMS}
    common = ec.Resistance_Mechanism.mode().iloc[0]
    fig, ax = plt.subplots(figsize=(10, 0.6 * len(ec) + 2.2))
    used = set()
    for i, r in ec.iterrows():
        for j, c in enumerate(cols):
            m = full.get(r[c], r[c])
            used.add(m)
            ax.add_patch(plt.Rectangle((j - 0.47, i - 0.44), 0.94, 0.88, color=MECHANISM_COLORS.get(m, "0.8")))
    ax.set_xlim(-0.5, len(cols) - 0.5)
    ax.set_ylim(len(ec) - 0.5, -0.5)
    ax.axvline(len(MODEL_NAMES) - 0.5, color="black", lw=1.5)
    for k, name in enumerate(("Pipeline A", "Pipeline B")):
        ax.text(k * len(MODEL_NAMES) + (len(MODEL_NAMES) - 1) / 2, -0.6, name, ha="center", va="bottom", fontsize=10, fontweight="bold")
    ax.set_xticks(range(len(cols)), [MODEL_LABELS[c[2:]] for c in cols], rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(ec)), ["\n".join(textwrap.wrap(r.AMR_Gene_Family, 42)) + f" ({r.ARO_Accession})"
                                   + (" *" if r.Resistance_Mechanism != common else "") for r in ec.itertuples()], fontsize=8)
    ax.spines[:].set_visible(False)
    ax.tick_params(length=0)
    ax.set_title("Mecanismo previsto pelos 8 modelos para os genes dos mecanismos raros", fontweight="bold", pad=34)
    others = sorted({short(m) for m in ec.Resistance_Mechanism if m != common})
    note = f"Mecanismo no CARD: {short(common)}" + (f"; * {', '.join(others)}" if others else "")
    ax.legend(handles=[Patch(color=MECHANISM_COLORS[m], label=short(m)) for m in MECHANISMS if m in used], loc="upper center",
              bbox_to_anchor=(0.5, -0.2), ncol=5, frameon=False, fontsize=9, title=note, title_fontsize=9)
    save(fig, output_path)


# ── Execução e catálogo ──────────────────────────────────────────────────────────
FIGURES = [
    ("identidade_teste_treino.csv", plot_identity_by_split, DATA_DIR / "identidade_teste_treino"),
    ("grupos_resumo.csv", plot_groups_per_threshold, DATA_DIR / "grupos_por_limiar"),
    ("grupos_resumo.csv", plot_largest_group, DATA_DIR / "maior_grupo_por_limiar"),
    ("familias_treino_teste.csv", plot_families_train_test, DATA_DIR / "familias_treino_teste"),
    ("categoria_isa_por_mecanismo.csv", plot_isa_category, DATA_DIR / "categoria_isa_por_mecanismo"),
    ("ramos_aro_exemplos.csv", plot_aro_branches, DATA_DIR / "ramos_aro"),
    ("atributos_por_mecanismo.csv", plot_depth_by_mechanism, DATA_DIR / "profundidade_por_mecanismo"),
    ("informacao_mutua_ontologia.csv", plot_mutual_information, DATA_DIR / "informacao_mutua"),
    ("regra_simples_ontologia.csv", plot_simple_rule, DATA_DIR / "regra_simples_ontologia"),
    ("atributos_por_mecanismo.csv", plot_drug_classes_by_mechanism, DATA_DIR / "classes_antibiotico_por_mecanismo"),
    ("cv_folds.csv", plot_cv_folds, ANALYSIS_DIR / "cv_folds"),
    ("recortes_teste.csv", plot_test_subsets, ANALYSIS_DIR / "recortes_teste"),
    ("ablacao_ontologia.csv", lambda p: plot_ablation("anotacao_curada", p), ANALYSIS_DIR / "ablacao_anotacao_curada"),
    ("ablacao_ontologia.csv", lambda p: plot_ablation("gene_novo", p), ANALYSIS_DIR / "ablacao_gene_novo"),
    ("sensibilidade_representacao.csv", plot_sensitivity_representation, ANALYSIS_DIR / "sensibilidade_representacao"),
    ("sensibilidade_balanceamento.csv", plot_sensitivity_balancing, ANALYSIS_DIR / "sensibilidade_balanceamento"),
    ("sensibilidade_ajuste.csv", lambda p: plot_sensitivity_tuning("cv", p), ANALYSIS_DIR / "sensibilidade_ajuste_cv"),
    ("erros_principais.csv", plot_main_errors, ANALYSIS_DIR / "erros_inativacao_alteracao"),
    ("estudo_caso_resumo.csv", plot_case_study, ANALYSIS_DIR / "estudo_caso"),
]


def generate_analysis_figures() -> None:
    for source, fn, path in FIGURES:
        if _need(source):
            fn(path)
    for col, label, scale in SEQ_FEATURES:
        plot_sequence_feature(col, label, scale, DATA_DIR / f"sequencia_{col}")


def write_figure_catalog() -> None:
    """Grava ``results/figuras_catalogo.json`` (o que cada figura mostra, fonte, por que, seção e legenda sugerida)."""
    import re

    from src.figure_catalog import CATALOG

    out = {}
    for entry in CATALOG:
        pattern = re.sub(r"\{[^}]*\}", "*", entry["arquivo"]) + ".png"
        out[entry["arquivo"]] = entry | {"gerada": any(PLOTS_DIR.glob(pattern))}
    (RESULTS_DIR / "figuras_catalogo.json").write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Catálogo de figuras: %s (%d entradas)", RESULTS_DIR / "figuras_catalogo.json", len(out))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    generate_analysis_figures()
    write_figure_catalog()
