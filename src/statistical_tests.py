"""
statistical_tests.py — Teste t corrigido para validação cruzada repetida, com Holm (Seções 4.12 e 5.5.1).

Por que trocar o Wilcoxon (TCC2, ponto 5; apontamento D8 do Diego):
    * os resultados de folds diferentes **não são independentes**: os conjuntos de
      treino se sobrepõem (em 5 folds, dois treinos dividem 3/4 dos genes). O
      Wilcoxon e o teste t comuns tratam os folds como amostras independentes e
      subestimam a variância, dando p pequenos demais (na prévia: p ≈ 2 × 10⁻¹⁵ com
      o Wilcoxon contra p = 0,01 com o teste corrigido, para a mesma diferença);
    * com 5 folds, o Wilcoxon exato nem consegue chegar a p < 0,05 (mínimo 0,0625).

Teste t corrigido (Nadeau e Bengio, 2003; na forma para validação cruzada repetida
de Bouckaert e Frank, 2004 — referências a adicionar):
    d_j = F1_B − F1_A na estimativa j (mesma repetição e mesmo fold para A e B),
    j = 1..J, com J = 10 repetições × 5 folds = 50;
    t = média(d) / sqrt((1/J + n_val/n_ajuste) · s²),   s² = variância amostral de d;
    graus de liberdade = J − 1.
O termo n_val/n_ajuste é a correção: ele aumenta a variância para levar em conta a
sobreposição dos treinos. A razão é calculada dos tamanhos reais dos folds (média
de n_validacao/n_ajuste em ``results/cv_folds.csv``), porque com grupos os folds
não têm exatamente o mesmo tamanho (seria 1/4 com folds iguais).
O intervalo de 95% da diferença usa o mesmo erro padrão:
    média(d) ± t_{0,975; J−1} · sqrt((1/J + n_val/n_ajuste) · s²).
O mesmo intervalo, aplicado ao F1 de cada pipeline, dá o intervalo da média da CV.

Quais comparações são testadas (decisões 3 e 4 do ponto 5; decisão 3 do ponto 10):
    * só o **F1-macro** (uma métrica, definida antes, para não multiplicar testes);
    * B × A para os 4 classificadores no cenário "anotação curada" (o experimento
      atual). O cenário "gene novo" (ponto 4) acrescenta mais 4: a família
      confirmatória final tem 8 testes.
    * correção de Holm sobre a família (referência a adicionar): controla a chance
      de pelo menos um falso positivo entre os testes, sem ser tão conservadora
      quanto Bonferroni. **Por enquanto a família tem 4 testes**; os p corrigidos
      mudam quando o ponto 4 acrescentar os outros 4.
    * Hipóteses: H0: média(d) = 0; H1: média(d) ≠ 0 (bilateral); α = 0,05, aplicado ao
      p corrigido por Holm.
Todas as outras comparações (por classe, ablação, curva por limiar, sensibilidades)
são descritivas: intervalos, sem teste.

Saída:
    results/testes_estatisticos.csv

Uso:
    python -m src.statistical_tests
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from scipy import stats

from src.experiment_config import ALPHA_STAT, MODEL_NAMES, RESULTS_DIR

logger = logging.getLogger(__name__)
TESTED_METRIC = "f1_macro"
CONFIDENCE = 1 - ALPHA_STAT  # intervalo de 95%, coerente com α = 0,05


def corrected_std_error(values: np.ndarray, test_train_ratio: float) -> float:
    """Erro padrão corrigido de Nadeau e Bengio: sqrt((1/J + n_val/n_ajuste) · s²)."""
    values = np.asarray(values, dtype=float)
    return float(np.sqrt((1 / len(values) + test_train_ratio) * np.var(values, ddof=1)))


def corrected_interval(values, test_train_ratio: float, confidence: float = CONFIDENCE) -> tuple[float, float]:
    """Intervalo da média com o erro padrão corrigido e t com J − 1 graus de liberdade."""
    values = np.asarray(values, dtype=float)
    half = stats.t.ppf(0.5 + confidence / 2, df=len(values) - 1) * corrected_std_error(values, test_train_ratio)
    return float(values.mean() - half), float(values.mean() + half)


def corrected_resampled_ttest(scores_a, scores_b, test_train_ratio: float) -> dict:
    """
    Teste t corrigido para validação cruzada repetida, pareado (B − A), bilateral.

    ``scores_a`` e ``scores_b`` precisam estar na mesma ordem (mesma repetição e mesmo
    fold em cada posição).
    """
    a, b = np.asarray(scores_a, dtype=float), np.asarray(scores_b, dtype=float)
    if len(a) != len(b):
        raise ValueError("Amostras pareadas com tamanhos diferentes.")
    d = b - a
    df = len(d) - 1
    se = corrected_std_error(d, test_train_ratio)
    if se == 0:
        t_stat, p = (0.0, 1.0) if d.mean() == 0 else (float(np.sign(d.mean()) * np.inf), 0.0)
    else:
        t_stat = float(d.mean() / se)
        p = float(2 * stats.t.sf(abs(t_stat), df))
    low, high = corrected_interval(d, test_train_ratio)
    return {
        "n_estimativas": len(d),
        "media_a": float(a.mean()),
        "media_b": float(b.mean()),
        "diferenca_media": float(d.mean()),
        "ic95_inf": low,
        "ic95_sup": high,
        "t": t_stat,
        "graus_liberdade": df,
        "p_value": p,
        "razao_validacao_ajuste": test_train_ratio,
        "estimativas_b_maior": int((d > 0).sum()),
        "estimativas_a_maior": int((d < 0).sum()),
    }


def holm_adjust(p_values) -> np.ndarray:
    """
    Correção de Holm (passo a passo descendente): ordena os p, multiplica o i-ésimo menor
    por (m − i + 1), impõe monotonicidade e limita a 1. Devolve na ordem original.
    """
    p = np.asarray(p_values, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        adjusted[idx] = running
    return adjusted


def test_train_ratio(cv_folds: pd.DataFrame) -> float:
    """Média de n_validacao / n_ajuste nos folds (as divisões são as mesmas para todos os modelos)."""
    folds = cv_folds.drop_duplicates(["repeticao", "fold"])
    return float((folds["n_validacao"] / folds["n_ajuste"]).mean())


def run_all_tests(cv_folds: pd.DataFrame, alpha: float = ALPHA_STAT, metric: str = TESTED_METRIC) -> pd.DataFrame:
    """B × A para cada modelo, com Holm sobre a família de comparações deste cenário."""
    ratio = test_train_ratio(cv_folds)
    rows = []
    for model in MODEL_NAMES:
        sub = cv_folds[cv_folds["modelo"] == model]
        a = sub[sub["pipeline"] == "a"].sort_values(["repeticao", "fold"])
        b = sub[sub["pipeline"] == "b"].sort_values(["repeticao", "fold"])
        if not (a[["repeticao", "fold"]].to_numpy() == b[["repeticao", "fold"]].to_numpy()).all():
            raise ValueError(f"{model}: folds de A e B não estão pareados.")
        res = corrected_resampled_ttest(a[metric].to_numpy(), b[metric].to_numpy(), ratio)
        ic_a, ic_b = corrected_interval(a[metric], ratio), corrected_interval(b[metric], ratio)
        rows.append(
            {
                "cenario": "anotacao_curada",
                "modelo": model,
                "metrica": metric,
                **res,
                "ic95_media_a": f"[{ic_a[0]:.4f}; {ic_a[1]:.4f}]",
                "ic95_media_b": f"[{ic_b[0]:.4f}; {ic_b[1]:.4f}]",
            }
        )
    out = pd.DataFrame(rows)
    out["p_holm"] = holm_adjust(out["p_value"])
    out["testes_na_familia"] = len(out)
    out["alpha"] = alpha
    out["decisao"] = np.where(out["p_holm"] < alpha, "rejeita_H0", "nao_rejeita_H0")
    return out


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    folds = pd.read_csv(RESULTS_DIR / "cv_folds.csv")
    result = run_all_tests(folds)
    result.to_csv(RESULTS_DIR / "testes_estatisticos.csv", index=False)
    for _, r in result.iterrows():
        logger.info(
            "%-20s B−A = %+.4f [%.4f; %.4f] | t=%.2f (gl %d) p=%.2e p_Holm=%.2e -> %s (B>A em %d/%d)",
            r["modelo"], r["diferenca_media"], r["ic95_inf"], r["ic95_sup"], r["t"], r["graus_liberdade"],
            r["p_value"], r["p_holm"], r["decisao"], r["estimativas_b_maior"], r["n_estimativas"],
        )
