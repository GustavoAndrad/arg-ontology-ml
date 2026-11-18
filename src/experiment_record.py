"""
experiment_record.py — Configuração experimental: hiperparâmetros, versões, dados e sementes.

Por que: para o experimento ser reproduzível, o texto precisa informar todos os hiperparâmetros,
as versões das bibliotecas, a versão dos dados e as sementes. Tudo aqui é lido do
código, dos modelos treinados e dos arquivos de resultado, para evitar erro de
transcrição.

Hiperparâmetros:
    * ``get_params()`` de cada classificador, como construído em
      ``src.pipeline_builder`` — os mesmos nos Pipelines A e B;
    * ``definido_no_codigo`` separa o que foi escolhido à mão do que ficou no
      padrão da biblioteca;
    * no XGBoost, vários valores aparecem como ``None`` no ``get_params()`` ("use o
      padrão da biblioteca"). O valor efetivo é lido da configuração do modelo
      treinado (``Booster.save_config()``, ``models/pipeline_a_xgboost.joblib``),
      em ``valor_efetivo``. Ex.: taxa de aprendizado 0,3 e profundidade máxima 6.

Ambiente (``results/ambiente.json``): versão do Python e de cada biblioteca do
``requirements.txt`` instalada; MMseqs2 (de ``results/mmseqs_parametros.json``);
dados (de ``results/data_provenance.json``); sementes (modelos, partição de teste
e validação cruzada, de ``results/particao_teste.json``).

Saídas:
    results/hiperparametros.csv                       — todos os hiperparâmetros
    reports/tabelas_latex/tabela_hiperparametros.tex  — os principais (seção 5.1)
    results/ambiente.json                             — versões, dados e sementes

Uso (depois de ``src.train_evaluate``):
    python -m src.experiment_record
"""

from __future__ import annotations

import json
import logging
import platform
import sys
from importlib import metadata
from pathlib import Path

import joblib
import pandas as pd

from src.config import _PROJECT_ROOT
from src.experiment_config import MODEL_NAMES, MODELS_DIR, RANDOM_STATE, RESULTS_DIR, SIMILARITY_THRESHOLD
from src.pipeline_builder import CLASS_BALANCING, MODEL_LABELS, _model_params, classifier_params
from src.report_generator import _latex_table, _write

logger = logging.getLogger(__name__)

# Hiperparâmetros mostrados na tabela do texto (a lista completa fica no CSV).
# Escolhidos por controlarem a capacidade do modelo ou o balanceamento; os demais
# são de execução (n_jobs, verbose, cache) ou não se aplicam à configuração usada.
MAIN_PARAMS: dict[str, list[str]] = {
    "random_forest": [
        "n_estimators", "criterion", "max_depth", "max_features", "min_samples_split",
        "min_samples_leaf", "bootstrap", "class_weight",
    ],
    "xgboost": [
        "n_estimators", "learning_rate", "max_depth", "min_child_weight", "subsample",
        "colsample_bytree", "reg_lambda", "reg_alpha", "gamma", "objective", "eval_metric",
    ],
    # tree_method fica só no CSV: a configuração do XGBoost 3.2.0 informa "auto", e o
    # método que "auto" escolhe não está documentado nas fontes do trabalho.
    "logistic_regression": ["penalty", "C", "solver", "max_iter", "multi_class", "class_weight"],
    "svm": ["kernel", "C", "gamma", "probability", "decision_function_shape", "class_weight"],
}

# Nome no get_params() do XGBoost → nome na configuração interna (save_config).
XGB_CONFIG_KEYS: dict[str, str] = {
    "learning_rate": "eta",
    "max_depth": "max_depth",
    "min_child_weight": "min_child_weight",
    "subsample": "subsample",
    "colsample_bytree": "colsample_bytree",
    "colsample_bylevel": "colsample_bylevel",
    "colsample_bynode": "colsample_bynode",
    "reg_lambda": "lambda",
    "reg_alpha": "alpha",
    "gamma": "gamma",
    "max_delta_step": "max_delta_step",
    "max_bin": "max_bin",
    "max_leaves": "max_leaves",
    "grow_policy": "grow_policy",
    "tree_method": "tree_method",
    "objective": "name",  # learner.objective.name
    "booster": "name",  # learner.gradient_booster.name
}


def _find_key(config, key: str):
    """Primeiro valor de ``key`` numa configuração aninhada (dicts e listas)."""
    if isinstance(config, dict):
        if key in config and not isinstance(config[key], (dict, list)):
            return config[key]
        for v in config.values():
            found = _find_key(v, key)
            if found is not None:
                return found
    elif isinstance(config, list):
        for v in config:
            found = _find_key(v, key)
            if found is not None:
                return found
    return None


def _clean_number(v):
    """'0.300000012' (float32 do XGBoost) → '0.3'; outros textos ficam como estão."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return v
    return f"{f:.6g}"


def xgboost_effective_params(model_path: Path = MODELS_DIR / "pipeline_a_xgboost.joblib") -> dict[str, str]:
    """Valores efetivos dos hiperparâmetros do XGBoost treinado (vazio se não houver modelo)."""
    if not model_path.exists():
        logger.warning("Modelo %s não encontrado; valores efetivos do XGBoost não registrados.", model_path)
        return {}
    clf = joblib.load(model_path).named_steps["clf"]
    config = json.loads(clf.get_booster().save_config())
    learner = config["learner"]
    out = {}
    for param, key in XGB_CONFIG_KEYS.items():
        if param == "objective":
            value = learner["objective"]["name"]
        elif param == "booster":
            value = learner["gradient_booster"]["name"]
        else:
            value = _find_key(learner, key)
        if value is not None:
            out[param] = _clean_number(value)
    return out


def hyperparameter_table() -> pd.DataFrame:
    """Uma linha por (modelo, hiperparâmetro), com valor, origem e valor efetivo."""
    effective = {"xgboost": xgboost_effective_params()}
    rows = []
    for name in MODEL_NAMES:
        chosen = _model_params()[name]
        for param, value in classifier_params(name).items():
            rows.append(
                {
                    "modelo": name,
                    "parametro": param,
                    "valor": value,
                    "definido_no_codigo": param in chosen,
                    "valor_efetivo": effective.get(name, {}).get(param, value),
                    "na_tabela_do_texto": param in MAIN_PARAMS[name],
                }
            )
    return pd.DataFrame(rows)


def _tex(v) -> str:
    text = str(v).replace("_", r"\_")
    try:
        float(text)
        return text.replace(".", ",")
    except ValueError:
        return text


def hyperparameter_latex(table: pd.DataFrame) -> str:
    rows = []
    for name in MODEL_NAMES:
        sub = table[(table["modelo"] == name) & table["na_tabela_do_texto"]].set_index("parametro")
        for i, param in enumerate(MAIN_PARAMS[name]):
            r = sub.loc[param]
            mark = r"$^{*}$" if r["definido_no_codigo"] else ""
            rows.append([MODEL_LABELS[name] if i == 0 else "", r"\texttt{" + _tex(param) + "}", _tex(r["valor_efetivo"]) + mark])
        if name == "xgboost":
            rows.append(["", r"peso por classe", r"\texttt{sample\_weight} balanceado$^{*}$"])
    caption = (
        "Principais hiperparâmetros dos classificadores, iguais nos Pipelines A e B. "
        "$^{*}$Definido no código; os demais são os valores padrão das bibliotecas. "
        f"Semente (\\texttt{{random\\_state}}) = {RANDOM_STATE} em todos."
    )
    return _latex_table(["Modelo", "Hiperparâmetro", "Valor"], rows, caption, "tab:hiperparametros")


def _installed_versions() -> dict[str, str]:
    """Versão instalada de cada pacote do requirements.txt."""
    versions = {}
    req = _PROJECT_ROOT / "requirements.txt"
    for line in req.read_text(encoding="utf-8").splitlines():
        name = line.split("==")[0].strip()
        if not name or name.startswith("#"):
            continue
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = "não instalado"
    return versions


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def environment_record() -> dict:
    """Versões, dados e sementes do experimento."""
    mmseqs = _read_json(RESULTS_DIR / "mmseqs_parametros.json")
    prov = _read_json(RESULTS_DIR / "data_provenance.json")
    part = _read_json(RESULTS_DIR / "particao_teste.json")
    return {
        "python": platform.python_version(),
        "sistema": platform.platform(),
        "bibliotecas": _installed_versions(),
        "mmseqs2": {k: mmseqs.get(k) for k in ("versao_release", "versao_commit", "sha256_executavel")},
        "dados": {
            "card_versao": prov.get("card_version"),
            "data_de_acesso": prov.get("access_date"),
            "aro_owl_sha256": prov.get("files", {}).get("aro_owl", {}).get("sha256"),
            "aro_index_sha256": prov.get("files", {}).get("aro_index", {}).get("sha256"),
        },
        "sementes": {
            "modelos_random_state": RANDOM_STATE,
            "particao_teste": part.get("semente"),
            "particao_teste_fold": part.get("fold"),
            "validacao_cruzada_repeticoes": part.get("sementes_cv"),
            "regra_validacao_cruzada": part.get("regra_sementes_cv"),
            "candidatas_testadas": "0 a 99 (escolha pela distribuição das classes, sem olhar o desempenho)",
            "bootstrap_teste": RANDOM_STATE,
        },
        "limiar_similaridade": SIMILARITY_THRESHOLD,
        "balanceamento": CLASS_BALANCING,
    }


def run() -> None:
    table = hyperparameter_table()
    table.to_csv(RESULTS_DIR / "hiperparametros.csv", index=False)
    logger.info("Hiperparâmetros: %s (%d linhas)", RESULTS_DIR / "hiperparametros.csv", len(table))
    _write("tabela_hiperparametros.tex", hyperparameter_latex(table))
    env = environment_record()
    (RESULTS_DIR / "ambiente.json").write_text(json.dumps(env, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Ambiente: %s", RESULTS_DIR / "ambiente.json")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )
    run()
