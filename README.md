# Classificação Semântica de Genes de Resistência a Antibióticos

Código do experimento **"Avaliação do impacto da incorporação de ontologias em modelos de aprendizado de máquina para detecção de genes de resistência a antibióticos"** — Ciência da Computação, CEFET/RJ (2026).

**Autor:** Gustavo Barros Rosa de Andrade ·
 **Orientadora:** Kele Teixeira Belloze, D.Sc.

📄 **O trabalho completo está em [`Artigo.pdf`](Artigo.pdf).** Este README resume o que foi feito e explica como executar o código; fundamentação, trabalhos relacionados, justificativas metodológicas e discussão dos resultados estão no artigo, referenciado ao longo do texto.

---

## Sobre o Projeto

Genes de resistência a antibióticos (ARGs) são normalmente classificados por modelos que usam apenas informações extraídas da sequência biológica. Este projeto investiga se acrescentar a esses modelos o conhecimento estruturado de uma ontologia especializada (**Antibiotic Resistance Ontology (ARO)**) melhora a classificação do **mecanismo de resistência** de cada gene.

Para isso, o código implementa um experimento controlado com dados do **CARD 4.0.1** (Comprehensive Antibiotic Resistance Database), no qual dois pipelines idênticos são comparados — um com atributos apenas de sequência e outro acrescido de atributos derivados da ARO —, com quatro classificadores e teste estatístico de significância.

### Pergunta de pesquisa

> *Qual é o impacto da incorporação de conhecimento semântico derivado de ontologias biológicas no desempenho de modelos de aprendizado de máquina para detecção e classificação de genes de resistência a antibióticos?*

(Artigo, Seção 1.2.)

---

## Metodologia em Resumo

Visão geral no Capítulo 4 do artigo (Figura 4.1).

1. **Dados** (Seções 4.2 e 4.3) — integração do `aro_index.tsv` com as sequências proteicas do modelo homólogo do CARD, resultando em **5.801 genes** distribuídos em 4 classes de mecanismo de resistência, com desbalanceamento extremo (654,9:1):

   | Classe | Instâncias |
   |---|---|
   | `antibiotic inactivation` | 5.239 |
   | `antibiotic target alteration` | 285 |
   | `antibiotic efflux` | 269 |
   | `reduced permeability to antibiotic` | 8 |

2. **Atributos** (Seções 4.4 e 4.5):
   - **Pipeline A (baseline):** 426 atributos de sequência — 6 propriedades físico-químicas, composição de 20 aminoácidos e de 400 dipeptídeos.
   - **Pipeline B (enriquecido):** os mesmos 426 + 39 atributos ontológicos — profundidade, número de ancestrais e de filhos na hierarquia *is-a* da ARO e 36 indicadores de classe de antibiótico.
3. **Controle de leakage** (Seção 4.6) — correlação de Spearman entre atributos ontológicos e rótulo e um experimento de sanidade com os atributos ontológicos isolados.
4. **Treinamento e avaliação** (Seções 4.7 a 4.12) — partição treino/teste 80/20 estratificada, validação cruzada estratificada de 10 folds, pré-processamento `MinMaxScaler → VarianceThreshold → SelectKBest(k=100)` ajustado só no treino, e quatro classificadores: Random Forest, XGBoost, Regressão Logística e SVM (RBF). A métrica principal é o **F1-score macro**, e a comparação A × B usa o **teste de Wilcoxon Signed-Rank** (α = 0,05).

Os dois pipelines diferem **apenas** nos atributos de entrada; todas as demais etapas são idênticas.

---

## Principais Resultados

F1-score macro no conjunto de teste e teste de Wilcoxon sobre os 10 folds de validação cruzada (artigo, Tabelas 5.3 e 5.6):

| Modelo | Pipeline A (sequência) | Pipeline B (+ ontologia) | Ganho | Wilcoxon (p) |
|---|---|---|---|---|
| Random Forest | 0,6550 | 0,7228 | +10,35% | 0,0020 ✓ |
| XGBoost | 0,6787 | **0,7305** | +7,63% | 0,0020 ✓ |
| Regressão Logística | 0,6271 | 0,7182 | +14,52% | 0,0645 |
| SVM (RBF) | 0,6605 | 0,7243 | +9,66% | 0,0020 ✓ |

✓ diferença estatisticamente significativa (α = 0,05).

- A incorporação dos atributos ontológicos melhora o desempenho nos quatro classificadores, com ganho médio de **+10,54%** no F1 macro.
- A melhora é estatisticamente significativa para Random Forest, XGBoost e SVM. Na Regressão Logística, o Pipeline B é superior em 9 dos 10 folds, sem atingir significância.

Demais métricas, curvas ROC, validação cruzada e ameaças à validade estão no Capítulo 5 do artigo.

---

## Instalação

Requer **Python 3.12**. As versões fixadas em `requirements.txt` — em especial `scikit-learn==1.4.1.post1` e `xgboost==3.2.0` — são as que reproduzem os resultados do artigo; outras versões podem produzir pequenas diferenças numéricas.

```bash
git clone <url-do-repositorio>
cd <pasta-do-repositorio>

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

---

## Obtenção dos Dados

Os arquivos do CARD e da ARO não são distribuídos neste repositório. O uso é regido pelos termos da McMaster University: gratuito para pesquisa acadêmica, e uso comercial exige licença (ver [card.mcmaster.ca/about](https://card.mcmaster.ca/about)).

1. Em [card.mcmaster.ca/download](https://card.mcmaster.ca/download), baixe os **dados do CARD versão 4.0.1** e extraia o conteúdo em `data/dataset/card-data/`. São usados `aro_index.tsv` e `protein_fasta_protein_homolog_model.fasta`.
2. Baixe a **ontologia (ARO) da mesma versão** e salve o arquivo `aro.owl` em `data/ontologia/`.
3. Confira se os arquivos são os mesmos usados no artigo comparando os hashes MD5 com os registrados em `results/data_provenance.json`:

   ```bash
   # Windows (PowerShell)
   Get-FileHash data\dataset\card-data\aro_index.tsv -Algorithm MD5
   # Linux/macOS
   md5sum data/dataset/card-data/aro_index.tsv
   ```

---

## Execução

Execute os comandos **na raiz do repositório**, com o ambiente virtual ativo. A execução completa leva entre 10 e 20 minutos em um computador pessoal; a etapa de interpretabilidade é a mais demorada.

```bash
python -m src.data_loader          # 1. integra CARD e sequências
python -m src.eda                  # 2. análise exploratória
python -m src.feature_extraction   # 3. atributos de sequência (Pipeline A)
python -m src.ontology_features    # 4. atributos ontológicos (Pipeline B)
python -m src.leakage_check        # 5. verificação de leakage e experimento de sanidade
python -m src.pipeline_builder     # 6. registra a configuração dos pipelines
python -m src.train_evaluate       # 7. treino, validação cruzada e teste
python -m src.statistical_tests    # 8. teste de Wilcoxon
python -m src.interpretability     # 9. importância de atributos
python -m src.visualization        # 10. figuras
python -m src.report_generator     # 11. tabelas comparativas, LaTeX e conclusões
```

Ao final, `results/metrics_pipeline_a.csv`, `results/metrics_pipeline_b.csv`, `results/cv_folds.csv` e `results/testes_estatisticos.csv` contêm os valores das Tabelas 5.1 a 5.6 do artigo.

### O que cada etapa faz

| Etapa | Módulo | Descrição | Saídas | Artigo |
|---|---|---|---|---|
| 1 | `data_loader` | Integra `aro_index.tsv` e o FASTA do modelo homólogo pelo ARO Accession (inner join), remove instâncias sem mecanismo de resistência e valida o resultado | `data/processed/dataset_integrado.csv` | 4.2 |
| 2 | `eda` | Distribuição de classes, razão de desbalanceamento, valores ausentes e estatísticas da ontologia | `plots/eda/` | 4.3 |
| 3 | `feature_extraction` | 426 atributos por sequência: propriedades físico-químicas (Biopython `ProtParam`), composição de aminoácidos e de dipeptídeos | `data/processed/features_sequencia.csv` | 4.4 |
| 4 | `ontology_features` | 39 atributos: profundidade, ancestrais e filhos na hierarquia *is-a* da ARO e 36 indicadores de classe de antibiótico (coluna `Drug Class` do CARD) | `data/processed/features_ontologicas.csv` | 4.5 |
| 5 | `leakage_check` | Spearman entre atributos ontológicos e rótulo (remoção se \|ρ\| ≥ 0,95) e experimento de sanidade com Random Forest só com atributos ontológicos | `results/sanidade_leakage.json`, `results/spearman_features_ontologicas.csv` | 4.6 |
| 6 | `pipeline_builder` | Registra o pré-processamento e os hiperparâmetros (os pipelines são instanciados na etapa 7) | `models/hyperparameters.json` | 4.7, 4.9, 4.10 |
| 7 | `train_evaluate` | Partição 80/20 estratificada, validação cruzada de 10 folds, treino final e avaliação no teste para 4 modelos × 2 pipelines | `results/metrics_pipeline_{a,b}.csv`, `results/cv_folds.csv`, `models/*.joblib` | 4.8, 4.11, 5.2–5.5 |
| 8 | `statistical_tests` | Wilcoxon Signed-Rank pareado sobre os 10 folds (A vs B) | `results/testes_estatisticos.csv` | 4.12, 5.5.1 |
| 9 | `interpretability` | Importância Gini (Random Forest) e permutation importance de todos os modelos | `results/feature_importance_*.csv`, `results/permutation_importance.csv` | — |
| 10 | `visualization` | Curvas ROC macro One-vs-Rest, matrizes de confusão, F1 A vs B e ganho percentual (PNG 300 DPI e PDF) | `plots/` | 5.3, 5.4 |
| 11 | `report_generator` | Tabela comparativa, tabelas LaTeX e texto com a decisão sobre H0/H1 | `results/tabela_comparativa.csv`, `reports/tabelas_latex/`, `results/conclusoes_experimento.txt` | 5.3, 5.6 |

O módulo `provenance` (`python -m src.provenance`) gera `results/data_provenance.json` com a versão do CARD e os hashes MD5 dos dados brutos. O arquivo versionado registra os dados usados no artigo; reexecutá-lo sobrescreve o registro com a data do dia.

#### Detalhes de implementação

- **Resíduos ambíguos:** 5 sequências contêm o resíduo `X`. Ele é excluído das propriedades físico-químicas e da composição de aminoácidos; nos dipeptídeos, pares que o contêm são descartados e as frequências são normalizadas pelo total de pares válidos.
- **Classes:** `dataset_integrado.csv` contém 6.052 instâncias; o experimento usa apenas as 4 classes da Seção 4.3.1 (filtro em `src/experiment_config.py`). Ficam fora `antibiotic target protection` (170), `antibiotic target replacement` (70) e rótulos múltiplos ou raros (11). A classe `resistance by absence` não tem instâncias no FASTA do modelo homólogo (Seção 4.3.3).
- **Ontologia:** o `aro.owl` do CARD 4.0.1 está em sintaxe OWL/XML e é lido com `xml.etree`, considerando apenas axiomas `SubClassOf` entre classes nomeadas.

---

## Estrutura do Repositório

```
.
├── Artigo.pdf                         # Trabalho completo
├── data/
│   ├── dataset/card-data/             # Dados brutos do CARD 4.0.1 (baixar — ver "Obtenção dos Dados")
│   ├── ontologia/aro.owl              # Ontologia ARO 4.0.1 (baixar)
│   └── processed/                     # Gerados pelo pipeline (dataset integrado, features, índices treino/teste)
├── models/                            # hyperparameters.json e modelos treinados (.joblib, gerados)
├── plots/                             # Figuras: EDA, curvas ROC, matrizes de confusão, comparação A vs B, importância
├── reports/tabelas_latex/             # Tabelas .tex no formato do artigo
├── results/                           # Métricas, validação cruzada, testes estatísticos e conclusões
├── src/
│   ├── config.py                      # Constantes globais e caminhos
│   ├── experiment_config.py           # Parâmetros do experimento (10 folds, 4 classes) e carga do dataset
│   ├── provenance.py                  # Versão e hashes MD5 dos dados brutos
│   ├── data_loader.py                 # Integração aro_index.tsv + FASTA
│   ├── eda.py                         # Análise exploratória
│   ├── feature_extraction.py          # Atributos de sequência (Pipeline A)
│   ├── ontology_features.py           # Atributos ontológicos (Pipeline B)
│   ├── leakage_check.py               # Spearman + experimento de sanidade
│   ├── pipeline_builder.py            # Pré-processamento e classificadores
│   ├── train_evaluate.py              # Partição, validação cruzada e avaliação
│   ├── statistical_tests.py           # Wilcoxon Signed-Rank
│   ├── interpretability.py            # Gini e permutation importance
│   ├── visualization.py               # Figuras
│   └── report_generator.py            # Tabelas comparativas, LaTeX e conclusões
├── tests/                             # Testes unitários e de propriedade (pytest + hypothesis)
├── requirements.txt
└── README.md
```

Os arquivos versionados em `results/`, `plots/` e `reports/` são os resultados de referência do artigo e são regenerados pela execução.

---

## Testes

```bash
pytest tests/ -v
```

A suíte (124 testes) cobre a integração dos dados, a EDA, a extração de atributos de sequência e ontológicos, a verificação de leakage, os pipelines, a partição treino/teste e o teste estatístico, além de testes de propriedade (`hypothesis`) para a integridade do dataset final.

---

## Reprodutibilidade e Controle de Leakage

- `random_state=42` é fixado em todos os pontos de aleatoriedade (marcados com `# SEED FIXA`), e a execução é determinística.
- Os dados de entrada são identificados por versão e hash MD5 em `results/data_provenance.json`.
- A partição treino/teste é determinística e seus índices são salvos em `data/processed/train_indices.npy` e `test_indices.npy`.
- Todos os transformadores (MinMaxScaler, VarianceThreshold, SelectKBest) são ajustados exclusivamente no treino de cada fold, via `sklearn.Pipeline`; a não-sobreposição entre treino e teste é verificada por `assert`.
- Nenhum atributo ontológico é construído a partir do mecanismo de resistência. Pontos críticos são marcados com `# ANTI-LEAKAGE`.
- Com os dados do CARD 4.0.1 e as versões de `requirements.txt`, a execução completa reproduz os resultados reportados no artigo.

Discussão das limitações e ameaças à validade: artigo, Seção 5.7.

---

## Observação sobre o Código

Parte do código-fonte original do experimento deixou de estar acessível após a execução que gerou os resultados do TCC. Por isso, os módulos correspondentes foram reimplementados a partir da metodologia descrita no trabalho, mantendo os mesmos dados (CARD 4.0.1), parâmetros, sementes aleatórias e etapas experimentais. Um detalhe de pré-processamento não especificado no texto foi definido de forma a manter a compatibilidade com a execução original. Com o ambiente definido em `requirements.txt`, a reimplementação reproduz integralmente os resultados reportados no TCC.
