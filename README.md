# Classificação Semântica de Genes de Resistência a Antibióticos

Código do experimento **"Avaliação do impacto da incorporação de ontologias em modelos de aprendizado de máquina para detecção de genes de resistência a antibióticos"** — Ciência da Computação, CEFET/RJ (2026).

**Autor:** Gustavo Barros Rosa de Andrade ·
 **Orientadora:** Kele Teixeira Belloze, D.Sc.

📄 **O trabalho completo está em [`Artigo.pdf`](Artigo.pdf).** Este README resume o que foi feito e explica como executar o código; fundamentação, trabalhos relacionados, justificativas metodológicas e discussão dos resultados estão no artigo, referenciado ao longo do texto.

> **Em desenvolvimento.** O trabalho está sendo revisado e o código atualizado; o artigo e partes deste README ainda podem não refletir a versão atual do código.

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

1. **Dados** (Seções 4.2 e 4.3) — integração do `aro_index.tsv` com as sequências proteicas do modelo homólogo do CARD (6.052 genes). Cada gene vai para um de três destinos:

   | Mecanismo | Genes | Destino |
   |---|---|---|
   | `antibiotic inactivation` | 5.239 | experimento |
   | `antibiotic target alteration` | 285 | experimento |
   | `antibiotic efflux` | 269 | experimento |
   | `antibiotic target protection` | 170 | experimento |
   | `antibiotic target replacement` | 70 | experimento |
   | `reduced permeability to antibiotic` | 8 | estudo de caso |
   | `resistance by host-dependent nutrient acquisition` | 1 | estudo de caso |
   | genes com dois mecanismos | 10 | fora (tarefa de rótulo único) |

   O experimento tem **6.033 genes em 5 classes**, com desbalanceamento de cerca de 75:1. As classes com 8 e 1 genes não têm genes suficientes para os folds de validação (0 ou 1 por fold) e viram um **estudo de caso qualitativo**: os modelos treinados com as 5 classes são aplicados a esses 9 genes, sem cálculo de F1.

2. **Atributos** (Seções 4.4 e 4.5):
   - **Pipeline A (baseline):** 73 atributos de sequência — 4 propriedades físico-químicas (comprimento, ponto isoelétrico, índice de instabilidade e carga em pH 7), composição de 20 aminoácidos e frequência de 49 pares de grupos de aminoácidos vizinhos, com um alfabeto reduzido de 7 grupos: {AGV}, {ILFP}, {YMTS}, {HNQW}, {RK}, {DE}, {C} (Shen et al. 2007). Peso molecular e aromaticidade não entram, por serem redundantes com o comprimento e com a soma das frações de F, W e Y.
   - **Pipeline B (enriquecido):** os mesmos 73 + 41 atributos ontológicos (114 no total) — profundidade, número de ancestrais e de filhos na hierarquia *is-a* da ARO e 38 indicadores de classe de antibiótico (coluna `Drug Class` do `aro_index.tsv`).
3. **Controle de leakage** (Seção 4.6) — correlação de Spearman entre atributos ontológicos e rótulo e um experimento de sanidade com os atributos ontológicos isolados.
4. **Treinamento e avaliação** (Seções 4.7 a 4.12) — partição treino/teste **por grupos de similaridade de sequência** (genes com identidade ≥ 70% e cobertura ≥ 50%, calculadas com o MMseqs2, ficam no mesmo grupo, e o grupo inteiro vai para o treino ou para o teste; ~80/20), validação cruzada por grupos (`StratifiedGroupKFold`, 5 folds), pré-processamento `MinMaxScaler` ajustado só no treino, sem seleção de atributos (todos os atributos entram no modelo), e quatro classificadores: Random Forest, XGBoost, Regressão Logística e SVM (RBF). A métrica principal é o **F1-score macro**, e a comparação A × B usa o **teste de Wilcoxon Signed-Rank** (α = 0,05).

Os dois pipelines diferem **apenas** nos atributos de entrada; todas as demais etapas são idênticas.

---

## Principais Resultados

F1-score macro no conjunto de teste (5 classes, 1.051 genes; divisão por grupos de similaridade de 70%) e teste de Wilcoxon sobre os 5 folds da validação cruzada por grupos (`results/tabela_comparativa.csv` e `results/testes_estatisticos.csv`):

| Modelo | Pipeline A (sequência) | Pipeline B (+ ontologia) | Ganho | Wilcoxon: p (folds com B > A) |
|---|---|---|---|---|
| Random Forest | 0,6407 | 0,8884 | +38,66% | 0,0625 (5/5) |
| XGBoost | 0,8212 | 0,9255 | +12,70% | 0,0625 (5/5) |
| Regressão Logística | 0,6110 | 0,9611 | +57,30% | 0,0625 (5/5) |
| SVM (RBF) | 0,6802 | **0,9733** | +43,10% | 0,0625 (5/5) |

Com 5 pares, o menor p possível do Wilcoxon é 0,0625: o teste não pode rejeitar H0 a 5%, mesmo com B > A em todos os folds.

O desempenho por divisão (aleatória e grupos de 90, 70 e 50%) está em `results/curva_limiar.csv` e na figura `plots/comparacao_pipelines/curva_limiar`.

---

## Instalação

Requer **Python 3.12**. As versões fixadas em `requirements.txt` — em especial `scikit-learn==1.4.1.post1` e `xgboost==3.2.0` — são as usadas para gerar os resultados; outras versões podem produzir pequenas diferenças numéricas.

```bash
git clone <url-do-repositorio>
cd <pasta-do-repositorio>

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # Linux/macOS

pip install -r requirements.txt
```

A divisão por grupos usa o **MMseqs2** (programa externo), na versão `18-8cc5c`. No Windows, baixe `mmseqs-win64.zip` da [página de releases](https://github.com/soedinglab/MMseqs2/releases/tag/18-8cc5c) e extraia em `tools/mmseqs_18-8cc5c/`, de modo que o executável fique em `tools/mmseqs_18-8cc5c/mmseqs/bin/mmseqs.exe`. Em outro sistema, ou em outro local, aponte a variável de ambiente `MMSEQS_BIN` para o executável. A versão e o SHA-256 usados ficam em `results/mmseqs_parametros.json`.

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
python -m src.homology_groups      # 2b. grupos de similaridade (MMseqs2, ~12 min)
python -m src.feature_extraction   # 3. atributos de sequência (Pipeline A)
python -m src.ontology_features    # 4. atributos ontológicos (Pipeline B)
python -m src.leakage_check        # 5. verificação de leakage e experimento de sanidade
python -m src.pipeline_builder     # 6. registra a configuração dos pipelines
python -m src.train_evaluate       # 7. treino, validação cruzada e teste
python -m src.statistical_tests    # 8. teste de Wilcoxon
python -m src.interpretability     # 9. importância de atributos
python -m src.visualization        # 10. figuras
python -m src.report_generator     # 11. tabelas comparativas, LaTeX e conclusões
python -m src.case_study           # 12. estudo de caso dos mecanismos raros
python -m src.sensitivity_representation  # 13. sensibilidade à representação da sequência (~20 min)
python -m src.threshold_curve      # 14. desempenho por limiar de similaridade (~20 min)
```

Ao final, `results/metrics_pipeline_a.csv`, `results/metrics_pipeline_b.csv`, `results/cv_folds.csv` e `results/testes_estatisticos.csv` contêm os valores no formato das Tabelas 5.1 a 5.6 do artigo.

### O que cada etapa faz

| Etapa | Módulo | Descrição | Saídas | Artigo |
|---|---|---|---|---|
| 1 | `data_loader` | Integra `aro_index.tsv` e o FASTA do modelo homólogo pelo ARO Accession (inner join), remove instâncias sem mecanismo de resistência e valida o resultado | `data/processed/dataset_integrado.csv` | 4.2 |
| 2 | `eda` | Distribuição de classes, razão de desbalanceamento, valores ausentes e estatísticas da ontologia | `plots/eda/` | 4.3 |
| 2b | `homology_groups` | Busca de todos contra todos com o MMseqs2 (identidade ≥ 40%, sensibilidade 7,5) e grupos por limiar (40, 50, 70 e 90%) como componentes conexos dos pares com identidade ≥ limiar e cobertura ≥ 50% das duas sequências; confere que nenhum par acima do limiar fica entre grupos diferentes | `results/grupos_similaridade.csv`, `results/grupos_resumo.csv`, `results/mmseqs_parametros.json` | — |
| 3 | `feature_extraction` | 73 atributos por sequência: propriedades físico-químicas (Biopython `ProtParam`), composição de aminoácidos e de pares de grupos do alfabeto reduzido. Gera também a representação de 426 atributos (com os 400 dipeptídeos), usada só na análise de sensibilidade | `data/processed/features_sequencia.csv`, `features_sequencia_estudo_caso.csv` e `features_sequencia_completa_426.csv` | 4.4 |
| 4 | `ontology_features` | 41 atributos: profundidade, ancestrais e filhos na hierarquia *is-a* da ARO e 38 indicadores de classe de antibiótico (coluna `Drug Class` do CARD). O estudo de caso recebe as mesmas colunas | `data/processed/features_ontologicas.csv` e `features_ontologicas_estudo_caso.csv` | 4.5 |
| 5 | `leakage_check` | Spearman entre atributos ontológicos e rótulo (remoção se \|ρ\| ≥ 0,95) e experimento de sanidade com Random Forest só com atributos ontológicos | `results/sanidade_leakage.json`, `results/spearman_features_ontologicas.csv` | 4.6 |
| 6 | `pipeline_builder` | Registra o pré-processamento e os hiperparâmetros (os pipelines são instanciados na etapa 7) | `models/hyperparameters.json` | 4.7, 4.9, 4.10 |
| 7 | `train_evaluate` | Partição por grupos de similaridade (70%; teste escolhido entre 100 sementes só pela proporção das classes, ~20% de cada), validação cruzada por grupos de 5 folds, treino final e avaliação no teste para 4 modelos × 2 pipelines. Métricas macro sobre as 5 classes declaradas; confere que cada fold tem genes de todas as classes | `results/metrics_pipeline_{a,b}.csv`, `results/cv_folds.csv`, `results/cv_contagem_classes.csv`, `results/composicao_dataset.csv`, `models/*.joblib` | 4.8, 4.11, 5.2–5.5 |
| 8 | `statistical_tests` | Wilcoxon Signed-Rank pareado sobre os folds (A vs B); com 5 folds, o menor p possível é 0,0625 | `results/testes_estatisticos.csv` | 4.12, 5.5.1 |
| 9 | `interpretability` | Importância Gini (Random Forest) e permutation importance de todos os modelos | `results/feature_importance_*.csv`, `results/permutation_importance.csv` | — |
| 10 | `visualization` | Curvas ROC macro One-vs-Rest, matrizes de confusão, F1 A vs B e ganho percentual (PNG 300 DPI e PDF) | `plots/` | 5.3, 5.4 |
| 11 | `report_generator` | Tabela comparativa, tabelas LaTeX e texto com a decisão sobre H0/H1 | `results/tabela_comparativa.csv`, `reports/tabelas_latex/`, `results/conclusoes_experimento.txt` | 5.3, 5.6 |
| 12 | `case_study` | Aplica os 8 modelos finais aos 9 genes dos mecanismos raros: classe prevista e probabilidade de cada uma das 5 classes (qualitativo, sem F1) | `results/estudo_caso_predicoes.csv`, `results/estudo_caso_resumo.csv` | — |
| 13 | `sensitivity_representation` | Repete treino, validação cruzada e teste com representações alternativas da sequência (426 atributos com e sem SelectKBest, k = 20/50/100, só as 26 básicas), com a mesma partição e os mesmos modelos; descritiva | `results/sensibilidade_representacao.csv`, `results/sensibilidade_representacao_folds.csv` | — |
| 14 | `threshold_curve` | Repete treino, validação cruzada e teste com quatro divisões (aleatória e grupos de 90, 70 e 50%) e mede, em cada uma, a identidade de cada gene do teste com o mais parecido do treino; descritiva | `results/curva_limiar.csv`, `results/curva_limiar_particoes.csv` | — |

O módulo `provenance` (`python -m src.provenance`) gera `results/data_provenance.json` com a versão do CARD e os hashes MD5 dos dados brutos. O arquivo versionado registra os dados usados no artigo; reexecutá-lo sobrescreve o registro com a data do dia.

#### Detalhes de implementação

- **Resíduos ambíguos:** 5 sequências contêm o resíduo `X`. Ele é excluído das propriedades físico-químicas e da composição de aminoácidos; nos pares de resíduos vizinhos (de grupos ou de aminoácidos), pares que o contêm são descartados e as frequências são normalizadas pelo total de pares válidos.
- **Classes:** `dataset_integrado.csv` contém 6.052 instâncias. As listas de classes ficam em `src/config.py` (`TARGET_CLASSES` e `CASE_STUDY_CLASSES`), e `src/experiment_config.py` separa os genes em experimento (6.033), estudo de caso (9) e multirrótulo (10), conferindo que a soma fecha. Um mecanismo que não esteja em nenhuma lista gera erro em vez de ser descartado. A classe `resistance by absence` não tem instâncias no FASTA do modelo homólogo (Seção 4.3.3).
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
│   ├── experiment_config.py           # Parâmetros do experimento, separação dos genes por classe, divisão por grupos e carga do dataset
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
│   ├── report_generator.py            # Tabelas comparativas, LaTeX e conclusões
│   ├── case_study.py                  # Estudo de caso dos mecanismos raros (9 genes)
│   ├── sensitivity_representation.py  # Sensibilidade à representação da sequência
│   ├── homology_groups.py             # Grupos de similaridade de sequência (MMseqs2)
│   ├── threshold_curve.py             # Desempenho por limiar de similaridade
│   └── compare_with_tcc.py            # Confronto com os valores publicados no artigo
├── tests/                             # Testes unitários e de propriedade (pytest + hypothesis)
├── requirements.txt
└── README.md
```

Os arquivos versionados em `results/`, `plots/` e `reports/` são regenerados pela execução, exceto `results/comparacao_tcc.csv` (confronto com os valores do artigo).

---

## Testes

```bash
pytest tests/ -v
```

A suíte (141 testes) cobre a separação dos genes por classe, a integração dos dados, a EDA, a extração de atributos de sequência e ontológicos, a verificação de leakage, os pipelines, a partição treino/teste e o teste estatístico, além de testes de propriedade (`hypothesis`) para a integridade do dataset final.

---

## Reprodutibilidade e Controle de Leakage

- `random_state=42` é fixado em todos os pontos de aleatoriedade (marcados com `# SEED FIXA`), e a execução é determinística.
- Os dados de entrada são identificados por versão e hash MD5 em `results/data_provenance.json`.
- A partição treino/teste é feita por grupos de similaridade, é determinística e seus índices são salvos em `data/processed/train_indices.npy` e `test_indices.npy`; a semente escolhida e a fração de cada classe no teste ficam em `results/particao_teste.json`. Nenhum grupo aparece no treino e no teste.
- O MinMaxScaler é ajustado exclusivamente no treino de cada fold, via `sklearn.Pipeline`; a não-sobreposição entre treino e teste é verificada por `assert`.
- Nenhum atributo ontológico é construído a partir do mecanismo de resistência. Pontos críticos são marcados com `# ANTI-LEAKAGE`.
- Com os dados do CARD 4.0.1 e as versões de `requirements.txt`, a execução completa é reproduzível.

Discussão das limitações e ameaças à validade: artigo, Seção 5.7.

---

## Observação sobre o Código

Parte do código-fonte original do experimento deixou de estar acessível após a execução que gerou os resultados do TCC. Por isso, os módulos correspondentes foram reimplementados a partir da metodologia descrita no trabalho, mantendo os mesmos dados (CARD 4.0.1), parâmetros, sementes aleatórias e etapas experimentais. Um detalhe de pré-processamento não especificado no texto foi definido de forma a manter a compatibilidade com a execução original. Com o ambiente definido em `requirements.txt`, a reimplementação reproduziu integralmente os resultados reportados no artigo.
