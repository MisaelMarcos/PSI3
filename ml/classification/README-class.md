# Classification — Organização futura

Este arquivo documenta temporariamente a organização planejada para a pasta `classification/`.
A estrutura poderá ser ajustada conforme os experimentos e decisões metodológicas do projeto evoluírem.

## Objetivo

Desenvolver e comparar modelos de **classificação supervisionada** para categorizar a intensidade da chuva, usando dados meteorológicos do Recife.

> **Atenção metodológica:** a definição das classes, do horizonte de previsão e das variáveis de entrada deve ser formalizada antes do treinamento. Evitar Data Leakage, especialmente se o objetivo for prever chuva futura. Se a precipitação observada for usada para construir o target, ela não deve ser incluída como feature do mesmo instante.

## Modelos planejados

Os experimentos serão organizados e desenvolvidos por três grupos distintos. Cuidado ao dividir as tarefas entre a equipe.

### 1. Modelos tradicionais

- Logistic Regression — baseline linear.
- Decision Tree — modelo interpretável baseado em regras.
- K-Nearest Neighbors (KNN) — classificação por proximidade.
- Support Vector Machine (SVM) — modelo baseado em fronteiras de separação.

### 2. Ensembles de árvores

- Random Forest.
- Extra Trees.

### 3. Gradient Boosting

- XGBoost.
- LightGBM.
- CatBoost.

A proposta inicial é comparar os nove algoritmos, sem assumir previamente qual será o melhor.

## Estrutura proposta

```text
classification/
├── README.md                  # Este documento: planejamento temporário
├── preprocessing.py           # Preparação, seleção de features e divisão temporal
├── train_baselines.py         # Logistic Regression, Decision Tree, KNN e SVM
├── train_ensembles.py         # Random Forest e Extra Trees
├── train_boosting.py          # XGBoost, LightGBM e CatBoost
├── evaluation.py              # Métricas, matrizes de confusão e avaliação
├── comparison.py              # Consolidação e comparação dos resultados
└── results/
    ├── metrics/                # Tabelas/arquivos de métricas
    ├── confusion_matrices/     # Matrizes de confusão
    └── figures/                # Gráficos comparativos
```

A estrutura acima é uma **proposta** inicial. Arquivos e diretórios poderão ser adicionados quando necessário.
