# Auditoria de Features, Redundancia, Risco de Leakage e Dataset de Trabalho (Recife - INMET A301)

Gerado em: 2026-11-22T00:42:52.366128+00:00
Base bruta (nao modificada): `dados/df_recife.csv` (17544 registros horarios)

## 1. Inventario do Pipeline Auditado

- **Features derivadas candidatas consolidadas**: 29 (minimo exigido: 8).
- **Variaveis de entrada em Clustering (K-Means)**: 8 -> `['pressao_mb', 'temp_c', 'temp_orvalho_c', 'umid_pct', 'hora_sin', 'hora_cos', 'mes_sin', 'mes_cos']`.
- **Variaveis de entrada em Classificacao**: 33 (4 fisicas originais + 29 candidatas).
- **Colunas excluidas por redundancia bruta**: `['pressao_max_mb', 'pressao_min_mb', 'temp_max_c', 'temp_min_c', 'orvalho_max_c', 'orvalho_min_c', 'umid_max_pct', 'umid_min_pct']`.
- **Colunas excluidas por excesso de nulos (~66%)**: `['vento_dir_gr', 'vento_rajada_ms', 'vento_vel_ms']`.

## 2. Features Candidatas Consolidadas (todas estritamente <= t)

1. **Ciclos temporais continuos (4)**: `hora_sin`, `hora_cos`, `mes_sin`, `mes_cos`.
2. **Sazonalidade regional (1)**: `estacao_chuvosa` (meses de abril a julho em Recife).
3. **Derivadas termodinamicas e amplitudes intra-horarias (4)**: `orvalho_depressao`, `amp_temp_1h`, `amp_pressao_1h`, `amp_umid_1h`.
4. **Tendencias meteorologicas ate t (8)**: `pressao_delta_1h`, `pressao_delta_3h`, `pressao_delta_6h`, `temp_delta_1h`, `temp_delta_3h`, `orvalho_delta_3h`, `umid_delta_1h`, `umid_delta_3h`.
5. **Memoria e acumulados de chuva observada ate t (12 - apenas classificacao)**: `chuva_1h`, `chuva_3h`, `chuva_6h`, `chuva_12h`, `chuva_24h`, `chuva_max_3h`, `chuva_max_6h`, `chuva_lag_1h`, `chuva_lag_3h`, `chuva_lag_6h`, `chuva_lag_12h`, `chuva_lag_24h`.

## 3. Relevancia e Correlacoes (Top 15 por importancia RF em 2019)

| feature | tipo | corr_precip_atual | corr_acum_fut_3h | corr_alvo_3h_acum_10mm | rf_imp_2019_acum_10mm | anova_f_clusters | entra_clustering | entra_classificacao |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chuva_6h | Candidata derivada | 0.5679 | 0.3606 | 0.2683 | 0.0770 | 194.7100 | False | True |
| chuva_1h | Candidata derivada | 1.0000 | 0.3614 | 0.2585 | 0.0729 | 129.4000 | False | True |
| chuva_12h | Candidata derivada | 0.4173 | 0.2898 | 0.2297 | 0.0724 | 142.7700 | False | True |
| chuva_24h | Candidata derivada | 0.3050 | 0.2372 | 0.1771 | 0.0590 | 213.7400 | False | True |
| chuva_3h | Candidata derivada | 0.7155 | 0.3922 | 0.2812 | 0.0549 | 196.9700 | False | True |
| chuva_max_6h | Candidata derivada | 0.5389 | 0.2926 | 0.2095 | 0.0500 | 194.1500 | False | True |
| chuva_max_3h | Candidata derivada | 0.6977 | 0.3344 | 0.2358 | 0.0469 | 203.6100 | False | True |
| pressao_delta_3h | Candidata derivada | 0.0256 | 0.0402 | 0.0265 | 0.0384 | 940.5000 | False | True |
| temp_orvalho_c | Original | 0.0860 | 0.1162 | 0.0792 | 0.0377 | 3585.0100 | True | True |
| chuva_lag_1h | Candidata derivada | 0.3566 | 0.2722 | 0.1937 | 0.0367 | 104.1700 | False | True |
| temp_c | Original | -0.1371 | -0.1453 | -0.0719 | 0.0345 | 15242.5700 | True | True |
| orvalho_depressao | Candidata derivada | -0.1794 | -0.2031 | -0.1117 | 0.0336 | 16700.6300 | False | True |
| pressao_mb | Original | 0.0658 | 0.0751 | 0.0314 | 0.0332 | 5737.4200 | True | True |
| pressao_delta_6h | Candidata derivada | 0.0406 | 0.0477 | 0.0283 | 0.0322 | 541.1400 | False | True |
| umid_pct | Original | 0.1900 | 0.2131 | 0.1187 | 0.0309 | 15681.3100 | True | True |

Arquivo completo de correlacoes: `ml/outputs/correlacoes_features.csv`.
Arquivo completo de relevancia: `ml/outputs/relevancia_features.csv`.

## 4. Redundancia (|r| >= 0,70) e Decisao: Clustering vs. Classificacao

Foram identificados **51 pares** com |r| >= 0,70 (registrados em `ml/outputs/redundancia_pares.csv`). Principais pares:

| variavel_a | variavel_b | pearson_r | decisao_tecnica |
| --- | --- | --- | --- |
| umid_pct | orvalho_depressao | -0.9957 | orvalho_depressao excluida do clustering (|r|=0,9957 evitaria dupla contagem euclidiana) e mantida apenas na classificacao. |
| pressao_max_mb | pressao_min_mb | 0.9941 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| pressao_mb | pressao_min_mb | 0.9926 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| pressao_mb | pressao_max_mb | 0.9910 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| umid_pct | umid_min_pct | 0.9833 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_c | temp_max_c | 0.9826 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| umid_min_pct | orvalho_depressao | -0.9768 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_max_c | temp_min_c | 0.9663 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_c | temp_min_c | 0.9636 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| umid_max_pct | umid_min_pct | 0.9614 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| umid_pct | umid_max_pct | 0.9573 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| umid_max_pct | orvalho_depressao | -0.9532 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| chuva_3h | chuva_max_3h | 0.9513 | Variaveis de chuva excluidas do clustering (leakage de alvo); mantidas apenas na classificacao como memoria multi-horizonte <= t. |
| temp_orvalho_c | orvalho_min_c | 0.9365 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_orvalho_c | orvalho_max_c | 0.9275 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| chuva_6h | chuva_max_6h | 0.9024 | Variaveis de chuva excluidas do clustering (leakage de alvo); mantidas apenas na classificacao como memoria multi-horizonte <= t. |
| temp_delta_3h | umid_delta_3h | -0.9005 | Derivada excluida do clustering para manter espaco enxuto; mantida na classificacao (Random Forest tolera correlacao entre derivadas). |
| orvalho_max_c | orvalho_min_c | 0.8731 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_max_c | umid_min_pct | -0.8710 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_max_c | orvalho_depressao | 0.8658 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |

### Resumo da separacao Clustering vs. Classificacao
- **Clustering (K-Means, k=3, StandardScaler)**: utiliza apenas as **8 features** de estado atmosferico instantaneo e ciclos (`pressao_mb`, `temp_c`, `temp_orvalho_c`, `umid_pct`, `hora_sin`, `hora_cos`, `mes_sin`, `mes_cos`). Exclui toda variavel de chuva (evita viciar os grupos pelo alvo), exclui `orvalho_depressao` (`r = -0,9957` com `umid_pct`, evitando dupla contagem na distancia euclidiana) e exclui deltas/amplitudes redundantes.
- **Classificacao Supervisionada (Random Forest)**: utiliza **33 features** (4 fisicas originais + 29 derivadas <= t), pois arvores de decisao sao invariantes a escala monotona e se beneficiam de sinais de persistencia (`chuva_*`), tendencia barometrica/termica (`*_delta_*`) e amplitude intra-horaria (`amp_*_1h`).

### Perfil dos 3 Clusters Obtidos (Validacao Externa com Chuva)

| cluster | horas | pressao_media | temp_media | orvalho_medio | umidade_media | pct_horas_chuva | chuva_acumulada_mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 5945 | 1011.9940 | 29.1420 | 21.2730 | 62.9800 | 1.8500 | 64.6000 |
| 1 | 6076 | 1011.7420 | 25.4160 | 22.5720 | 84.6260 | 15.3720 | 1817.0000 |
| 2 | 5523 | 1014.7180 | 23.7060 | 20.6480 | 83.5300 | 21.7090 | 1856.8000 |

## 5. Governanca e Risco de Data Leakage

1. **Leakage de Alvo no Clustering**: `precip_mm`, `precip_mm_limpa` e `chuva_*` sao proibidas em `FEATURES_CLUSTERING` (verificado por `validar_sem_leakage(..., modo='clustering')`).
2. **Leakage Temporal Futuro na Classificacao**: `max_fut_2h`, `acum_fut_2h`, `max_fut_3h`, `acum_fut_3h` e todas as 8 colunas `alvo_*` usam `shift(-1..-3)` e estao estritamente fora de `X` (verificado por `validar_sem_leakage(..., modo='classificacao')`).
3. **Leakage de Identificadores e Metadados Temporais**: `Unnamed: 0`, `Data`, `Hora UTC`, `datetime`, `datetime_utc`, `datetime_local`, `ano`, `mes`, `dia`, `hora` ficam isolados no bloco identificador `df_id`.
4. **Leakage de Treino/Teste e Hiperparametros**: importancias de RF, KMeans de comparacao e escolha de limiares usam apenas 2019 (com `TimeSeriesSplit(gap=3)` na validacao cruzada), mantendo 2020 como holdout cego.

## 6. Datasets de Trabalho Atualizados

- `dados/df_recife_features.csv`: [17544, 40] — identificadores separados (`datetime_utc`, `datetime_local`, `ano`), 33 features de entrada + `cluster`/`cluster_0..2`, **0 nulos** e **0 colunas de alvo/futuro/chuva bruta**.
- `dados/df_recife_cluster_features.csv`: [17544, 10] — `datetime`, 8 features de clustering (padrao `kmeans_recife.ipynb`) e `cluster`, sem chuva e sem alvos.
- `dados/df_recife_alvos.csv`: [17544, 17] — identificadores + precipitacao limpa + agregados futuros e 8 colunas `alvo_*` isolados em arquivo proprio.

## 7. Checklist Final de Evidencias

- [x] **Item 1**: `ml/features.py` e notebooks de Recife auditados; alteracoes preexistentes preservadas e CSV bruto (`dados/df_recife.csv`) intacto.
- [x] **Item 2**: 29 features candidatas consolidadas em `ml/features.py` (ciclos, sazonalidade, derivadas meteorologicas e lags/acumulados).
- [x] **Item 3**: Correlacoes (Pearson/Spearman) e relevancia (RF 2019 + ANOVA F) registradas em `ml/outputs/correlacoes_features.csv` e `ml/outputs/relevancia_features.csv`.
- [x] **Item 4**: Redundancia (51 pares com |r| >= 0,70) documentada em `ml/outputs/redundancia_pares.csv` com justificativa clustering vs. classificacao.
- [x] **Item 5**: Risco de leakage documentado e garantido por `validar_sem_leakage` e `separar_matrizes_modelo`.
- [x] **Item 6**: Datasets de trabalho (`dados/df_recife_features.csv`, `dados/df_recife_cluster_features.csv`) e alvos (`dados/df_recife_alvos.csv`) gerados com separacao estrita.
- [x] **Item 7**: Evidencias consolidadas em `ml/outputs/auditoria_features.md` e `ml/outputs/auditoria_features.json`.
