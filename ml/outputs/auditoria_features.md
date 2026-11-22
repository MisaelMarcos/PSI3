# Auditoria de Features, Redundancia, Risco de Leakage e Dataset de Trabalho (Recife - INMET A301)

Gerado em: 2026-11-22T01:34:12.822232+00:00
Base bruta (nao modificada): `dados/df_recife_datetime.csv` (33371 registros horarios)

## 1. Inventario do Pipeline Auditado

- **Features derivadas candidatas consolidadas**: 29 (minimo exigido: 8).
- **Variaveis de entrada em Clustering (K-Means)**: 8 -> `['pressao_mb', 'temp_c', 'temp_orvalho_c', 'umid_pct', 'hora_sin', 'hora_cos', 'mes_sin', 'mes_cos']`.
- **Variaveis de entrada em Classificacao**: 33 (4 fisicas originais + 29 candidatas).
- **Colunas excluidas por redundancia bruta**: `['pressao_max_mb', 'pressao_min_mb', 'temp_max_c', 'temp_min_c', 'orvalho_max_c', 'orvalho_min_c', 'umid_max_pct', 'umid_min_pct']`.
- **Colunas excluidas por excesso de nulos**: `['vento_dir_gr', 'vento_rajada_ms', 'vento_vel_ms']`.

## 2. Features Candidatas Consolidadas (todas estritamente <= t)

1. **Ciclos temporais continuos (4)**: `hora_sin`, `hora_cos`, `mes_sin`, `mes_cos`.
2. **Sazonalidade regional (1)**: `estacao_chuvosa` (meses de abril a julho em Recife).
3. **Derivadas termodinamicas e amplitudes intra-horarias (4)**: `orvalho_depressao`, `amp_temp_1h`, `amp_pressao_1h`, `amp_umid_1h`.
4. **Tendencias meteorologicas ate t (8)**: `pressao_delta_1h`, `pressao_delta_3h`, `pressao_delta_6h`, `temp_delta_1h`, `temp_delta_3h`, `orvalho_delta_3h`, `umid_delta_1h`, `umid_delta_3h`.
5. **Memoria e acumulados de chuva observada ate t (12 - apenas classificacao)**: `chuva_1h`, `chuva_3h`, `chuva_6h`, `chuva_12h`, `chuva_24h`, `chuva_max_3h`, `chuva_max_6h`, `chuva_lag_1h`, `chuva_lag_3h`, `chuva_lag_6h`, `chuva_lag_12h`, `chuva_lag_24h`.

## 3. Relevancia e Correlacoes (Top 15 por importancia RF em 2019)

| feature | tipo | corr_precip_atual | corr_acum_fut_3h | corr_alvo_3h_acum_10mm | rf_imp_2019_acum_10mm | anova_f_clusters | entra_clustering | entra_classificacao |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| chuva_1h | Candidata derivada | 1.0000 | 0.3650 | 0.2721 | 0.0804 | 264.3700 | False | True |
| chuva_6h | Candidata derivada | 0.5689 | 0.3573 | 0.2670 | 0.0741 | 395.1500 | False | True |
| chuva_12h | Candidata derivada | 0.4247 | 0.3011 | 0.2313 | 0.0702 | 287.7800 | False | True |
| chuva_24h | Candidata derivada | 0.3342 | 0.2794 | 0.2012 | 0.0606 | 414.4500 | False | True |
| chuva_3h | Candidata derivada | 0.7229 | 0.3809 | 0.2844 | 0.0564 | 397.9100 | False | True |
| chuva_max_3h | Candidata derivada | 0.7160 | 0.3470 | 0.2554 | 0.0464 | 433.1700 | False | True |
| chuva_max_6h | Candidata derivada | 0.5556 | 0.3105 | 0.2255 | 0.0461 | 426.1200 | False | True |
| temp_c | Original | -0.1341 | -0.1429 | -0.0693 | 0.0381 | 30067.2000 | True | True |
| temp_orvalho_c | Original | 0.1104 | 0.1393 | 0.0926 | 0.0364 | 5325.6900 | True | True |
| pressao_delta_3h | Candidata derivada | 0.0357 | 0.0418 | 0.0339 | 0.0348 | 1855.0800 | False | True |
| pressao_delta_6h | Candidata derivada | 0.0453 | 0.0401 | 0.0295 | 0.0345 | 990.9100 | False | True |
| orvalho_depressao | Candidata derivada | -0.1827 | -0.2053 | -0.1118 | 0.0340 | 32055.0900 | False | True |
| chuva_lag_1h | Candidata derivada | 0.3904 | 0.2658 | 0.1969 | 0.0333 | 224.1100 | False | True |
| umid_pct | Original | 0.1936 | 0.2150 | 0.1179 | 0.0316 | 30163.5000 | True | True |
| pressao_mb | Original | 0.0402 | 0.0349 | -0.0020 | 0.0308 | 10470.1400 | True | True |

Arquivo completo de correlacoes: `ml/outputs/correlacoes_features.csv`.
Arquivo completo de relevancia: `ml/outputs/relevancia_features.csv`.

## 4. Redundancia (|r| >= 0,70) e Decisao: Clustering vs. Classificacao

Foram identificados **52 pares** com |r| >= 0,70 (registrados em `ml/outputs/redundancia_pares.csv`). Principais pares:

| variavel_a | variavel_b | pearson_r | decisao_tecnica |
| --- | --- | --- | --- |
| umid_pct | orvalho_depressao | -0.9952 | orvalho_depressao excluida do clustering por correlacao quase perfeita com umid_pct; evita dupla contagem euclidiana. |
| pressao_max_mb | pressao_min_mb | 0.9947 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| pressao_mb | pressao_min_mb | 0.9933 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| pressao_mb | pressao_max_mb | 0.9917 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| umid_pct | umid_min_pct | 0.9827 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_c | temp_max_c | 0.9825 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| umid_min_pct | orvalho_depressao | -0.9756 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_max_c | temp_min_c | 0.9672 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_c | temp_min_c | 0.9645 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| umid_max_pct | umid_min_pct | 0.9595 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| umid_pct | umid_max_pct | 0.9565 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| chuva_3h | chuva_max_3h | 0.9542 | Variaveis de chuva excluidas do clustering (leakage de alvo); mantidas apenas na classificacao como memoria multi-horizonte <= t. |
| umid_max_pct | orvalho_depressao | -0.9518 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_orvalho_c | orvalho_min_c | 0.9342 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_orvalho_c | orvalho_max_c | 0.9200 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| chuva_6h | chuva_max_6h | 0.9158 | Variaveis de chuva excluidas do clustering (leakage de alvo); mantidas apenas na classificacao como memoria multi-horizonte <= t. |
| temp_delta_3h | umid_delta_3h | -0.9019 | Derivada excluida do clustering para manter espaco enxuto; mantida na classificacao (Random Forest tolera correlacao entre derivadas). |
| temp_max_c | umid_min_pct | -0.8842 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_max_c | orvalho_depressao | 0.8781 | Extremos max/min excluidos de ambos os modelos; mantida leitura horaria principal e amplitude intra-horaria derivada (amp_*_1h). |
| temp_c | orvalho_depressao | 0.8763 | Derivada excluida do clustering para manter espaco enxuto; mantida na classificacao (Random Forest tolera correlacao entre derivadas). |

### Resumo da separacao Clustering vs. Classificacao
- **Clustering (K-Means, k=3, StandardScaler)**: utiliza apenas as **8 features** de estado atmosferico instantaneo e ciclos (`pressao_mb`, `temp_c`, `temp_orvalho_c`, `umid_pct`, `hora_sin`, `hora_cos`, `mes_sin`, `mes_cos`). Exclui toda variavel de chuva (evita viciar os grupos pelo alvo), exclui `orvalho_depressao` (quase perfeitamente correlacionada com `umid_pct`, evitando dupla contagem na distancia euclidiana) e exclui deltas/amplitudes redundantes.
- **Classificacao Supervisionada (Random Forest)**: utiliza **33 features** (4 fisicas originais + 29 derivadas <= t), pois arvores de decisao sao invariantes a escala monotona e se beneficiam de sinais de persistencia (`chuva_*`), tendencia barometrica/termica (`*_delta_*`) e amplitude intra-horaria (`amp_*_1h`).

### Perfil dos 3 Clusters Obtidos (Validacao Externa com Chuva)

| cluster | horas | pressao_media | temp_media | orvalho_medio | umidade_media | pct_horas_chuva | chuva_acumulada_mm |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 11476 | 1012.1700 | 28.8970 | 21.1890 | 63.5560 | 2.3090 | 147.2000 |
| 1 | 11409 | 1011.6500 | 25.2450 | 22.4650 | 84.9570 | 17.7230 | 4188.0000 |
| 2 | 10486 | 1014.8130 | 23.4930 | 20.8290 | 85.4800 | 21.3520 | 3171.4000 |

## 5. Governanca e Risco de Data Leakage

1. **Leakage de Alvo no Clustering**: `precip_mm`, `precip_mm_limpa` e `chuva_*` sao proibidas em `FEATURES_CLUSTERING` (verificado por `validar_sem_leakage(..., modo='clustering')`).
2. **Leakage Temporal Futuro na Classificacao**: `max_fut_2h`, `acum_fut_2h`, `max_fut_3h`, `acum_fut_3h` e todas as 8 colunas `alvo_*` usam `shift(-1..-3)` e estao estritamente fora de `X` (verificado por `validar_sem_leakage(..., modo='classificacao')`).
3. **Leakage de Identificadores e Metadados Temporais**: `Unnamed: 0`, `Data`, `Hora UTC`, `datetime`, `datetime_utc`, `datetime_local`, `ano`, `mes`, `dia`, `hora` ficam isolados no bloco identificador `df_id`.
4. **Separacao temporal da classificacao**: importancias de RF e escolha de limiares usam apenas 2019 (com `TimeSeriesSplit(gap=3)` na validacao cruzada), mantendo 2020 como holdout. A auditoria de clustering e os perfis descritivos usam a base completa.

## 6. Datasets de Trabalho Atualizados

- `dados/df_recife_features.csv`: [33371, 40] — identificadores separados (`datetime_utc`, `datetime_local`, `ano`), 33 features de entrada + `cluster`/`cluster_0..2`, **0 nulos** e **0 colunas de alvo/futuro/chuva bruta**.
- `dados/df_recife_cluster_features.csv`: [33371, 10] — `datetime`, 8 features de clustering (padrao `kmeans_recife.ipynb`) e `cluster`, sem chuva e sem alvos.
- `dados/df_recife_alvos.csv`: [33371, 17] — identificadores + precipitacao limpa + agregados futuros e 8 colunas `alvo_*` isolados em arquivo proprio.

## 7. Checklist Final de Evidencias

- [x] **Item 1**: `ml/features.py` e notebooks de Recife auditados; CSV bruto (`dados/df_recife_datetime.csv`) preservado.
- [x] **Item 2**: 29 features candidatas consolidadas em `ml/features.py` (ciclos, sazonalidade, derivadas meteorologicas e lags/acumulados).
- [x] **Item 3**: Correlacoes (Pearson/Spearman) e relevancia (RF 2019 + ANOVA F) registradas em `ml/outputs/correlacoes_features.csv` e `ml/outputs/relevancia_features.csv`.
- [x] **Item 4**: Redundancia (52 pares com |r| >= 0,70) documentada em `ml/outputs/redundancia_pares.csv` com justificativa clustering vs. classificacao.
- [x] **Item 5**: Risco de leakage documentado e garantido por `validar_sem_leakage` e `separar_matrizes_modelo`.
- [x] **Item 6**: Datasets de trabalho (`dados/df_recife_features.csv`, `dados/df_recife_cluster_features.csv`) e alvos (`dados/df_recife_alvos.csv`) gerados com separacao estrita.
- [x] **Item 7**: Evidencias consolidadas em `ml/outputs/auditoria_features.md` e `ml/outputs/auditoria_features.json`.
