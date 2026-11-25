# Experimento HDBSCAN — Densidade Variavel, Estabilidade e Comparacao com DBSCAN (Recife, INMET A301)

## 1. Base e Pipeline de Preparacao

- **Base:** `dados/df_recife_datetime.csv` (**33,371** registros horarios, 2018–2021).

- **Features (8, identicas ao K-Means e ao `dbscan_recife.ipynb`):** `pressao_mb, temp_c, temp_orvalho_c, umid_pct, hora_sin, hora_cos, mes_sin, mes_cos` (precipitacao excluida da entrada e usada apenas para validacao externa).

- **Pre-processamento:** `StandardScaler` + `PCA(n_components=6)` retendo **97.61%** da variancia total.

- **Implementacao:** `sklearn.cluster.HDBSCAN` (scikit-learn `1.9.1`), distancia euclidiana.

## 2. Grade de Parametros (`min_cluster_size`, `min_samples` e `method`)

| method | min_cluster_size | min_samples | clusters | ruido (%) | cobertura (%) | Silhouette | Davies-Bouldin | Calinski-Harabasz | Prob. media |

|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| eom | 25 | None (= mcs) | 4 | 31.12% | 68.88% | 0.0287 | 1.958 | 2677.9 | 0.981 |
| eom | 25 | 5 | 6 | 5.54% | 94.46% | 0.0083 | 3.775 | 2914.7 | 0.963 |
| eom | 25 | 10 | 6 | 14.94% | 85.06% | 0.0132 | 3.648 | 2745.9 | 0.955 |
| eom | 25 | 18 | 6 | 29.44% | 70.56% | 0.0269 | 3.392 | 2463.0 | 0.951 |
| eom | 25 | 24 | 6 | 39.67% | 60.33% | 0.0379 | 3.227 | 2252.0 | 0.969 |
| eom | 50 | None (= mcs) | 2 | 46.85% | 53.15% | 0.2073 | 1.928 | 4379.5 | 0.984 |
| eom | 50 | 5 | 6 | 5.54% | 94.46% | 0.0083 | 3.775 | 2914.7 | 0.957 |
| eom | 50 | 10 | 6 | 14.94% | 85.06% | 0.0132 | 3.648 | 2745.9 | 0.948 |
| eom | 50 | 18 | 6 | 29.44% | 70.56% | 0.0269 | 3.392 | 2463.0 | 0.948 |
| eom | 50 | 24 | 6 | 39.67% | 60.33% | 0.0379 | 3.227 | 2252.0 | 0.959 |
| eom | 100 | None (= mcs) | 4 | 86.72% | 13.28% | 0.3072 | 0.885 | 1984.3 | 0.906 |
| eom | 100 | 5 | 6 | 5.54% | 94.46% | 0.0083 | 3.775 | 2914.7 | 0.957 |
| eom | 100 | 10 | 6 | 14.94% | 85.06% | 0.0132 | 3.648 | 2745.9 | 0.936 |
| eom | 100 | 18 | 6 | 29.44% | 70.56% | 0.0269 | 3.392 | 2463.0 | 0.940 |
| eom | 100 | 24 | 6 | 39.67% | 60.33% | 0.0379 | 3.227 | 2252.0 | 0.929 |
| eom | 200 | None (= mcs) | 2 | 91.08% | 8.92% | 0.4902 | 0.782 | 2630.4 | 0.929 |
| eom | 200 | 5 | 6 | 5.54% | 94.46% | 0.0083 | 3.775 | 2914.7 | 0.956 |
| eom | 200 | 10 | 6 | 14.94% | 85.06% | 0.0132 | 3.648 | 2745.9 | 0.938 |
| eom | 200 | 18 | 6 | 29.44% | 70.56% | 0.0269 | 3.392 | 2463.0 | 0.935 |
| eom | 200 | 24 | 6 | 39.67% | 60.33% | 0.0379 | 3.227 | 2252.0 | 0.917 |
| eom | 400 | None (= mcs) | 2 | 92.54% | 7.46% | 0.5178 | 0.693 | 2209.9 | 0.960 |
| eom | 400 | 5 | 6 | 5.54% | 94.46% | 0.0083 | 3.775 | 2914.7 | 0.946 |
| eom | 400 | 10 | 6 | 14.94% | 85.06% | 0.0132 | 3.648 | 2745.9 | 0.931 |
| eom | 400 | 18 | 6 | 29.44% | 70.56% | 0.0269 | 3.392 | 2463.0 | 0.934 |
| eom | 400 | 24 | 6 | 39.67% | 60.33% | 0.0379 | 3.227 | 2252.0 | 0.907 |
| leaf | 50 | 10 | 46 | 73.52% | 26.48% | 0.0985 | 1.903 | 1063.8 | 0.918 |
| leaf | 100 | 10 | 26 | 67.62% | 32.38% | 0.0194 | 2.127 | 1078.7 | 0.891 |
| leaf | 100 | 18 | 19 | 78.84% | 21.16% | 0.0326 | 1.872 | 1065.0 | 0.919 |
| leaf | 200 | 18 | 16 | 70.68% | 29.32% | -0.0257 | 2.348 | 1026.2 | 0.907 |

## 3. Avaliacao de Estabilidade (Subamostragem 80% x 5 rodadas)

| Configuracao | Clusters | Ruido (%) | Silhouette | Prob. media | ARI subamostra (media +/- DP) | ARI nao-ruido | NMI | Score |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| eom mcs=25, ms=5 | 6 | 5.54% | 0.0083 | 0.963 | 0.967 +/- 0.002 | 1.000 | 0.950 | 0.769 |
| eom mcs=25, ms=10 | 6 | 14.94% | 0.0132 | 0.955 | 0.908 +/- 0.003 | 1.000 | 0.904 | 0.722 |
| eom mcs=25, ms=18 | 6 | 29.44% | 0.0269 | 0.951 | 0.812 +/- 0.005 | 1.000 | 0.846 | 0.649 |

## 4. Configuracao Escolhida e Justificativa: `eom mcs=25, ms=5`

- **Configuracao escolhida:** `min_cluster_size=25`, `min_samples=5`, `cluster_selection_method='eom'`.
- **Resultado:** **6 clusters**, **5.54% de ruido** (94.46% de cobertura), Silhouette = **0.0083**, Davies-Bouldin = **3.775**, probabilidade media de pertinencia = **0.963** (92.8% dos pontos agrupados com prob >= 0,80) e **ARI de estabilidade = 0.967 +/- 0.002**.
- **Justificativa:** Configuracoes com `min_samples=None` e `min_cluster_size >= 100` produzem Silhouette maior (>0,30), mas descartam mais de 85% dos dados como ruido — o que inviabiliza a caracterizacao dos regimes climaticos da cidade. A configuracao `eom mcs=25, ms=5` entrega o melhor compromisso entre estabilidade estrutural (ARI elevado), alta cobertura e interpretabilidade fisica dos regimes.

## 5. Comparacao Direta: HDBSCAN vs. DBSCAN (`dbscan_recife.ipynb`)

| Algoritmo | Parametros | Clusters | Ruido (%) | Cobertura (%) | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI vs DBSCAN C6 | NMI vs DBSCAN C6 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| HDBSCAN (eom mcs=25, ms=5) | eom mcs=25, ms=5 | 6 | 5.54% | 94.46% | 0.0083 | 3.775 | 2914.7 | 0.697 | 0.738 |
| DBSCAN_C6_escolhida | eps=0.66, min_samples=24 | 4 | 10.06% | 89.94% | 0.0222 | 2.131 | 3271.9 | 1.000 | 1.000 |
| DBSCAN_C1_aprovada | eps=0.62, min_samples=12 | 4 | 6.28% | 93.72% | 0.0185 | 2.163 | 3299.5 | 0.920 | 0.880 |

- **Concordancia HDBSCAN x DBSCAN C6:** `ARI global = 0.697`, `ARI nao-ruido = 0.752`, `NMI = 0.738`.
- **Densidade variavel:** A razao entre o cluster menos denso e o mais denso (medida pela mediana de `k-dist 15`) foi de **1.20x** no HDBSCAN contra **1.08x** no DBSCAN C6.
- **Pontos resgatados vs. descartados:** O HDBSCAN agrupou **1,793** horas que o DBSCAN C6 classificou como ruido, e classificou como ruido **287** horas que o DBSCAN C6 havia agrupado.

## 6. Perfis Meteorologicos, Densidade Local e Validacao Externa com Chuva (HDBSCAN)

| Cluster | Registros | Partic. (%) | k-dist 15 | Prob. media | Temp. (°C) | Orvalho (°C) | Umidade (%) | Pressao (mB) | % Horas Chuva |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| -1 | 1,850 | 5.54% | 0.889 | 0.000 | 26.25 | 21.32 | 75.86 | 1013.11 | 23.73% |
| 0 | 2,790 | 8.36% | 0.512 | 0.919 | 23.97 | 20.83 | 83.66 | 1014.96 | 23.44% |
| 1 | 2,736 | 8.20% | 0.567 | 0.952 | 24.10 | 20.43 | 80.96 | 1015.70 | 18.02% |
| 2 | 11,128 | 33.35% | 0.471 | 0.981 | 25.85 | 22.28 | 81.81 | 1012.31 | 17.91% |
| 3 | 9,868 | 29.57% | 0.506 | 0.972 | 26.37 | 20.93 | 73.14 | 1012.67 | 5.82% |
| 4 | 2,521 | 7.55% | 0.482 | 0.931 | 27.29 | 21.56 | 72.09 | 1011.76 | 6.66% |
| 5 | 2,478 | 7.43% | 0.491 | 0.940 | 27.41 | 22.36 | 75.10 | 1011.00 | 8.68% |

## 7. Figuras Geradas
- `figuras/grade_parametros_hdbscan.png`
- `figuras/estabilidade_hdbscan.png`
- `figuras/clusters_pca_fisico_hdbscan.png`
- `figuras/comparacao_hdbscan_vs_dbscan.png`
- `figuras/perfil_clusters_hdbscan.png`
