"""DIA 4 — Mean Shift e avaliacao de clustering por densidade (Recife, INMET A301).

Usa a mesma matriz do K-Means, DBSCAN e HDBSCAN (8 features, StandardScaler + PCA >= 95%)
e a mesma funcao de metricas do HDBSCAN (`_evaluate_partition`), para que a comparacao
seja direta. Gera `reports/meanshift_recife/` com relatorio, CSVs, rotulos por registro
e figuras para o artigo.
"""

from __future__ import annotations

import json
from pathlib import Path
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.cluster import DBSCAN, KMeans, MeanShift, estimate_bandwidth
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, RAW_CSV
from experimento_hdbscan_recife import (
    DBSCAN_REFS,
    _canonical_labels,
    _evaluate_partition,
    _sha256_file,
)
from experimento_kmeans import _integer, _number
from features import (
    COLUNAS_ORIGINAIS_METEOROLOGICAS,
    features_kmeans_disponiveis,
    preparar_base,
    validar_sem_leakage,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "meanshift_recife"
FIGURES_DIR = OUTPUT_DIR / "figuras"

PCA_VARIANCE = 0.95
SILHOUETTE_SAMPLE_SIZE = 5000  # mesmo tamanho e semente do experimento HDBSCAN
BANDWIDTH_SAMPLE = 3000
# O bandwidth e derivado do quantil das distancias entre pares (estimate_bandwidth):
# quantis menores => raio menor => mais clusters.
QUANTIS = (0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30)
MIN_BIN_FREQ = 5
# Regra de escolha (fixada antes de olhar os resultados): entre as configuracoes com
# 3 a 10 clusters e nenhum cluster menor que 1% dos registros, maior Silhouette;
# empate desfeito pelo menor Davies-Bouldin.
K_MIN, K_MAX, MENOR_CLUSTER_MIN_PCT = 3, 10, 1.0
K_KMEANS = 5  # K final de reports/kmeans_recife
DBSCAN_REF = "DBSCAN_C6_escolhida"

NOMES_ATRIB = {
    "pressao_mb": ("pressão", "mB"),
    "temp_c": ("temperatura", "°C"),
    "temp_orvalho_c": ("ponto de orvalho", "°C"),
    "umid_pct": ("umidade", "%"),
}


def _save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _fmt(v, casas: int = 3) -> str:
    return "—" if v is None or (isinstance(v, float) and np.isnan(v)) else _number(float(v), casas)


def _orfaos_pct(X: np.ndarray, centros: np.ndarray, bandwidth: float) -> float:
    """% de pontos a mais de `bandwidth` do centro mais proximo (seriam -1 com cluster_all=False)."""
    d2 = ((X[:, None, :] - centros[None, :, :]) ** 2).sum(axis=2).min(axis=1)
    return float((np.sqrt(d2) > bandwidth).mean() * 100.0)


def _descrever(perfil_z: pd.Series) -> str:
    """Descricao textual a partir dos desvios (z-score) em relacao a media geral."""
    termos = {
        "temp_c": ("mais quente", "mais fresco"),
        "umid_pct": ("mais úmido", "mais seco"),
        "pressao_mb": ("pressão mais alta", "pressão mais baixa"),
        "temp_orvalho_c": ("orvalho mais alto", "orvalho mais baixo"),
    }
    partes = [
        f"{termos[c][0] if z > 0 else termos[c][1]} ({_number(z, 1)}σ)"
        for c, z in perfil_z.reindex(perfil_z.abs().sort_values(ascending=False).index).items()
        if abs(z) >= 0.5
    ]
    return ", ".join(partes) if partes else "próximo da média geral em todos os atributos"


def _grade(X: np.ndarray, sample_idx: np.ndarray) -> tuple[pd.DataFrame, dict]:
    linhas, rotulos = [], {}
    for q in QUANTIS:
        t0 = time.perf_counter()
        bw = float(estimate_bandwidth(X, quantile=q, n_samples=BANDWIDTH_SAMPLE, random_state=RANDOM_STATE))
        t_bw = time.perf_counter() - t0
        t0 = time.perf_counter()
        modelo = MeanShift(bandwidth=bw, bin_seeding=True, min_bin_freq=MIN_BIN_FREQ, cluster_all=True, n_jobs=-1)
        rot = modelo.fit_predict(X)
        t_fit = time.perf_counter() - t0
        ev = _evaluate_partition(X, rot, sample_idx)
        linhas.append(
            {
                "quantil": q,
                "bandwidth": bw,
                "tempo_bandwidth_s": t_bw,
                "tempo_fit_s": t_fit,
                "orfaos_pct": _orfaos_pct(X, modelo.cluster_centers_, bw),
                **ev,
            }
        )
        rotulos[q] = (rot, modelo.cluster_centers_)
        print(
            f"      q={q:<5} bw={bw:.3f} | k={ev['n_clusters']:<2} | sil={_fmt(ev['silhouette_nao_ruido'])} "
            f"| DBI={_fmt(ev['davies_bouldin_nao_ruido'], 2)} | {t_fit:5.1f}s"
        )
    return pd.DataFrame(linhas), rotulos


def _escolher(grade: pd.DataFrame) -> pd.Series:
    elegiveis = grade[
        grade["n_clusters"].between(K_MIN, K_MAX) & (grade["menor_cluster_pct"] >= MENOR_CLUSTER_MIN_PCT)
    ]
    if elegiveis.empty:
        raise ValueError("Nenhuma configuracao atende a regra de escolha; revise a grade de quantis.")
    return elegiveis.sort_values(["silhouette_nao_ruido", "davies_bouldin_nao_ruido"], ascending=[False, True]).iloc[0]


def _perfis(base: pd.DataFrame, rot: np.ndarray) -> pd.DataFrame:
    df = base[COLUNAS_ORIGINAIS_METEOROLOGICAS].copy()
    df["cluster"] = rot
    df["chuva"] = base["precip_mm_limpa"].to_numpy()
    df["hora"] = base["datetime_local"].dt.hour.to_numpy()
    df["estacao_chuvosa"] = base["datetime_local"].dt.month.isin([4, 5, 6, 7]).to_numpy()
    media, desvio = df[COLUNAS_ORIGINAIS_METEOROLOGICAS].mean(), df[COLUNAS_ORIGINAIS_METEOROLOGICAS].std()
    linhas = []
    for c, g in df.groupby("cluster"):
        z = (g[COLUNAS_ORIGINAIS_METEOROLOGICAS].mean() - media) / desvio
        linha = {"cluster": int(c), "registros": len(g), "participacao_pct": len(g) / len(df) * 100}
        for col in COLUNAS_ORIGINAIS_METEOROLOGICAS:
            linha[f"{col}_media"] = g[col].mean()
            linha[f"{col}_mediana"] = g[col].median()
            linha[f"{col}_desvio"] = g[col].std()
            linha[f"{col}_z"] = z[col]
        linha["hora_local_mediana"] = g["hora"].median()
        linha["estacao_chuvosa_pct"] = g["estacao_chuvosa"].mean() * 100
        linha["horas_com_chuva_pct"] = (g["chuva"] > 0).mean() * 100
        linha["chuva_media_mm"] = g["chuva"].mean()
        linha["descricao"] = _descrever(z)
        linhas.append(linha)
    return pd.DataFrame(linhas)


def _comparar(X: np.ndarray, sample_idx: np.ndarray, rot_ms: np.ndarray, t_ms: float, desc_ms: str, base: pd.DataFrame):
    t0 = time.perf_counter()
    rot_km = KMeans(n_clusters=K_KMEANS, n_init=10, random_state=RANDOM_STATE).fit_predict(X)
    t_km = time.perf_counter() - t0
    t0 = time.perf_counter()
    rot_db = DBSCAN(**DBSCAN_REFS[DBSCAN_REF], n_jobs=-1).fit_predict(X)
    t_db = time.perf_counter() - t0
    chuva = base["precip_mm_limpa"].to_numpy() > 0

    def linha(nome, tipo, params, rot, t):
        ev = _evaluate_partition(X, rot, sample_idx)
        validos = rot != -1
        taxa = pd.Series(chuva[validos]).groupby(rot[validos]).mean() * 100
        return {
            "algoritmo": nome,
            "tipo": tipo,
            "parametros": params,
            "tempo_s": t,
            **{k: v for k, v in ev.items() if k not in ("prob_media_nao_ruido", "alta_confianca_pct")},
            "chuva_pct_min_cluster": taxa.min(),
            "chuva_pct_max_cluster": taxa.max(),
            "ari_vs_meanshift": adjusted_rand_score(rot_ms, rot),
            "nmi_vs_meanshift": normalized_mutual_info_score(rot_ms, rot),
        }

    tabela = pd.DataFrame(
        [
            linha("Mean Shift", "Densidade (busca de modas)", desc_ms, rot_ms, t_ms),
            linha("KMeans", "Particional (centroides)", f"K={K_KMEANS}, n_init=10", rot_km, t_km),
            linha("DBSCAN", "Densidade fixa (eps global)",
                  f"eps={DBSCAN_REFS[DBSCAN_REF]['eps']}, min_samples={DBSCAN_REFS[DBSCAN_REF]['min_samples']}", rot_db, t_db),
        ]
    )
    return tabela, rot_km, rot_db


def _figuras(grade, escolhido, X, rot, centros, perfis, rot_km, rot_db):
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    paineis = [
        ("n_clusters", "Número de clusters", "Clusters"),
        ("silhouette_nao_ruido", "Silhouette (maior é melhor)", "Coeficiente"),
        ("davies_bouldin_nao_ruido", "Davies–Bouldin (menor é melhor)", "Índice"),
        ("tempo_fit_s", "Tempo de execução do Mean Shift", "Segundos"),
    ]
    for ax, (col, titulo, ylabel) in zip(axes.flat, paineis):
        ax.plot(grade["bandwidth"], grade[col], marker="o", linewidth=1.7)
        ax.axvline(escolhido["bandwidth"], color="#d95f02", linestyle="--",
                   label=f"bandwidth escolhido = {_number(escolhido['bandwidth'], 2)}")
        ax.set(title=titulo, xlabel="bandwidth (raio no espaço PCA)", ylabel=ylabel)
        ax.legend(fontsize=8)
    fig.suptitle("Mean Shift — efeito do bandwidth (StandardScaler + PCA)", y=1.02)
    fig.tight_layout()
    _save_figure(fig, "grade_bandwidth")

    k = len(centros)
    cores = sns.color_palette("colorblind", max(k, 6))
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharex=True, sharey=True)
    amostra = np.random.default_rng(RANDOM_STATE).choice(len(X), size=min(8000, len(X)), replace=False)
    for ax, (titulo, r) in zip(axes, [("Mean Shift", rot), (f"KMeans (K={K_KMEANS})", rot_km), ("DBSCAN (eps=0,66, min_samples=24)", rot_db)]):
        ra = r[amostra]
        for c in np.unique(ra):
            m = ra == c
            ax.scatter(X[amostra][m, 0], X[amostra][m, 1], s=3, alpha=0.5,
                       color="#bdbdbd" if c == -1 else cores[int(c) % len(cores)],
                       label="ruído" if c == -1 else f"C{int(c)}")
        ax.set(title=titulo, xlabel="PC1")
        ax.legend(markerscale=4, fontsize=7, loc="upper right")
    axes[0].scatter(centros[:, 0], centros[:, 1], marker="X", s=120, color="black", label="modas")
    axes[0].set_ylabel("PC2")
    fig.suptitle("Grupos no plano PC1 × PC2 (amostra fixa de 8.000 registros)", y=1.02)
    fig.tight_layout()
    _save_figure(fig, "clusters_pca_comparacao")

    zcols = [f"{c}_z" for c in COLUNAS_ORIGINAIS_METEOROLOGICAS]
    medias = perfis.set_index("cluster")[[f"{c}_media" for c in COLUNAS_ORIGINAIS_METEOROLOGICAS]]
    z = perfis.set_index("cluster")[zcols]
    fig, ax = plt.subplots(figsize=(9, 0.6 * k + 1.8))
    sns.heatmap(z, cmap="RdBu_r", center=0, linewidths=0.5, cbar_kws={"label": "Desvio da média geral (z-score)"}, ax=ax)
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j + 0.5, i + 0.5, f"{medias.iloc[i, j]:.1f}", ha="center", va="center",
                    color="white" if abs(z.iloc[i, j]) >= 0.85 else "#202020")
    ax.set_xticklabels([f"{NOMES_ATRIB[c][0].capitalize()}\n({NOMES_ATRIB[c][1]})" for c in COLUNAS_ORIGINAIS_METEOROLOGICAS], rotation=0)
    ax.set_yticklabels([f"C{c} ({_number(p, 1)}%)" for c, p in zip(perfis["cluster"], perfis["participacao_pct"])], rotation=0)
    ax.set(title="Mean Shift — médias dos atributos meteorológicos por cluster", xlabel="", ylabel="")
    fig.tight_layout()
    _save_figure(fig, "perfil_meteorologico")


def _relatorio(grade, escolhido, perfis, comp, meta) -> str:
    tab_grade = "\n".join(
        f"| {_number(r.quantil, 3)} | {_number(r.bandwidth, 3)} | {int(r.n_clusters)} | {_number(r.menor_cluster_pct, 1)}% | "
        f"{_number(r.maior_cluster_pct, 1)}% | {_fmt(r.silhouette_nao_ruido, 4)} | {_fmt(r.davies_bouldin_nao_ruido, 3)} | "
        f"{_fmt(r.calinski_harabasz_nao_ruido, 0)} | {_number(r.orfaos_pct, 1)}% | {_number(r.tempo_fit_s, 1)} s |"
        + (" **← escolhido**" if r.quantil == escolhido["quantil"] else "")
        for r in grade.itertuples()
    )
    tab_perfis = "\n".join(
        f"| C{int(r.cluster)} | {_integer(int(r.registros))} ({_number(r.participacao_pct, 1)}%) | "
        f"{_number(r.temp_c_media, 1)} °C | {_number(r.umid_pct_media, 0)}% | {_number(r.temp_orvalho_c_media, 1)} °C | "
        f"{_number(r.pressao_mb_media, 1)} mB | {int(r.hora_local_mediana)}h | {_number(r.estacao_chuvosa_pct, 0)}% | "
        f"{_number(r.horas_com_chuva_pct, 1)}% | {r.descricao} |"
        for r in perfis.itertuples()
    )
    tab_comp = "\n".join(
        f"| {r.algoritmo} | {r.parametros} | {int(r.n_clusters)} | {_number(r.ruido_pct, 1)}% | "
        f"{_fmt(r.silhouette_nao_ruido, 4)} | {_fmt(r.davies_bouldin_nao_ruido, 3)} | {_fmt(r.calinski_harabasz_nao_ruido, 0)} | "
        f"{_number(r.chuva_pct_min_cluster, 1)}–{_number(r.chuva_pct_max_cluster, 1)}% | {_number(r.tempo_s, 2)} s | "
        f"{_fmt(r.ari_vs_meanshift, 3)} |"
        for r in comp.itertuples()
    )
    c = comp.set_index("algoritmo")
    ms, km, db = c.loc["Mean Shift"], c.loc["KMeans"], c.loc["DBSCAN"]
    k1 = grade.loc[grade["n_clusters"] == 1, "bandwidth"]
    kmax = grade.loc[grade["n_clusters"].idxmax()]
    chuvoso = perfis.loc[perfis["horas_com_chuva_pct"].idxmax()]
    seco = perfis.loc[perfis["horas_com_chuva_pct"].idxmin()]
    melhor_sil = c["silhouette_nao_ruido"].idxmax()
    mais_rapido = c["tempo_s"].idxmin()

    return f"""# Experimento Mean Shift — Recife (INMET A301)

## Método e reprodutibilidade

- Mesma matriz do K-Means, DBSCAN e HDBSCAN: `preparar_base(usar_open_meteo=False, filtrar_anos=False)`, as 8 features de clustering ({', '.join(meta['features'])}), `StandardScaler` e PCA com ≥ {PCA_VARIANCE:.0%} da variância ({meta['componentes_pca']} componentes, {_number(meta['variancia_pca_pct'])}%). Chuva e alvos não entram (`validar_sem_leakage(modo="clustering")`).
- {_integer(meta['registros'])} registros ({meta['periodo_inicio']} a {meta['periodo_fim']}).
- Métricas calculadas pela mesma função do experimento HDBSCAN (`_evaluate_partition`): Silhouette numa amostra fixa de {_integer(SILHOUETTE_SAMPLE_SIZE)} registros (mesma semente), Davies–Bouldin e Calinski–Harabasz na base inteira.
- Mean Shift com `bin_seeding=True` (sementes numa grade de lado igual ao bandwidth), `min_bin_freq={MIN_BIN_FREQ}` e `cluster_all=True` (todo registro recebe o cluster da moda mais próxima, então não há ruído).
- O Mean Shift é determinístico com essas opções: rodar de novo dá os mesmos grupos, por isso não há avaliação de estabilidade por sementes como no K-Means.
- Python {meta['python']}, scikit-learn {meta['scikit_learn']}, pandas {meta['pandas']}, NumPy {meta['numpy']}. SHA-256 do CSV: `{meta['sha256_csv']}`.

## Avaliação do bandwidth

O Mean Shift não recebe o número de clusters. Ele desloca cada semente na direção de maior densidade até chegar a uma moda, e cada moda vira um cluster. O único parâmetro relevante é o **bandwidth**, o raio da vizinhança usada para estimar a densidade. Ele foi derivado de quantis das distâncias entre pares de pontos (`estimate_bandwidth`, amostra de {_integer(BANDWIDTH_SAMPLE)}).

| Quantil | Bandwidth | Clusters | Menor cluster | Maior cluster | Silhouette | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | Órfãos* | Tempo |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{tab_grade}

\\* Registros a mais de um bandwidth da moda mais próxima. Com `cluster_all=False` eles seriam marcados como ruído (−1). Mostram quanto da base fica longe dos picos de densidade.

- O número de clusters cai rápido conforme o bandwidth cresce: {int(kmax['n_clusters'])} clusters com bandwidth {_number(kmax['bandwidth'], 2)}{f" e apenas 1 cluster a partir de {_number(k1.min(), 2)}" if len(k1) else ""}. Isso mostra que a base não tem várias modas bem separadas; os regimes meteorológicos formam um contínuo, como o K-Means e o DBSCAN já indicavam (Silhouette baixa em todos).
- Bandwidths pequenos são mais lentos, porque geram mais sementes para deslocar.

**Configuração escolhida: quantil {_number(escolhido['quantil'], 3)}, bandwidth {_number(escolhido['bandwidth'], 3)}, {int(escolhido['n_clusters'])} clusters.** Regra fixada antes de olhar os resultados: entre as configurações com {K_MIN} a {K_MAX} clusters e nenhum cluster menor que {_number(MENOR_CLUSTER_MIN_PCT, 0)}% dos registros, a de maior Silhouette (empate pelo menor Davies–Bouldin). Abaixo de {K_MIN} grupos o resultado não descreve regimes; acima de {K_MAX}, fragmenta demais para interpretar; clusters com menos de 1% são pequenos demais para caracterizar.

![Efeito do bandwidth](figuras/grade_bandwidth.png)

## Clusters encontrados e perfil meteorológico

Clusters ordenados pela temperatura média crescente, no mesmo critério do HDBSCAN. A chuva não entrou no ajuste; aparece só como validação externa.

| Cluster | Registros | Temp. | Umidade | Orvalho | Pressão | Hora mediana | % quadra chuvosa | % horas com chuva | Leitura (desvios ≥ 0,5σ) |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
{tab_perfis}

O cluster com mais horas de chuva é o C{int(chuvoso['cluster'])} ({_number(chuvoso['horas_com_chuva_pct'], 1)}% das horas, {chuvoso['descricao']}); o mais seco é o C{int(seco['cluster'])} ({_number(seco['horas_com_chuva_pct'], 1)}%, {seco['descricao']}). Ou seja, mesmo sem usar chuva no ajuste, o Mean Shift separa regimes associados a chuva de regimes quase sem chuva.

![Perfil meteorológico](figuras/perfil_meteorologico.png)

## Comparação inicial com KMeans e DBSCAN

As três partições foram recalculadas na **mesma matriz** e avaliadas pela **mesma função**, então os números são comparáveis entre si. KMeans com o K final do `reports/kmeans_recife` (K={K_KMEANS}); DBSCAN com a configuração escolhida no `Notebooks/dbscan_recife.ipynb` (C6). ARI mede a concordância com o Mean Shift (1 = partições idênticas, 0 = acaso).

| Algoritmo | Parâmetros | Clusters | Ruído | Silhouette | Davies–Bouldin ↓ | Calinski–Harabasz ↑ | % horas com chuva por cluster (mín–máx) | Tempo | ARI vs. Mean Shift |
|---|---|---:|---:|---:|---:|---:|---|---:|---:|
{tab_comp}

- **Separação:** a maior Silhouette é do {melhor_sil}. Os valores dos três continuam baixos (todos abaixo de 0,3), coerente com regimes que se sobrepõem.
- **Nota sobre a Silhouette do KMeans:** aqui ela é {_number(km['silhouette_nao_ruido'], 4)}, e em `reports/kmeans_recife` é 0,2242. A partição é a mesma configuração (K=5, `random_state=42`); a diferença vem só da amostra usada no cálculo (5.000 registros com a semente do HDBSCAN aqui, 4.000 com outra semente lá). Para comparar algoritmos, usar os valores desta tabela, que vêm todos da mesma amostra.
- **Concordância:** ARI de {_number(km['ari_vs_meanshift'], 3)} com o KMeans e {_number(db['ari_vs_meanshift'], 3)} com o DBSCAN.
- **Ruído:** o Mean Shift, como o KMeans, atribui todos os registros. O DBSCAN deixa {_number(db['ruido_pct'], 1)}% como ruído. A coluna de órfãos da grade mostra o equivalente no Mean Shift se o modo `cluster_all=False` fosse usado.
- **Chuva (validação externa):** a % de horas com chuva varia de {_number(ms['chuva_pct_min_cluster'], 1)}% a {_number(ms['chuva_pct_max_cluster'], 1)}% entre os clusters do Mean Shift, contra {_number(km['chuva_pct_min_cluster'], 1)}–{_number(km['chuva_pct_max_cluster'], 1)}% no KMeans e {_number(db['chuva_pct_min_cluster'], 1)}–{_number(db['chuva_pct_max_cluster'], 1)}% no DBSCAN. Quanto maior a diferença, mais os grupos separam situações com e sem chuva.
- **Tempo:** o mais rápido foi o {mais_rapido}. O Mean Shift levou {_number(ms['tempo_s'], 1)} s com o bandwidth escolhido; a grade inteira levou {_number(meta['tempo_grade_s'], 0)} s.

![Comparação no plano PCA](figuras/clusters_pca_comparacao.png)

## Para o cartão "DIA 4 — Comparação KMeans, DBSCAN, HDBSCAN e Mean Shift"

- Métricas, tempo e parâmetros do Mean Shift: `comparacao_kmeans_dbscan.csv` (linha "Mean Shift") e `metricas_bandwidth.csv`.
- Rótulo de cada registro, para cruzar com os outros algoritmos pela coluna `datetime`: `registros_clusterizados.csv`.
- Perfil de cada grupo: `perfis_clusters.csv`.
- Figuras em PNG (300 dpi) e PDF em `figuras/`.

## Arquivos

- `metricas_bandwidth.csv`: grade completa (quantil, bandwidth, clusters, métricas, órfãos, tempos).
- `distribuicao_clusters.csv` e `perfis_clusters.csv`: tamanho e perfil de cada cluster.
- `comparacao_kmeans_dbscan.csv`: Mean Shift × KMeans × DBSCAN na mesma matriz.
- `registros_clusterizados.csv`: `datetime`, atributos originais e `cluster`.
- `reproducibilidade.json`: configuração, versões e hash da base.

Para reproduzir, na raiz do repositório: `python ml/experimento_meanshift_recife.py`.
"""


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/5] Preparando matriz StandardScaler + PCA(0.95)...")
    base = preparar_base(usar_open_meteo=False, filtrar_anos=False)
    features = features_kmeans_disponiveis(base)
    validar_sem_leakage(features, modo="clustering")
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X = pca.fit_transform(StandardScaler().fit_transform(base[features].to_numpy(dtype=float)))
    sample_idx = np.random.default_rng(RANDOM_STATE).choice(len(X), size=min(SILHOUETTE_SAMPLE_SIZE, len(X)), replace=False)

    print("[2/5] Grade de bandwidth...")
    t0 = time.perf_counter()
    grade, rotulos = _grade(X, sample_idx)
    tempo_grade = time.perf_counter() - t0
    escolhido = _escolher(grade)
    rot_raw, centros_raw = rotulos[escolhido["quantil"]]
    rot = _canonical_labels(rot_raw, base)
    # Centros (modas) na mesma ordem dos rotulos canonicos.
    centros = np.array([centros_raw[rot_raw[rot == c][0]] for c in range(int(escolhido["n_clusters"]))])
    assert len(np.unique(rot)) == escolhido["n_clusters"] and (rot >= 0).all()

    print("[3/5] Perfis dos clusters...")
    perfis = _perfis(base, rot)
    assert perfis["registros"].sum() == len(base)

    print("[4/5] Comparacao com KMeans e DBSCAN...")
    desc = f"bandwidth={_number(escolhido['bandwidth'], 3)} (quantil {_number(escolhido['quantil'], 3)})"
    comp, rot_km, rot_db = _comparar(X, sample_idx, rot, float(escolhido["tempo_fit_s"]), desc, base)
    assert abs(comp.iloc[0]["silhouette_nao_ruido"] - escolhido["silhouette_nao_ruido"]) < 1e-9

    print("[5/5] Salvando resultados...")
    grade.to_csv(OUTPUT_DIR / "metricas_bandwidth.csv", index=False, float_format="%.6f")
    perfis[["cluster", "registros", "participacao_pct"]].to_csv(OUTPUT_DIR / "distribuicao_clusters.csv", index=False, float_format="%.6f")
    perfis.to_csv(OUTPUT_DIR / "perfis_clusters.csv", index=False, float_format="%.6f")
    comp.to_csv(OUTPUT_DIR / "comparacao_kmeans_dbscan.csv", index=False, float_format="%.6f")
    reg = base[["datetime", *COLUNAS_ORIGINAIS_METEOROLOGICAS]].copy()
    reg["cluster"] = rot
    reg.to_csv(OUTPUT_DIR / "registros_clusterizados.csv", index=False, float_format="%.6f")
    _figuras(grade, escolhido, X, rot, centros, perfis, rot_km, rot_db)

    meta = {
        "dataset": str(RAW_CSV.relative_to(ROOT)).replace("\\", "/"),
        "sha256_csv": _sha256_file(RAW_CSV),
        "registros": len(base),
        "periodo_inicio": str(base["datetime_utc"].min().date()),
        "periodo_fim": str(base["datetime_utc"].max().date()),
        "features": features,
        "pca_variancia_minima": PCA_VARIANCE,
        "componentes_pca": int(pca.n_components_),
        "variancia_pca_pct": float(pca.explained_variance_ratio_.sum() * 100),
        "quantis": list(QUANTIS),
        "bandwidth_escolhido": float(escolhido["bandwidth"]),
        "quantil_escolhido": float(escolhido["quantil"]),
        "n_clusters": int(escolhido["n_clusters"]),
        "min_bin_freq": MIN_BIN_FREQ,
        "regra_escolha": f"{K_MIN}<=k<={K_MAX}, menor cluster >= {MENOR_CLUSTER_MIN_PCT}%, maior Silhouette",
        "tempo_grade_s": tempo_grade,
        "random_state": RANDOM_STATE,
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "comando_reproducao": "python ml/experimento_meanshift_recife.py",
    }
    (OUTPUT_DIR / "reproducibilidade.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (OUTPUT_DIR / "relatorio.md").write_text(_relatorio(grade, escolhido, perfis, comp, meta), encoding="utf-8")
    print(f"Mean Shift: bandwidth={escolhido['bandwidth']:.3f}, k={int(escolhido['n_clusters'])}. Relatorio: {OUTPUT_DIR.relative_to(ROOT)}")
    print(comp[["algoritmo", "n_clusters", "ruido_pct", "silhouette_nao_ruido", "davies_bouldin_nao_ruido", "tempo_s", "ari_vs_meanshift"]].to_string(index=False))


if __name__ == "__main__":
    main()
