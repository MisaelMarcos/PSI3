"""Experimento reproduzivel de selecao e interpretacao do K-Means de Recife."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import sklearn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_samples,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from config import RAW_CSV, RANDOM_STATE
from features import (
    COLUNAS_AGREGADAS_FUTURAS,
    COLUNAS_ALVO,
    COLUNAS_CHUVA_OBSERVADA,
    COLUNAS_ORIGINAIS_METEOROLOGICAS,
    features_kmeans_disponiveis,
    preparar_base,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "kmeans_recife"
FIGURES_DIR = OUTPUT_DIR / "figuras"
PCA_VARIANCE = 0.95
SILHOUETTE_SAMPLE_SIZE = 4000
N_INIT = 10
K_FINAL = 5
# Names, K choice and prose were reviewed against this exact Recife snapshot.
EXPECTED_DATASET_SHA256 = "c5eb0aa028e8ff569159a5ee83455834cebc81f51e051a165784063a8d83a2b0"

UNITS = {
    "pressao_mb": "mB",
    "temp_c": "°C",
    "temp_orvalho_c": "°C",
    "umid_pct": "%",
}
DISPLAY_NAMES = {
    "pressao_mb": "Pressão atmosférica",
    "temp_c": "Temperatura do ar",
    "temp_orvalho_c": "Ponto de orvalho",
    "umid_pct": "Umidade relativa",
}
SHORT_NAMES = {
    "pressao_mb": "Pressão",
    "temp_c": "Temp. do ar",
    "temp_orvalho_c": "Ponto de orvalho",
    "umid_pct": "Umidade",
}
CLUSTER_NAMES = {
    0: "Fresco e muito úmido",
    1: "Morno e abafado",
    2: "Morno e umidade intermediária",
    3: "Quente e menos úmido",
    4: "Muito quente e mais seco",
}
INTERPRETATIONS = {
    0: (
        "É o grupo mais fresco e tem a maior pressão média. A umidade relativa é alta "
        "e o ponto de orvalho fica abaixo dos grupos mais abafados."
    ),
    1: (
        "Apesar da temperatura amena, combina umidade relativa alta com o maior ponto "
        "de orvalho entre os grupos. A pressão média é baixa; o conjunto caracteriza "
        "ar morno e abafado, com bastante vapor d'água."
    ),
    2: (
        "Tem temperatura intermediária e umidade relativa entre os grupos úmidos e os "
        "mais secos. A pressão é semelhante à do grupo morno e abafado, mas o ponto de "
        "orvalho é menor e a umidade relativa intermediária caracteriza uma condição de transição."
    ),
    3: (
        "É quente e menos úmido que os três grupos de menor temperatura. A pressão "
        "média é relativamente alta, o que o distingue do grupo mais quente, mesmo com "
        "umidade relativa igualmente baixa."
    ),
    4: (
        "É o grupo mais quente e tem a menor umidade relativa. Como o ponto de orvalho "
        "não aumenta na mesma proporção que a temperatura do ar, apresenta a maior "
        "diferença entre essas duas medidas e caracteriza o regime mais quente e seco."
    ),
}


def _save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _number(value: float, decimals: int = 2) -> str:
    return f"{value:,.{decimals}f}".replace(",", "_").replace(".", ",").replace("_", ".")


def _integer(value: int) -> str:
    return f"{value:,}".replace(",", ".")


def _profile_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cluster, group in df.groupby("cluster", sort=True):
        for column in COLUNAS_ORIGINAIS_METEOROLOGICAS:
            values = group[column]
            rows.append(
                {
                    "cluster": cluster,
                    "regime": CLUSTER_NAMES[cluster],
                    "coluna": column,
                    "atributo": DISPLAY_NAMES[column],
                    "unidade": UNITS[column],
                    "media": values.mean(),
                    "mediana": values.median(),
                    "desvio_padrao": values.std(),
                    "q1": values.quantile(0.25),
                    "q3": values.quantile(0.75),
                    "iqr": values.quantile(0.75) - values.quantile(0.25),
                    "minimo": values.min(),
                    "maximo": values.max(),
                }
            )
    return pd.DataFrame(rows)


def _markdown_report(
    metrics: pd.DataFrame,
    distribution: pd.DataFrame,
    profiles: pd.DataFrame,
    metadata: dict,
) -> str:
    by_k = metrics.set_index("k")

    def metric_row(k: int) -> str:
        row = by_k.loc[k]
        return (
            f"| {k} | {_number(row['inercia'], 1)} | {_number(row['reducao_inercia_pct'])}% | "
            f"{_number(row['silhouette'], 4)} | {_number(row['davies_bouldin'], 4)} | "
            f"{_number(row['calinski_harabasz'], 1)} |"
        )

    candidate_ks = [k for k in (2, 3, 4, 5, 6, 10) if k in by_k.index]
    metric_table = "\n".join(metric_row(k) for k in candidate_ks)
    distribution_table = "\n".join(
        f"| {int(row.cluster)} — {row.regime} | {_integer(row.registros)} | {_number(row.participacao_pct)}% |"
        for row in distribution.itertuples()
    )
    profile_table = "\n".join(
        "| {cluster} — {regime} | {atributo} | {media} {unidade} | "
        "{mediana} {unidade} | {desvio_padrao} {unidade} | {iqr} {unidade} |".format(
            **{
                **row._asdict(),
                "media": _number(row.media),
                "mediana": _number(row.mediana),
                "desvio_padrao": _number(row.desvio_padrao),
                "iqr": _number(row.iqr),
            }
        )
        for row in profiles.itertuples(index=False)
    )
    cluster_notes = "\n\n".join(
        f"### Cluster {cluster} — {CLUSTER_NAMES[cluster]}\n\n{INTERPRETATIONS[cluster]}"
        for cluster in range(K_FINAL)
    )
    k2, k3, k4, k5, k6, k10 = (by_k.loc[k] for k in (2, 3, 4, 5, 6, 10))

    return f"""# Experimento K-Means — Recife (INMET A301)

## Método e reprodutibilidade

- Base preparada por `ml/features.py::preparar_base`, com a série completa disponível; linhas sem entradas completas são removidas.
- `ler_csv` ordena a série e preenche ausências meteorológicas com o último valor anterior. Neste CSV, 109 registros tinham ponto de orvalho e umidade ausentes; ambos foram imputados por esse forward-fill causal.
- {_integer(metadata['registros'])} registros usados ({metadata['periodo_inicio']} a {metadata['periodo_fim']}).
- Entradas do K-Means: {', '.join(metadata['features'])}. Chuva observada e alvos futuros foram excluídos.
- `StandardScaler` seguido de PCA antes de calcular Elbow, Silhouette, Davies-Bouldin e Calinski-Harabasz. O limiar de retenção foi {PCA_VARIANCE:.0%}: {metadata['componentes_pca']} componentes retiveram {_number(metadata['variancia_pca_pct'])}% da variância.
- KMeans avaliado para K=2–{metadata['k_max']}, com `n_init={N_INIT}` e `random_state={RANDOM_STATE}`. A inércia também foi calculada para K=1 como referência do Elbow.
- Silhouette calculada no espaço PCA sobre uma amostra fixa aleatória de {_integer(metadata['amostra_silhueta'])} registros; os demais índices usam os registros completos.
- Ambiente: Python {metadata['python']}, scikit-learn {metadata['scikit_learn']}, pandas {metadata['pandas']}, NumPy {metadata['numpy']}.
- SHA-256 do CSV de entrada: `{metadata['sha256_csv']}`.
- A escolha de K e as interpretações estão vinculadas a esse hash; o script interrompe a execução se o CSV mudar.

## Comparação quantitativa

| K | Inércia PCA | Redução da inércia vs. K anterior | Silhouette | Davies–Bouldin ↓ | Calinski–Harabasz ↑ |
|---:|---:|---:|---:|---:|---:|
{metric_table}

A Silhouette e o Calinski–Harabasz atingem seus maiores valores em K=2 ({_number(k2['silhouette'], 4)} e {_number(k2['calinski_harabasz'], 1)}). K=5 mantém Silhouette próxima ({_number(k5['silhouette'], 4)}, {_number(100 * k5['silhouette'] / k2['silhouette'], 1)}% do máximo), supera K=3 ({_number(k3['silhouette'], 4)}) e reduz Davies–Bouldin de {_number(k3['davies_bouldin'], 4)} para {_number(k5['davies_bouldin'], 4)}. No Elbow, a redução marginal da inércia cai de {_number(k5['reducao_inercia_pct'])}% em K=5 para {_number(k6['reducao_inercia_pct'])}% em K=6. K=10 aumenta Silhouette em apenas {_number(k10['silhouette'] - k5['silhouette'], 4)} sobre K=5, ao custo de dobrar o número de grupos.

**K final: {K_FINAL}.** K=2 apresenta o maior valor isolado de Silhouette, mas forma apenas dois regimes amplos. K=5 equilibra separação e interpretação: seus cinco grupos têm perfis meteorológicos distintos e participação entre {_number(distribution['participacao_pct'].min())}% e {_number(distribution['participacao_pct'].max())}%. A melhoria marginal da inércia diminui a partir de K=5; K=10 oferece ganho pequeno de Silhouette, mas fragmenta os perfis em mais grupos. Assim, a escolha combina Elbow, Silhouette, Davies–Bouldin, distribuição e leitura física; nenhum índice isolado ou aparência do gráfico determinou K.

![Elbow e métricas quantitativas](figuras/criterios_k.png)

![Variância explicada pela PCA](figuras/variancia_pca.png)

## Distribuição dos grupos

| Grupo interpretado | Registros | Participação |
|---|---:|---:|
{distribution_table}

## Perfil estatístico nos atributos meteorológicos originais

Os clusters foram ordenados pela temperatura média crescente para tornar seus rótulos reproduzíveis. A leitura abaixo usa pressão, temperatura do ar, ponto de orvalho e umidade nas unidades originais. IQR é o intervalo interquartil e, junto ao desvio-padrão, mostra a variabilidade dentro de cada grupo.

| Grupo | Atributo | Média | Mediana | Desvio-padrão | IQR |
|---|---|---:|---:|---:|---:|
{profile_table}

{cluster_notes}

![Silhouette por grupo para K final](figuras/silhueta_final.png)

![Médias meteorológicas por grupo](figuras/perfil_meteorologico.png)

## Arquivos

- `metricas_k.csv`: inércia e índices para todos os K avaliados.
- `distribuicao_clusters.csv`: contagens e participação dos registros.
- `perfis_clusters.csv`: médias, medianas e variabilidade por atributo original.
- `registros_clusterizados.csv`: timestamp, atributos originais, rótulo e nome do regime.
- `reproducibilidade.json`: configuração, versões e hash da base.
- `figuras/`: versões PNG (300 dpi) e PDF dos gráficos para o artigo.

Para reproduzir a execução na raiz do repositório: `venv/bin/python ml/experimento_kmeans.py`.
"""


def main() -> None:
    csv_hash = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    if csv_hash != EXPECTED_DATASET_SHA256:
        raise ValueError(
            "O CSV de Recife mudou. Revise a escolha de K e as interpretações antes de atualizar EXPECTED_DATASET_SHA256."
        )

    sns.set_theme(style="whitegrid", context="paper")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    base = preparar_base(caminho=RAW_CSV, usar_open_meteo=False, filtrar_anos=False)
    features = features_kmeans_disponiveis(base)
    forbidden = set(COLUNAS_CHUVA_OBSERVADA + COLUNAS_AGREGADAS_FUTURAS + COLUNAS_ALVO)
    if forbidden.intersection(features):
        raise ValueError("Chuva ou alvos futuros não podem entrar no K-Means deste experimento.")
    missing_columns = set(COLUNAS_ORIGINAIS_METEOROLOGICAS).difference(base.columns)
    if missing_columns:
        raise ValueError(f"Atributos meteorológicos ausentes na base: {sorted(missing_columns)}")

    registros_originais = len(base)
    base = base.dropna(subset=features).reset_index(drop=True)
    if len(base) < 21:
        raise ValueError("A base preparada precisa de ao menos 21 registros para avaliar K até 20.")
    X = base[features]

    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)
    k_max = min(20, len(base) - 1)
    sample_idx = np.random.RandomState(RANDOM_STATE).permutation(len(base))[
        : min(SILHOUETTE_SAMPLE_SIZE, len(base))
    ]

    metric_rows = []
    inertia_previous = None
    final_raw_labels = None
    for k in range(1, k_max + 1):
        model = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=N_INIT)
        labels = model.fit_predict(X_pca)
        inertia = float(model.inertia_)
        reduction = (
            (inertia_previous - inertia) / inertia_previous * 100
            if inertia_previous is not None
            else np.nan
        )
        row = {
            "k": k,
            "inercia": inertia,
            "reducao_inercia_pct": reduction,
            "silhouette": np.nan,
            "davies_bouldin": np.nan,
            "calinski_harabasz": np.nan,
        }
        if k >= 2:
            sample_labels = labels[sample_idx]
            assert np.unique(sample_labels).size == k
            row["silhouette"] = silhouette_score(X_pca[sample_idx], sample_labels)
            row["davies_bouldin"] = davies_bouldin_score(X_pca, labels)
            row["calinski_harabasz"] = calinski_harabasz_score(X_pca, labels)
        metric_rows.append(row)
        if k == K_FINAL:
            final_raw_labels = labels.copy()
        inertia_previous = inertia

    if final_raw_labels is None:
        raise ValueError(f"K final={K_FINAL} não foi avaliado.")

    # Canonicaliza os IDs pela temperatura média; os nomes e perfis explicam cada grupo.
    order = sorted(
        range(K_FINAL),
        key=lambda cluster: base.loc[final_raw_labels == cluster, "temp_c"].mean(),
    )
    remap = {old: new for new, old in enumerate(order)}
    labels_final = np.array([remap[label] for label in final_raw_labels])
    clustered = base.copy()
    clustered["cluster"] = labels_final
    clustered["regime"] = clustered["cluster"].map(CLUSTER_NAMES)

    metrics = pd.DataFrame(metric_rows)
    distribution = (
        clustered.groupby(["cluster", "regime"], sort=True)
        .size()
        .rename("registros")
        .reset_index()
    )
    distribution["participacao_pct"] = distribution["registros"] / len(clustered) * 100
    profiles = _profile_table(clustered)
    assert len(clustered) == len(base)
    assert distribution["registros"].sum() == len(base)
    assert distribution["cluster"].nunique() == K_FINAL
    assert pca.explained_variance_ratio_.sum() >= PCA_VARIANCE

    metrics.to_csv(OUTPUT_DIR / "metricas_k.csv", index=False, float_format="%.6f")
    distribution.to_csv(OUTPUT_DIR / "distribuicao_clusters.csv", index=False, float_format="%.6f")
    profiles.to_csv(OUTPUT_DIR / "perfis_clusters.csv", index=False, float_format="%.6f")
    clustered[["datetime", *COLUNAS_ORIGINAIS_METEOROLOGICAS, "cluster", "regime"]].to_csv(
        OUTPUT_DIR / "registros_clusterizados.csv", index=False, float_format="%.6f"
    )

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    plots = [
        ("inercia", "Inércia no espaço PCA", "Inércia", False),
        ("silhouette", "Silhouette (amostra fixa)", "Coeficiente", True),
        ("davies_bouldin", "Davies–Bouldin (menor é melhor)", "Índice", True),
        ("calinski_harabasz", "Calinski–Harabasz (maior é melhor)", "Índice", True),
    ]
    for ax, (column, title, ylabel, skip_k1) in zip(axes.flat, plots):
        values = metrics[metrics["k"] >= 2] if skip_k1 else metrics
        ax.plot(values["k"], values[column], marker="o", linewidth=1.7)
        ax.axvline(K_FINAL, color="#d95f02", linestyle="--", label=f"K final = {K_FINAL}")
        ax.set(title=title, xlabel="Número de clusters (K)", ylabel=ylabel)
        ax.set_xticks(range(1 if not skip_k1 else 2, k_max + 1, 2))
        ax.legend()
    fig.suptitle("Seleção quantitativa de K após StandardScaler e PCA", y=1.02)
    fig.tight_layout()
    _save_figure(fig, "criterios_k")

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ratios = pca.explained_variance_ratio_
    x = np.arange(1, len(ratios) + 1)
    ax.bar(x, ratios * 100, alpha=0.75, label="Individual")
    ax.plot(x, np.cumsum(ratios) * 100, marker="o", color="#d95f02", label="Acumulada")
    ax.axhline(PCA_VARIANCE * 100, color="gray", linestyle="--", label=f"Limiar {PCA_VARIANCE:.0%}")
    ax.set(title="Variância explicada pela PCA", xlabel="Componente principal", ylabel="Variância (%)")
    ax.set_xticks(x)
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, "variancia_pca")

    silhouette_values = silhouette_samples(X_pca[sample_idx], final_raw_labels[sample_idx])
    silhouette_labels = labels_final[sample_idx]
    silhouette_mean = silhouette_score(X_pca[sample_idx], final_raw_labels[sample_idx])
    fig, ax = plt.subplots(figsize=(8, 5))
    y_start = 0
    colors = sns.color_palette("colorblind", K_FINAL)
    for cluster in range(K_FINAL):
        values = np.sort(silhouette_values[silhouette_labels == cluster])
        y_end = y_start + len(values)
        ax.fill_betweenx(np.arange(y_start, y_end), 0, values, color=colors[cluster], alpha=0.85)
        ax.text(-0.02, (y_start + y_end) / 2, f"C{cluster}: {CLUSTER_NAMES[cluster]}", ha="right", va="center")
        y_start = y_end + 12
    ax.axvline(silhouette_mean, color="black", linestyle="--", label=f"Média = {silhouette_mean:.4f}")
    ax.set(title=f"Silhouette por grupo — K={K_FINAL}", xlabel="Coeficiente de Silhouette", yticks=[])
    ax.legend(loc="lower right")
    fig.tight_layout()
    _save_figure(fig, "silhueta_final")

    means = clustered.groupby("cluster")[COLUNAS_ORIGINAIS_METEOROLOGICAS].mean().astype(float)
    overall_std = clustered[COLUNAS_ORIGINAIS_METEOROLOGICAS].std()
    z_means = ((means - clustered[COLUNAS_ORIGINAIS_METEOROLOGICAS].mean()) / overall_std).astype(float)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    sns.heatmap(
        z_means,
        cmap="RdBu_r",
        center=0,
        linewidths=0.5,
        cbar_kws={"label": "Desvio da média geral (z-score)"},
        ax=ax,
    )
    for row in range(K_FINAL):
        for column in range(len(COLUNAS_ORIGINAIS_METEOROLOGICAS)):
            z_value = z_means.iloc[row, column]
            color = "white" if abs(z_value) >= 0.85 else "#202020"
            ax.text(
                column + 0.5,
                row + 0.5,
                f"{means.iloc[row, column]:.1f}",
                ha="center",
                va="center",
                color=color,
            )
    ax.set(title="Médias dos atributos meteorológicos originais por cluster", xlabel="Atributo", ylabel="")
    ax.set_xticklabels(
        [f"{SHORT_NAMES[name]}\n({UNITS[name]})" for name in COLUNAS_ORIGINAIS_METEOROLOGICAS],
        rotation=0,
    )
    ax.set_yticklabels([f"C{i} — {CLUSTER_NAMES[i]}" for i in range(K_FINAL)], rotation=0)
    fig.tight_layout()
    _save_figure(fig, "perfil_meteorologico")

    metadata = {
        "dataset": str(RAW_CSV.relative_to(ROOT)),
        "sha256_csv": csv_hash,
        "registros": len(base),
        "registros_descartados_sem_entrada_completa": registros_originais - len(base),
        "periodo_inicio": str(pd.to_datetime(base["datetime"], utc=True).min().date()),
        "periodo_fim": str(pd.to_datetime(base["datetime"], utc=True).max().date()),
        "features": features,
        "atributos_interpretacao": COLUNAS_ORIGINAIS_METEOROLOGICAS,
        "pca_variancia_minima": PCA_VARIANCE,
        "componentes_pca": int(pca.n_components_),
        "variancia_pca_pct": float(pca.explained_variance_ratio_.sum() * 100),
        "amostra_silhueta": len(sample_idx),
        "k_avaliados": list(range(2, k_max + 1)),
        "k_max": k_max,
        "k_final": K_FINAL,
        "n_init": N_INIT,
        "random_state": RANDOM_STATE,
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "comando_reproducao": "venv/bin/python ml/experimento_kmeans.py",
    }
    (OUTPUT_DIR / "reproducibilidade.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "relatorio.md").write_text(
        _markdown_report(metrics, distribution, profiles, metadata), encoding="utf-8"
    )
    print(f"Experimento concluído: K=2..{k_max}; K final={K_FINAL}; registros={_integer(len(base))}.")
    print(f"PCA: {pca.n_components_} componentes, {_number(metadata['variancia_pca_pct'])}% da variância.")
    print(f"Relatório e arquivos: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
