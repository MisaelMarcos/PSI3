"""Experimento reproduzível de HDBSCAN com dados meteorológicos de Recife."""

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
import sklearn
from sklearn.cluster import HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, RAW_CSV
from features import (
    COLUNAS_ORIGINAIS_METEOROLOGICAS,
    features_kmeans_disponiveis,
    preparar_base,
    validar_sem_leakage,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "hdbscan_recife"
FIGURES_DIR = OUTPUT_DIR / "figuras"
PCA_VARIANCE = 0.95
SILHOUETTE_SAMPLE_SIZE = 4000
MIN_CLUSTER_SIZES = (25, 50, 100, 200)
MIN_SAMPLES_VALUES = (None, 5, 10, 25)


def _save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _evaluate_partition(
    X: np.ndarray, labels: np.ndarray, random_state: int
) -> dict[str, float | int]:
    noise = labels == -1
    clustered_labels = labels[~noise]
    clusters = np.unique(clustered_labels)
    sample_idx = np.flatnonzero(~noise)
    if len(sample_idx) > SILHOUETTE_SAMPLE_SIZE:
        sample_idx = np.random.RandomState(random_state).choice(
            sample_idx, SILHOUETTE_SAMPLE_SIZE, replace=False
        )

    silhouette = np.nan
    davies_bouldin = np.nan
    calinski_harabasz = np.nan
    if len(clusters) >= 2:
        sampled_labels = labels[sample_idx]
        if len(np.unique(sampled_labels)) >= 2:
            silhouette = float(silhouette_score(X[sample_idx], sampled_labels))
        davies_bouldin = float(davies_bouldin_score(X[~noise], clustered_labels))
        calinski_harabasz = float(calinski_harabasz_score(X[~noise], clustered_labels))

    return {
        "n_clusters": int(len(clusters)),
        "registros_ruido": int(noise.sum()),
        "ruido_pct": float(noise.mean() * 100),
        "cobertura_pct": float((~noise).mean() * 100),
        "silhouette_nao_ruido": silhouette,
        "davies_bouldin_nao_ruido": davies_bouldin,
        "calinski_harabasz_nao_ruido": calinski_harabasz,
    }


def _canonical_labels(labels: np.ndarray, base: pd.DataFrame) -> np.ndarray:
    clusters = np.unique(labels[labels != -1])
    order = sorted(
        clusters,
        key=lambda cluster: (
            base.loc[labels == cluster, "temp_c"].mean(),
            base.loc[labels == cluster, "umid_pct"].mean(),
            int(cluster),
        ),
    )
    remap = {int(old): new for new, old in enumerate(order)}
    return np.array([-1 if label == -1 else remap[int(label)] for label in labels])


def _plot_grid(metrics: pd.DataFrame) -> None:
    min_samples_order = [
        "None (= min_cluster_size)",
        *(str(value) for value in MIN_SAMPLES_VALUES[1:]),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, title, cmap in (
        (axes[0], "silhouette_nao_ruido", "Silhouette (somente pontos agrupados)", "viridis"),
        (axes[1], "cobertura_pct", "Cobertura dos dados (%)", "magma"),
    ):
        matrix = metrics.pivot(
            index="min_cluster_size", columns="min_samples_label", values=metric
        ).reindex(index=MIN_CLUSTER_SIZES, columns=min_samples_order)
        values = matrix.to_numpy(dtype=float)
        finite = values[np.isfinite(values)]
        image = ax.imshow(
            values,
            aspect="auto",
            cmap=cmap,
            vmin=float(finite.min()) if finite.size else 0,
            vmax=float(finite.max()) if finite.size else 1,
        )
        fig.colorbar(image, ax=ax)
        ax.set(
            title=title,
            xlabel="min_samples",
            ylabel="min_cluster_size",
            xticks=range(len(min_samples_order)),
            xticklabels=min_samples_order,
            yticks=range(len(MIN_CLUSTER_SIZES)),
            yticklabels=MIN_CLUSTER_SIZES,
        )
        ax.tick_params(axis="x", rotation=25)
        for row in range(values.shape[0]):
            for column in range(values.shape[1]):
                value = values[row, column]
                label = (
                    "—"
                    if np.isnan(value)
                    else f"{value:.3f}"
                    if metric.startswith("silhouette")
                    else f"{value:.1f}"
                )
                text_color = "white"
                if np.isfinite(value):
                    red, green, blue, _ = image.cmap(image.norm(value))
                    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                    if luminance > 0.55:
                        text_color = "#202020"
                ax.text(column, row, label, ha="center", va="center", color=text_color)
    fig.suptitle("HDBSCAN em Recife — grade de parâmetros")
    fig.tight_layout()
    _save_figure(fig, "grade_parametros_hdbscan")


def _plot_profiles(profiles: pd.DataFrame) -> None:
    if profiles.empty:
        for suffix in ("png", "pdf"):
            (FIGURES_DIR / f"perfil_clusters_hdbscan.{suffix}").unlink(missing_ok=True)
        return
    means = profiles.xs("mean", axis=1, level=1).astype(float)
    overall_mean = profiles.attrs["overall_mean"].astype(float)
    overall_std = profiles.attrs["overall_std"].astype(float).replace(0, np.nan)
    z_means = ((means - overall_mean) / overall_std).fillna(0)
    fig, ax = plt.subplots(figsize=(9, max(3.5, len(z_means) * 0.55)))
    image = ax.imshow(z_means.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2)
    fig.colorbar(image, ax=ax, label="Desvio da média geral (z-score)")
    ax.set(
        title="Perfil meteorológico por cluster (configuração de referência)",
        xlabel="Atributo",
        ylabel="Cluster",
        xticks=range(len(z_means.columns)),
        xticklabels=z_means.columns,
        yticks=range(len(z_means.index)),
        yticklabels=z_means.index,
    )
    ax.tick_params(axis="x", rotation=25)
    for row in range(len(z_means.index)):
        for column in range(len(z_means.columns)):
            ax.text(
                column,
                row,
                f"{means.iloc[row, column]:.1f}",
                ha="center",
                va="center",
                fontsize=8,
            )
    fig.tight_layout()
    _save_figure(fig, "perfil_clusters_hdbscan")


def _markdown_report(
    metrics: pd.DataFrame,
    distribution: pd.DataFrame,
    profiles: pd.DataFrame,
    metadata: dict,
    selection_note: str,
) -> str:
    metrics_rows = "\n".join(
        f"| {int(row.min_cluster_size)} | {row.min_samples_label} | {int(row.n_clusters)} | "
        f"{int(row.registros_ruido):,} | {row.ruido_pct:.2f}% | {row.cobertura_pct:.2f}% | "
        f"{row.silhouette_nao_ruido:.4f} | {row.davies_bouldin_nao_ruido:.4f} | "
        f"{row.calinski_harabasz_nao_ruido:.1f} |"
        if pd.notna(row.silhouette_nao_ruido)
        else f"| {int(row.min_cluster_size)} | {row.min_samples_label} | {int(row.n_clusters)} | "
        f"{int(row.registros_ruido):,} | {row.ruido_pct:.2f}% | {row.cobertura_pct:.2f}% | — | — | — |"
        for row in metrics.itertuples(index=False)
    )
    distribution_rows = "\n".join(
        f"| {int(row.cluster)} | {int(row.registros):,} | {row.participacao_pct:.2f}% |"
        for row in distribution.itertuples(index=False)
    )
    if profiles.empty:
        profile_rows = "Nenhum cluster sem ruído foi identificado na configuração de referência."
    else:
        means = profiles.xs("mean", axis=1, level=1)
        profile_rows = "\n".join(
            f"| {int(cluster)} | {int(distribution.set_index('cluster').loc[cluster, 'registros']):,} | "
            f"{means.loc[cluster, 'temp_c']:.2f} | {means.loc[cluster, 'temp_orvalho_c']:.2f} | "
            f"{means.loc[cluster, 'umid_pct']:.2f} | {means.loc[cluster, 'pressao_mb']:.2f} |"
            for cluster in means.index
        )

    profile_figure = (
        "![Perfis meteorológicos](figuras/perfil_clusters_hdbscan.png)"
        if not profiles.empty
        else ""
    )

    return f"""# Experimento HDBSCAN — Recife (INMET A301)

## Base e preparação

- CSV de entrada: {metadata['dataset']} (SHA-256 {metadata['sha256_csv']}).
- Preparação: ml/features.py::preparar_base, sem Open-Meteo e sem filtro de anos; o leitor preenche lacunas meteorológicas usando apenas valores anteriores.
- Registros preparados: {metadata['registros_preparados']:,}; usados: {metadata['registros']:,}; removidos sem entradas completas: {metadata['registros_descartados']:,}.
- Período: {metadata['periodo_inicio']} a {metadata['periodo_fim']}.
- Features, alinhadas ao K-Means de Recife: {', '.join(metadata['features'])}.
- Pré-processamento: StandardScaler e PCA até {PCA_VARIANCE:.0%} de variância ({metadata['componentes_pca']} componentes; {metadata['variancia_pca_pct']:.2f}% retidos).
- HDBSCAN do scikit-learn {metadata['scikit_learn']}, distância euclidiana. min_samples=None usa o padrão igual a min_cluster_size.

## Grade de parâmetros

Silhouette, Davies–Bouldin e Calinski–Harabasz foram calculados somente nos pontos agrupados; Silhouette usa uma amostra fixa de até {SILHOUETTE_SAMPLE_SIZE:,} pontos não-ruído. A cobertura e o percentual de ruído usam a base toda.

| min_cluster_size | min_samples | clusters | ruído (n) | ruído | cobertura | Silhouette | Davies–Bouldin | Calinski–Harabasz |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
{metrics_rows}

## Referência de alta cobertura

{selection_note}

O rótulo -1 representa ruído; os outros clusters foram numerados pela temperatura média crescente.

| Cluster | Registros | Participação |
|---:|---:|---:|
{distribution_rows}

| Cluster | Registros | Temp. do ar (°C) | Ponto de orvalho (°C) | Umidade (%) | Pressão (mB) |
|---:|---:|---:|---:|---:|---:|
{profile_rows}

![Grade de parâmetros](figuras/grade_parametros_hdbscan.png)

{profile_figure}

## Reprodutibilidade

- Python {metadata['python']}; scikit-learn {metadata['scikit_learn']}; pandas {metadata['pandas']}; NumPy {metadata['numpy']}.
- PCA e HDBSCAN usam as entradas completas após a preparação; somente Silhouette é amostrada.
- Comando: venv/bin/python ml/experimento_hdbscan_recife.py.
- Arquivos: metricas_hdbscan.csv, distribuicao_clusters.csv, perfis_clusters.csv, registros_clusterizados.csv e reproducibilidade.json.
"""


def main() -> None:
    csv_hash = hashlib.sha256(RAW_CSV.read_bytes()).hexdigest()
    base = preparar_base(caminho=RAW_CSV, usar_open_meteo=False, filtrar_anos=False)
    features = features_kmeans_disponiveis(base)
    validar_sem_leakage(features, modo="clustering")
    missing = set(features).difference(base.columns)
    if missing:
        raise ValueError(f"Features de clustering ausentes na base: {sorted(missing)}")

    registros_preparados = len(base)
    base = base.dropna(subset=features).reset_index(drop=True)
    if len(base) < 2:
        raise ValueError("A base de Recife precisa de ao menos dois registros completos.")
    X = base[features].to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)
    pca = PCA(n_components=PCA_VARIANCE, svd_solver="full")
    X_pca = pca.fit_transform(X_scaled)

    rows = []
    labels_by_config = {}
    for min_cluster_size in MIN_CLUSTER_SIZES:
        for min_samples in MIN_SAMPLES_VALUES:
            label = "None (= min_cluster_size)" if min_samples is None else str(min_samples)
            print(f"HDBSCAN: min_cluster_size={min_cluster_size}, min_samples={label}")
            labels = HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
            ).fit_predict(X_pca)
            labels_by_config[(min_cluster_size, label)] = labels
            rows.append(
                {
                    "min_cluster_size": min_cluster_size,
                    "min_samples_label": label,
                    **_evaluate_partition(X_pca, labels, RANDOM_STATE),
                }
            )

    metrics = pd.DataFrame(rows)
    candidates = metrics[metrics["n_clusters"] >= 2]
    if not candidates.empty:
        chosen = candidates.sort_values(
            ["cobertura_pct", "silhouette_nao_ruido"],
            ascending=False,
            na_position="last",
            kind="stable",
        ).iloc[0]
        best_silhouette = metrics.dropna(subset=["silhouette_nao_ruido"]).sort_values(
            ["silhouette_nao_ruido", "cobertura_pct"],
            ascending=False,
            kind="stable",
        ).iloc[0]
        selection_note = (
            f"Referência de alta cobertura: min_cluster_size={int(chosen.min_cluster_size)}, "
            f"min_samples={chosen.min_samples_label}; cobertura {chosen.cobertura_pct:.2f}% "
            f"e Silhouette {chosen.silhouette_nao_ruido:.4f}. A maior Silhouette foi "
            f"{best_silhouette.silhouette_nao_ruido:.4f} em "
            f"min_cluster_size={int(best_silhouette.min_cluster_size)}, "
            f"min_samples={best_silhouette.min_samples_label}, mas cobre apenas "
            f"{best_silhouette.cobertura_pct:.2f}% dos registros. Nenhuma configuração "
            "deve ser tratada como melhor sem considerar conjuntamente ruído, cobertura "
            "e separação."
        )
    else:
        chosen = metrics.sort_values(
            ["cobertura_pct", "n_clusters"], ascending=False, kind="stable"
        ).iloc[0]
        best_silhouette = None
        selection_note = (
            "Nenhuma configuração formou dois ou mais clusters; "
            "não há partição HDBSCAN validada para selecionar."
        )

    selected_labels = _canonical_labels(
        labels_by_config[(int(chosen.min_cluster_size), chosen.min_samples_label)], base
    )
    clustered = base.copy()
    clustered["cluster"] = selected_labels
    counts = clustered["cluster"].value_counts().sort_index()
    distribution = counts.rename("registros").rename_axis("cluster").reset_index()
    distribution["participacao_pct"] = distribution["registros"] / len(clustered) * 100
    assert int(distribution["registros"].sum()) == len(base)
    profiles = clustered.loc[clustered["cluster"] != -1].groupby("cluster")[
        COLUNAS_ORIGINAIS_METEOROLOGICAS
    ].agg(["mean", "median", "std"])
    profiles.attrs["overall_mean"] = clustered[COLUNAS_ORIGINAIS_METEOROLOGICAS].mean()
    profiles.attrs["overall_std"] = clustered[COLUNAS_ORIGINAIS_METEOROLOGICAS].std()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(OUTPUT_DIR / "metricas_hdbscan.csv", index=False, float_format="%.6f")
    distribution.to_csv(
        OUTPUT_DIR / "distribuicao_clusters.csv", index=False, float_format="%.6f"
    )
    profiles.to_csv(OUTPUT_DIR / "perfis_clusters.csv", float_format="%.6f")
    clustered[["datetime", *COLUNAS_ORIGINAIS_METEOROLOGICAS, "cluster"]].to_csv(
        OUTPUT_DIR / "registros_clusterizados.csv", index=False, float_format="%.6f"
    )
    _plot_grid(metrics)
    _plot_profiles(profiles)

    metadata = {
        "dataset": str(RAW_CSV.relative_to(ROOT)),
        "sha256_csv": csv_hash,
        "registros_preparados": registros_preparados,
        "registros": len(base),
        "registros_descartados": registros_preparados - len(base),
        "periodo_inicio": str(pd.to_datetime(base["datetime"], utc=True).min().date()),
        "periodo_fim": str(pd.to_datetime(base["datetime"], utc=True).max().date()),
        "features": features,
        "pca_variancia_minima": PCA_VARIANCE,
        "componentes_pca": int(pca.n_components_),
        "variancia_pca_pct": float(pca.explained_variance_ratio_.sum() * 100),
        "amostra_silhueta": SILHOUETTE_SAMPLE_SIZE,
        "min_cluster_sizes": list(MIN_CLUSTER_SIZES),
        "min_samples": list(MIN_SAMPLES_VALUES),
        "configuracao_referencia_alta_cobertura": {
            "min_cluster_size": int(chosen.min_cluster_size),
            "min_samples": (
                None
                if chosen.min_samples_label.startswith("None")
                else int(chosen.min_samples_label)
            ),
        },
        "configuracao_maior_silhueta": (
            None
            if best_silhouette is None
            else {
                "min_cluster_size": int(best_silhouette.min_cluster_size),
                "min_samples": (
                    None
                    if best_silhouette.min_samples_label.startswith("None")
                    else int(best_silhouette.min_samples_label)
                ),
                "silhouette": float(best_silhouette.silhouette_nao_ruido),
                "cobertura_pct": float(best_silhouette.cobertura_pct),
            }
        ),
        "random_state": RANDOM_STATE,
        "python": sys.version.split()[0],
        "scikit_learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "comando_reproducao": "venv/bin/python ml/experimento_hdbscan_recife.py",
    }
    (OUTPUT_DIR / "reproducibilidade.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "relatorio.md").write_text(
        _markdown_report(metrics, distribution, profiles, metadata, selection_note),
        encoding="utf-8",
    )
    print(
        f"Experimento concluído: {len(metrics)} configurações; "
        f"{len(base):,} registros; referência "
        f"(min_cluster_size={int(chosen.min_cluster_size)}, "
        f"min_samples={chosen.min_samples_label})."
    )
    print(f"Relatório e arquivos: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
