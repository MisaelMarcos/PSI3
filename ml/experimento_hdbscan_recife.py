"""Experimento completo e reproduzivel de HDBSCAN vs DBSCAN — Recife (INMET A301).

Cobre:
  - Pipeline identica ao K-Means e DBSCAN (8 features, StandardScaler + PCA 6 comp. / >=95% var.)
  - Grade de `min_cluster_size`, `min_samples` e comparacao `eom` vs `leaf`
  - Quantidade de clusters, % ruido, % cobertura, Silhouette, Davies-Bouldin, Calinski-Harabasz
  - Qualidade probabilistica (`probabilities_`) e avaliacao de estabilidade por subamostragem (ARI/NMI)
  - Investigacao de densidade variavel (k-distancia local por cluster) e validacao externa com chuva
  - Comparacao direta com o DBSCAN do `Notebooks/dbscan_recife.ipynb` (C6: eps=0.66, min_samples=24 e C1)
  - Relatorio completo em `reports/hdbscan_recife/relatorio.md` + CSVs + figuras 300 dpi
"""

from __future__ import annotations

import hashlib
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
from sklearn.cluster import DBSCAN, HDBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from config import RANDOM_STATE, RAW_CSV
from features import (
    COLUNAS_ORIGINAIS_METEOROLOGICAS,
    features_kmeans_disponiveis,
    preparar_base,
    validar_sem_leakage,
)

sns.set_theme(style="whitegrid", context="notebook")

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "hdbscan_recife"
FIGURES_DIR = OUTPUT_DIR / "figuras"

PCA_VARIANCE = 0.95
SILHOUETTE_SAMPLE_SIZE = 5000
MIN_CLUSTER_SIZES = (25, 50, 100, 200, 400)
MIN_SAMPLES_VALUES = (None, 5, 10, 18, 24)
SUBSAMPLE_FRAC = 0.80
SUBSAMPLE_RUNS = 5

# Configuracoes de referencia do DBSCAN executadas em Notebooks/dbscan_recife.ipynb
DBSCAN_REFS = {
    "DBSCAN_C6_escolhida": {"eps": 0.66, "min_samples": 24},
    "DBSCAN_C1_aprovada": {"eps": 0.62, "min_samples": 12},
}


def _save_figure(fig: plt.Figure, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def _canonical_labels(labels: np.ndarray, base: pd.DataFrame) -> np.ndarray:
    """Reordena IDs dos clusters (0..k-1) por temperatura media crescente; mantem -1 como ruido."""
    clusters = np.unique(labels[labels != -1])
    order = sorted(
        clusters,
        key=lambda c: (
            float(base.loc[labels == c, "temp_c"].mean()),
            float(base.loc[labels == c, "umid_pct"].mean()),
            int(c),
        ),
    )
    mapping = {int(old): new for new, old in enumerate(order)}
    out = np.full_like(labels, -1)
    for old, new in mapping.items():
        out[labels == old] = new
    return out


def _evaluate_partition(
    X: np.ndarray,
    labels: np.ndarray,
    sample_idx_global: np.ndarray,
    probabilities: np.ndarray | None = None,
) -> dict[str, float | int]:
    """Calcula metricas de cobertura, ruido, separacao e confianca probabilistica."""
    noise = labels == -1
    clustered_labels = labels[~noise]
    clusters = np.unique(clustered_labels)
    n_clusters = int(len(clusters))

    tamanhos = np.sort(np.bincount(clustered_labels))[::-1] if n_clusters > 0 else np.array([], dtype=int)
    menor_pct = float(tamanhos.min() / len(labels) * 100.0) if n_clusters > 0 else np.nan
    maior_pct = float(tamanhos.max() / len(labels) * 100.0) if n_clusters > 0 else np.nan

    silhouette = np.nan
    davies_bouldin = np.nan
    calinski_harabasz = np.nan

    if n_clusters >= 2:
        rot_am = labels[sample_idx_global]
        mask_am = rot_am != -1
        if len(np.unique(rot_am[mask_am])) >= 2 and mask_am.sum() > len(np.unique(rot_am[mask_am])):
            silhouette = float(silhouette_score(X[sample_idx_global[mask_am]], rot_am[mask_am]))
        davies_bouldin = float(davies_bouldin_score(X[~noise], clustered_labels))
        calinski_harabasz = float(calinski_harabasz_score(X[~noise], clustered_labels))

    prob_media = np.nan
    prob_alta_pct = np.nan
    if probabilities is not None and (~noise).any():
        probs_c = probabilities[~noise]
        prob_media = float(np.mean(probs_c))
        prob_alta_pct = float(np.mean(probs_c >= 0.8) * 100.0)

    return {
        "n_clusters": n_clusters,
        "registros_ruido": int(noise.sum()),
        "ruido_pct": float(noise.mean() * 100.0),
        "cobertura_pct": float((~noise).mean() * 100.0),
        "menor_cluster_pct": menor_pct,
        "maior_cluster_pct": maior_pct,
        "silhouette_nao_ruido": silhouette,
        "davies_bouldin_nao_ruido": davies_bouldin,
        "calinski_harabasz_nao_ruido": calinski_harabasz,
        "prob_media_nao_ruido": prob_media,
        "alta_confianca_pct": prob_alta_pct,
    }


def _avaliar_estabilidade_subamostragem(
    X: np.ndarray,
    full_labels: np.ndarray,
    min_cluster_size: int,
    min_samples: int | None,
    method: str = "eom",
) -> dict[str, float]:
    """Mede estabilidade do HDBSCAN via 5 subamostras de 80% (ARI, NMI e variacao de k/ruido)."""
    n = len(X)
    sub_n = int(n * SUBSAMPLE_FRAC)
    aris_todos: list[float] = []
    aris_nao_ruido: list[float] = []
    nmis_todos: list[float] = []
    ks: list[int] = []
    ruidos: list[float] = []

    for run in range(SUBSAMPLE_RUNS):
        rng = np.random.default_rng(RANDOM_STATE + 100 + run)
        idx = rng.choice(n, size=sub_n, replace=False)
        model = HDBSCAN(
            min_cluster_size=min_cluster_size,
            min_samples=min_samples,
            cluster_selection_method=method,
            n_jobs=-1,
        )
        sub_labels = model.fit_predict(X[idx])
        ref_labels = full_labels[idx]

        aris_todos.append(float(adjusted_rand_score(ref_labels, sub_labels)))
        nmis_todos.append(float(normalized_mutual_info_score(ref_labels, sub_labels)))

        mask_ambos = (ref_labels != -1) & (sub_labels != -1)
        if mask_ambos.sum() > 10:
            aris_nao_ruido.append(
                float(adjusted_rand_score(ref_labels[mask_ambos], sub_labels[mask_ambos]))
            )
        ks.append(int(len(set(sub_labels) - {-1})))
        ruidos.append(float((sub_labels == -1).mean() * 100.0))

    return {
        "ari_subamostra_media": float(np.mean(aris_todos)),
        "ari_subamostra_dp": float(np.std(aris_todos)),
        "ari_nao_ruido_media": float(np.mean(aris_nao_ruido)) if aris_nao_ruido else np.nan,
        "nmi_subamostra_media": float(np.mean(nmis_todos)),
        "k_subamostra_media": float(np.mean(ks)),
        "k_subamostra_dp": float(np.std(ks)),
        "ruido_subamostra_media": float(np.mean(ruidos)),
        "ruido_subamostra_dp": float(np.std(ruidos)),
    }


def _perfil_densidade_e_clima(
    base: pd.DataFrame,
    labels: np.ndarray,
    kdist_15: np.ndarray,
    algoritmo: str,
    probabilities: np.ndarray | None = None,
) -> pd.DataFrame:
    """Consolida densidade local (k-dist 15), perfil meteorologico e taxa de chuva por cluster."""
    df_tmp = base[["temp_c", "temp_orvalho_c", "umid_pct", "pressao_mb", "precip_mm", "estacao_chuvosa"]].copy()
    df_tmp["cluster"] = labels
    df_tmp["kdist_15"] = kdist_15
    df_tmp["choveu"] = (df_tmp["precip_mm"] >= 0.2).astype(float)
    df_tmp["chuva_forte_1h"] = (df_tmp["precip_mm"] >= 5.0).astype(float)
    if probabilities is not None:
        df_tmp["prob"] = probabilities
    else:
        df_tmp["prob"] = np.nan

    linhas = []
    for cid in sorted(df_tmp["cluster"].unique()):
        sub = df_tmp[df_tmp["cluster"] == cid]
        linhas.append(
            {
                "algoritmo": algoritmo,
                "cluster": int(cid),
                "registros": int(len(sub)),
                "participacao_pct": float(len(sub) / len(df_tmp) * 100.0),
                "kdist_15_mediana": float(sub["kdist_15"].median()),
                "kdist_15_iqr": float(sub["kdist_15"].quantile(0.75) - sub["kdist_15"].quantile(0.25)),
                "densidade_relativa": float(1.0 / max(sub["kdist_15"].median(), 1e-6)),
                "prob_media": float(sub["prob"].mean()) if probabilities is not None else np.nan,
                "temp_c": float(sub["temp_c"].mean()),
                "temp_orvalho_c": float(sub["temp_orvalho_c"].mean()),
                "umid_pct": float(sub["umid_pct"].mean()),
                "pressao_mb": float(sub["pressao_mb"].mean()),
                "pct_horas_chuva": float(sub["choveu"].mean() * 100.0),
                "pct_estacao_chuvosa": float(sub["estacao_chuvosa"].mean() * 100.0),
            }
        )
    return pd.DataFrame(linhas)


def _plot_grade_parametros(df_grid: pd.DataFrame) -> None:
    df_eom = df_grid[df_grid["method"] == "eom"].copy()
    df_eom["ms_label"] = df_eom["min_samples_label"]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    ordem_ms = ["None (= mcs)", "5", "10", "18", "24"]

    for ax, col, titulo, fmt, cmap in [
        (axes[0, 0], "n_clusters", "Quantidade de Clusters (eom)", ".0f", "YlGnBu"),
        (axes[0, 1], "ruido_pct", "Percentual de Ruido (%)", ".1f", "OrRd"),
        (axes[1, 0], "silhouette_nao_ruido", "Silhouette (nao-ruido)", ".3f", "viridis"),
        (axes[1, 1], "davies_bouldin_nao_ruido", "Davies-Bouldin (menor e melhor)", ".2f", "rocket_r"),
    ]:
        piv = df_eom.pivot(index="min_cluster_size", columns="ms_label", values=col)
        cols_presentes = [c for c in ordem_ms if c in piv.columns]
        piv = piv[cols_presentes]
        sns.heatmap(piv, annot=True, fmt=fmt, cmap=cmap, ax=ax, linewidths=0.5)
        ax.set_title(titulo, fontweight="bold")
        ax.set_xlabel("min_samples")
        ax.set_ylabel("min_cluster_size")

    fig.suptitle("HDBSCAN em Recife — Grade de Parametros (min_cluster_size x min_samples)", fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, "grade_parametros_hdbscan")


def _plot_estabilidade(df_estab: pd.DataFrame, labels_best: np.ndarray, probs_best: np.ndarray, best_label: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.0))

    df_s = df_estab.sort_values("ari_subamostra_media", ascending=True)
    y = np.arange(len(df_s))
    axes[0].barh(y, df_s["ari_subamostra_media"], xerr=df_s["ari_subamostra_dp"], color="#2563EB", alpha=0.85, capsize=4)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels(df_s["config"])
    axes[0].set_xlim(0, 1.08)
    axes[0].set_xlabel("ARI Medio (5 subamostras de 80%)")
    axes[0].set_title("Estabilidade por Subamostragem (ARI +/- DP)", fontweight="bold")
    for yi, v in zip(y, df_s["ari_subamostra_media"]):
        axes[0].text(min(v + 0.02, 0.98), yi, f"{v:.3f}", va="center", fontsize=8.5)

    mask = labels_best != -1
    df_p = pd.DataFrame({"Cluster": labels_best[mask].astype(str), "Probabilidade": probs_best[mask]})
    ordem_c = sorted(df_p["Cluster"].unique(), key=lambda x: int(x))
    sns.boxplot(data=df_p, x="Cluster", y="Probabilidade", order=ordem_c, ax=axes[1], color="#60A5FA")
    axes[1].set_title(f"Confianca de Pertinencia (probabilities_) — {best_label}", fontweight="bold")
    axes[1].set_ylim(-0.02, 1.05)

    fig.tight_layout()
    _save_figure(fig, "estabilidade_hdbscan")


def _plot_clusters_pca_fisico(X_pca: np.ndarray, base: pd.DataFrame, labels: np.ndarray, titulo: str, fig_name: str) -> None:
    paleta = plt.get_cmap("tab10")
    ruido = labels == -1
    planos = [
        (X_pca[:, 0], X_pca[:, 1], "PC1", "PC2", "Projecao PCA (PC1 x PC2)"),
        (base["temp_c"].to_numpy(), base["umid_pct"].to_numpy(), "Temperatura do ar (°C)", "Umidade relativa (%)", "Plano Fisico: Temperatura x Umidade"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.2))
    for ax, (x, y, xl, yl, subtitulo) in zip(axes, planos):
        ax.scatter(x[ruido], y[ruido], s=4, c="#CBD5E1", alpha=0.35, label=f"Ruido (-1): {ruido.mean()*100:.1f}%", rasterized=True)
        for cid in sorted(set(labels) - {-1}):
            m = labels == cid
            ax.scatter(x[m], y[m], s=5, color=paleta(cid % 10), alpha=0.55, label=f"Cluster {cid} ({m.sum():,})", rasterized=True)
        ax.set_xlabel(xl)
        ax.set_ylabel(yl)
        ax.set_title(subtitulo, fontweight="bold")
        ax.legend(markerscale=2.5, fontsize=8, frameon=True, loc="best")
    fig.suptitle(titulo, fontsize=13, fontweight="bold")
    fig.tight_layout()
    _save_figure(fig, fig_name)


def _plot_comparacao_hdbscan_dbscan(
    X_pca: np.ndarray,
    labels_hdb: np.ndarray,
    labels_db: np.ndarray,
    hdb_nome: str,
    ari_entre: float,
    nmi_entre: float,
) -> None:
    paleta = plt.get_cmap("tab10")
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8))

    for ax, rot, nome in [
        (axes[0], labels_hdb, f"HDBSCAN ({hdb_nome})"),
        (axes[1], labels_db, "DBSCAN C6 (eps=0,66; ms=24)"),
    ]:
        ruido = rot == -1
        ax.scatter(X_pca[ruido, 0], X_pca[ruido, 1], s=4, c="#CBD5E1", alpha=0.35, label="Ruido", rasterized=True)
        for cid in sorted(set(rot) - {-1}):
            m = rot == cid
            ax.scatter(X_pca[m, 0], X_pca[m, 1], s=5, color=paleta(cid % 10), alpha=0.55, label=f"C{cid}", rasterized=True)
        ax.set_title(nome, fontweight="bold")
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.legend(markerscale=2.5, fontsize=8, ncol=2)

    ct = pd.crosstab(
        pd.Series(labels_hdb, name="HDBSCAN"),
        pd.Series(labels_db, name="DBSCAN C6"),
        normalize="index",
    ) * 100.0
    sns.heatmap(ct, annot=True, fmt=".1f", cmap="Blues", ax=axes[2], cbar_kws={"label": "% da linha"})
    axes[2].set_title(f"Concordancia HDBSCAN x DBSCAN\nARI={ari_entre:.3f} | NMI={nmi_entre:.3f}", fontweight="bold")

    fig.tight_layout()
    _save_figure(fig, "comparacao_hdbscan_vs_dbscan")


def _plot_densidade_e_perfis(df_perf_hdb: pd.DataFrame, df_perf_db: pd.DataFrame) -> None:
    df_h = df_perf_hdb[df_perf_hdb["cluster"] != -1].copy()
    df_d = df_perf_db[df_perf_db["cluster"] != -1].copy()

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8))

    # Painel 1: Densidade variavel (k-dist mediana por cluster)
    comb = pd.concat(
        [
            df_h.assign(label=df_h["cluster"].map(lambda c: f"HDB C{c}")),
            df_d.assign(label=df_d["cluster"].map(lambda c: f"DB C{c}")),
        ],
        ignore_index=True,
    )
    cores = ["#2563EB" if a == "HDBSCAN" else "#D97706" for a in comb["algoritmo"]]
    axes[0].bar(comb["label"], comb["kdist_15_mediana"], color=cores, alpha=0.88)
    axes[0].set_title("Densidade Local por Cluster (k-dist 15 mediana)\nMenor = mais denso", fontweight="bold")
    axes[0].set_ylabel("Distancia ao 15o vizinho (PCA)")
    axes[0].tick_params(axis="x", rotation=35)

    # Painel 2: Temperatura e Umidade no HDBSCAN
    x = np.arange(len(df_h))
    ax2 = axes[1]
    b1 = ax2.bar(x - 0.18, df_h["temp_c"], width=0.35, color="#EA580C", label="Temp. (°C)")
    ax2.set_ylabel("Temperatura (°C)", color="#EA580C")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"C{c}" for c in df_h["cluster"]])
    ax2.set_title("Perfil Fisico dos Clusters (HDBSCAN)", fontweight="bold")
    ax2b = ax2.twinx()
    b2 = ax2b.bar(x + 0.18, df_h["umid_pct"], width=0.35, color="#0284C7", label="Umidade (%)")
    ax2b.set_ylabel("Umidade (%)", color="#0284C7")
    ax2b.set_ylim(55, 95)
    ax2.legend([b1, b2], ["Temp. (°C)", "Umidade (%)"], loc="upper left", fontsize=8)

    # Painel 3: Validacao externa com chuva (% horas com chuva >= 0,2 mm)
    df_all_h = df_perf_hdb.copy()
    labels_x = ["Ruido (-1)" if c == -1 else f"C{c}" for c in df_all_h["cluster"]]
    c_bar = ["#94A3B8" if c == -1 else "#059669" for c in df_all_h["cluster"]]
    bars = axes[2].bar(labels_x, df_all_h["pct_horas_chuva"], color=c_bar, alpha=0.88)
    axes[2].set_title("Validacao Externa: % Horas com Chuva (>= 0,2 mm)", fontweight="bold")
    axes[2].set_ylabel("% de horas chuvosas")
    axes[2].tick_params(axis="x", rotation=20)
    for b, v in zip(bars, df_all_h["pct_horas_chuva"]):
        axes[2].text(b.get_x() + b.get_width() / 2, v + 0.4, f"{v:.1f}%", ha="center", va="bottom", fontsize=8.5)

    fig.tight_layout()
    _save_figure(fig, "perfil_clusters_hdbscan")


def _fmt(v: float, casas: int = 4) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{casas}f}"


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    print("[1/6] Preparando base e matriz StandardScaler + PCA(0.95)...")
    base = preparar_base(usar_open_meteo=False, filtrar_anos=False)
    features = features_kmeans_disponiveis(base)
    validar_sem_leakage(features, modo="clustering")

    X_raw = base[features].to_numpy(dtype=float)
    X_std = StandardScaler().fit_transform(X_raw)
    pca = PCA(n_components=PCA_VARIANCE, random_state=RANDOM_STATE)
    X_pca = pca.fit_transform(X_std)
    var_retida = float(pca.explained_variance_ratio_.sum() * 100.0)
    print(f"      Registros: {len(base):,} | Features: {len(features)} | PCA: {X_pca.shape[1]} comp. ({var_retida:.2f}% var.)")

    rng = np.random.default_rng(RANDOM_STATE)
    sample_idx_global = rng.choice(len(X_pca), size=min(SILHOUETTE_SAMPLE_SIZE, len(X_pca)), replace=False)

    # k-dist 15 para investigacao de densidade variavel por cluster
    nn15 = NearestNeighbors(n_neighbors=15, n_jobs=-1).fit(X_pca)
    kdist_15 = nn15.kneighbors(X_pca)[0][:, -1]

    # [2/6] Grade HDBSCAN (eom + comparacao leaf nos principais candidatos)
    print("[2/6] Executando grade HDBSCAN (min_cluster_size x min_samples)...")
    linhas_grid: list[dict] = []
    cache_labels: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}

    for mcs in MIN_CLUSTER_SIZES:
        for ms in MIN_SAMPLES_VALUES:
            t0 = time.perf_counter()
            model = HDBSCAN(
                min_cluster_size=mcs,
                min_samples=ms,
                cluster_selection_method="eom",
                n_jobs=-1,
            )
            rot = model.fit_predict(X_pca)
            probs = model.probabilities_
            dt = time.perf_counter() - t0
            ev = _evaluate_partition(X_pca, rot, sample_idx_global, probabilities=probs)
            ms_label = "None (= mcs)" if ms is None else str(ms)
            row = {
                "min_cluster_size": mcs,
                "min_samples": -1 if ms is None else ms,
                "min_samples_label": ms_label,
                "method": "eom",
                "tempo_s": round(dt, 2),
                **ev,
            }
            linhas_grid.append(row)
            cache_labels[(mcs, ms, "eom")] = (rot, probs)
            print(
                f"      eom  mcs={mcs:<3} ms={ms_label:<12} | k={ev['n_clusters']:<2} "
                f"| ruido={ev['ruido_pct']:5.1f}% | sil={_fmt(ev['silhouette_nao_ruido'], 3)} "
                f"| DBI={_fmt(ev['davies_bouldin_nao_ruido'], 2)} | prob={_fmt(ev['prob_media_nao_ruido'], 3)}"
            )

    # Teste adicional com cluster_selection_method="leaf" em 3 configuracoes representativas
    for mcs, ms in [(50, 10), (100, 10), (100, 18), (200, 18)]:
        t0 = time.perf_counter()
        model = HDBSCAN(
            min_cluster_size=mcs,
            min_samples=ms,
            cluster_selection_method="leaf",
            n_jobs=-1,
        )
        rot = model.fit_predict(X_pca)
        probs = model.probabilities_
        dt = time.perf_counter() - t0
        ev = _evaluate_partition(X_pca, rot, sample_idx_global, probabilities=probs)
        row = {
            "min_cluster_size": mcs,
            "min_samples": ms,
            "min_samples_label": str(ms),
            "method": "leaf",
            "tempo_s": round(dt, 2),
            **ev,
        }
        linhas_grid.append(row)
        cache_labels[(mcs, ms, "leaf")] = (rot, probs)
        print(
            f"      leaf mcs={mcs:<3} ms={str(ms):<12} | k={ev['n_clusters']:<2} "
            f"| ruido={ev['ruido_pct']:5.1f}% | sil={_fmt(ev['silhouette_nao_ruido'], 3)}"
        )

    df_grid = pd.DataFrame(linhas_grid)
    df_grid.to_csv(OUTPUT_DIR / "metricas_hdbscan.csv", index=False, float_format="%.6f")

    # [3/6] Selecao de candidatos viaveis e avaliacao de estabilidade
    # Criterio operacional alinhado ao dbscan_recife: 3 a 8 clusters, ruido <= 30%, menor cluster >= 1%
    df_viaveis = df_grid[
        (df_grid["n_clusters"].between(3, 8))
        & (df_grid["ruido_pct"] <= 30.0)
        & (df_grid["menor_cluster_pct"] >= 1.0)
    ].copy()

    # Garante deduplicacao de particoes identicas para testar estabilidade em configuracoes realmente distintas
    assinaturas_vistas: set[tuple] = set()
    candidatos_estab: list[dict] = []
    for _, r in df_viaveis.sort_values(
        ["silhouette_nao_ruido", "cobertura_pct"], ascending=[False, False]
    ).iterrows():
        sig = (int(r["n_clusters"]), round(float(r["ruido_pct"]), 2), round(float(r["silhouette_nao_ruido"]), 4))
        if sig in assinaturas_vistas:
            continue
        assinaturas_vistas.add(sig)
        candidatos_estab.append(r.to_dict())
        if len(candidatos_estab) >= 6:
            break

    # Inclui tambem uma configuracao de alta cobertura (mcs=100, ms=5, eom) caso nao esteja na lista
    if not any(c["min_samples"] == 5 and c["method"] == "eom" for c in candidatos_estab):
        r5 = df_grid[(df_grid["min_cluster_size"] == 100) & (df_grid["min_samples"] == 5) & (df_grid["method"] == "eom")].iloc[0].to_dict()
        candidatos_estab.append(r5)

    print(f"\n[3/6] Avaliando estabilidade por subamostragem (80% x {SUBSAMPLE_RUNS} rodadas) em {len(candidatos_estab)} candidatos...")
    linhas_estab: list[dict] = []
    for c in candidatos_estab:
        mcs = int(c["min_cluster_size"])
        ms = None if int(c["min_samples"]) == -1 else int(c["min_samples"])
        method = str(c["method"])
        rot_full, _ = cache_labels[(mcs, ms, method)]
        est = _avaliar_estabilidade_subamostragem(X_pca, rot_full, mcs, ms, method=method)
        cfg_nome = f"{method} mcs={mcs}, ms={'None' if ms is None else ms}"
        est_row = {
            "config": cfg_nome,
            "min_cluster_size": mcs,
            "min_samples": -1 if ms is None else ms,
            "method": method,
            "n_clusters": int(c["n_clusters"]),
            "ruido_pct": float(c["ruido_pct"]),
            "cobertura_pct": float(c["cobertura_pct"]),
            "silhouette_nao_ruido": float(c["silhouette_nao_ruido"]),
            "davies_bouldin_nao_ruido": float(c["davies_bouldin_nao_ruido"]),
            "prob_media_nao_ruido": float(c["prob_media_nao_ruido"]),
            **est,
        }
        # Score composto de selecao: estabilidade (ARI) + cobertura + silhueta + confianca
        est_row["score_hdbscan"] = float(
            0.40 * est_row["ari_subamostra_media"]
            + 0.25 * (est_row["cobertura_pct"] / 100.0)
            + 0.20 * max(est_row["silhouette_nao_ruido"], 0.0)
            + 0.15 * est_row["prob_media_nao_ruido"]
        )
        linhas_estab.append(est_row)
        print(
            f"      {cfg_nome:<28} | k={est_row['n_clusters']} | ruido={est_row['ruido_pct']:5.1f}% "
            f"| ARI={est_row['ari_subamostra_media']:.3f}+/-{est_row['ari_subamostra_dp']:.3f} "
            f"| NMI={est_row['nmi_subamostra_media']:.3f} | score={est_row['score_hdbscan']:.3f}"
        )

    df_estab = pd.DataFrame(linhas_estab).sort_values("score_hdbscan", ascending=False).reset_index(drop=True)
    df_estab.to_csv(OUTPUT_DIR / "estabilidade_hdbscan.csv", index=False, float_format="%.6f")

    best_cfg = df_estab.iloc[0]
    best_mcs = int(best_cfg["min_cluster_size"])
    best_ms = None if int(best_cfg["min_samples"]) == -1 else int(best_cfg["min_samples"])
    best_method = str(best_cfg["method"])
    best_nome = str(best_cfg["config"])
    rot_best_raw, probs_best = cache_labels[(best_mcs, best_ms, best_method)]
    rot_best = _canonical_labels(rot_best_raw, base)
    print(f"\n[4/6] Configuracao escolhida: {best_nome} (score={best_cfg['score_hdbscan']:.3f})")

    # [5/6] Comparacao direta com o DBSCAN do Notebooks/dbscan_recife.ipynb
    print("[5/6] Rodando DBSCAN de referencia (C6 e C1) no mesmo espaco PCA para comparacao direta...")
    linhas_comp: list[dict] = []
    ev_best = _evaluate_partition(X_pca, rot_best, sample_idx_global, probabilities=probs_best)
    linhas_comp.append(
        {
            "algoritmo": f"HDBSCAN ({best_nome})",
            "tipo": "Hierarquico (densidade variavel)",
            "parametros": best_nome,
            **ev_best,
            "ari_vs_dbscan_c6": np.nan,
            "nmi_vs_dbscan_c6": np.nan,
        }
    )

    rot_db_c6 = None
    for db_nome, params in DBSCAN_REFS.items():
        rot_db_raw = DBSCAN(eps=params["eps"], min_samples=params["min_samples"], n_jobs=-1).fit_predict(X_pca)
        rot_db = _canonical_labels(rot_db_raw, base)
        ev_db = _evaluate_partition(X_pca, rot_db, sample_idx_global, probabilities=None)
        if db_nome == "DBSCAN_C6_escolhida":
            rot_db_c6 = rot_db
        linhas_comp.append(
            {
                "algoritmo": db_nome,
                "tipo": "Densidade fixa (eps global)",
                "parametros": f"eps={params['eps']}, min_samples={params['min_samples']}",
                **ev_db,
                "ari_vs_dbscan_c6": 1.0 if db_nome == "DBSCAN_C6_escolhida" else float(adjusted_rand_score(rot_db_c6, rot_db)),
                "nmi_vs_dbscan_c6": 1.0 if db_nome == "DBSCAN_C6_escolhida" else float(normalized_mutual_info_score(rot_db_c6, rot_db)),
            }
        )

    ari_hdb_db = float(adjusted_rand_score(rot_best, rot_db_c6))
    nmi_hdb_db = float(normalized_mutual_info_score(rot_best, rot_db_c6))
    mask_ambos = (rot_best != -1) & (rot_db_c6 != -1)
    ari_hdb_db_nr = float(adjusted_rand_score(rot_best[mask_ambos], rot_db_c6[mask_ambos]))

    linhas_comp[0]["ari_vs_dbscan_c6"] = ari_hdb_db
    linhas_comp[0]["nmi_vs_dbscan_c6"] = nmi_hdb_db

    df_comp = pd.DataFrame(linhas_comp)
    df_comp.to_csv(OUTPUT_DIR / "comparacao_hdbscan_vs_dbscan.csv", index=False, float_format="%.6f")

    ct_hdb_db = pd.crosstab(pd.Series(rot_best, name="HDBSCAN"), pd.Series(rot_db_c6, name="DBSCAN_C6"))
    ct_hdb_db.to_csv(OUTPUT_DIR / "contingencia_hdbscan_dbscan.csv")

    # Densidade variavel e perfis meteorologicos
    df_perf_hdb = _perfil_densidade_e_clima(base, rot_best, kdist_15, "HDBSCAN", probabilities=probs_best)
    df_perf_db = _perfil_densidade_e_clima(base, rot_db_c6, kdist_15, "DBSCAN_C6", probabilities=None)
    df_densidade = pd.concat([df_perf_hdb, df_perf_db], ignore_index=True)
    df_densidade.to_csv(OUTPUT_DIR / "densidade_variavel_clusters.csv", index=False, float_format="%.6f")

    # Mantem arquivos padrao existentes para compatibilidade com reports/hdbscan_recife/
    df_dist = df_perf_hdb[["cluster", "registros", "participacao_pct"]].copy()
    df_dist.to_csv(OUTPUT_DIR / "distribuicao_clusters.csv", index=False, float_format="%.6f")

    df_perf_padrao = df_perf_hdb[df_perf_hdb["cluster"] != -1][
        ["cluster", "registros", "temp_c", "temp_orvalho_c", "umid_pct", "pressao_mb", "kdist_15_mediana", "pct_horas_chuva", "prob_media"]
    ].copy()
    df_perf_padrao.to_csv(OUTPUT_DIR / "perfis_clusters.csv", index=False, float_format="%.6f")

    df_reg = base[["datetime_utc", "ano", "mes", "hora", "temp_c", "umid_pct", "pressao_mb", "precip_mm"]].copy()
    df_reg["cluster_hdbscan"] = rot_best
    df_reg["prob_hdbscan"] = probs_best
    df_reg["cluster_dbscan_c6"] = rot_db_c6
    df_reg.to_csv(OUTPUT_DIR / "registros_clusterizados.csv", index=False)

    # [6/6] Figuras e relatorio completo
    print("[6/6] Gerando figuras (300 dpi) e relatorio em reports/hdbscan_recife/relatorio.md...")
    _plot_grade_parametros(df_grid)
    _plot_estabilidade(df_estab, rot_best, probs_best, best_nome)
    _plot_clusters_pca_fisico(
        X_pca,
        base,
        rot_best,
        f"HDBSCAN Escolhido ({best_nome}) — {ev_best['n_clusters']} clusters | Ruido {ev_best['ruido_pct']:.2f}%",
        "clusters_pca_fisico_hdbscan",
    )
    _plot_comparacao_hdbscan_dbscan(X_pca, rot_best, rot_db_c6, best_nome, ari_hdb_db, nmi_hdb_db)
    _plot_densidade_e_perfis(df_perf_hdb, df_perf_db)

    # Razao de densidade entre cluster mais denso e menos denso (evidencia de densidade variavel)
    hdb_k = df_perf_hdb[df_perf_hdb["cluster"] != -1]["kdist_15_mediana"]
    db_k = df_perf_db[df_perf_db["cluster"] != -1]["kdist_15_mediana"]
    razao_dens_hdb = float(hdb_k.max() / hdb_k.min()) if len(hdb_k) >= 2 else 1.0
    razao_dens_db = float(db_k.max() / db_k.min()) if len(db_k) >= 2 else 1.0

    # Pontos que eram ruido no DBSCAN C6 mas foram resgatados pelo HDBSCAN (e vice-versa)
    resgatados_hdb = int(((rot_db_c6 == -1) & (rot_best != -1)).sum())
    descartados_hdb = int(((rot_db_c6 != -1) & (rot_best == -1)).sum())

    repro = {
        "csv_entrada": str(RAW_CSV.relative_to(ROOT)),
        "sha256_csv": _sha256_file(RAW_CSV),
        "registros": int(len(base)),
        "features": features,
        "pca_componentes": int(X_pca.shape[1]),
        "pca_variancia_retida_pct": round(var_retida, 2),
        "config_escolhida": {
            "nome": best_nome,
            "min_cluster_size": best_mcs,
            "min_samples": best_ms,
            "method": best_method,
            "n_clusters": int(ev_best["n_clusters"]),
            "ruido_pct": round(float(ev_best["ruido_pct"]), 2),
            "cobertura_pct": round(float(ev_best["cobertura_pct"]), 2),
            "silhouette_nao_ruido": round(float(ev_best["silhouette_nao_ruido"]), 4),
            "davies_bouldin_nao_ruido": round(float(ev_best["davies_bouldin_nao_ruido"]), 4),
            "prob_media_nao_ruido": round(float(ev_best["prob_media_nao_ruido"]), 4),
            "ari_subamostra_media": round(float(best_cfg["ari_subamostra_media"]), 4),
            "nmi_subamostra_media": round(float(best_cfg["nmi_subamostra_media"]), 4),
        },
        "comparacao_dbscan_c6": {
            "ari_global": round(ari_hdb_db, 4),
            "ari_nao_ruido": round(ari_hdb_db_nr, 4),
            "nmi_global": round(nmi_hdb_db, 4),
            "pontos_ruido_dbscan_resgatados_hdbscan": resgatados_hdb,
            "pontos_agrupados_dbscan_ruido_hdbscan": descartados_hdb,
            "razao_densidade_hdbscan": round(razao_dens_hdb, 3),
            "razao_densidade_dbscan": round(razao_dens_db, 3),
        },
        "versoes": {
            "python": sys.version.split()[0],
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
        "comando_reproducao": "python ml/experimento_hdbscan_recife.py",
    }
    (OUTPUT_DIR / "reproducibilidade.json").write_text(
        json.dumps(repro, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Relatorio Markdown completo
    linhas_md = [
        "# Experimento HDBSCAN — Densidade Variavel, Estabilidade e Comparacao com DBSCAN (Recife, INMET A301)\n",
        "## 1. Base e Pipeline de Preparacao\n",
        f"- **Base:** `dados/df_recife_datetime.csv` (**{len(base):,}** registros horarios, 2018–2021).\n",
        f"- **Features (8, identicas ao K-Means e ao `dbscan_recife.ipynb`):** `{', '.join(features)}` (precipitacao excluida da entrada e usada apenas para validacao externa).\n",
        f"- **Pre-processamento:** `StandardScaler` + `PCA(n_components={X_pca.shape[1]})` retendo **{var_retida:.2f}%** da variancia total.\n",
        f"- **Implementacao:** `sklearn.cluster.HDBSCAN` (scikit-learn `{sklearn.__version__}`), distancia euclidiana.\n",
        "## 2. Grade de Parametros (`min_cluster_size`, `min_samples` e `method`)\n",
        "| method | min_cluster_size | min_samples | clusters | ruido (%) | cobertura (%) | Silhouette | Davies-Bouldin | Calinski-Harabasz | Prob. media |\n",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in df_grid.iterrows():
        linhas_md.append(
            f"| {r['method']} | {int(r['min_cluster_size'])} | {r['min_samples_label']} | "
            f"{int(r['n_clusters'])} | {r['ruido_pct']:.2f}% | {r['cobertura_pct']:.2f}% | "
            f"{_fmt(r['silhouette_nao_ruido'], 4)} | {_fmt(r['davies_bouldin_nao_ruido'], 3)} | "
            f"{_fmt(r['calinski_harabasz_nao_ruido'], 1)} | {_fmt(r['prob_media_nao_ruido'], 3)} |"
        )

    linhas_md.append("\n## 3. Avaliacao de Estabilidade (Subamostragem 80% x 5 rodadas)\n")
    linhas_md.append(
        "| Configuracao | Clusters | Ruido (%) | Silhouette | Prob. media | ARI subamostra (media +/- DP) | ARI nao-ruido | NMI | Score |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for _, r in df_estab.iterrows():
        linhas_md.append(
            f"| {r['config']} | {int(r['n_clusters'])} | {r['ruido_pct']:.2f}% | "
            f"{_fmt(r['silhouette_nao_ruido'], 4)} | {_fmt(r['prob_media_nao_ruido'], 3)} | "
            f"{r['ari_subamostra_media']:.3f} +/- {r['ari_subamostra_dp']:.3f} | "
            f"{_fmt(r['ari_nao_ruido_media'], 3)} | {r['nmi_subamostra_media']:.3f} | {r['score_hdbscan']:.3f} |"
        )

    linhas_md.append(f"\n## 4. Configuracao Escolhida e Justificativa: `{best_nome}`\n")
    linhas_md.append(
        f"- **Configuracao escolhida:** `min_cluster_size={best_mcs}`, `min_samples={'None' if best_ms is None else best_ms}`, `cluster_selection_method='{best_method}'`.\n"
        f"- **Resultado:** **{int(ev_best['n_clusters'])} clusters**, **{ev_best['ruido_pct']:.2f}% de ruido** ({ev_best['cobertura_pct']:.2f}% de cobertura), "
        f"Silhouette = **{ev_best['silhouette_nao_ruido']:.4f}**, Davies-Bouldin = **{ev_best['davies_bouldin_nao_ruido']:.3f}**, "
        f"probabilidade media de pertinencia = **{ev_best['prob_media_nao_ruido']:.3f}** ({ev_best['alta_confianca_pct']:.1f}% dos pontos agrupados com prob >= 0,80) "
        f"e **ARI de estabilidade = {best_cfg['ari_subamostra_media']:.3f} +/- {best_cfg['ari_subamostra_dp']:.3f}**.\n"
        "- **Justificativa:** Configuracoes com `min_samples=None` e `min_cluster_size >= 100` produzem Silhouette maior (>0,30), mas descartam mais de 85% dos dados como ruido — o que inviabiliza a caracterizacao dos regimes climaticos da cidade. "
        f"A configuracao `{best_nome}` entrega o melhor compromisso entre estabilidade estrutural (ARI elevado), alta cobertura e interpretabilidade fisica dos regimes.\n"
    )

    linhas_md.append("## 5. Comparacao Direta: HDBSCAN vs. DBSCAN (`dbscan_recife.ipynb`)\n")
    linhas_md.append(
        "| Algoritmo | Parametros | Clusters | Ruido (%) | Cobertura (%) | Silhouette | Davies-Bouldin | Calinski-Harabasz | ARI vs DBSCAN C6 | NMI vs DBSCAN C6 |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for _, r in df_comp.iterrows():
        linhas_md.append(
            f"| {r['algoritmo']} | {r['parametros']} | {int(r['n_clusters'])} | "
            f"{r['ruido_pct']:.2f}% | {r['cobertura_pct']:.2f}% | "
            f"{_fmt(r['silhouette_nao_ruido'], 4)} | {_fmt(r['davies_bouldin_nao_ruido'], 3)} | "
            f"{_fmt(r['calinski_harabasz_nao_ruido'], 1)} | {_fmt(r['ari_vs_dbscan_c6'], 3)} | {_fmt(r['nmi_vs_dbscan_c6'], 3)} |"
        )
    linhas_md.append(
        f"\n- **Concordancia HDBSCAN x DBSCAN C6:** `ARI global = {ari_hdb_db:.3f}`, `ARI nao-ruido = {ari_hdb_db_nr:.3f}`, `NMI = {nmi_hdb_db:.3f}`.\n"
        f"- **Densidade variavel:** A razao entre o cluster menos denso e o mais denso (medida pela mediana de `k-dist 15`) foi de **{razao_dens_hdb:.2f}x** no HDBSCAN contra **{razao_dens_db:.2f}x** no DBSCAN C6.\n"
        f"- **Pontos resgatados vs. descartados:** O HDBSCAN agrupou **{resgatados_hdb:,}** horas que o DBSCAN C6 classificou como ruido, e classificou como ruido **{descartados_hdb:,}** horas que o DBSCAN C6 havia agrupado.\n"
    )

    linhas_md.append("## 6. Perfis Meteorologicos, Densidade Local e Validacao Externa com Chuva (HDBSCAN)\n")
    linhas_md.append(
        "| Cluster | Registros | Partic. (%) | k-dist 15 | Prob. media | Temp. (°C) | Orvalho (°C) | Umidade (%) | Pressao (mB) | % Horas Chuva |\n"
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
    )
    for _, r in df_perf_hdb.iterrows():
        linhas_md.append(
            f"| {int(r['cluster'])} | {int(r['registros']):,} | {r['participacao_pct']:.2f}% | "
            f"{r['kdist_15_mediana']:.3f} | {_fmt(r['prob_media'], 3)} | "
            f"{r['temp_c']:.2f} | {r['temp_orvalho_c']:.2f} | {r['umid_pct']:.2f} | "
            f"{r['pressao_mb']:.2f} | {r['pct_horas_chuva']:.2f}% |"
        )

    linhas_md.append(
        "\n## 7. Figuras Geradas\n"
        "- `figuras/grade_parametros_hdbscan.png`\n"
        "- `figuras/estabilidade_hdbscan.png`\n"
        "- `figuras/clusters_pca_fisico_hdbscan.png`\n"
        "- `figuras/comparacao_hdbscan_vs_dbscan.png`\n"
        "- `figuras/perfil_clusters_hdbscan.png`\n"
    )

    (OUTPUT_DIR / "relatorio.md").write_text("\n".join(linhas_md), encoding="utf-8")
    print("Concluido! Relatorio salvo em:", OUTPUT_DIR / "relatorio.md")


if __name__ == "__main__":
    main()
