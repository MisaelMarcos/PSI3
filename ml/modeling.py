"""Treino dos classificadores, KMeans e avaliacao temporal."""

from __future__ import annotations

from typing import Iterable

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ALVOS, RANDOM_STATE
from features import FEATURES_KMEANS
from metrics import avaliar_completo, escolher_limiar, escolher_limiar_f1


def _matriz(df: pd.DataFrame, features: Iterable[str]) -> pd.DataFrame:
    X = df[list(features)].astype(float).copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    return X.fillna(0.0)


def criar_modelo(
    n_estimators: int = 300,
    max_depth: int | None = 8,
    random_state: int = RANDOM_STATE,
) -> RandomForestClassifier:
    # Sem class_weight: a calibracao fica por conta do limiar escolhido na
    # validacao. Nos experimentos, isso deu melhor PR-AUC que balanced.
    return RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_leaf=5,
        min_samples_split=10,
        max_features="sqrt",
        class_weight=None,
        n_jobs=-1,
        random_state=random_state,
    )


def treinar_kmeans(
    X: pd.DataFrame,
    k_min: int = 2,
    k_max: int = 6,
    random_state: int = RANDOM_STATE,
) -> tuple[Pipeline, dict]:
    """Ajusta StandardScaler+KMeans e escolhe k por silhouette.

    A silhueta e calculada em uma amostra de ate 4.000 linhas para manter o
    custo baixo. O pipeline final e treinado em todas as linhas recebidas.
    """
    X = _matriz(X, X.columns)
    if len(X) < k_max:
        raise ValueError("Poucas linhas para treinar o KMeans.")

    amostra_idx = np.random.default_rng(random_state).choice(
        len(X), size=min(4000, len(X)), replace=False
    )
    avaliacoes = []
    melhor_pipeline = None
    melhor_silhueta = -np.inf
    melhor_k = None

    for k in range(k_min, k_max + 1):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                (
                    "kmeans",
                    KMeans(
                        n_clusters=k,
                        random_state=random_state,
                        n_init=10,
                    ),
                ),
            ]
        )
        pipeline.fit(X)
        rotulos = pipeline.named_steps["kmeans"].labels_
        X_escalado = pipeline.named_steps["scaler"].transform(X)
        from sklearn.metrics import silhouette_score

        silhueta = float(
            silhouette_score(X_escalado[amostra_idx], rotulos[amostra_idx])
        )
        avaliacoes.append(
            {
                "k": k,
                "silhouette": silhueta,
                "inercia": float(pipeline.named_steps["kmeans"].inertia_),
            }
        )
        if silhueta > melhor_silhueta:
            melhor_silhueta = silhueta
            melhor_pipeline = pipeline
            melhor_k = k

    assert melhor_pipeline is not None and melhor_k is not None
    return melhor_pipeline, {
        "k": int(melhor_k),
        "silhouette": float(melhor_silhueta),
        "avaliacoes": avaliacoes,
    }


def adicionar_clusters(
    df: pd.DataFrame,
    kmeans: Pipeline,
    features_kmeans: Iterable[str] = FEATURES_KMEANS,
) -> tuple[pd.DataFrame, list[str]]:
    """Adiciona rotulo e one-hot do cluster ao dataframe."""
    df = df.copy()
    X = _matriz(df, features_kmeans)
    rotulos = kmeans.predict(X).astype(int)
    df["cluster"] = rotulos
    n_clusters = int(kmeans.named_steps["kmeans"].n_clusters)
    colunas = []
    for i in range(n_clusters):
        coluna = f"cluster_{i}"
        df[coluna] = (df["cluster"] == i).astype(int)
        colunas.append(coluna)
    return df, colunas


def _base_2019(df: pd.DataFrame, alvo: str) -> pd.DataFrame:
    coluna = ALVOS[alvo]["coluna"]
    return (
        df[(df["ano"] == 2019) & df[coluna].notna()]
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )


def _base_teste_2020(df: pd.DataFrame, alvo: str) -> pd.DataFrame:
    coluna = ALVOS[alvo]["coluna"]
    return (
        df[(df["ano"] == 2020) & df[coluna].notna()]
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )


def avaliar_cv_2019(
    df: pd.DataFrame,
    alvo: str,
    features_base: list[str],
    usar_cluster: bool,
    kmeans_2019: Pipeline | None = None,
    features_kmeans: list[str] | None = None,
    n_splits: int = 4,
    n_estimators: int = 250,
) -> dict:
    """Validação temporal dentro de 2019, com gap de 3 horas.

    O gap evita que o alvo de uma linha de treino (que olha ate t+3) se
    sobreponha ao periodo de validação.
    """
    info_alvo = ALVOS[alvo]
    coluna = info_alvo["coluna"]
    base = _base_2019(df, alvo)
    if len(base) < 500:
        raise ValueError(f"Poucas linhas de 2019 para o alvo {alvo}: {len(base)}")

    features_kmeans = features_kmeans or FEATURES_KMEANS
    if usar_cluster:
        if kmeans_2019 is None:
            kmeans_2019, _ = treinar_kmeans(base[features_kmeans])
        base, colunas_cluster = adicionar_clusters(
            base, kmeans_2019, features_kmeans
        )
        features = features_base + colunas_cluster
    else:
        features = list(features_base)

    y = base[coluna].astype(int).to_numpy()
    X = _matriz(base, features)
    tscv = TimeSeriesSplit(n_splits=n_splits, gap=3)
    partes_y = []
    partes_proba = []
    folds = 0

    for idx_treino, idx_val in tscv.split(X):
        y_treino = y[idx_treino]
        y_val = y[idx_val]
        if len(np.unique(y_treino)) < 2 or len(np.unique(y_val)) < 2:
            continue
        modelo = criar_modelo(n_estimators=n_estimators)
        modelo.fit(X.iloc[idx_treino], y_treino)
        proba = modelo.predict_proba(X.iloc[idx_val])[:, 1]
        partes_y.append(y_val)
        partes_proba.append(proba)
        folds += 1

    if not partes_y:
        raise RuntimeError(f"Nao foi possivel validar o alvo {alvo}.")

    y_oof = np.concatenate(partes_y)
    proba_oof = np.concatenate(partes_proba)
    corte = escolher_limiar(y_oof, proba_oof, alvo_recall=0.8)
    corte_forte = escolher_limiar_f1(y_oof, proba_oof)
    # O nivel forte nunca pode ser mais sensivel que o nivel de atencao.
    corte_forte["limiar"] = max(corte_forte["limiar"], corte["limiar"])
    metricas = avaliar_completo(
        y_oof,
        proba_oof,
        horizonte=info_alvo["horizonte_h"],
        limiar=corte["limiar"],
    )
    metricas_forte = avaliar_completo(
        y_oof,
        proba_oof,
        horizonte=info_alvo["horizonte_h"],
        limiar=corte_forte["limiar"],
    )
    return {
        "alvo": alvo,
        "usar_cluster": bool(usar_cluster),
        "n_folds": folds,
        "y_oof": y_oof,
        "proba_oof": proba_oof,
        "corte": corte,
        "corte_forte": corte_forte,
        "pr_auc": float(average_precision_score(y_oof, proba_oof)),
        "roc_auc": float(roc_auc_score(y_oof, proba_oof)),
        "metricas_oof": metricas,
        "metricas_oof_forte": metricas_forte,
        "features": features,
        "features_kmeans": features_kmeans,
        "kmeans_2019": kmeans_2019,
    }


def avaliar_holdout_2020(
    df: pd.DataFrame,
    alvo: str,
    features_base: list[str],
    usar_cluster: bool,
    kmeans_2019: Pipeline | None,
    limiar: float,
    limiar_forte: float | None = None,
    features_kmeans: list[str] | None = None,
    n_estimators: int = 300,
) -> dict:
    """Treina em 2019 e avalia o ano de 2020 (holdout temporal)."""
    info_alvo = ALVOS[alvo]
    coluna = info_alvo["coluna"]
    treino = _base_2019(df, alvo)
    teste = _base_teste_2020(df, alvo)
    features_kmeans = features_kmeans or FEATURES_KMEANS

    if usar_cluster:
        if kmeans_2019 is None:
            raise ValueError("KMeans de 2019 e obrigatorio quando usar_cluster=True.")
        treino, colunas_cluster = adicionar_clusters(
            treino, kmeans_2019, features_kmeans
        )
        teste, _ = adicionar_clusters(teste, kmeans_2019, features_kmeans)
        features = features_base + colunas_cluster
    else:
        features = list(features_base)

    modelo = criar_modelo(n_estimators=n_estimators)
    modelo.fit(_matriz(treino, features), treino[coluna].astype(int))
    proba = modelo.predict_proba(_matriz(teste, features))[:, 1]
    y_teste = teste[coluna].astype(int).to_numpy()
    metricas = avaliar_completo(
        y_teste,
        proba,
        horizonte=info_alvo["horizonte_h"],
        limiar=limiar,
    )
    limiar_forte = float(limiar_forte if limiar_forte is not None else limiar)
    metricas_forte = avaliar_completo(
        y_teste,
        proba,
        horizonte=info_alvo["horizonte_h"],
        limiar=limiar_forte,
    )
    return {
        "alvo": alvo,
        "usar_cluster": bool(usar_cluster),
        "modelo": modelo,
        "features": features,
        "limiar": float(limiar),
        "limiar_forte": limiar_forte,
        "metricas_teste": metricas,
        "metricas_teste_forte": metricas_forte,
        "proba_teste": proba,
        "datas_teste": teste["datetime_local"].astype(str).to_list(),
        "y_teste": y_teste,
    }


def treinar_modelo_final(
    df: pd.DataFrame,
    alvo: str,
    features_base: list[str],
    usar_cluster: bool,
    features_kmeans: list[str] | None = None,
    n_estimators: int = 400,
) -> dict:
    """Treina o modelo de producao com todos os dados disponiveis."""
    info_alvo = ALVOS[alvo]
    coluna = info_alvo["coluna"]
    base = (
        df[df[coluna].notna()]
        .sort_values("datetime_utc")
        .reset_index(drop=True)
    )
    features_kmeans = features_kmeans or FEATURES_KMEANS

    kmeans_final = None
    info_kmeans = None
    if usar_cluster:
        kmeans_final, info_kmeans = treinar_kmeans(base[features_kmeans])
        base, colunas_cluster = adicionar_clusters(
            base, kmeans_final, features_kmeans
        )
        features = features_base + colunas_cluster
    else:
        features = list(features_base)

    modelo = criar_modelo(n_estimators=n_estimators)
    modelo.fit(_matriz(base, features), base[coluna].astype(int))
    return {
        "alvo": alvo,
        "modelo": modelo,
        "features": features,
        "features_kmeans": features_kmeans,
        "usar_cluster": bool(usar_cluster),
        "kmeans": kmeans_final,
        "info_kmeans": info_kmeans,
        "n_treino": int(len(base)),
        "periodo_treino": "2019-2020",
    }


def salvar_modelo(pacote: dict, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pacote, caminho)


def carregar_modelo(caminho: Path) -> dict:
    return joblib.load(caminho)


def importancia_features(modelo: RandomForestClassifier, features: list[str]) -> list[dict]:
    importancias = getattr(modelo, "feature_importances_", None)
    if importancias is None:
        return []
    pares = sorted(
        zip(features, importancias), key=lambda par: par[1], reverse=True
    )
    return [
        {"feature": nome, "importancia": float(valor)}
        for nome, valor in pares
    ]
