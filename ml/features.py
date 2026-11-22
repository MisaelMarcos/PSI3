"""Engenharia de features, governanca anti-leakage e construcao dos alvos de chuva futura."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import ALVOS, HORIZONTES, LIMIARES_MM, OPEN_METEO_CSV, RANDOM_STATE
from data import COLUNAS_OPEN_METEO, ler_csv, ler_open_meteo

# =============================================================================
# Governanca de colunas: identificadores, excluidas, alvos e conjuntos finais
# =============================================================================

COLUNAS_IDENTIFICADORAS = [
    "datetime",
    "datetime_utc",
    "datetime_local",
    "ano",
    "mes",
    "dia",
    "hora",
]

COLUNAS_IDENTIFICADORAS_BRUTAS = [
    "Unnamed: 0",
    "indice",
    "Data",
    "data",
    "Hora UTC",
    "hora_utc",
]

COLUNAS_ORIGINAIS_METEOROLOGICAS = [
    "pressao_mb",
    "temp_c",
    "temp_orvalho_c",
    "umid_pct",
]

COLUNAS_EXCLUIDAS_REDUNDANCIA_BRUTA = [
    "pressao_max_mb",
    "pressao_min_mb",
    "temp_max_c",
    "temp_min_c",
    "orvalho_max_c",
    "orvalho_min_c",
    "umid_max_pct",
    "umid_min_pct",
]

# Vento tem ~66% de valores ausentes em 2019-2020 na estacao A301.
COLUNAS_EXCLUIDAS_EXCESSO_NULOS = [
    "vento_dir_gr",
    "vento_rajada_ms",
    "vento_vel_ms",
]

COLUNAS_CHUVA_OBSERVADA = [
    "chuva_1h",
    "chuva_3h",
    "chuva_6h",
    "chuva_12h",
    "chuva_24h",
    "chuva_max_3h",
    "chuva_max_6h",
    "chuva_lag_1h",
    "chuva_lag_3h",
    "chuva_lag_6h",
    "chuva_lag_12h",
    "chuva_lag_24h",
]

COLUNAS_AGREGADAS_FUTURAS = [
    f"max_fut_{horizonte}h" for horizonte in HORIZONTES
] + [
    f"acum_fut_{horizonte}h" for horizonte in HORIZONTES
]

COLUNAS_ALVO = [
    f"alvo_{horizonte}h_{limiar}mm"
    for horizonte in HORIZONTES
    for limiar in LIMIARES_MM
] + [
    f"alvo_{horizonte}h_acum_{limiar}mm"
    for horizonte in HORIZONTES
    for limiar in (10, 20)
]

COLUNAS_PROIBIDAS_ENTRADA = (
    COLUNAS_IDENTIFICADORAS
    + COLUNAS_IDENTIFICADORAS_BRUTAS
    + ["precip_mm", "precip_mm_limpa"]
    + COLUNAS_EXCLUIDAS_REDUNDANCIA_BRUTA
    + COLUNAS_EXCLUIDAS_EXCESSO_NULOS
    + COLUNAS_AGREGADAS_FUTURAS
    + COLUNAS_ALVO
)

# Conjunto consolidado de 29 features derivadas candidatas (todas <= t).
FEATURES_CANDIDATAS = [
    # Ciclos diarios e anuais (4)
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
    # Sazonalidade regional de Recife (1)
    "estacao_chuvosa",
    # Derivadas termodinamicas e amplitudes intra-horarias (4)
    "orvalho_depressao",
    "amp_temp_1h",
    "amp_pressao_1h",
    "amp_umid_1h",
    # Tendencias / variacoes meteorologicas ate o instante t (8)
    "pressao_delta_1h",
    "pressao_delta_3h",
    "pressao_delta_6h",
    "temp_delta_1h",
    "temp_delta_3h",
    "orvalho_delta_3h",
    "umid_delta_1h",
    "umid_delta_3h",
    # Memoria e acumulados de chuva observada ate o instante t (12)
    *COLUNAS_CHUVA_OBSERVADA,
]

# Subconjunto usado na clusterizacao nao-supervisionada (K-Means).
# Alinhado a Notebooks/kmeans_recife.ipynb: exclui chuva observada (evita viciar
# grupos pelo fenomeno-alvo), exclui orvalho_depressao (|r| = 0,9957 com
# umid_pct) e exclui deltas/amplitudes para manter espaco euclidiano enxuto e
# equilibrado entre estado termodinamico e ciclos temporais.
FEATURES_CLUSTERING = [
    "pressao_mb",
    "temp_c",
    "temp_orvalho_c",
    "umid_pct",
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
]

# Mantido para compatibilidade com ml/modeling.py e ml/train.py.
FEATURES_KMEANS = list(FEATURES_CLUSTERING)

# Conjunto usado pelos classificadores supervisionados (33 features = 4 originais
# + 29 derivadas). Exclui identificadores temporais lineares (hora, mes, dia,
# ano), extremos brutos redundantes (max/min) e todas as colunas futuras/alvos.
FEATURES_CLASSIFICACAO = [
    # Chuva observada / acumulada ate t (12)
    *COLUNAS_CHUVA_OBSERVADA,
    # Pressao e derivadas (5)
    "pressao_mb",
    "pressao_delta_1h",
    "pressao_delta_3h",
    "pressao_delta_6h",
    "amp_pressao_1h",
    # Temperatura / ponto de orvalho e derivadas (7)
    "temp_c",
    "temp_orvalho_c",
    "orvalho_depressao",
    "temp_delta_1h",
    "temp_delta_3h",
    "orvalho_delta_3h",
    "amp_temp_1h",
    # Umidade e derivadas (4)
    "umid_pct",
    "umid_delta_1h",
    "umid_delta_3h",
    "amp_umid_1h",
    # Sazonalidade / ciclo diario (5)
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
    "estacao_chuvosa",
]

# Alias preservado para compatibilidade com ml/modeling.py e ml/train.py.
FEATURES = list(FEATURES_CLASSIFICACAO)

# Features atmosfericas do Open-Meteo. Nao substituem a chuva observada do
# INMET: entram como contexto de nuvens, radiacao, vento e camada limite.
FEATURES_OPEN_METEO = [
    "om_wind",
    "om_wind_dir_sin",
    "om_wind_dir_cos",
    "om_gust",
    "om_cloud",
    "om_cloud_low",
    "om_cloud_mid",
    "om_cloud_high",
    "om_rad",
    "om_rad_lag_1h",
    "om_blh",
    "om_cloud_delta_3h",
    "om_rad_delta_3h",
    "om_precip",
    "om_precip_3h",
]

FEATURES_KMEANS_OM = FEATURES_KMEANS + [
    "om_cloud",
    "om_rad",
    "om_wind",
    "om_blh",
]

MAPA_COLUNAS_CLUSTER_NOTEBOOK = {
    "pressao_mb": "PRESSAO ATMOSFERICA AO NIVEL DA ESTACAO, HORARIA (mB)",
    "temp_c": "TEMPERATURA DO AR - BULBO SECO, HORARIA (°C)",
    "temp_orvalho_c": "TEMPERATURA DO PONTO DE ORVALHO (°C)",
    "umid_pct": "UMIDADE RELATIVA DO AR, HORARIA (%)",
    "hora_sin": "hora_sin",
    "hora_cos": "hora_cos",
    "mes_sin": "mes_sin",
    "mes_cos": "mes_cos",
}


def _rolling(serie: pd.Series, janela: int, funcao: str = "sum") -> pd.Series:
    obj = serie.rolling(janela, min_periods=1)
    return getattr(obj, funcao)()


def construir_features(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features candidatas (<= t) e alvos futuros (> t) a base horaria."""
    df = df_bruto.copy()
    chuva = df["precip_mm_limpa"].astype(float)

    # Memoria e acumulados de chuva observada ate o instante t
    df["chuva_1h"] = chuva
    df["chuva_3h"] = _rolling(chuva, 3)
    df["chuva_6h"] = _rolling(chuva, 6)
    df["chuva_12h"] = _rolling(chuva, 12)
    df["chuva_24h"] = _rolling(chuva, 24)
    df["chuva_max_3h"] = _rolling(chuva, 3, "max")
    df["chuva_max_6h"] = _rolling(chuva, 6, "max")
    for lag in (1, 3, 6, 12, 24):
        df[f"chuva_lag_{lag}h"] = chuva.shift(lag).fillna(0.0)

    # Pressao e variacoes barometricas ate t
    pressao = df["pressao_mb"].astype(float)
    df["pressao_delta_1h"] = pressao.diff(1).fillna(0.0)
    df["pressao_delta_3h"] = pressao.diff(3).fillna(0.0)
    df["pressao_delta_6h"] = pressao.diff(6).fillna(0.0)

    # Temperatura, ponto de orvalho e derivadas termodinamicas ate t
    temp = df["temp_c"].astype(float)
    orvalho = df["temp_orvalho_c"].astype(float)
    df["orvalho_depressao"] = temp - orvalho
    df["temp_delta_1h"] = temp.diff(1).fillna(0.0)
    df["temp_delta_3h"] = temp.diff(3).fillna(0.0)
    df["orvalho_delta_3h"] = orvalho.diff(3).fillna(0.0)

    # Umidade e variacoes ate t
    umidade = df["umid_pct"].astype(float)
    df["umid_delta_1h"] = umidade.diff(1).fillna(0.0)
    df["umid_delta_3h"] = umidade.diff(3).fillna(0.0)

    # Amplitudes intra-horarias (consolidadas de clusterizacao_gmm.ipynb)
    if {"temp_max_c", "temp_min_c"}.issubset(df.columns):
        df["amp_temp_1h"] = (
            df["temp_max_c"].astype(float) - df["temp_min_c"].astype(float)
        ).fillna(0.0)
    else:
        df["amp_temp_1h"] = 0.0

    if {"pressao_max_mb", "pressao_min_mb"}.issubset(df.columns):
        df["amp_pressao_1h"] = (
            df["pressao_max_mb"].astype(float) - df["pressao_min_mb"].astype(float)
        ).fillna(0.0)
    else:
        df["amp_pressao_1h"] = 0.0

    if {"umid_max_pct", "umid_min_pct"}.issubset(df.columns):
        df["amp_umid_1h"] = (
            df["umid_max_pct"].astype(float) - df["umid_min_pct"].astype(float)
        ).fillna(0.0)
    else:
        df["amp_umid_1h"] = 0.0

    # Ciclos temporais (hora local Recife e mes) e sazonalidade regional
    hora = df["datetime_local"].dt.hour.astype(float)
    mes = df["datetime_local"].dt.month.astype(float)
    df["hora_sin"] = np.sin(2 * np.pi * hora / 24.0)
    df["hora_cos"] = np.cos(2 * np.pi * hora / 24.0)
    df["mes_sin"] = np.sin(2 * np.pi * mes / 12.0)
    df["mes_cos"] = np.cos(2 * np.pi * mes / 12.0)
    df["estacao_chuvosa"] = mes.isin([4, 5, 6, 7]).astype(int)

    # Features do Open-Meteo (quando o CSV/coletor fornecer as colunas).
    if "om_cloud" in df.columns:
        for coluna in COLUNAS_OPEN_METEO:
            df[coluna] = (
                pd.to_numeric(df[coluna], errors="coerce")
                .interpolate(method="linear", limit_direction="both")
                .fillna(0.0)
            )
        direcao = np.deg2rad(df["om_wind_dir"].astype(float))
        df["om_wind_dir_sin"] = np.sin(direcao)
        df["om_wind_dir_cos"] = np.cos(direcao)
        df["om_rad_lag_1h"] = df["om_rad"].shift(1).fillna(0.0)
        df["om_cloud_delta_3h"] = df["om_cloud"].diff(3).fillna(0.0)
        df["om_rad_delta_3h"] = df["om_rad"].diff(3).fillna(0.0)
        df["om_precip_3h"] = _rolling(df["om_precip"].astype(float), 3)

    # Alvos futuros: maior precipitacao horaria e acumulado entre t+1 e t+horizonte.
    for horizonte in HORIZONTES:
        futuros = pd.concat(
            [chuva.shift(-passo) for passo in range(1, horizonte + 1)],
            axis=1,
        )
        futuros.columns = [f"futuro_{passo}h" for passo in range(1, horizonte + 1)]
        # skipna=False garante NaN quando qualquer hora futura estiver ausente.
        max_futuro = futuros.max(axis=1, skipna=False)
        soma_futuro = futuros.sum(axis=1, skipna=False)
        df[f"max_fut_{horizonte}h"] = max_futuro
        df[f"acum_fut_{horizonte}h"] = soma_futuro
        for limiar in LIMIARES_MM:
            coluna = f"alvo_{horizonte}h_{limiar}mm"
            df[coluna] = np.where(
                max_futuro.notna(),
                (max_futuro >= limiar).astype(float),
                np.nan,
            )
        for limiar in (10, 20):
            coluna = f"alvo_{horizonte}h_acum_{limiar}mm"
            df[coluna] = np.where(
                soma_futuro.notna(),
                (soma_futuro >= limiar).astype(float),
                np.nan,
            )

    return df


def preparar_base(
    caminho=None,
    caminho_open_meteo=OPEN_METEO_CSV,
    usar_open_meteo: bool = True,
    filtrar_anos: bool = True,
) -> pd.DataFrame:
    """Le o CSV, opcionalmente junta Open-Meteo, cria features/alvos."""
    df = ler_csv(caminho) if caminho else ler_csv()
    if usar_open_meteo:
        try:
            om = ler_open_meteo(caminho_open_meteo)
            df = df.merge(om, on="datetime_utc", how="left")
            print(f"[ok] Open-Meteo: {len(om)} horas de features")
        except FileNotFoundError as exc:
            print(f"[aviso] {exc}")
            print("[aviso] seguindo apenas com features do INMET.")
    df = construir_features(df)
    if filtrar_anos:
        df = df[df["ano"].isin([2019, 2020])].copy()
    return df.reset_index(drop=True)


def features_modelo(df: pd.DataFrame) -> list[str]:
    """Lista de features disponiveis no dataframe (INMET + Open-Meteo)."""
    return FEATURES + [c for c in FEATURES_OPEN_METEO if c in df.columns]


def features_kmeans_disponiveis(df: pd.DataFrame) -> list[str]:
    if "om_cloud" in df.columns:
        return FEATURES_KMEANS_OM
    return FEATURES_KMEANS


def validar_sem_leakage(
    colunas_entrada: list[str],
    modo: str = "classificacao",
) -> None:
    """Garante programaticamente ausencia de alvos, futuros e identificadores na matriz X."""
    if modo not in {"classificacao", "clustering"}:
        raise ValueError(f"Modo invalido: {modo}. Use 'classificacao' ou 'clustering'.")

    assert len(colunas_entrada) == len(set(colunas_entrada)), (
        "Existem colunas duplicadas na matriz de entrada!"
    )

    proibidas = set(COLUNAS_PROIBIDAS_ENTRADA)
    encontradas = sorted(proibidas.intersection(colunas_entrada))
    assert not encontradas, (
        f"Risco de leakage/contaminacao: colunas proibidas em X ({modo}): {encontradas}"
    )

    prefixos_futuros = ("alvo_", "max_fut_", "acum_fut_", "futuro_")
    colunas_futuras = [
        c for c in colunas_entrada if c.startswith(prefixos_futuros)
    ]
    assert not colunas_futuras, (
        f"Leakage temporal: colunas futuras presentes em X: {colunas_futuras}"
    )

    if modo == "clustering":
        colunas_chuva = [
            c
            for c in colunas_entrada
            if c.startswith("chuva_") or c in {"precip_mm", "precip_mm_limpa", "om_precip", "om_precip_3h"}
        ]
        assert not colunas_chuva, (
            f"Vazamento de alvo no clustering: variaveis de chuva em X: {colunas_chuva}"
        )


def separar_matrizes_modelo(
    df: pd.DataFrame,
    alvo: str = "3h_acum_10mm",
    modo: str = "classificacao",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series | None]:
    """Separa identificadores temporais, matriz X validada e vetor alvo y."""
    colunas_id = [
        c for c in ["datetime_utc", "datetime_local", "ano"] if c in df.columns
    ]
    df_id = df[colunas_id].copy()

    if modo == "clustering":
        colunas_x = [c for c in FEATURES_CLUSTERING if c in df.columns]
        validar_sem_leakage(colunas_x, modo="clustering")
        X = df[colunas_x].astype(float).copy()
        return df_id, X, None

    if modo == "classificacao":
        colunas_x = [c for c in features_modelo(df) if c in df.columns]
        validar_sem_leakage(colunas_x, modo="classificacao")
        X = df[colunas_x].astype(float).copy()
        col_alvo = ALVOS[alvo]["coluna"] if alvo in ALVOS else alvo
        y = df[col_alvo].copy() if col_alvo in df.columns else None
        return df_id, X, y

    raise ValueError(f"Modo invalido: {modo}. Use 'classificacao' ou 'clustering'.")


def ajustar_clusters_regimes(
    df: pd.DataFrame,
    n_clusters: int = 3,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, Pipeline, pd.DataFrame]:
    """Ajusta StandardScaler + KMeans (sem chuva) e ordena clusters por temperatura."""
    _, X_cluster, _ = separar_matrizes_modelo(df, modo="clustering")
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "kmeans",
                KMeans(
                    n_clusters=n_clusters,
                    n_init=10,
                    random_state=random_state,
                ),
            ),
        ]
    )
    rotulos_raw = pipeline.fit_predict(X_cluster)

    out = df.copy()
    out["_cluster_raw"] = rotulos_raw
    ordem_temp = (
        out.groupby("_cluster_raw")["temp_c"]
        .mean()
        .sort_values(ascending=False)
        .index
    )
    mapa = {int(antigo): int(novo) for novo, antigo in enumerate(ordem_temp)}
    out["cluster"] = out["_cluster_raw"].map(mapa).astype(int)
    out = out.drop(columns=["_cluster_raw"])
    for idx in range(n_clusters):
        out[f"cluster_{idx}"] = (out["cluster"] == idx).astype(int)

    perfil = (
        out.groupby("cluster")[
            ["pressao_mb", "temp_c", "temp_orvalho_c", "umid_pct", "precip_mm_limpa"]
        ]
        .agg(
            horas=("temp_c", "size"),
            pressao_media=("pressao_mb", "mean"),
            temp_media=("temp_c", "mean"),
            orvalho_medio=("temp_orvalho_c", "mean"),
            umidade_media=("umid_pct", "mean"),
            pct_horas_chuva=("precip_mm_limpa", lambda s: (s > 0).mean() * 100.0),
            chuva_acumulada_mm=("precip_mm_limpa", "sum"),
        )
        .round(3)
    )
    return out, pipeline, perfil


def _justificativa_feature(nome: str) -> str:
    if nome in FEATURES_CLUSTERING:
        return (
            "Entra em clustering e classificacao: estado termodinamico/barometrico "
            "ou ciclo temporal continuo sem redundancia critica."
        )
    if nome in COLUNAS_CHUVA_OBSERVADA:
        return (
            "Exclusiva de classificacao: memoria de chuva observada ate t (sinal "
            "mais forte para persistencia em t+1..t+3); vetada no clustering para "
            "nao viciar grupos pelo fenomeno-alvo."
        )
    if nome == "orvalho_depressao":
        return (
            "Exclusiva de classificacao: r = -0,9957 com umid_pct (duplicaria "
            "peso na distancia euclidiana do K-Means), mas util em arvores."
        )
    if nome == "estacao_chuvosa":
        return (
            "Exclusiva de classificacao: indicador binario da quadra chuvosa "
            "(abr-jul), ja representado continuamente por mes_sin/mes_cos no clustering."
        )
    if nome.startswith(("pressao_delta_", "temp_delta_", "orvalho_delta_", "umid_delta_", "amp_")):
        return (
            "Exclusiva de classificacao: derivada/amplitude de variacao recente "
            "(<= t) que captura instabilidade pre-evento para o classificador, "
            "preservando espaco enxuto no clustering."
        )
    return "Incluida na classificacao."


def auditar_features(df: pd.DataFrame) -> dict:
    """Calcula correlacoes, pares redundantes e relevancia para clustering e classificacao."""
    df_proc, _, perfil_clusters = ajustar_clusters_regimes(df)

    originais_e_alvos = [
        *COLUNAS_ORIGINAIS_METEOROLOGICAS,
        "precip_mm_limpa",
        "acum_fut_3h",
        "max_fut_3h",
        "alvo_3h_5mm",
        "alvo_3h_10mm",
        "alvo_3h_acum_10mm",
    ]
    linhas_corr = [
        c for c in [*COLUNAS_ORIGINAIS_METEOROLOGICAS, *FEATURES_CANDIDATAS] if c in df_proc.columns
    ]
    matriz_corr_pearson = (
        df_proc[linhas_corr + [c for c in originais_e_alvos if c not in linhas_corr]]
        .corr(method="pearson")
        .loc[linhas_corr, originais_e_alvos]
    )
    matriz_corr_spearman = (
        df_proc[linhas_corr + [c for c in originais_e_alvos if c not in linhas_corr]]
        .corr(method="spearman")
        .loc[linhas_corr, originais_e_alvos]
    )

    # Analise de redundancia sobre originais + excluidas + candidatas + lineares hora/mes
    cols_redundancia = [
        c
        for c in [
            *COLUNAS_ORIGINAIS_METEOROLOGICAS,
            *COLUNAS_EXCLUIDAS_REDUNDANCIA_BRUTA,
            "hora",
            "mes",
            *FEATURES_CANDIDATAS,
        ]
        if c in df_proc.columns
    ]
    corr_completa = df_proc[cols_redundancia].corr(method="pearson")
    pares_redundantes = []
    for i in range(len(cols_redundancia)):
        for j in range(i + 1, len(cols_redundancia)):
            a = cols_redundancia[i]
            b = cols_redundancia[j]
            r = float(corr_completa.iloc[i, j])
            if abs(r) >= 0.70:
                pares_redundantes.append(
                    {
                        "variavel_a": a,
                        "variavel_b": b,
                        "pearson_r": round(r, 4),
                        "abs_r": round(abs(r), 4),
                    }
                )
    df_redundancia = pd.DataFrame(pares_redundantes).sort_values(
        "abs_r", ascending=False
    ).reset_index(drop=True)

    # Importancia Random Forest (apenas 2019 -> sem leakage do holdout 2020)
    _, X_cls, _ = separar_matrizes_modelo(df_proc, alvo="3h_acum_10mm", modo="classificacao")
    mask_2019 = (df_proc["ano"] == 2019) & df_proc["alvo_3h_acum_10mm"].notna() & df_proc["alvo_3h_10mm"].notna()
    rf_acum = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_acum.fit(X_cls.loc[mask_2019], df_proc.loc[mask_2019, "alvo_3h_acum_10mm"].astype(int))

    rf_int = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=5,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    rf_int.fit(X_cls.loc[mask_2019], df_proc.loc[mask_2019, "alvo_3h_10mm"].astype(int))

    f_vals, p_vals = f_classif(X_cls, df_proc["cluster"])

    registros_rel = []
    for idx, feat in enumerate(X_cls.columns):
        registros_rel.append(
            {
                "feature": feat,
                "tipo": "Original" if feat in COLUNAS_ORIGINAIS_METEOROLOGICAS else "Candidata derivada",
                "corr_precip_atual": round(float(matriz_corr_pearson.loc[feat, "precip_mm_limpa"]), 4),
                "corr_acum_fut_3h": round(float(matriz_corr_pearson.loc[feat, "acum_fut_3h"]), 4),
                "corr_alvo_3h_acum_10mm": round(float(matriz_corr_pearson.loc[feat, "alvo_3h_acum_10mm"]), 4),
                "spearman_alvo_3h_acum_10mm": round(float(matriz_corr_spearman.loc[feat, "alvo_3h_acum_10mm"]), 4),
                "corr_alvo_3h_10mm": round(float(matriz_corr_pearson.loc[feat, "alvo_3h_10mm"]), 4),
                "rf_imp_2019_acum_10mm": round(float(rf_acum.feature_importances_[idx]), 4),
                "rf_imp_2019_3h_10mm": round(float(rf_int.feature_importances_[idx]), 4),
                "anova_f_clusters": round(float(f_vals[idx]), 2),
                "anova_p_clusters": float(p_vals[idx]),
                "entra_clustering": feat in FEATURES_CLUSTERING,
                "entra_classificacao": feat in FEATURES_CLASSIFICACAO,
                "justificativa": _justificativa_feature(feat),
            }
        )

    df_relevancia = pd.DataFrame(registros_rel).sort_values(
        "rf_imp_2019_acum_10mm", ascending=False
    ).reset_index(drop=True)

    return {
        "df_com_clusters": df_proc,
        "perfil_clusters": perfil_clusters,
        "correlacoes_pearson": matriz_corr_pearson.round(4),
        "correlacoes_spearman": matriz_corr_spearman.round(4),
        "redundancia_pares": df_redundancia,
        "relevancia_features": df_relevancia,
    }


def resumo_alvos(df: pd.DataFrame) -> dict:
    """Contagens de positivos por alvo, para documentacao e logs."""
    resumo = {}
    for nome, info in ALVOS.items():
        coluna = info["coluna"]
        valido = df[coluna].notna()
        resumo[nome] = {
            "horizonte_h": info["horizonte_h"],
            "limiar_mm": info["limiar_mm"],
            "linhas_validas": int(valido.sum()),
            "positivos": int((df.loc[valido, coluna] == 1).sum()),
            "prevalencia": float(df.loc[valido, coluna].mean()),
            "positivos_2019": int(
                (df.loc[valido & (df["ano"] == 2019), coluna] == 1).sum()
            ),
            "positivos_2020": int(
                (df.loc[valido & (df["ano"] == 2020), coluna] == 1).sum()
            ),
        }
    return resumo
