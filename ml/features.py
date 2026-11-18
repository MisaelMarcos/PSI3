"""Engenharia de features e construcao dos alvos de chuva futura."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import ALVOS, OPEN_METEO_CSV, LIMIARES_MM, HORIZONTES
from data import COLUNAS_OPEN_METEO, ler_csv, ler_open_meteo

# Features usadas pelos classificadores. Todas usam apenas informacao ate o
# instante t (chuva acumulada, lags, variacao de pressao etc.); os alvos olham
# apenas para t+1 ... t+horizonte.
FEATURES = [
    # Chuva observada / acumulada
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
    # Pressao
    "pressao_mb",
    "pressao_delta_1h",
    "pressao_delta_3h",
    "pressao_delta_6h",
    # Temperatura / umidade
    "temp_c",
    "temp_orvalho_c",
    "orvalho_depressao",
    "temp_delta_3h",
    "umid_pct",
    "umid_delta_1h",
    "umid_delta_3h",
    # Sazonalidade / ciclo diario
    "hora",
    "mes",
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
    "estacao_chuvosa",
]

# Subconjunto usado pelo KMeans. Evita dezenas de lags altamente
# correlacionados e prioriza o estado meteorologico do momento.
FEATURES_KMEANS = [
    "chuva_1h",
    "chuva_3h",
    "chuva_6h",
    "chuva_12h",
    "pressao_mb",
    "pressao_delta_3h",
    "temp_c",
    "orvalho_depressao",
    "umid_pct",
    "hora_sin",
    "hora_cos",
    "mes_sin",
    "mes_cos",
]

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


def _rolling(serie: pd.Series, janela: int, funcao: str = "sum") -> pd.Series:
    obj = serie.rolling(janela, min_periods=1)
    return getattr(obj, funcao)()


def construir_features(df_bruto: pd.DataFrame) -> pd.DataFrame:
    """Adiciona features e alvos a uma base horaria ja tratada."""
    df = df_bruto.copy()
    chuva = df["precip_mm_limpa"].astype(float)

    df["chuva_1h"] = chuva
    df["chuva_3h"] = _rolling(chuva, 3)
    df["chuva_6h"] = _rolling(chuva, 6)
    df["chuva_12h"] = _rolling(chuva, 12)
    df["chuva_24h"] = _rolling(chuva, 24)
    df["chuva_max_3h"] = _rolling(chuva, 3, "max")
    df["chuva_max_6h"] = _rolling(chuva, 6, "max")
    for lag in (1, 3, 6, 12, 24):
        df[f"chuva_lag_{lag}h"] = chuva.shift(lag).fillna(0.0)

    pressao = df["pressao_mb"].astype(float)
    df["pressao_delta_1h"] = pressao.diff(1).fillna(0.0)
    df["pressao_delta_3h"] = pressao.diff(3).fillna(0.0)
    df["pressao_delta_6h"] = pressao.diff(6).fillna(0.0)

    temp = df["temp_c"].astype(float)
    orvalho = df["temp_orvalho_c"].astype(float)
    df["orvalho_depressao"] = temp - orvalho
    df["temp_delta_3h"] = temp.diff(3).fillna(0.0)

    umidade = df["umid_pct"].astype(float)
    df["umid_delta_1h"] = umidade.diff(1).fillna(0.0)
    df["umid_delta_3h"] = umidade.diff(3).fillna(0.0)

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

    # Alvos: maior precipitacao horaria e acumulado entre t+1 e t+horizonte.
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
    df = df[df["ano"].isin([2019, 2020])].copy()
    return df.reset_index(drop=True)


def features_modelo(df: pd.DataFrame) -> list[str]:
    """Lista de features disponiveis no dataframe (INMET + Open-Meteo)."""
    return FEATURES + [c for c in FEATURES_OPEN_METEO if c in df.columns]


def features_kmeans_disponiveis(df: pd.DataFrame) -> list[str]:
    if "om_cloud" in df.columns:
        return FEATURES_KMEANS_OM
    return FEATURES_KMEANS


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
