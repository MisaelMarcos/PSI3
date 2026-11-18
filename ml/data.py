"""Leitura e limpeza do CSV horario do INMET (estacao A301 - Recife)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from config import LIVE_CSV, OPEN_METEO_CSV, RAW_CSV, TIMEZONE

# O arquivo original tem 20 colunas, mas os nomes vem com acentos quebrados
# pelo encoding latin1. A renomeacao por posicao e mais segura.
COLUNAS_ORIGINAIS = [
    "indice",
    "data",
    "hora_utc",
    "precip_mm",
    "pressao_mb",
    "pressao_max_mb",
    "pressao_min_mb",
    "temp_c",
    "temp_orvalho_c",
    "temp_max_c",
    "temp_min_c",
    "orvalho_max_c",
    "orvalho_min_c",
    "umid_max_pct",
    "umid_min_pct",
    "umid_pct",
    "vento_dir_gr",
    "vento_rajada_ms",
    "vento_vel_ms",
    "datetime",
]

COLUNAS_NUMERICAS = [
    "precip_mm",
    "pressao_mb",
    "pressao_max_mb",
    "pressao_min_mb",
    "temp_c",
    "temp_orvalho_c",
    "temp_max_c",
    "temp_min_c",
    "orvalho_max_c",
    "orvalho_min_c",
    "umid_max_pct",
    "umid_min_pct",
    "umid_pct",
    "vento_dir_gr",
    "vento_rajada_ms",
    "vento_vel_ms",
]

# Vento tem cerca de 66% de nulos no periodo e por isso nao entra nas features.
COLUNAS_INTERPOLADAS = [
    "pressao_mb",
    "temp_c",
    "temp_orvalho_c",
    "umid_pct",
    "umid_max_pct",
    "umid_min_pct",
]

COLUNAS_OPEN_METEO = [
    "om_temp",
    "om_umid",
    "om_dew",
    "om_pressao",
    "om_precip",
    "om_wind",
    "om_wind_dir",
    "om_gust",
    "om_cloud",
    "om_cloud_low",
    "om_cloud_mid",
    "om_cloud_high",
    "om_rad",
    "om_blh",
]


def _converter_numero(serie: pd.Series) -> pd.Series:
    """Converte numeros no formato brasileiro (virgula decimal)."""
    texto = serie.astype("string").str.strip()
    texto = texto.replace(
        {
            "": pd.NA,
            "-": pd.NA,
            "--": pd.NA,
            "NaN": pd.NA,
            "nan": pd.NA,
            "null": pd.NA,
            "None": pd.NA,
            "-9999": pd.NA,
            "-9999,0": pd.NA,
            "-9999.0": pd.NA,
        }
    )
    # Quando ha virgula, ela e o separador decimal. Remove pontos de milhar.
    tem_virgula = texto.str.contains(",", na=False)
    texto = texto.mask(
        tem_virgula,
        texto.str.replace(".", "", regex=False).str.replace(",", ".", regex=False),
    )
    return pd.to_numeric(texto, errors="coerce")


def ler_csv(caminho: str | Path = RAW_CSV) -> pd.DataFrame:
    """Le o CSV do INMET e devolve uma base horaria tratada.

    - O CSV original esta em latin1.
    - A coluna ``datetime`` ja vem em UTC.
    - Valores fisicos sao interpolados apenas para features; a precipitacao
      original e preservada em ``precip_mm`` e a versao usada nos calculos
      fica em ``precip_mm_limpa`` (nulos viram 0).
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"CSV nao encontrado: {caminho}. "
            "Coloque o arquivo em dados/clima/recife_inmet_2019_2020.csv."
        )

    df = pd.read_csv(caminho, encoding="latin1", dtype=str)
    if len(df.columns) != len(COLUNAS_ORIGINAIS):
        raise ValueError(
            f"Esperava {len(COLUNAS_ORIGINAIS)} colunas no CSV, "
            f"recebi {len(df.columns)}."
        )
    df.columns = COLUNAS_ORIGINAIS

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")

    for coluna in COLUNAS_NUMERICAS:
        df[coluna] = _converter_numero(df[coluna])

    # Precipitacao: preserva o valor bruto para diagnosticos e usa 0 quando a
    # leitura esta ausente. O alvo tambem usa a serie limpa; linhas sem futuro
    # valido sao descartadas na construcao dos alvos.
    df["precip_mm_limpa"] = df["precip_mm"].fillna(0.0)

    # Sensores de pressao/temperatura/umidade tem poucos nulos (13 no total).
    for coluna in COLUNAS_INTERPOLADAS:
        mediana = df[coluna].median()
        df[coluna] = (
            df[coluna]
            .interpolate(method="linear", limit_direction="both")
            .fillna(mediana)
            .fillna(0.0)
        )

    # Vento nao e usado no modelo, mas mantemos as colunas para auditoria.
    for coluna in ["vento_dir_gr", "vento_rajada_ms", "vento_vel_ms"]:
        df[coluna] = df[coluna].fillna(df[coluna].median()).fillna(0.0)

    df["datetime_utc"] = df["datetime"]
    df["datetime_local"] = df["datetime"].dt.tz_convert(TIMEZONE)
    df["ano"] = df["datetime_local"].dt.year
    df["mes"] = df["datetime_local"].dt.month
    df["dia"] = df["datetime_local"].dt.day
    df["hora"] = df["datetime_local"].dt.hour

    # Remove colunas auxiliares que nao agregam ao modelo.
    df = df.drop(columns=["indice", "data", "hora_utc"], errors="ignore")
    return df.reset_index(drop=True)


def ler_open_meteo(caminho: str | Path = OPEN_METEO_CSV) -> pd.DataFrame:
    """Le as features historicas compactadas do Open-Meteo."""
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"CSV do Open-Meteo nao encontrado: {caminho}. "
            "Rode baixar_open_meteo.py ou treine apenas com o INMET."
        )
    df = pd.read_csv(caminho)
    df["datetime_utc"] = pd.to_datetime(
        df["datetime_utc"], errors="coerce", utc=True
    )
    df = df.dropna(subset=["datetime_utc"]).sort_values("datetime_utc")
    df = df.drop_duplicates("datetime_utc", keep="last")
    for coluna in COLUNAS_OPEN_METEO:
        if coluna not in df.columns:
            df[coluna] = 0.0
        df[coluna] = pd.to_numeric(df[coluna], errors="coerce")
        mediana = df[coluna].median()
        df[coluna] = (
            df[coluna]
            .interpolate(method="linear", limit_direction="both")
            .fillna(mediana)
            .fillna(0.0)
        )
    return df[["datetime_utc", *COLUNAS_OPEN_METEO]].reset_index(drop=True)


def ler_observacoes_csv(caminho: str | Path = LIVE_CSV) -> pd.DataFrame:
    """Le o CSV incremental gerado pelo coletor ao vivo.

    O coletor grava apenas as colunas usadas na inferencia. O tratamento e o
    mesmo do CSV historico para que ``features.construir_features`` funcione
    sem caminho especial.
    """
    caminho = Path(caminho)
    if not caminho.exists():
        raise FileNotFoundError(
            f"Observacoes nao encontradas: {caminho}. "
            "Rode collect_inmet.py antes de prever."
        )

    df = pd.read_csv(caminho, dtype=str)
    if "datetime" not in df.columns:
        raise ValueError("O CSV ao vivo precisa ter a coluna 'datetime'.")

    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True)
    df = df.dropna(subset=["datetime"]).copy()
    df = df.sort_values("datetime").drop_duplicates("datetime", keep="last")

    for coluna in [
        "precip_mm",
        "pressao_mb",
        "temp_c",
        "temp_orvalho_c",
        "umid_pct",
        "vento_dir_gr",
        "vento_rajada_ms",
        "vento_vel_ms",
        *COLUNAS_OPEN_METEO,
    ]:
        if coluna in df.columns:
            df[coluna] = _converter_numero(df[coluna])

    if "precip_mm" not in df.columns:
        df["precip_mm"] = 0.0
    df["precip_mm_limpa"] = df["precip_mm"].fillna(0.0)

    for coluna in COLUNAS_INTERPOLADAS:
        if coluna not in df.columns:
            df[coluna] = 0.0
        mediana = df[coluna].median()
        df[coluna] = (
            df[coluna]
            .interpolate(method="linear", limit_direction="both")
            .fillna(mediana)
            .fillna(0.0)
        )

    for coluna in ["vento_dir_gr", "vento_rajada_ms", "vento_vel_ms"]:
        if coluna not in df.columns:
            df[coluna] = 0.0
        df[coluna] = df[coluna].fillna(0.0)

    # Se o coletor nao conseguiu buscar o Open-Meteo, mantem as colunas com 0.
    # O modelo continua funcionando, mas com menor poder preditivo.
    for coluna in COLUNAS_OPEN_METEO:
        if coluna not in df.columns:
            df[coluna] = 0.0
        df[coluna] = df[coluna].fillna(df[coluna].median()).fillna(0.0)

    df["datetime_utc"] = df["datetime"]
    df["datetime_local"] = df["datetime"].dt.tz_convert(TIMEZONE)
    df["ano"] = df["datetime_local"].dt.year
    df["mes"] = df["datetime_local"].dt.month
    df["dia"] = df["datetime_local"].dt.day
    df["hora"] = df["datetime_local"].dt.hour
    return df.reset_index(drop=True)


def resumo_base(df: pd.DataFrame) -> dict:
    """Resumo usado no relatorio de treino."""
    return {
        "linhas": int(len(df)),
        "inicio_utc": str(df["datetime_utc"].min()),
        "fim_utc": str(df["datetime_utc"].max()),
        "inicio_local": str(df["datetime_local"].min()),
        "fim_local": str(df["datetime_local"].max()),
        "precipitacao_max_mm": float(np.nanmax(df["precip_mm_limpa"])),
        "horas_chuva_5mm": int((df["precip_mm_limpa"] >= 5).sum()),
        "horas_chuva_10mm": int((df["precip_mm_limpa"] >= 10).sum()),
        "dias_chuva_5mm": int(
            df.loc[df["precip_mm_limpa"] >= 5, "datetime_local"]
            .dt.date.nunique()
        ),
        "dias_chuva_10mm": int(
            df.loc[df["precip_mm_limpa"] >= 10, "datetime_local"]
            .dt.date.nunique()
        ),
    }
