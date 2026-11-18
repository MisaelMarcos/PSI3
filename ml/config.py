"""Configuracoes do modulo de previsao de chuva.

Todos os caminhos sao resolvidos a partir deste arquivo para que os scripts
funcionem tanto a partir da pasta do modulo quanto da raiz do repositorio.
"""

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parents[2]

DATA_DIR = REPO_ROOT / "dados" / "clima"
RAW_CSV = DATA_DIR / "recife_inmet_2019_2020.csv"
OPEN_METEO_CSV = DATA_DIR / "open_meteo_recife_2019_2020.csv"
LIVE_CSV = DATA_DIR / "observacoes_inmet_a301.csv"

PIPELINE_DATA_DIR = REPO_ROOT / "src" / "backend" / "pipeline" / "data"
SCORE_BAIRRO_CSV = PIPELINE_DATA_DIR / "score_bairro.csv"
BAIRROS_GEO_CSV = PIPELINE_DATA_DIR / "bairros_geo.csv"

MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
METADATA_PATH = MODELS_DIR / "metadata.json"

MODEL_VERSION = "1.0.0"
RANDOM_STATE = 42
TIMEZONE = "America/Recife"

ESTACAO = "A301"
MUNICIPIO_CODIGO = "2611606"
INMET_ESTACAO_URL = f"https://apiprevmet3.inmet.gov.br/estacao/proxima/{MUNICIPIO_CODIGO}"
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
LATITUDE = -8.0476
LONGITUDE = -34.877

HORIZONTES = (2, 3)
LIMIARES_MM = (5, 10)

# Alvos de intensidade: maximo horario nas proximas h horas.
# Alvos operacionais de volume: acumulado nas proximas h horas.
ALVOS: dict[str, dict] = {}
for horizonte in HORIZONTES:
    for limiar in LIMIARES_MM:
        nome = f"{horizonte}h_{limiar}mm"
        ALVOS[nome] = {
            "horizonte_h": horizonte,
            "limiar_mm": limiar,
            "tipo": "max",
            "coluna": f"alvo_{horizonte}h_{limiar}mm",
            "descricao": (
                f"chuva maxima horaria nas proximas {horizonte} horas "
                f"maior ou igual a {limiar} mm/h"
            ),
        }

for horizonte, limiar in [(2, 10), (3, 10), (3, 20)]:
    nome = f"{horizonte}h_acum_{limiar}mm"
    ALVOS[nome] = {
        "horizonte_h": horizonte,
        "limiar_mm": limiar,
        "tipo": "acum",
        "coluna": f"alvo_{horizonte}h_acum_{limiar}mm",
        "descricao": (
            f"chuva acumulada nas proximas {horizonte} horas "
            f"maior ou igual a {limiar} mm"
        ),
    }

# Alvo principal do alerta operacional (volume de chuva em 3h).
ALVO_PRINCIPAL = "3h_acum_10mm"
# Alvo de intensidade usado como confirmacao de chuva forte.
ALVO_INTENSIDADE = "3h_10mm"
