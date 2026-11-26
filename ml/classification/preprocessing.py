"""Preparacao dos dados de classificacao, comum aos tres grupos do DIA 5.

Reaproveita as decisoes do DIA 2 (`ml/experimento_target.py`):
- target `classe_chuva_3h`: chuva acumulada em t+1..t+3, 4 classes
  (0 Sem chuva < 0,2 mm | 1 Fraca < 2,5 | 2 Moderada < 10 | 3 Forte >= 10);
- 33 features de `FEATURES_CLASSIFICACAO`, todas observadas ate t;
- divisao out-of-time: treino 2018-2019, validacao 2020, teste 2021.

Uso nos scripts de treino:

    from preprocessing import carregar_dados
    dados = carregar_dados()
    dados.X["treino"], dados.y["treino"]  # idem "validacao" e "teste"

Escalonamento e balanceamento NAO sao feitos aqui: entram no Pipeline de cada
modelo e sao ajustados apenas no treino.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

ML_DIR = Path(__file__).resolve().parents[1]
if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))

import pandas as pd  # noqa: E402

from config import RAW_CSV  # noqa: E402
from experimento_target import (  # noqa: E402
    CLASSES,
    LIMITES_MM,
    PARTICOES,
    VARIAVEL_BASE,
    _particao,
    _verificar_janela_futura,
    classificar,
)
from features import FEATURES_CLASSIFICACAO, preparar_base, validar_sem_leakage  # noqa: E402

TARGET = "classe_chuva_3h"
NOMES_CLASSES = [CLASSES[i] for i in sorted(CLASSES)]


@dataclass
class DadosClassificacao:
    X: dict[str, pd.DataFrame]
    y: dict[str, pd.Series]
    datas: dict[str, pd.Series]
    features: list[str]

    def resumo(self) -> pd.DataFrame:
        linhas = []
        for nome in PARTICOES:
            contagem = self.y[nome].value_counts().reindex(range(len(CLASSES)), fill_value=0)
            linhas.append(
                {
                    "particao": nome,
                    "anos": ", ".join(map(str, PARTICOES[nome])),
                    "inicio": self.datas[nome].min(),
                    "fim": self.datas[nome].max(),
                    "registros": len(self.y[nome]),
                    **{CLASSES[c]: int(n) for c, n in contagem.items()},
                }
            )
        return pd.DataFrame(linhas)


def carregar_dados() -> DadosClassificacao:
    """Monta X/y por particao temporal, com as verificacoes anti-leakage do DIA 2."""
    base = preparar_base(caminho=RAW_CSV, usar_open_meteo=False, filtrar_anos=False)

    # 1) O alvo usa apenas t+1..t+3 e nenhuma janela cruza a fronteira entre anos/particoes.
    _verificar_janela_futura(base)
    # 2) Nenhuma coluna futura, alvo, identificador ou rotulo de cluster entra em X.
    validar_sem_leakage(FEATURES_CLASSIFICACAO, modo="classificacao")

    base[TARGET] = classificar(base[VARIAVEL_BASE])
    base["particao"] = _particao(base["ano"])
    base = base.dropna(subset=[TARGET]).copy()
    base[TARGET] = base[TARGET].astype(int)
    assert base["particao"].notna().all(), "Ano fora das particoes definidas"
    assert base[FEATURES_CLASSIFICACAO].notna().all().all(), "Features com valores ausentes"

    X, y, datas = {}, {}, {}
    for nome in PARTICOES:
        parte = base[base["particao"] == nome]
        X[nome] = parte[FEATURES_CLASSIFICACAO].astype(float)
        y[nome] = parte[TARGET].rename(TARGET)
        datas[nome] = parte["datetime_local"]

    # 3) Ordem temporal estrita: treino < validacao < teste, sem sobreposicao.
    nomes = list(PARTICOES)
    for anterior, seguinte in zip(nomes, nomes[1:]):
        assert datas[anterior].max() < datas[seguinte].min(), f"{anterior} sobrepoe {seguinte}"
    # 4) Todas as classes presentes em todas as particoes.
    for nome in nomes:
        assert set(y[nome].unique()) == set(CLASSES), f"Classe ausente em {nome}"

    return DadosClassificacao(X=X, y=y, datas=datas, features=list(FEATURES_CLASSIFICACAO))


if __name__ == "__main__":
    dados = carregar_dados()
    print(f"Target: {TARGET} | limites (mm em 3h): {LIMITES_MM} | classes: {NOMES_CLASSES}")
    print(f"Features: {len(dados.features)} (todas observadas ate t; validar_sem_leakage OK)")
    print(dados.resumo().to_string(index=False))
