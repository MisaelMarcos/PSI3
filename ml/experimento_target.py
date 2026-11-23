"""DIA 2 — Definicao do target, das faixas de intensidade e da estrategia temporal.

Gera `reports/target_classificacao/` com a justificativa do alvo de classificacao,
a distribuicao por classe e por particao temporal, a analise de desbalanceamento,
a lista de features permitidas e os riscos de vazamento verificados no codigo.

Nao altera nenhum dataset existente: le a base por `features.preparar_base`.
"""

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

from config import ALVO_PRINCIPAL, RAW_CSV, RANDOM_STATE
from experimento_kmeans import _integer, _number
from features import (
    COLUNAS_AGREGADAS_FUTURAS,
    COLUNAS_ALVO,
    COLUNAS_CHUVA_OBSERVADA,
    COLUNAS_PROIBIDAS_ENTRADA,
    FEATURES_CLASSIFICACAO,
    FEATURES_OPEN_METEO,
    preparar_base,
    validar_sem_leakage,
)

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "reports" / "target_classificacao"
FIGURES_DIR = OUTPUT_DIR / "figuras"

# Variavel continua que origina o alvo: chuva acumulada em t+1, t+2 e t+3.
VARIAVEL_BASE = "acum_fut_3h"
HORIZONTE_H = 3

# Faixas escolhidas (limite inferior inclusivo, em mm acumulados em 3 h).
LIMITES_MM = (0.2, 2.5, 10.0)
CLASSES = {0: "Sem chuva", 1: "Fraca", 2: "Moderada", 3: "Forte"}

# Esquemas candidatos comparados antes da escolha.
ESQUEMAS = {
    "A — binario (>= 10 mm)": (10.0,),
    "B — 3 classes (0,2 / 10)": (0.2, 10.0),
    "C — 4 classes (0,2 / 2,5 / 10)": LIMITES_MM,
    "D — 5 classes (0,2 / 2,5 / 10 / 20)": (0.2, 2.5, 10.0, 20.0),
}
ESQUEMA_ESCOLHIDO = "C — 4 classes (0,2 / 2,5 / 10)"
MIN_EXEMPLOS_TESTE = 30  # minimo para a menor classe ter metricas estaveis no teste

# Particao out-of-time por ano civil local (America/Recife).
PARTICOES = {"treino": (2018, 2019), "validacao": (2020,), "teste": (2021,)}


def classificar(acumulado: pd.Series, limites: tuple[float, ...] = LIMITES_MM) -> pd.Series:
    """Converte chuva acumulada futura em classe ordinal (NaN quando a janela e invalida)."""
    bins = [-np.inf, *limites, np.inf]
    classes = pd.cut(acumulado, bins=bins, right=False, labels=False)
    return classes.astype("Float64")


def _particao(ano: pd.Series) -> pd.Series:
    mapa = {a: nome for nome, anos in PARTICOES.items() for a in anos}
    return ano.map(mapa)


def _save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(FIGURES_DIR / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIGURES_DIR / f"{name}.pdf", bbox_inches="tight")
    plt.close(fig)


def _verificar_janela_futura(base: pd.DataFrame) -> int:
    """Recalcula o alvo a partir de precip_mm e confirma que t nao entra na janela."""
    serie = pd.Series(
        base["precip_mm"].to_numpy(), index=pd.DatetimeIndex(base["datetime_utc"])
    ).asfreq("h")
    soma = sum(serie.shift(-p) for p in range(1, HORIZONTE_H + 1))
    soma = soma.reindex(pd.DatetimeIndex(base["datetime_utc"])).to_numpy()
    validos = base[VARIAVEL_BASE].notna().to_numpy()
    assert np.allclose(base.loc[validos, VARIAVEL_BASE], soma[validos]), (
        "acum_fut_3h nao corresponde a soma de t+1..t+3"
    )
    # Janelas que cruzam o ano (e portanto a fronteira entre particoes) ficam invalidas.
    fim_janela = (base["datetime_local"] + pd.Timedelta(hours=HORIZONTE_H)).dt.year
    assert (fim_janela[validos] == base.loc[validos, "ano"]).all(), (
        "Existe janela de alvo cruzando a fronteira entre anos/particoes"
    )
    return int(validos.sum())


def _tabela_esquemas(df: pd.DataFrame) -> pd.DataFrame:
    linhas = []
    for nome, limites in ESQUEMAS.items():
        classes = classificar(df[VARIAVEL_BASE], limites)
        contagem = classes.value_counts()
        teste = classes[df["particao"] == "teste"].value_counts()
        linhas.append(
            {
                "esquema": nome,
                "n_classes": len(limites) + 1,
                "limites_mm": " / ".join(_number(l, 1) for l in limites),
                "menor_classe_registros": int(contagem.min()),
                "menor_classe_pct": contagem.min() / contagem.sum() * 100,
                "menor_classe_no_teste": int(teste.reindex(range(len(limites) + 1), fill_value=0).min()),
                "razao_desbalanceamento": contagem.max() / contagem.min(),
            }
        )
    return pd.DataFrame(linhas)


def _distribuicao(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    geral = df["classe"].value_counts().sort_index().rename("registros").reset_index()
    geral["classe_nome"] = geral["classe"].map(CLASSES)
    geral["participacao_pct"] = geral["registros"] / geral["registros"].sum() * 100
    faixas = [f"< {_number(LIMITES_MM[0], 1)} mm"] + [
        f"{_number(a, 1)} a < {_number(b, 1)} mm" for a, b in zip(LIMITES_MM, LIMITES_MM[1:])
    ] + [f">= {_number(LIMITES_MM[-1], 1)} mm"]
    geral["faixa_acum_3h"] = geral["classe"].map(dict(enumerate(faixas)))
    geral["razao_vs_majoritaria"] = geral["registros"].max() / geral["registros"]
    stats = df.groupby("classe")[VARIAVEL_BASE].agg(["mean", "median", "std", "max"])
    geral = geral.join(stats, on="classe").rename(
        columns={"mean": "media_mm", "median": "mediana_mm", "std": "desvio_mm", "max": "maximo_mm"}
    )

    por_split = (
        df.groupby(["particao", "classe"]).size().unstack(fill_value=0).reindex(PARTICOES)
    )
    por_split.columns = [CLASSES[int(c)] for c in por_split.columns]
    por_split["total"] = por_split.sum(axis=1)
    por_split["periodo"] = [
        f"{df.loc[df['particao'] == p, 'datetime_local'].min():%Y-%m-%d} a "
        f"{df.loc[df['particao'] == p, 'datetime_local'].max():%Y-%m-%d}"
        for p in por_split.index
    ]
    return geral, por_split.reset_index()


def _tabela_features(base: pd.DataFrame, treino: pd.DataFrame) -> pd.DataFrame:
    """Lista as features permitidas e a correlacao de Spearman com a classe no treino."""
    corr = treino[FEATURES_CLASSIFICACAO].corrwith(treino["classe"].astype(float), method="spearman")
    grupos = []
    for f in FEATURES_CLASSIFICACAO:
        if f in COLUNAS_CHUVA_OBSERVADA:
            grupo = "Chuva observada ate t"
        elif f.startswith(("hora_", "mes_")) or f == "estacao_chuvosa":
            grupo = "Ciclo temporal / sazonalidade"
        elif "delta" in f or f.startswith("amp_"):
            grupo = "Tendencia / amplitude ate t"
        else:
            grupo = "Estado atmosferico em t"
        grupos.append(grupo)
    return pd.DataFrame(
        {
            "feature": FEATURES_CLASSIFICACAO,
            "grupo": grupos,
            "spearman_classe_treino": corr.reindex(FEATURES_CLASSIFICACAO).to_numpy(),
            "nulos_pct": base[FEATURES_CLASSIFICACAO].isna().mean().to_numpy() * 100,
        }
    )


def _figuras(df: pd.DataFrame, geral: pd.DataFrame, por_split: pd.DataFrame) -> None:
    cores = sns.color_palette("Blues", len(CLASSES) + 1)[1:]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax = axes[0]
    barras = ax.bar(geral["classe_nome"], geral["registros"], color=cores)
    ax.set_yscale("log")
    ax.set(
        title="Registros por classe (escala log)",
        xlabel="Classe de chuva acumulada nas próximas 3 h",
        ylabel="Registros (horas)",
    )
    for barra, pct in zip(barras, geral["participacao_pct"]):
        ax.annotate(f"{_number(pct)}%", (barra.get_x() + barra.get_width() / 2, barra.get_height()),
                    ha="center", va="bottom", fontsize=9)
    ax = axes[1]
    chuvosos = df.loc[df[VARIAVEL_BASE] > 0, VARIAVEL_BASE]
    ax.hist(chuvosos, bins=np.logspace(np.log10(0.2), np.log10(chuvosos.max()), 40), color="#6baed6")
    ax.set_xscale("log")
    for limite in LIMITES_MM[1:]:
        ax.axvline(limite, color="#d95f02", linestyle="--")
        ax.text(limite, ax.get_ylim()[1] * 0.92, f" {_number(limite, 1)} mm", color="#d95f02")
    ax.set(
        title="Janelas com chuva: acumulado em 3 h e limites",
        xlabel="Chuva acumulada nas próximas 3 h (mm, escala log)",
        ylabel="Registros",
    )
    fig.tight_layout()
    _save_figure(fig, "distribuicao_classes")

    nomes = list(CLASSES.values())
    pct = por_split.set_index("particao")[nomes].div(por_split.set_index("particao")["total"], axis=0) * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    pct.T.plot(kind="bar", ax=ax, color=sns.color_palette("colorblind", 3), rot=0)
    ax.set_yscale("log")
    ax.set(
        title="Participação de cada classe por partição temporal (escala log)",
        xlabel="Classe",
        ylabel="Participação na partição (%)",
    )
    ax.legend(title="Partição")
    fig.tight_layout()
    _save_figure(fig, "distribuicao_particoes")

    mensal = (
        df.assign(mes=df["datetime_local"].dt.month)
        .groupby("mes")["classe"]
        .apply(lambda s: (s >= 2).mean() * 100)
    )
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.bar(mensal.index, mensal.values, color="#3182bd")
    ax.set(
        title="Sazonalidade: % de horas seguidas de chuva moderada ou forte",
        xlabel="Mês",
        ylabel="% das horas do mês",
        xticks=range(1, 13),
    )
    fig.tight_layout()
    _save_figure(fig, "sazonalidade_classes")


def _relatorio(
    esquemas: pd.DataFrame,
    geral: pd.DataFrame,
    por_split: pd.DataFrame,
    feats: pd.DataFrame,
    meta: dict,
) -> str:
    tab_esq = "\n".join(
        f"| {r.esquema} | {r.n_classes} | {_integer(r.menor_classe_registros)} "
        f"({_number(r.menor_classe_pct)}%) | {_integer(r.menor_classe_no_teste)} | "
        f"{_number(r.razao_desbalanceamento, 1)} |"
        for r in esquemas.itertuples()
    )
    tab_classes = "\n".join(
        f"| {int(r.classe)} — {r.classe_nome} | {r.faixa_acum_3h} | {_integer(r.registros)} | "
        f"{_number(r.participacao_pct)}% | {_number(r.mediana_mm, 1)} | {_number(r.razao_vs_majoritaria, 1)} |"
        for r in geral.itertuples()
    )
    nomes = list(CLASSES.values())
    tab_split = "\n".join(
        f"| {r['particao']} | {r['periodo']} | {_integer(r['total'])} | "
        + " | ".join(f"{_integer(r[n])} ({_number(r[n] / r['total'] * 100)}%)" for n in nomes)
        + " |"
        for _, r in por_split.iterrows()
    )
    tab_feats = "\n".join(
        f"| {grupo} | {len(g)} | {', '.join(f'`{f}`' for f in g['feature'])} |"
        for grupo, g in feats.groupby("grupo", sort=False)
    )
    top = feats.reindex(feats["spearman_classe_treino"].abs().sort_values(ascending=False).index).head(8)
    tab_top = "\n".join(
        f"| `{r.feature}` | {_number(r.spearman_classe_treino, 3)} |" for r in top.itertuples()
    )
    g = geral.set_index("classe")
    escolhido = esquemas.set_index("esquema").loc[ESQUEMA_ESCOLHIDO]
    esq_d = esquemas.set_index("esquema").loc["D — 5 classes (0,2 / 2,5 / 10 / 20)"]
    teste_forte = int(por_split.set_index("particao").loc["teste", "Forte"])
    acuracia_trivial = float(g.loc[0, "participacao_pct"])

    return f"""# DIA 2 — Definição de target e estratégia de classificação (Recife, INMET A301)

## 1. Definição do target

**Pergunta de previsão:** dado o estado atmosférico observado até a hora *t*, qual será a intensidade da chuva **acumulada nas próximas {HORIZONTE_H} horas** (*t*+1, *t*+2 e *t*+3)?

- **Variável base:** `{VARIAVEL_BASE}` — soma de `precip_mm` em *t*+1..*t*+{HORIZONTE_H}, criada em `ml/features.py::construir_features`. A hora *t* **não** entra na soma; isso é verificado pelo script ao recalcular a soma a partir de `precip_mm`.
- **Target multiclasse ordinal:** `classe_chuva_3h` ∈ {{0, 1, 2, 3}}, obtido por `classificar()` em `ml/experimento_target.py`.
- **Janelas inválidas:** se alguma das {HORIZONTE_H} horas futuras não tiver medição, ou se a janela cruzar a virada do ano, o alvo fica `NaN` e o registro sai da modelagem. Foram {_integer(meta['registros_base'] - meta['registros_validos'])} registros descartados por isso, de {_integer(meta['registros_base'])}.
- **Relação com o trabalho anterior:** a classe 3 (Forte) equivale exatamente ao alvo binário principal já usado pelo grupo (`ALVO_PRINCIPAL = "{ALVO_PRINCIPAL}"`, chuva acumulada ≥ 10 mm em 3 h). O alerta binário continua disponível como `classe == 3`.

O horizonte de 3 h e o acumulado (e não o máximo horário) seguem o alvo operacional definido em `ml/config.py`. O acumulado representa o volume que chega ao solo, que é o que causa alagamentos.

## 2. Classes candidatas

Foram comparados quatro esquemas sobre os {_integer(meta['registros_validos'])} registros válidos:

| Esquema | Nº de classes | Menor classe (registros) | Menor classe no teste (2021) | Razão maior/menor |
|---|---:|---:|---:|---:|
{tab_esq}

**Escolhido: {ESQUEMA_ESCOLHIDO}.**

- **A (binário)** perde a gradação: chuva fraca e ausência de chuva ficam na mesma classe, e o app não consegue diferenciar "vai chover pouco" de "não vai chover".
- **B (3 classes)** junta chuvas de 0,2 mm e de 9,8 mm na mesma classe, o que é pouco útil para quem decide sair de casa.
- **D (5 classes)** cria uma classe "Muito forte" (≥ 20 mm) com só {_integer(esq_d['menor_classe_registros'])} registros ({_number(esq_d['menor_classe_pct'])}%) e {_integer(esq_d['menor_classe_no_teste'])} no teste. Isso não basta para estimar precisão e recall dessa classe com estabilidade (mínimo adotado: {MIN_EXEMPLOS_TESTE}).
- **C** mantém a classe Forte com {_integer(escolhido['menor_classe_registros'])} registros, sendo {_integer(escolhido['menor_classe_no_teste'])} no teste, e distingue os três níveis que importam para o usuário.

## 3. Limites das classes e justificativa

| Classe | Faixa (acumulado em 3 h) | Registros | Participação | Mediana (mm) | Razão vs. majoritária |
|---|---|---:|---:|---:|---:|
{tab_classes}

- **0,2 mm (Sem chuva × Fraca):** é a resolução do pluviômetro da estação automática. O menor valor não nulo da série é {_number(meta['menor_valor_nao_nulo'], 1)} mm, então abaixo disso não existe medição de chuva.
- **2,5 mm (Fraca × Moderada):** fica próximo do percentil 75 das janelas com chuva ({_number(meta['p75_chuvosos'], 1)} mm). Assim, as três quartas partes mais fracas das janelas com chuva ficam na classe Fraca e o quarto superior segue para as classes de maior impacto.
- **10 mm (Moderada × Forte):** é o limiar do `ALVO_PRINCIPAL` já adotado no projeto e fica próximo do percentil 95 das janelas com chuva ({_number(meta['p95_chuvosos'], 1)} mm). Isso mantém compatibilidade com o pipeline e o dashboard existentes.

Os percentis foram calculados sobre a série completa só para justificar os cortes. Os limites são fixos e não são reajustados por partição, então nenhum parâmetro do target é aprendido a partir dos dados de teste.

![Distribuição das classes](figuras/distribuicao_classes.png)

## 4. Distribuição por classe e por partição

| Partição | Período (hora local) | Total | {' | '.join(nomes)} |
|---|---|---:|{'---:|' * len(nomes)}
{tab_split}

![Participação por partição](figuras/distribuicao_particoes.png)

![Sazonalidade](figuras/sazonalidade_classes.png)

## 5. Desbalanceamento

- A classe majoritária (Sem chuva) tem {_number(g.loc[0, 'participacao_pct'])}% dos registros, e a minoritária (Forte) tem {_number(g.loc[3, 'participacao_pct'])}%, uma razão de **{_number(g.loc[3, 'razao_vs_majoritaria'], 1)}:1**.
- Um modelo que sempre responde "Sem chuva" alcançaria **{_number(acuracia_trivial, 1)}% de acurácia** sem prever nenhum evento. Por isso a acurácia **não** será usada como métrica principal, como recomendam as notas do professor.
- **Métricas adotadas para os DIAS 5 e 6:** F1-macro como métrica de seleção; precisão, recall e F1 por classe, com foco no **recall da classe Forte**; matriz de confusão; PR-AUC da classe Forte. Uma diferença muito grande entre precisão e recall será tratada como sinal de vazamento.
- **Tratamento previsto:** `class_weight="balanced"` como primeira opção e, como alternativa, SMOTE. Todas as features são numéricas, então SMOTE se aplica (SMOTENC só seria necessário com variáveis categóricas). O reamostrador é ajustado **somente no treino** e dentro do pipeline, nunca na validação ou no teste.
- A prevalência das classes muda entre os anos e ao longo dos meses (figura de sazonalidade, com concentração entre abril e julho). A partição por ano civil completo mantém todas as estações em cada conjunto.

## 6. Estratégia temporal de treino e teste (out-of-time)

| Partição | Anos | Uso |
|---|---|---|
| Treino | {', '.join(map(str, PARTICOES['treino']))} | Ajuste dos modelos, escalonamento e reamostragem. Validação cruzada interna com `TimeSeriesSplit(gap={HORIZONTE_H})`. |
| Validação | {', '.join(map(str, PARTICOES['validacao']))} | Escolha de hiperparâmetros, limiar de decisão e comparação de algoritmos. |
| Teste | {', '.join(map(str, PARTICOES['teste']))} | Avaliação final, usada **uma única vez** após todas as escolhas. |

- Sem `train_test_split` aleatório: horas vizinhas são quase idênticas (autocorrelação), e a divisão aleatória colocaria a mesma chuva no treino e no teste.
- O `gap={HORIZONTE_H}` no `TimeSeriesSplit` impede que o alvo da última hora de um fold de treino use horas do fold de validação.
- Nas fronteiras entre partições, janelas que cruzam o ano já ficam inválidas pela construção do alvo. O script verifica isso com `assert`.
- 2021 vai até {meta['fim_teste']}. Por isso o teste tem menos horas que os outros anos, mas inclui a quadra chuvosa completa (abril a julho).
- O teste tem proporção um pouco maior de chuva Moderada e Forte que o treino (tabela acima). Avaliar num ano diferente mede se o modelo generaliza para anos mais ou menos chuvosos.

## 7. Features permitidas para previsão

Todas as {len(FEATURES_CLASSIFICACAO)} features de `FEATURES_CLASSIFICACAO` usam apenas informação observada até *t*. `validar_sem_leakage(..., modo="classificacao")` é executada pelo script e passa.

| Grupo | Nº | Features |
|---|---:|---|
{tab_feats}

**Maiores correlações de Spearman com a classe (somente no treino):**

| Feature | Spearman |
|---|---:|
{tab_top}

Nenhuma feature tem |ρ| ≥ 0,9 com o target. Uma correlação assim indicaria que a feature "contém" o alvo. As mais fortes são de chuva observada até *t*, o que é esperado, já que chuva tende a persistir.

**Proibidas como entrada ({len(COLUNAS_PROIBIDAS_ENTRADA)}):** identificadores temporais lineares (`ano`, `mes`, `dia`, `hora`, timestamps), `precip_mm`/`precip_mm_limpa` crus (já representados por `chuva_1h`), extremos max/min redundantes, vento (excesso de nulos), agregados futuros (`{'`, `'.join(COLUNAS_AGREGADAS_FUTURAS)}`) e todas as colunas `alvo_*`. Rótulos de cluster (`cluster`, `cluster_*`) também ficam fora, porque o K-Means foi ajustado na base completa, incluindo 2021.

## 8. Riscos de vazamento documentados

| # | Risco | Como é evitado / verificado |
|---|---|---|
| 1 | O target incluir a chuva da própria hora *t* | Soma recalculada de *t*+1..*t*+3 a partir de `precip_mm` e comparada com `acum_fut_3h` (`assert` no script). |
| 2 | Agregados futuros ou outras colunas `alvo_*` entrarem em X | `validar_sem_leakage` bloqueia `max_fut_*`, `acum_fut_*`, `alvo_*` e `futuro_*`. |
| 3 | Divisão aleatória com horas vizinhas no treino e no teste | Partição por ano (out-of-time) e `TimeSeriesSplit(gap={HORIZONTE_H})` no treino. |
| 4 | Janela do alvo cruzando a fronteira entre partições | Janelas que cruzam o ano ficam `NaN`; `assert` no script. |
| 5 | Scaler, SMOTE ou seleção de features ajustados com validação/teste | Todas as transformações ficam dentro de um `Pipeline` ajustado só no treino (DIA 5). |
| 6 | Rótulos de cluster usados como feature | O K-Means foi ajustado em 2018–2021; usar o rótulo levaria informação do teste. Vetado em `validar_sem_leakage`. |
| 7 | Imputação olhando para o futuro | `ler_csv` usa só forward-fill (valor passado). Nenhuma interpolação centrada ou backfill. |
| 8 | Features do Open-Meteo (`{FEATURES_OPEN_METEO[0]}`, ..., `om_precip`) | Vêm de arquivo histórico/reanálise, que pode incorporar observações posteriores a *t*, e `om_precip` é praticamente a chuva modelada. **Ficam fora** até se confirmar que estão disponíveis no instante da previsão. |
| 9 | Limiar de decisão ou hiperparâmetros escolhidos olhando o teste | Escolhas feitas só na validação (2020); o teste (2021) é avaliado uma vez. |
| 10 | Limites das classes ajustados por partição | Limites fixos (0,2 / 2,5 / 10 mm), definidos por critério físico e pelo limiar do projeto. |

## 9. Reprodutibilidade

- Base: `{meta['dataset']}` (SHA-256 `{meta['sha256_csv']}`), lida por `preparar_base(usar_open_meteo=False, filtrar_anos=False)`.
- Python {meta['python']}, pandas {meta['pandas']}, scikit-learn {meta['scikit_learn']}, NumPy {meta['numpy']}.
- Para reproduzir, na raiz do repositório: `python ml/experimento_target.py`.

## Arquivos

- `esquemas_candidatos.csv`: comparação dos quatro esquemas de classes.
- `distribuicao_classes.csv`: registros, participação e estatísticas do acumulado por classe.
- `distribuicao_particoes.csv`: registros por classe em treino, validação e teste.
- `features_permitidas.csv`: features de entrada, grupo, Spearman no treino e % de nulos.
- `reproducibilidade.json`: configuração, versões e hash da base.
- `figuras/`: PNG (300 dpi) e PDF para o artigo.
"""


def main() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    base = preparar_base(caminho=RAW_CSV, usar_open_meteo=False, filtrar_anos=False)
    registros_validos = _verificar_janela_futura(base)
    validar_sem_leakage(FEATURES_CLASSIFICACAO, modo="classificacao")
    assert not set(FEATURES_CLASSIFICACAO) & set(COLUNAS_ALVO + COLUNAS_AGREGADAS_FUTURAS)

    base["particao"] = _particao(base["ano"])
    base["classe"] = classificar(base[VARIAVEL_BASE])
    df = base.dropna(subset=["classe"]).copy()
    df["classe"] = df["classe"].astype(int)
    assert df["particao"].notna().all(), "Ano fora das particoes definidas"
    assert len(df) == registros_validos

    # Coerencia com o alvo binario existente: classe Forte == ALVO_PRINCIPAL.
    alvo_bin = df["alvo_3h_acum_10mm"].astype(int)
    assert ((df["classe"] == 3).astype(int) == alvo_bin).all()

    # Ordem temporal estrita entre particoes.
    limites_tempo = df.groupby("particao")["datetime_utc"].agg(["min", "max"]).reindex(PARTICOES)
    assert (limites_tempo["max"].iloc[:-1].to_numpy() < limites_tempo["min"].iloc[1:].to_numpy()).all()

    esquemas = _tabela_esquemas(df)
    geral, por_split = _distribuicao(df)
    assert por_split.set_index("particao").loc["teste", "Forte"] >= MIN_EXEMPLOS_TESTE

    treino = df[df["particao"] == "treino"]
    feats = _tabela_features(df, treino)
    assert (feats["spearman_classe_treino"].abs() < 0.9).all(), "Correlacao suspeita de leakage"

    chuvosos = df.loc[df[VARIAVEL_BASE] > 0, VARIAVEL_BASE]
    meta = {
        "dataset": str(RAW_CSV.relative_to(ROOT)).replace("\\", "/"),
        "sha256_csv": hashlib.sha256(RAW_CSV.read_bytes()).hexdigest(),
        "registros_base": len(base),
        "registros_validos": registros_validos,
        "variavel_base": VARIAVEL_BASE,
        "horizonte_h": HORIZONTE_H,
        "limites_mm": list(LIMITES_MM),
        "classes": CLASSES,
        "particoes": {k: list(v) for k, v in PARTICOES.items()},
        "fim_teste": f"{df.loc[df['particao'] == 'teste', 'datetime_local'].max():%d/%m/%Y}",
        "menor_valor_nao_nulo": float(chuvosos.min()),
        "p75_chuvosos": float(chuvosos.quantile(0.75)),
        "p95_chuvosos": float(chuvosos.quantile(0.95)),
        "features": FEATURES_CLASSIFICACAO,
        "random_state": RANDOM_STATE,
        "python": sys.version.split()[0],
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
        "comando_reproducao": "python ml/experimento_target.py",
    }

    esquemas.to_csv(OUTPUT_DIR / "esquemas_candidatos.csv", index=False, float_format="%.6f")
    geral.to_csv(OUTPUT_DIR / "distribuicao_classes.csv", index=False, float_format="%.6f")
    por_split.to_csv(OUTPUT_DIR / "distribuicao_particoes.csv", index=False)
    feats.to_csv(OUTPUT_DIR / "features_permitidas.csv", index=False, float_format="%.6f")
    _figuras(df, geral, por_split)
    (OUTPUT_DIR / "reproducibilidade.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (OUTPUT_DIR / "relatorio.md").write_text(
        _relatorio(esquemas, geral, por_split, feats, meta), encoding="utf-8"
    )
    print(f"Target definido: {len(CLASSES)} classes, {_integer(registros_validos)} registros válidos.")
    print(geral[["classe_nome", "registros", "participacao_pct"]].to_string(index=False))
    print(f"Relatório: {OUTPUT_DIR.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
