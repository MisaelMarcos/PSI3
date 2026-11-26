"""Comparacao dos classificadores ja treinados e selecao de candidatos ao modelo final.

Junta todos os `results/metrics/resumo_*.csv` (um por grupo) e gera:
- `results/metrics/comparativo_modelos.csv`
- `results/figures/comparativo_modelos.png/.pdf`
- `results/relatorio_comparacao.md`

A selecao usa SOMENTE a validacao (2020); o teste (2021) e reportado para conferir a
generalizacao, sem influenciar a escolha. Quando os Grupos 2 e 3 salvarem seus
resumos, basta rodar de novo:

    python ml/classification/comparison.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from evaluation import FIGURES_DIR, GAP_FORTE, GAP_MODERADO, METRICS_DIR, RESULTS_DIR
from preprocessing import PARTICOES

# Um modelo so e candidato pelo recall da classe Forte se nao perder mais que isso de
# F1 macro em relacao ao melhor (evita escolher um modelo que "chuta Forte" para tudo).
TOLERANCIA_F1 = 0.05


def _n(v: float, casas: int = 3) -> str:
    return f"{v:.{casas}f}".replace(".", ",")


def carregar_resumos() -> pd.DataFrame:
    arquivos = sorted(METRICS_DIR.glob("resumo_*.csv"))
    if not arquivos:
        raise FileNotFoundError("Nenhum results/metrics/resumo_*.csv encontrado. Rode um train_*.py antes.")
    return pd.concat([pd.read_csv(a) for a in arquivos], ignore_index=True)


def selecionar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values(["f1_macro_validacao", "recall_forte_validacao"], ascending=False).reset_index(drop=True)
    melhor_f1 = df.loc[0, "f1_macro_validacao"]
    df["ranking_validacao"] = np.arange(1, len(df) + 1)
    df["candidato"] = ""
    df.loc[0, "candidato"] = "melhor F1 macro (validação)"
    aceitaveis = df[df["f1_macro_validacao"] >= melhor_f1 - TOLERANCIA_F1]
    idx_recall = aceitaveis["recall_forte_validacao"].idxmax()
    if idx_recall != 0:
        df.loc[idx_recall, "candidato"] = "melhor recall Forte (validação)"
    return df


def figura(df: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.8))
    ordem = df["modelo"].tolist()
    cores = sns.color_palette("colorblind", 3)
    for ax, (prefixo, titulo) in zip(axes[:2], [("f1_macro", "F1 macro"), ("recall_forte", "Recall da classe Forte")]):
        largura = 0.38
        x = np.arange(len(ordem))
        for i, (particao, ano) in enumerate([("validacao", "2020"), ("teste", "2021")]):
            col = f"{prefixo}_{particao}"
            ax.bar(x + (i - 0.5) * largura, df[col], largura, color=cores[i], label=f"{particao} ({ano})")
        ax.set_xticks(x, ordem, rotation=20, ha="right")
        ax.set(title=titulo, ylabel="Valor", ylim=(0, 1))
        ax.legend()
    ax = axes[2]
    ax.barh(ordem[::-1], df["tempo_fit_s"][::-1], color=cores[2])
    ax.set_xscale("log")
    ax.set(title="Custo: tempo de treino (escala log)", xlabel="Segundos")
    fig.suptitle("Comparação dos classificadores já treinados", y=1.02)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES_DIR / f"comparativo_modelos.{ext}", dpi=300 if ext == "png" else None, bbox_inches="tight")
    plt.close(fig)


def relatorio(df: pd.DataFrame) -> str:
    grupos = ", ".join(sorted(df["grupo"].unique()))
    tab = "\n".join(
        f"| {r.ranking_validacao} | {r.modelo} | {r.grupo} | {_n(r.f1_macro_treino)} | {_n(r.f1_macro_validacao)} | "
        f"{_n(r.f1_macro_teste)} | {_n(r.f1_ponderado_teste)} | {_n(r.accuracy_teste)} | {_n(r.recall_forte_validacao)} | "
        f"{_n(r.recall_forte_teste)} | {_n(r.precision_forte_teste)} | {_n(r.tempo_fit_s, 1)} s | "
        f"{_n(r.tempo_predicao_teste_s, 2)} s | {r.overfitting} ({_n(r.gap_f1_treino_validacao)}) | {r.candidato or '—'} |"
        for r in df.itertuples()
    )
    params = "\n".join(
        f"| {r.modelo} | `{', '.join(f'{k.split('__')[-1]}={v}' for k, v in json.loads(r.parametros).items())}` | {r.observacao} |"
        for r in df.itertuples()
    )
    melhor = df.iloc[0]
    rec = df.loc[df["recall_forte_validacao"].idxmax()]
    rapido = df.loc[df["tempo_fit_s"].idxmin()]
    lento = df.loc[df["tempo_fit_s"].idxmax()]
    over = df[df["overfitting"] != "nao"]["modelo"].tolist()
    candidatos = df[df["candidato"] != ""]
    lista_cand = "\n".join(f"- **{r.modelo}** — {r.candidato}: F1 macro {_n(r.f1_macro_validacao)} na validação e "
                           f"{_n(r.f1_macro_teste)} no teste; recall Forte {_n(r.recall_forte_validacao)} / {_n(r.recall_forte_teste)}."
                           for r in candidatos.itertuples())

    return f"""# Comparação dos classificadores — DIA 5

Modelos incluídos: {len(df)} ({grupos}). Quando os outros grupos salvarem `results/metrics/resumo_<grupo>.csv`, rodar `python ml/classification/comparison.py` de novo atualiza esta comparação.

## Como foi feito

- **Dados:** `preprocessing.carregar_dados()`, igual para todos os grupos. Target `classe_chuva_3h` (chuva acumulada nas próximas 3 h em 4 classes: Sem chuva, Fraca, Moderada e Forte), 33 features observadas até *t*, divisão out-of-time: treino {', '.join(map(str, PARTICOES['treino']))}, validação {', '.join(map(str, PARTICOES['validacao']))} e teste {', '.join(map(str, PARTICOES['teste']))}.
- **Hiperparâmetros:** grade pequena por modelo, ajuste no treino e escolha pelo F1 macro na validação. O teste não participa de nenhuma escolha.
- **Desbalanceamento:** `class_weight="balanced"` nos modelos que aceitam. Escalonamento (`StandardScaler`) dentro do Pipeline, ajustado só no treino.
- **Overfitting:** diferença de F1 macro entre treino e validação; acima de {_n(GAP_MODERADO, 2)} = moderado, acima de {_n(GAP_FORTE, 2)} = sim.
- **Seleção:** candidato principal = maior F1 macro na validação. Também entra como candidato o maior recall da classe Forte na validação, desde que o F1 macro fique a no máximo {_n(TOLERANCIA_F1, 2)} do melhor.
- Accuracy aparece só como referência: um modelo que sempre diz "Sem chuva" já teria ~70% de accuracy no teste.

## Tabela comparativa

| # | Modelo | Grupo | F1 macro treino | F1 macro validação | F1 macro teste | F1 ponderado teste | Accuracy teste | Recall Forte validação | Recall Forte teste | Precision Forte teste | Tempo de treino | Tempo de predição (teste) | Overfitting (gap) | Candidato |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|
{tab}

![Comparação](figures/comparativo_modelos.png)

## Leitura

- **Melhor desempenho geral:** {melhor['modelo']}, com F1 macro {_n(melhor['f1_macro_validacao'])} na validação e {_n(melhor['f1_macro_teste'])} no teste.
- **Chuva mais intensa:** o maior recall da classe Forte na validação é do {rec['modelo']} ({_n(rec['recall_forte_validacao'])}; no teste, {_n(rec['recall_forte_teste'])}). Para um sistema de alerta, deixar passar chuva forte é o erro mais caro, por isso esse recall pesa na escolha.
- **Falsos alarmes:** a precisão da classe Forte no teste fica entre {_n(df['precision_forte_teste'].min())} e {_n(df['precision_forte_teste'].max())}. Ou seja, a maioria dos avisos de chuva forte não se confirma. Recall alto com precisão baixa é o padrão esperado com `class_weight="balanced"` numa classe de ~1,5% dos registros, e deve ser considerado na hora de definir o limiar do alerta.
- **Generalização:** {('indícios de overfitting em ' + ', '.join(over) + '.') if over else 'nenhum modelo passou do limite de overfitting.'} A comparação validação × teste mostra se o desempenho se mantém num ano que o modelo nunca viu. Ver a coluna "Configuração" abaixo para modelos em que o gap não se aplica (KNN).
- **Custo computacional:** o mais rápido para treinar foi o {rapido['modelo']} ({_n(rapido['tempo_fit_s'], 1)} s) e o mais lento o {lento['modelo']} ({_n(lento['tempo_fit_s'], 1)} s). Tempos medidos numa única máquina; servem para comparar ordem de grandeza.

## Candidatos ao modelo final

{lista_cand}

A decisão final sai depois que os Grupos 2 e 3 adicionarem os resultados de ensembles e boosting.

## Parâmetros registrados

| Modelo | Hiperparâmetros escolhidos | Configuração |
|---|---|---|
{params}

## Arquivos

- `metrics/resumo_<grupo>.csv`: uma linha por modelo (métricas, tempos, parâmetros).
- `metrics/<modelo>_metricas.csv`: todas as métricas por partição e por classe.
- `metrics/<modelo>_grade.csv`: resultado de cada hiperparâmetro testado na validação.
- `confusion_matrices/<modelo>.png` e `.csv`: matrizes de confusão de validação e teste.
- `metrics/comparativo_modelos.csv` e `figures/comparativo_modelos.png/.pdf`: esta comparação.
"""


def main() -> None:
    df = selecionar(carregar_resumos())
    df.to_csv(METRICS_DIR / "comparativo_modelos.csv", index=False, float_format="%.6f")
    figura(df)
    (RESULTS_DIR / "relatorio_comparacao.md").write_text(relatorio(df), encoding="utf-8")
    print(df[["ranking_validacao", "modelo", "f1_macro_validacao", "f1_macro_teste", "recall_forte_teste", "candidato"]]
          .round(3).to_string(index=False))


if __name__ == "__main__":
    main()
