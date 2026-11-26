"""Avaliacao padronizada dos classificadores do DIA 5 (comum aos tres grupos).

`avaliar_modelo` faz, para um modelo:
1. busca dos hiperparametros numa grade pequena, ajustando no TREINO (2018-2019) e
   escolhendo pelo F1 macro na VALIDACAO (2020) - o teste nao participa da escolha;
2. metricas no treino, validacao e teste (2021): accuracy, precision/recall/F1 por
   classe, F1 macro e ponderado;
3. matriz de confusao (validacao e teste), tempo de ajuste e de predicao;
4. diagnostico de overfitting pela diferenca de F1 macro entre treino e validacao.

Grava em `results/metrics/` e `results/confusion_matrices/`. Cada script de treino
salva um `results/metrics/resumo_<grupo>.csv`, que o `comparison.py` junta.
"""

from __future__ import annotations

import json
import re
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.base import clone
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_recall_fscore_support

from preprocessing import CLASSES, ML_DIR, NOMES_CLASSES, DadosClassificacao

RESULTS_DIR = ML_DIR / "classification" / "results"
METRICS_DIR = RESULTS_DIR / "metrics"
CM_DIR = RESULTS_DIR / "confusion_matrices"
FIGURES_DIR = RESULTS_DIR / "figures"

CLASSE_FORTE = 3
CLASSE_MODERADA = 2
# Diferenca de F1 macro treino - validacao acima destes limites indica overfitting.
GAP_MODERADO, GAP_FORTE = 0.05, 0.10


def _slug(nome: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", nome.lower()).strip("_")


def metricas(y_true, y_pred) -> dict[str, float]:
    """Accuracy, F1 macro/ponderado e precision/recall/F1/suporte por classe."""
    labels = sorted(CLASSES)
    p, r, f, s = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    out = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro", labels=labels, zero_division=0),
        "f1_ponderado": f1_score(y_true, y_pred, average="weighted", labels=labels, zero_division=0),
    }
    for i, nome in enumerate(NOMES_CLASSES):
        chave = _slug(nome)
        out[f"precision_{chave}"] = p[i]
        out[f"recall_{chave}"] = r[i]
        out[f"f1_{chave}"] = f[i]
        out[f"suporte_{chave}"] = int(s[i])
    return out


def _diagnostico_overfitting(gap: float) -> str:
    if gap > GAP_FORTE:
        return "sim"
    if gap > GAP_MODERADO:
        return "moderado"
    return "nao"


def _matriz_confusao(nome: str, dados: DadosClassificacao, preds: dict[str, np.ndarray]) -> None:
    CM_DIR.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, particao in zip(axes, ("validacao", "teste")):
        cm = confusion_matrix(dados.y[particao], preds[particao], labels=sorted(CLASSES))
        pct = cm / cm.sum(axis=1, keepdims=True) * 100
        anot = np.array([[f"{pct[i, j]:.0f}%\n({cm[i, j]})" for j in range(cm.shape[1])] for i in range(cm.shape[0])])
        sns.heatmap(pct, annot=anot, fmt="", cmap="Blues", vmin=0, vmax=100, cbar=False, ax=ax,
                    xticklabels=NOMES_CLASSES, yticklabels=NOMES_CLASSES)
        titulo = "Validação (2020)" if particao == "validacao" else "Teste (2021)"
        ax.set(title=titulo, xlabel="Classe prevista", ylabel="Classe real")
        pd.DataFrame(cm, index=NOMES_CLASSES, columns=NOMES_CLASSES).to_csv(CM_DIR / f"{_slug(nome)}_{particao}.csv")
    fig.suptitle(f"{nome} — matriz de confusão (% da classe real e contagem)", y=1.02)
    fig.tight_layout()
    fig.savefig(CM_DIR / f"{_slug(nome)}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def avaliar_modelo(
    nome: str,
    grupo: str,
    estimador,
    grade: list[dict],
    dados: DadosClassificacao,
    observacao: str = "",
) -> dict:
    """Escolhe hiperparametros na validacao e avalia treino/validacao/teste. Devolve a linha do resumo."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    linhas_grade, melhor, melhor_modelo = [], None, None
    for params in grade:
        modelo = clone(estimador).set_params(**params)
        t0 = time.perf_counter()
        modelo.fit(dados.X["treino"], dados.y["treino"])
        t_fit = time.perf_counter() - t0
        m_val = metricas(dados.y["validacao"], modelo.predict(dados.X["validacao"]))
        linha = {"parametros": json.dumps(params), "tempo_fit_s": t_fit,
                 "f1_macro_validacao": m_val["f1_macro"], "recall_forte_validacao": m_val["recall_forte"]}
        linhas_grade.append(linha)
        print(f"   {nome:<20} {json.dumps(params):<45} F1-macro val={m_val['f1_macro']:.3f} "
              f"recall Forte val={m_val['recall_forte']:.3f} ({t_fit:.1f}s)")
        chave = (m_val["f1_macro"], m_val["recall_forte"])
        if melhor is None or chave > melhor[0]:
            melhor, melhor_modelo = (chave, params, t_fit), modelo
    pd.DataFrame(linhas_grade).to_csv(METRICS_DIR / f"{_slug(nome)}_grade.csv", index=False, float_format="%.6f")

    _, params, t_fit = melhor
    preds, tempos_pred, por_particao = {}, {}, {}
    for particao in ("treino", "validacao", "teste"):
        t0 = time.perf_counter()
        preds[particao] = melhor_modelo.predict(dados.X[particao])
        tempos_pred[particao] = time.perf_counter() - t0
        por_particao[particao] = metricas(dados.y[particao], preds[particao])
    pd.DataFrame(por_particao).T.rename_axis("particao").to_csv(
        METRICS_DIR / f"{_slug(nome)}_metricas.csv", float_format="%.6f")
    _matriz_confusao(nome, dados, preds)

    tr, va, te = por_particao["treino"], por_particao["validacao"], por_particao["teste"]
    gap = tr["f1_macro"] - va["f1_macro"]
    return {
        "modelo": nome,
        "grupo": grupo,
        "parametros": json.dumps(params),
        "tempo_fit_s": t_fit,
        "tempo_predicao_teste_s": tempos_pred["teste"],
        "f1_macro_treino": tr["f1_macro"],
        "f1_macro_validacao": va["f1_macro"],
        "f1_macro_teste": te["f1_macro"],
        "f1_ponderado_teste": te["f1_ponderado"],
        "accuracy_teste": te["accuracy"],
        "recall_forte_validacao": va["recall_forte"],
        "recall_forte_teste": te["recall_forte"],
        "precision_forte_teste": te["precision_forte"],
        "recall_moderada_teste": te["recall_moderada"],
        "gap_f1_treino_validacao": gap,
        "overfitting": _diagnostico_overfitting(gap),
        "observacao": observacao,
    }


def salvar_resumo(linhas: list[dict], grupo_slug: str) -> pd.DataFrame:
    resumo = pd.DataFrame(linhas)
    resumo.to_csv(METRICS_DIR / f"resumo_{grupo_slug}.csv", index=False, float_format="%.6f")
    return resumo
