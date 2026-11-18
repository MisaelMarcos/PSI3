"""Metricas de classificacao e avaliacao por evento de chuva."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)


def metricas_binarias(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    limiar: float = 0.5,
) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    y_pred = (y_proba >= limiar).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    resultado = {
        "n": int(len(y_true)),
        "positivos": int(y_true.sum()),
        "prevalencia": float(y_true.mean()) if len(y_true) else 0.0,
        "limiar_probabilidade": float(limiar),
        "acuracia": float((tp + tn) / len(y_true)) if len(y_true) else 0.0,
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "falsos_alarmes_por_dia": float(fp / max(len(y_true) / 24.0, 1.0)),
    }
    # PR-AUC e ROC-AUC so fazem sentido quando ha as duas classes.
    if len(np.unique(y_true)) == 2:
        resultado["pr_auc"] = float(average_precision_score(y_true, y_proba))
        resultado["roc_auc"] = float(roc_auc_score(y_true, y_proba))
    else:
        resultado["pr_auc"] = None
        resultado["roc_auc"] = None
    return resultado


def _curva_f1(y_true: np.ndarray, y_proba: np.ndarray):
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
    precisao = precision[:-1]
    cobertura = recall[:-1]
    f1 = np.divide(
        2 * precisao * cobertura,
        precisao + cobertura,
        out=np.zeros_like(precisao),
        where=(precisao + cobertura) > 0,
    )
    return precisao, cobertura, f1, thresholds


def escolher_limiar_f1(
    y_true: np.ndarray,
    y_proba: np.ndarray,
) -> dict:
    """Corte que maximiza F1 na validacao."""
    if len(np.unique(np.asarray(y_true, dtype=int))) < 2:
        return {"limiar": 0.5, "recall": None, "precision": None, "f1": None}
    precisao, cobertura, f1, thresholds = _curva_f1(y_true, y_proba)
    melhor = int(np.argmax(f1))
    return {
        "limiar": float(thresholds[melhor]),
        "recall": float(cobertura[melhor]),
        "precision": float(precisao[melhor]),
        "f1": float(f1[melhor]),
        "criterio": "f1",
    }


def escolher_limiar(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    alvo_recall: float = 0.8,
) -> dict:
    """Escolhe probabilidade de corte priorizando recall.

    Entre os cortes que atingem o recall desejado, escolhe o de maior F1.
    Se nenhum corte atingir o recall, devolve o corte de maior F1.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_proba = np.asarray(y_proba, dtype=float)
    if len(np.unique(y_true)) < 2:
        return {"limiar": 0.5, "recall": None, "precision": None, "f1": None}

    precisao, cobertura, f1, thresholds = _curva_f1(y_true, y_proba)

    elegiveis = cobertura >= alvo_recall
    if elegiveis.any():
        indices = np.flatnonzero(elegiveis)
        melhor = indices[np.argmax(f1[indices])]
    else:
        melhor = int(np.argmax(f1))

    return {
        "limiar": float(thresholds[melhor]),
        "recall": float(cobertura[melhor]),
        "precision": float(precisao[melhor]),
        "f1": float(f1[melhor]),
        "alvo_recall": float(alvo_recall),
    }


def avaliar_eventos(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    horizonte: int,
) -> dict:
    """Avalia se cada evento de chuva recebeu alerta antes de comecar.

    Um evento e um bloco contiguo de horas com alvo positivo. Para cada inicio
    de evento, verificamos se houve predicao positiva em qualquer uma das
    ``horizonte`` horas anteriores.
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    inicios = np.flatnonzero(
        (y_true == 1) & np.r_[True, y_true[:-1] == 0]
    )
    acertos = 0
    acertos_2h = 0
    acertos_3h = 0
    antecedencias = []
    for inicio in inicios:
        janela_inicio = max(0, inicio - horizonte)
        janela = y_pred[janela_inicio:inicio]
        if len(janela) and janela.any():
            acertos += 1
            # Primeiro alerta da janela = maior antecedencia.
            primeiro_alerta = janela_inicio + int(np.flatnonzero(janela)[0])
            antecedencia = inicio - primeiro_alerta
            antecedencias.append(antecedencia)
            if antecedencia >= 2:
                acertos_2h += 1
            if antecedencia >= 3:
                acertos_3h += 1

    total = len(inicios)
    return {
        "eventos": int(total),
        "eventos_com_alerta": int(acertos),
        "eventos_com_2h_antecedencia": int(acertos_2h),
        "eventos_com_3h_antecedencia": int(acertos_3h),
        "taxa_eventos_com_alerta": float(acertos / total) if total else None,
        "taxa_eventos_com_2h_antecedencia": (
            float(acertos_2h / total) if total else None
        ),
        "taxa_eventos_com_3h_antecedencia": (
            float(acertos_3h / total) if total else None
        ),
        "antecedencia_media_h": (
            float(np.mean(antecedencias)) if antecedencias else None
        ),
        "antecedencia_min_h": (
            float(np.min(antecedencias)) if antecedencias else None
        ),
        "antecedencia_max_h": (
            float(np.max(antecedencias)) if antecedencias else None
        ),
    }


def avaliar_completo(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    horizonte: int,
    limiar: float = 0.5,
) -> dict:
    metricas = metricas_binarias(y_true, y_proba, limiar)
    y_pred = (np.asarray(y_proba) >= limiar).astype(int)
    metricas["eventos"] = avaliar_eventos(y_true, y_pred, horizonte)
    return metricas
