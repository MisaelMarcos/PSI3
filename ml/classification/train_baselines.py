"""DIA 5 — Grupo 1: modelos tradicionais (Logistic Regression, Decision Tree, KNN e SVM).

Usa `preprocessing.carregar_dados()` (mesmos dados dos Grupos 2 e 3) e
`evaluation.avaliar_modelo()` (mesmas metricas). Rodar na raiz do repositorio:

    python ml/classification/train_baselines.py
"""

from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from evaluation import avaliar_modelo, salvar_resumo
from preprocessing import carregar_dados

from config import RANDOM_STATE  # noqa: E402  (ml/ entra no path via preprocessing)

GRUPO = "Grupo 1 — Modelos tradicionais"

# O StandardScaler fica dentro do Pipeline: e ajustado so no treino de cada modelo.
# class_weight="balanced" compensa o desbalanceamento (52:1) sem gerar dados sinteticos.
MODELOS = [
    (
        "Logistic Regression",
        Pipeline([("scaler", StandardScaler()),
                  ("clf", LogisticRegression(class_weight="balanced", max_iter=3000))]),
        [{"clf__C": c} for c in (0.01, 0.1, 1.0, 10.0)],
        "Baseline linear; StandardScaler + class_weight='balanced'.",
    ),
    (
        "Decision Tree",
        DecisionTreeClassifier(class_weight="balanced", min_samples_leaf=20, random_state=RANDOM_STATE),
        [{"max_depth": d} for d in (4, 6, 8, 12, None)],
        "Sem escalonamento (arvore nao depende de escala); class_weight='balanced', min_samples_leaf=20.",
    ),
    (
        "KNN",
        Pipeline([("scaler", StandardScaler()),
                  ("clf", KNeighborsClassifier(weights="distance", n_jobs=-1))]),
        [{"clf__n_neighbors": k} for k in (5, 15, 31, 51)],
        "StandardScaler; KNN nao aceita class_weight (sem compensacao do desbalanceamento). Com "
        "weights='distance' cada ponto de treino e vizinho de si mesmo, entao o F1 de treino e sempre 1 "
        "e o gap treino-validacao nao indica overfitting real.",
    ),
    (
        "SVM",
        Pipeline([("scaler", StandardScaler()),
                  ("clf", SVC(kernel="rbf", gamma="scale", class_weight="balanced", cache_size=1000))]),
        [{"clf__C": c} for c in (0.3, 1.0, 3.0)],
        "Kernel RBF; StandardScaler + class_weight='balanced'.",
    ),
]


def main() -> None:
    dados = carregar_dados()
    print(dados.resumo()[["particao", "registros", "Forte"]].to_string(index=False))
    linhas = []
    for nome, estimador, grade, obs in MODELOS:
        print(f"\n>> {nome}")
        linhas.append(avaliar_modelo(nome, GRUPO, estimador, grade, dados, observacao=obs))
    resumo = salvar_resumo(linhas, "grupo1_tradicionais")
    colunas = ["modelo", "parametros", "f1_macro_treino", "f1_macro_validacao", "f1_macro_teste",
               "recall_forte_teste", "tempo_fit_s", "overfitting"]
    print("\n" + resumo[colunas].round(3).to_string(index=False))


if __name__ == "__main__":
    main()
