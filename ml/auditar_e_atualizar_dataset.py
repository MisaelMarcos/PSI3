"""Auditoria de features, redundancia, risco de leakage e geracao do dataset de trabalho."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from config import (
    ALVOS,
    CLUSTER_FEATURES_CSV,
    OUTPUT_DIR,
    RAW_CSV,
    TARGETS_CSV,
    WORKING_FEATURES_CSV,
)
from features import (
    COLUNAS_AGREGADAS_FUTURAS,
    COLUNAS_ALVO,
    COLUNAS_CHUVA_OBSERVADA,
    COLUNAS_EXCLUIDAS_EXCESSO_NULOS,
    COLUNAS_EXCLUIDAS_REDUNDANCIA_BRUTA,
    COLUNAS_IDENTIFICADORAS,
    COLUNAS_ORIGINAIS_METEOROLOGICAS,
    FEATURES_CANDIDATAS,
    FEATURES_CLASSIFICACAO,
    FEATURES_CLUSTERING,
    MAPA_COLUNAS_CLUSTER_NOTEBOOK,
    auditar_features,
    preparar_base,
    resumo_alvos,
    separar_matrizes_modelo,
    validar_sem_leakage,
)


def _classificar_decisao_redundancia(a: str, b: str) -> str:
    par = {a, b}
    if par.intersection(COLUNAS_EXCLUIDAS_REDUNDANCIA_BRUTA):
        return (
            "Extremos max/min excluidos de ambos os modelos; mantida leitura "
            "horaria principal e amplitude intra-horaria derivada (amp_*_1h)."
        )
    if par.issubset(FEATURES_CLUSTERING):
        return (
            "Ambas mantidas em clustering e classificacao: representam "
            "dimensoes fisicas distintas (ciclo termico vs. saturacao/hora), "
            "ao contrario de duplicacoes diretas (max/min ou orvalho_depressao)."
        )
    if par == {"umid_pct", "orvalho_depressao"}:
        return (
            "orvalho_depressao excluida do clustering (|r|=0,9957 evitaria "
            "dupla contagem euclidiana) e mantida apenas na classificacao."
        )
    if par.intersection({"hora", "mes"}):
        return (
            "Representacao linear inteira (hora/mes) excluida das entradas; "
            "substituida por senos/cossenos ciclicos continuos."
        )
    if par == {"mes_cos", "estacao_chuvosa"}:
        return (
            "estacao_chuvosa excluida do clustering (redundante com mes_cos) "
            "e mantida na classificacao para facilitar splits binarios."
        )
    if par.intersection(COLUNAS_CHUVA_OBSERVADA):
        return (
            "Variaveis de chuva excluidas do clustering (leakage de alvo); "
            "mantidas apenas na classificacao como memoria multi-horizonte <= t."
        )
    return (
        "Derivada excluida do clustering para manter espaco enxuto; mantida na "
        "classificacao (Random Forest tolera correlacao entre derivadas)."
    )


def _formatar_tabela_md(df: pd.DataFrame) -> str:
    colunas = list(df.columns)
    cabecalho = "| " + " | ".join(colunas) + " |"
    separador = "| " + " | ".join(["---"] * len(colunas)) + " |"
    linhas = [cabecalho, separador]
    for _, row in df.iterrows():
        valores = []
        for col in colunas:
            val = row[col]
            if isinstance(val, float):
                valores.append(f"{val:.4f}")
            else:
                valores.append(str(val))
        linhas.append("| " + " | ".join(valores) + " |")
    return "\n".join(linhas)


def gerar_relatorio_markdown(
    df_proc: pd.DataFrame,
    audit: dict,
    resumo_json: dict,
) -> str:
    rel = audit["relevancia_features"]
    red = audit["redundancia_pares"]
    perfil = audit["perfil_clusters"].reset_index()

    top_rel = rel[
        [
            "feature",
            "tipo",
            "corr_precip_atual",
            "corr_acum_fut_3h",
            "corr_alvo_3h_acum_10mm",
            "rf_imp_2019_acum_10mm",
            "anova_f_clusters",
            "entra_clustering",
            "entra_classificacao",
        ]
    ].head(15)

    top_red = red[["variavel_a", "variavel_b", "pearson_r", "decisao_tecnica"]].head(20)

    linhas = [
        "# Auditoria de Features, Redundancia, Risco de Leakage e Dataset de Trabalho (Recife - INMET A301)",
        "",
        f"Gerado em: {resumo_json['gerado_em']}",
        f"Base bruta (nao modificada): `{resumo_json['csv_bruto']}` ({resumo_json['n_registros']} registros horarios)",
        "",
        "## 1. Inventario do Pipeline Auditado",
        "",
        f"- **Features derivadas candidatas consolidadas**: {len(FEATURES_CANDIDATAS)} (minimo exigido: 8).",
        f"- **Variaveis de entrada em Clustering (K-Means)**: {len(FEATURES_CLUSTERING)} -> `{FEATURES_CLUSTERING}`.",
        f"- **Variaveis de entrada em Classificacao**: {len(FEATURES_CLASSIFICACAO)} (4 fisicas originais + {len(FEATURES_CANDIDATAS)} candidatas).",
        f"- **Colunas excluidas por redundancia bruta**: `{COLUNAS_EXCLUIDAS_REDUNDANCIA_BRUTA}`.",
        f"- **Colunas excluidas por excesso de nulos (~66%)**: `{COLUNAS_EXCLUIDAS_EXCESSO_NULOS}`.",
        "",
        "## 2. Features Candidatas Consolidadas (todas estritamente <= t)",
        "",
        "1. **Ciclos temporais continuos (4)**: `hora_sin`, `hora_cos`, `mes_sin`, `mes_cos`.",
        "2. **Sazonalidade regional (1)**: `estacao_chuvosa` (meses de abril a julho em Recife).",
        "3. **Derivadas termodinamicas e amplitudes intra-horarias (4)**: `orvalho_depressao`, `amp_temp_1h`, `amp_pressao_1h`, `amp_umid_1h`.",
        "4. **Tendencias meteorologicas ate t (8)**: `pressao_delta_1h`, `pressao_delta_3h`, `pressao_delta_6h`, `temp_delta_1h`, `temp_delta_3h`, `orvalho_delta_3h`, `umid_delta_1h`, `umid_delta_3h`.",
        "5. **Memoria e acumulados de chuva observada ate t (12 - apenas classificacao)**: `chuva_1h`, `chuva_3h`, `chuva_6h`, `chuva_12h`, `chuva_24h`, `chuva_max_3h`, `chuva_max_6h`, `chuva_lag_1h`, `chuva_lag_3h`, `chuva_lag_6h`, `chuva_lag_12h`, `chuva_lag_24h`.",
        "",
        "## 3. Relevancia e Correlacoes (Top 15 por importancia RF em 2019)",
        "",
        _formatar_tabela_md(top_rel),
        "",
        "Arquivo completo de correlacoes: `ml/outputs/correlacoes_features.csv`.",
        "Arquivo completo de relevancia: `ml/outputs/relevancia_features.csv`.",
        "",
        "## 4. Redundancia (|r| >= 0,70) e Decisao: Clustering vs. Classificacao",
        "",
        f"Foram identificados **{len(red)} pares** com |r| >= 0,70 (registrados em `ml/outputs/redundancia_pares.csv`). Principais pares:",
        "",
        _formatar_tabela_md(top_red),
        "",
        "### Resumo da separacao Clustering vs. Classificacao",
        "- **Clustering (K-Means, k=3, StandardScaler)**: utiliza apenas as **8 features** de estado atmosferico instantaneo e ciclos (`pressao_mb`, `temp_c`, `temp_orvalho_c`, `umid_pct`, `hora_sin`, `hora_cos`, `mes_sin`, `mes_cos`). Exclui toda variavel de chuva (evita viciar os grupos pelo alvo), exclui `orvalho_depressao` (`r = -0,9957` com `umid_pct`, evitando dupla contagem na distancia euclidiana) e exclui deltas/amplitudes redundantes.",
        "- **Classificacao Supervisionada (Random Forest)**: utiliza **33 features** (4 fisicas originais + 29 derivadas <= t), pois arvores de decisao sao invariantes a escala monotona e se beneficiam de sinais de persistencia (`chuva_*`), tendencia barometrica/termica (`*_delta_*`) e amplitude intra-horaria (`amp_*_1h`).",
        "",
        "### Perfil dos 3 Clusters Obtidos (Validacao Externa com Chuva)",
        "",
        _formatar_tabela_md(perfil),
        "",
        "## 5. Governanca e Risco de Data Leakage",
        "",
        "1. **Leakage de Alvo no Clustering**: `precip_mm`, `precip_mm_limpa` e `chuva_*` sao proibidas em `FEATURES_CLUSTERING` (verificado por `validar_sem_leakage(..., modo='clustering')`).",
        "2. **Leakage Temporal Futuro na Classificacao**: `max_fut_2h`, `acum_fut_2h`, `max_fut_3h`, `acum_fut_3h` e todas as 8 colunas `alvo_*` usam `shift(-1..-3)` e estao estritamente fora de `X` (verificado por `validar_sem_leakage(..., modo='classificacao')`).",
        "3. **Leakage de Identificadores e Metadados Temporais**: `Unnamed: 0`, `Data`, `Hora UTC`, `datetime`, `datetime_utc`, `datetime_local`, `ano`, `mes`, `dia`, `hora` ficam isolados no bloco identificador `df_id`.",
        "4. **Leakage de Treino/Teste e Hiperparametros**: importancias de RF, KMeans de comparacao e escolha de limiares usam apenas 2019 (com `TimeSeriesSplit(gap=3)` na validacao cruzada), mantendo 2020 como holdout cego.",
        "",
        "## 6. Datasets de Trabalho Atualizados",
        "",
        f"- `{resumo_json['arquivos_gerados']['working_features_csv']}`: {resumo_json['shapes']['working_features']} — identificadores separados (`datetime_utc`, `datetime_local`, `ano`), 33 features de entrada + `cluster`/`cluster_0..2`, **0 nulos** e **0 colunas de alvo/futuro/chuva bruta**.",
        f"- `{resumo_json['arquivos_gerados']['cluster_features_csv']}`: {resumo_json['shapes']['cluster_features']} — `datetime`, 8 features de clustering (padrao `kmeans_recife.ipynb`) e `cluster`, sem chuva e sem alvos.",
        f"- `{resumo_json['arquivos_gerados']['targets_csv']}`: {resumo_json['shapes']['targets']} — identificadores + precipitacao limpa + agregados futuros e 8 colunas `alvo_*` isolados em arquivo proprio.",
        "",
        "## 7. Checklist Final de Evidencias",
        "",
        "- [x] **Item 1**: `ml/features.py` e notebooks de Recife auditados; alteracoes preexistentes preservadas e CSV bruto (`dados/df_recife.csv`) intacto.",
        f"- [x] **Item 2**: {len(FEATURES_CANDIDATAS)} features candidatas consolidadas em `ml/features.py` (ciclos, sazonalidade, derivadas meteorologicas e lags/acumulados).",
        "- [x] **Item 3**: Correlacoes (Pearson/Spearman) e relevancia (RF 2019 + ANOVA F) registradas em `ml/outputs/correlacoes_features.csv` e `ml/outputs/relevancia_features.csv`.",
        f"- [x] **Item 4**: Redundancia ({len(red)} pares com |r| >= 0,70) documentada em `ml/outputs/redundancia_pares.csv` com justificativa clustering vs. classificacao.",
        "- [x] **Item 5**: Risco de leakage documentado e garantido por `validar_sem_leakage` e `separar_matrizes_modelo`.",
        "- [x] **Item 6**: Datasets de trabalho (`dados/df_recife_features.csv`, `dados/df_recife_cluster_features.csv`) e alvos (`dados/df_recife_alvos.csv`) gerados com separacao estrita.",
        "- [x] **Item 7**: Evidencias consolidadas em `ml/outputs/auditoria_features.md` e `ml/outputs/auditoria_features.json`.",
        "",
    ]
    return "\n".join(linhas)


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df = preparar_base(RAW_CSV, usar_open_meteo=False, filtrar_anos=False)
    audit = auditar_features(df)
    df_proc = audit["df_com_clusters"]

    df_redundancia = audit["redundancia_pares"].copy()
    df_redundancia["decisao_tecnica"] = [
        _classificar_decisao_redundancia(a, b)
        for a, b in zip(df_redundancia["variavel_a"], df_redundancia["variavel_b"])
    ]
    audit["redundancia_pares"] = df_redundancia

    # Separacao formal de matrizes (identificadores, X_cluster, X_classificacao, y)
    df_id_clu, X_clu, y_clu = separar_matrizes_modelo(df_proc, modo="clustering")
    df_id_cls, X_cls, y_cls = separar_matrizes_modelo(
        df_proc, alvo="3h_acum_10mm", modo="classificacao"
    )
    assert y_clu is None
    assert y_cls is not None
    assert set(df_id_cls.columns).isdisjoint(X_cls.columns)
    assert set(df_id_clu.columns).isdisjoint(X_clu.columns)
    assert set(COLUNAS_ALVO).isdisjoint(X_cls.columns)
    assert set(COLUNAS_AGREGADAS_FUTURAS).isdisjoint(X_cls.columns)

    # 1) Dataset de trabalho para modelagem (identificadores + features + cluster, SEM alvos/futuros/chuva bruta)
    colunas_cluster_onehot = ["cluster", "cluster_0", "cluster_1", "cluster_2"]
    df_working_features = pd.concat(
        [
            df_id_cls.reset_index(drop=True),
            X_cls.reset_index(drop=True),
            df_proc[colunas_cluster_onehot].reset_index(drop=True),
        ],
        axis=1,
    )
    feature_cols_only = [
        c for c in df_working_features.columns if c not in df_id_cls.columns
    ]
    validar_sem_leakage(
        [c for c in feature_cols_only if c not in colunas_cluster_onehot],
        modo="classificacao",
    )
    assert set(COLUNAS_ALVO).isdisjoint(df_working_features.columns)
    assert set(COLUNAS_AGREGADAS_FUTURAS).isdisjoint(df_working_features.columns)
    assert "precip_mm" not in df_working_features.columns
    assert "precip_mm_limpa" not in df_working_features.columns
    assert int(df_working_features.isna().sum().sum()) == 0
    df_working_features.to_csv(WORKING_FEATURES_CSV, index=False)

    # 2) Dataset de trabalho do clustering (esquema compativel com Notebooks/kmeans_recife.ipynb)
    df_cluster_out = pd.DataFrame(
        {"datetime": df_proc["datetime_utc"].dt.strftime("%Y-%m-%d %H:%M:%S")}
    )
    for col_int, col_nb in MAPA_COLUNAS_CLUSTER_NOTEBOOK.items():
        df_cluster_out[col_nb] = df_proc[col_int].to_numpy(dtype=float)
    df_cluster_out["cluster"] = df_proc["cluster"].to_numpy(dtype=int)
    assert "precip_mm" not in df_cluster_out.columns
    assert "precip_mm_limpa" not in df_cluster_out.columns
    assert set(COLUNAS_ALVO).isdisjoint(df_cluster_out.columns)
    assert int(df_cluster_out.isna().sum().sum()) == 0
    df_cluster_out.to_csv(CLUSTER_FEATURES_CSV, index=False)

    # 3) Arquivo separado de alvos e precipitacao observada (para rastreabilidade sem vazar em X)
    colunas_targets = [
        "datetime_utc",
        "datetime_local",
        "ano",
        "precip_mm",
        "precip_mm_limpa",
        *COLUNAS_AGREGADAS_FUTURAS,
        *COLUNAS_ALVO,
    ]
    df_targets = df_proc[colunas_targets].copy()
    df_targets.to_csv(TARGETS_CSV, index=False)

    # Salvar artefatos de auditoria em ml/outputs/
    audit["correlacoes_pearson"].to_csv(OUTPUT_DIR / "correlacoes_features.csv")
    audit["relevancia_features"].to_csv(
        OUTPUT_DIR / "relevancia_features.csv", index=False
    )
    df_redundancia.to_csv(OUTPUT_DIR / "redundancia_pares.csv", index=False)

    repo_root = Path(__file__).resolve().parent.parent
    resumo_json = {
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "csv_bruto": str(RAW_CSV.relative_to(repo_root)),
        "n_registros": int(len(df_proc)),
        "features_candidatas_count": len(FEATURES_CANDIDATAS),
        "features_candidatas": FEATURES_CANDIDATAS,
        "features_clustering": FEATURES_CLUSTERING,
        "features_classificacao": FEATURES_CLASSIFICACAO,
        "colunas_identificadoras": COLUNAS_IDENTIFICADORAS,
        "colunas_excluidas_redundancia_bruta": COLUNAS_EXCLUIDAS_REDUNDANCIA_BRUTA,
        "colunas_excluidas_excesso_nulos": COLUNAS_EXCLUIDAS_EXCESSO_NULOS,
        "colunas_alvo": COLUNAS_ALVO,
        "colunas_agregadas_futuras": COLUNAS_AGREGADAS_FUTURAS,
        "alvos_resumo": resumo_alvos(df_proc),
        "n_pares_redundantes_abs_r_070": int(len(df_redundancia)),
        "perfil_clusters": audit["perfil_clusters"].to_dict(orient="index"),
        "shapes": {
            "working_features": list(df_working_features.shape),
            "cluster_features": list(df_cluster_out.shape),
            "targets": list(df_targets.shape),
            "X_clustering": list(X_clu.shape),
            "X_classificacao": list(X_cls.shape),
        },
        "arquivos_gerados": {
            "working_features_csv": str(WORKING_FEATURES_CSV.relative_to(repo_root)),
            "cluster_features_csv": str(CLUSTER_FEATURES_CSV.relative_to(repo_root)),
            "targets_csv": str(TARGETS_CSV.relative_to(repo_root)),
            "correlacoes_csv": str((OUTPUT_DIR / "correlacoes_features.csv").relative_to(repo_root)),
            "relevancia_csv": str((OUTPUT_DIR / "relevancia_features.csv").relative_to(repo_root)),
            "redundancia_csv": str((OUTPUT_DIR / "redundancia_pares.csv").relative_to(repo_root)),
            "auditoria_md": str((OUTPUT_DIR / "auditoria_features.md").relative_to(repo_root)),
        },
    }

    (OUTPUT_DIR / "auditoria_features.json").write_text(
        json.dumps(resumo_json, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (OUTPUT_DIR / "auditoria_features.md").write_text(
        gerar_relatorio_markdown(df_proc, audit, resumo_json),
        encoding="utf-8",
    )

    print("=" * 78)
    print("AUDITORIA E ATUALIZACAO DO DATASET CONCLUIDAS COM SUCESSO")
    print("=" * 78)
    print(f"CSV bruto intacto:          {RAW_CSV} ({len(df_proc)} linhas)")
    print(f"Dataset features (ML):      {WORKING_FEATURES_CSV} {df_working_features.shape}")
    print(f"Dataset clustering:         {CLUSTER_FEATURES_CSV} {df_cluster_out.shape}")
    print(f"Dataset alvos (separado):   {TARGETS_CSV} {df_targets.shape}")
    print(f"Pares redundantes |r|>=0.7: {len(df_redundancia)}")
    print(f"Relatorio Markdown:         {OUTPUT_DIR / 'auditoria_features.md'}")
    print(f"Resumo JSON:                {OUTPUT_DIR / 'auditoria_features.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
