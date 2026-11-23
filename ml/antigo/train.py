"""Treina, compara com/sem KMeans e salva os modelos de previsao de chuva.

Uso (a partir de src/backend/ml):
    .venv\\Scripts\\python.exe train.py
    .venv\\Scripts\\python.exe train.py --alvos 3h_10mm --quick
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from config import (
    ALVO_PRINCIPAL,
    ALVOS,
    METADATA_PATH,
    MODELS_DIR,
    MODEL_VERSION,
    OUTPUT_DIR,
    RAW_CSV,
)
from data import resumo_base
from features import (
    features_kmeans_disponiveis,
    features_modelo,
    preparar_base,
    resumo_alvos,
)
from modeling import (
    avaliar_cv_2019,
    avaliar_holdout_2020,
    importancia_features,
    salvar_modelo,
    treinar_kmeans,
    treinar_modelo_final,
)


def _json_dump(obj, caminho: Path) -> None:
    caminho.parent.mkdir(parents=True, exist_ok=True)
    caminho.write_text(
        json.dumps(obj, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _fmt(valor, casas: int = 3) -> str:
    if valor is None:
        return "n/d"
    return f"{valor:.{casas}f}"


def _resumo_metricas(metricas: dict) -> str:
    eventos = metricas.get("eventos", {})
    return (
        f"PR-AUC={_fmt(metricas.get('pr_auc'))} | "
        f"precisao={_fmt(metricas.get('precision'))} | "
        f"recall={_fmt(metricas.get('recall'))} | "
        f"F1={_fmt(metricas.get('f1'))} | "
        f"eventos={eventos.get('eventos_com_alerta', 'n/d')}/"
        f"{eventos.get('eventos', 'n/d')}"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Treino do modelo de chuva.")
    parser.add_argument("--csv", type=Path, default=RAW_CSV)
    parser.add_argument(
        "--alvos",
        default=",".join(ALVOS.keys()),
        help="Lista separada por virgula. Ex.: 3h_10mm,3h_5mm",
    )
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--n-splits", type=int, default=4)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Menos arvores/folds para testes rapidos.",
    )
    args = parser.parse_args()

    if args.quick:
        args.n_estimators = min(args.n_estimators, 120)
        args.n_splits = min(args.n_splits, 3)

    alvos = [a.strip() for a in args.alvos.split(",") if a.strip()]
    desconhecidos = [a for a in alvos if a not in ALVOS]
    if desconhecidos:
        raise SystemExit(f"Alvos desconhecidos: {desconhecidos}")

    inicio = time.time()
    print("=" * 78)
    print("PREVISAO DE CHUVA RECIFE - INMET A301")
    print("=" * 78)
    df = preparar_base(args.csv)
    features = features_modelo(df)
    features_kmeans = features_kmeans_disponiveis(df)
    print(f"Base: {len(df)} linhas horarias (2019-2020)")
    print(f"Features do modelo: {len(features)} (KMeans: {len(features_kmeans)})")
    print(json.dumps(resumo_base(df), ensure_ascii=False, indent=2))
    print("Alvos:")
    print(json.dumps(resumo_alvos(df), ensure_ascii=False, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # KMeans usado na comparacao com/sem cluster (treinado apenas em 2019).
    base_2019 = df[df["ano"] == 2019]
    print("\nTreinando KMeans de comparacao (2019)...")
    kmeans_2019, info_kmeans_2019 = treinar_kmeans(base_2019[features_kmeans])
    print(
        f"KMeans 2019: k={info_kmeans_2019['k']} "
        f"silhouette={info_kmeans_2019['silhouette']:.3f}"
    )

    modelos_meta = {}
    for alvo in alvos:
        info_alvo = ALVOS[alvo]
        print("\n" + "-" * 78)
        print(f"ALVO {alvo}: {info_alvo['descricao']}")
        print("-" * 78)

        cv_base = avaliar_cv_2019(
            df,
            alvo,
            features,
            usar_cluster=False,
            features_kmeans=features_kmeans,
            n_splits=args.n_splits,
            n_estimators=args.n_estimators,
        )
        cv_cluster = avaliar_cv_2019(
            df,
            alvo,
            features,
            usar_cluster=True,
            kmeans_2019=kmeans_2019,
            features_kmeans=features_kmeans,
            n_splits=args.n_splits,
            n_estimators=args.n_estimators,
        )

        # Comparacao honesta: escolhe pelo PR-AUC fora da amostra em 2019.
        # Em caso de empate muito proximo, prefere o modelo sem cluster.
        usar_cluster = cv_cluster["pr_auc"] > cv_base["pr_auc"] + 0.001
        escolhido = cv_cluster if usar_cluster else cv_base
        limiar = float(escolhido["corte"]["limiar"])
        limiar_forte = float(escolhido["corte_forte"]["limiar"])

        print(
            f"CV 2019 sem KMeans: PR-AUC={cv_base['pr_auc']:.3f} "
            f"F1={cv_base['metricas_oof']['f1']:.3f} "
            f"limiar={cv_base['corte']['limiar']:.3f}"
        )
        print(
            f"CV 2019 com KMeans: PR-AUC={cv_cluster['pr_auc']:.3f} "
            f"F1={cv_cluster['metricas_oof']['f1']:.3f} "
            f"limiar={cv_cluster['corte']['limiar']:.3f}"
        )
        print(
            f"Escolhido: {'com KMeans' if usar_cluster else 'sem KMeans'} | "
            f"atencao={limiar:.3f} | forte={limiar_forte:.3f}"
        )

        holdout = avaliar_holdout_2020(
            df,
            alvo,
            features,
            usar_cluster=usar_cluster,
            kmeans_2019=kmeans_2019 if usar_cluster else None,
            limiar=limiar,
            limiar_forte=limiar_forte,
            features_kmeans=features_kmeans,
            n_estimators=max(args.n_estimators, 250),
        )
        print(f"Holdout 2020: {_resumo_metricas(holdout['metricas_teste'])}")

        final = treinar_modelo_final(
            df,
            alvo,
            features,
            usar_cluster=usar_cluster,
            features_kmeans=features_kmeans,
            n_estimators=max(args.n_estimators, 300),
        )
        arquivo = f"modelo_{alvo}.joblib"
        pacote = {
            "versao": MODEL_VERSION,
            "alvo": alvo,
            "horizonte_h": info_alvo["horizonte_h"],
            "limiar_mm": info_alvo["limiar_mm"],
            "descricao": info_alvo["descricao"],
            "modelo": final["modelo"],
            "features": final["features"],
            "features_base": features,
            "features_kmeans": final["features_kmeans"],
            "usar_cluster": final["usar_cluster"],
            "kmeans": final["kmeans"],
            "limiar_probabilidade": limiar,
            "limiar_forte": limiar_forte,
            "n_treino": final["n_treino"],
            "periodo_treino": final["periodo_treino"],
            "metricas_oof_2019": escolhido["metricas_oof"],
            "metricas_oof_2019_forte": escolhido["metricas_oof_forte"],
            "metricas_holdout_2020": holdout["metricas_teste"],
            "metricas_holdout_2020_forte": holdout["metricas_teste_forte"],
        }
        salvar_modelo(pacote, MODELS_DIR / arquivo)

        modelos_meta[alvo] = {
            "arquivo": arquivo,
            "horizonte_h": info_alvo["horizonte_h"],
            "limiar_mm": info_alvo["limiar_mm"],
            "descricao": info_alvo["descricao"],
            "usar_cluster": final["usar_cluster"],
            "limiar_probabilidade": limiar,
            "limiar_forte": limiar_forte,
            "n_treino": final["n_treino"],
            "features": final["features"],
            "features_base": features,
            "features_kmeans": final["features_kmeans"],
            "kmeans_final": final["info_kmeans"],
            "comparacao_kmeans": {
                "sem_cluster": {
                    "pr_auc_oof": cv_base["pr_auc"],
                    "roc_auc_oof": cv_base["roc_auc"],
                    "limiar": cv_base["corte"]["limiar"],
                    "metricas_oof": cv_base["metricas_oof"],
                },
                "com_cluster": {
                    "pr_auc_oof": cv_cluster["pr_auc"],
                    "roc_auc_oof": cv_cluster["roc_auc"],
                    "limiar": cv_cluster["corte"]["limiar"],
                    "metricas_oof": cv_cluster["metricas_oof"],
                },
            },
            "metricas_oof_2019": escolhido["metricas_oof"],
            "metricas_oof_2019_forte": escolhido["metricas_oof_forte"],
            "metricas_holdout_2020": holdout["metricas_teste"],
            "metricas_holdout_2020_forte": holdout["metricas_teste_forte"],
            "importancia_features": importancia_features(
                final["modelo"], final["features"]
            )[:15],
        }

    metadata = {
        "versao": MODEL_VERSION,
        "gerado_em": datetime.now(timezone.utc).isoformat(),
        "fonte": "INMET - estacao automatica A301 (Recife)",
        "csv_treino": str(RAW_CSV.relative_to(RAW_CSV.parents[2])),
        "periodo": "2019-2020",
        "alvo_principal": ALVO_PRINCIPAL,
        "features": features,
        "features_kmeans": features_kmeans,
        "open_meteo_disponivel": "om_cloud" in df.columns,
        "resumo_base": resumo_base(df),
        "resumo_alvos": resumo_alvos(df),
        "kmeans_comparacao_2019": info_kmeans_2019,
        "modelos": modelos_meta,
    }
    _json_dump(metadata, METADATA_PATH)
    _json_dump(metadata, OUTPUT_DIR / "metricas.json")

    linhas = [
        "# Resultados do modelo de chuva",
        "",
        f"Gerado em: {metadata['gerado_em']}",
        f"Fonte: {metadata['fonte']}",
        f"Periodo: {metadata['periodo']}",
        "",
        "## Comparacao KMeans (validacao temporal em 2019)",
        "",
        f"- k escolhido: {info_kmeans_2019['k']}",
        f"- silhouette: {info_kmeans_2019['silhouette']:.3f}",
        "",
        "## Metricas por alvo",
        "",
        "| Alvo | KMeans | PR-AUC OOF | Limiar atencao | P | R | F1 | Limiar forte | P forte | R forte | Eventos (2020) | 2h antes | 3h antes |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for alvo, meta in modelos_meta.items():
        m = meta["metricas_holdout_2020"]
        mf = meta["metricas_holdout_2020_forte"]
        ev = m.get("eventos", {})
        linhas.append(
            f"| {alvo} | {'sim' if meta['usar_cluster'] else 'nao'} | "
            f"{_fmt(meta['metricas_oof_2019'].get('pr_auc'))} | "
            f"{meta['limiar_probabilidade']:.3f} | "
            f"{_fmt(m.get('precision'))} | {_fmt(m.get('recall'))} | "
            f"{_fmt(m.get('f1'))} | "
            f"{meta['limiar_forte']:.3f} | "
            f"{_fmt(mf.get('precision'))} | {_fmt(mf.get('recall'))} | "
            f"{ev.get('eventos_com_alerta', 'n/d')}/{ev.get('eventos', 'n/d')} | "
            f"{ev.get('eventos_com_2h_antecedencia', 'n/d')} | "
            f"{ev.get('eventos_com_3h_antecedencia', 'n/d')} |"
        )
    linhas.extend(
        [
            "",
            "## Leitura",
            "",
            "- `PR-AUC OOF` e a area sob a curva precisao-recall na validacao",
            "  temporal dentro de 2019, com gap de 3 horas.",
            "- As metricas de precisao/recall/F1/eventos sao do holdout de 2020,",
            "  usando os limiares escolhidos no OOF de 2019.",
            "- `Limiar atencao` prioriza recall >= 0,80; `Limiar forte` maximiza F1.",
            "- `2h antes` e `3h antes` contam eventos com alerta nesses horizontes.",
            "- O modelo final salvo em `models/` e retreinado com 2019+2020.",
            "",
        ]
    )
    (OUTPUT_DIR / "metricas.md").write_text(
        "\n".join(linhas), encoding="utf-8"
    )

    print("\n" + "=" * 78)
    print(f"Modelos salvos em: {MODELS_DIR}")
    print(f"Metadata: {METADATA_PATH}")
    print(f"Relatorio: {OUTPUT_DIR / 'metricas.md'}")
    print(f"Tempo total: {time.time() - inicio:.1f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
