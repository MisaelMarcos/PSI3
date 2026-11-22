import sys
import tempfile
import unittest
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data import ler_observacoes_csv
from features import construir_features


def base_horaria(instantes, chuva=None):
    index = pd.DatetimeIndex(instantes, tz="UTC")
    chuva = chuva if chuva is not None else [0.0] * len(index)
    return pd.DataFrame(
        {
            "datetime_utc": index,
            "datetime_local": index.tz_convert("America/Recife"),
            "ano": index.tz_convert("America/Recife").year,
            "precip_mm": chuva,
            "precip_mm_limpa": pd.Series(chuva).fillna(0.0).to_numpy(),
            "pressao_mb": range(1000, 1000 + len(index)),
            "temp_c": range(20, 20 + len(index)),
            "temp_orvalho_c": range(18, 18 + len(index)),
            "umid_pct": range(70, 70 + len(index)),
        }
    )


class TestFeaturesTemporais(unittest.TestCase):
    def test_lacuna_nao_conta_como_hora_adjacente_nem_rotulo_valido(self):
        df = construir_features(
            base_horaria(
                ["2020-01-01 00:00", "2020-01-01 01:00", "2020-01-01 05:00"]
            )
        ).set_index("datetime_utc")

        self.assertEqual(df.loc["2020-01-01 05:00+00:00", "chuva_lag_1h"], 0)
        self.assertEqual(df.loc["2020-01-01 05:00+00:00", "pressao_delta_1h"], 0)
        self.assertTrue(pd.isna(df.loc["2020-01-01 01:00+00:00", "alvo_2h_5mm"]))

    def test_precipitacao_futura_ausente_invalida_o_alvo(self):
        df = construir_features(
            base_horaria(
                pd.date_range("2020-01-01", periods=4, freq="h"),
                [0.0, None, 10.0, 0.0],
            )
        )
        self.assertTrue(pd.isna(df.loc[0, "alvo_2h_5mm"]))
        self.assertTrue(pd.isna(df.loc[0, "max_fut_2h"]))
        self.assertTrue(pd.isna(df.loc[0, "acum_fut_2h"]))

    def test_agregados_futuros_sao_atribuidos_as_linhas_corretas(self):
        df = construir_features(
            base_horaria(
                pd.date_range("2020-01-01", periods=4, freq="h"),
                [0.0, 7.0, 1.0, 0.0],
            )
        )
        self.assertEqual(df.loc[0, "max_fut_2h"], 7.0)
        self.assertEqual(df.loc[0, "acum_fut_2h"], 8.0)
        self.assertEqual(df.loc[0, "alvo_2h_5mm"], 1.0)

    def test_janela_nao_cruza_ano_local(self):
        df = construir_features(
            base_horaria(
                pd.date_range("2020-01-01 01:00", periods=4, freq="h"),
                [0.0, 0.0, 10.0, 0.0],
            )
        )
        self.assertEqual(df.loc[1, "ano"], 2019)
        self.assertEqual(df.loc[2, "ano"], 2020)
        self.assertTrue(pd.isna(df.loc[1, "alvo_2h_5mm"]))

    def test_imputacao_ao_vivo_nao_usa_observacao_futura(self):
        with tempfile.TemporaryDirectory() as pasta:
            caminho = Path(pasta) / "observacoes.csv"
            pd.DataFrame(
                {
                    "datetime": pd.date_range("2020-01-01", periods=3, freq="h"),
                    "temp_c": [None, 20.0, 30.0],
                }
            ).to_csv(caminho, index=False)
            df = ler_observacoes_csv(caminho)
        self.assertEqual(df.loc[0, "temp_c"], 0.0)
        self.assertEqual(df.loc[1, "temp_c"], 20.0)


if __name__ == "__main__":
    unittest.main()
