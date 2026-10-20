"""Check da lógica de clusterização da página 9 (Agrupamento).

Execução (sklearn 1.3.2 não tem wheel para py3.13; usar 3.11):
    cd <raiz do projeto>
    uv run --no-project --python 3.11 --with scikit-learn==1.3.2 --with pandas --with plotly==5.18.0 python test_page9_clustering.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.resolve()))

import types

# px.scatter é stubado: com pandas novo + plotly 5.18, o matplotlib interno do plotly
# quebra no scatter (verificado também com a página ORIGINAL do git, neste mesmo ambiente);
# o alvo do check é a lógica de clusterização, não o runtime visual.
import plotly.express as _px
_px.scatter = lambda *a, **k: None


# --- stub mínimo do streamlit, antes de o módulo da página importar
class _Col:
    def __init__(self, select_values=None): self.select_values = select_values
    def multiselect(self, *a, **k): return ['idade', 'tarifa']
    def slider(self, *a, **k): return k.get('value', 0.5)
    def selectbox(self, *a, **k): return self.select_values or 'KMeans'
    def write(self, *a, **k): pass
    def dataframe(self, *a, **k): pass
    def plotly_chart(self, *a, **k): pass

class _Expander:
    def __enter__(self): return self
    def __exit__(self, *a): return False
    def write(self, *a, **k): pass
    def dataframe(self, *a, **k): pass
    def columns(self, *a, **k):
        span = a[0] if isinstance(a[0], int) else (len(a[0]) if isinstance(a[0], (list, tuple)) else len(a))
        return [_Col() for _ in range(span)]

fake = types.ModuleType('streamlit')
fake.selectbox = lambda *a, **k: 'KMeans'
fake.multiselect = lambda *a, **k: ['idade', 'tarifa']
fake.slider = lambda *a, **k: k.get('value', 0.5)
fake.columns = lambda *a, **k: [_Col() for _ in range(a[0] if isinstance(a[0], int) else (len(a[0]) if isinstance(a[0], (list, tuple)) else len(a)))]
fake.expander = lambda *a, **k: _Expander()
for name in ['error', 'title', 'header', 'subheader', 'write', 'dataframe', 'plotly_chart', 'markdown', 'text', 'warning', 'info', 'success', 'button', 'set_page_config', 'pyplot', 'table']:
    setattr(fake, name, lambda *a, **k: None)
sys.modules['streamlit'] = fake

from sklearn.preprocessing import StandardScaler, MinMaxScaler

import importlib.util
spec = importlib.util.spec_from_file_location('page9', 'pages/9_🫧_Agrupamento.py')
page = importlib.util.module_from_spec(spec)
spec.loader.exec_module(page)

# --- 1. KMeans: deve gerar os 3 clusters conforme slider/hardcode do build
page.clustering_cols = ['idade', 'tarifa']
page.algorithm = 'KMeans'
page.n_clusters = 3
enc = page.create_df_encoded(page.create_df_raw(['id', 'idade', 'tarifa', 'sobreviveu_val', 'classe_val', 'sexo']).dropna())
labels_default = page.clusterize(enc)
assert len(set(labels_default)) == 3, f'KMeans deveria gerar 3 clusters, gerou {set(labels_default)}'
labels_std = page.clusterize(enc, StandardScaler())
assert 0 < len(set(labels_std)) <= 3

# --- 2. DBSCAN: com eps minúsculo, pontos de ruído (-1) são esperados
page.algorithm = 'DBSCAN'
page.eps = 0.01
page.min_samples = 5
labels_db = page.clusterize(enc, StandardScaler())
noise = sum(1 for x in labels_db if x == -1)
assert noise > 0, 'DBSCAN com eps=0.01 deveria gerar pontos de ruído (-1)'

# --- 3. Hierarchical: número de clusters deve casar com o parâmetro
page.algorithm = 'Hierarquico (Agglomerative)'
page.n_clusters = 4
labels_h = page.clusterize(enc)
assert len(set(labels_h)) == 4, f'Hierarquico deveria gerar 4 clusters, gerou {set(labels_h)}'

# --- 4. silhouette: 3 scores (0..1) para KMeans; vazio p/ DBSCAN; guard para degenerados
page.algorithm = 'KMeans'
page.n_clusters = 3
clusters_df = page.create_df_clusters(enc)
scores = page.compute_scores(enc, clusters_df)
assert set(scores.keys()) == {'cluster', 'cluster_standard', 'cluster_minmax'}, scores
assert all(0.0 <= s <= 1.0 for s in scores.values()), scores

page.algorithm = 'DBSCAN'
assert page.compute_scores(enc, clusters_df) == {}

page.algorithm = 'KMeans'
degenerated = clusters_df.copy()
degenerated['cluster'] = 0
degenerated['cluster_standard'] = 0
degenerated['cluster_minmax'] = 0
assert page.compute_scores(enc, degenerated) == {}

# --- 5. cotovelo: roda sem exceção (k=2..10, 3 variantes de normalização)
page.algorithm = 'KMeans'
page.plot_elbow(enc)

print('OK: clusterize (3 algoritmos), silhouette com guard, e cotovelo funcionando.')
