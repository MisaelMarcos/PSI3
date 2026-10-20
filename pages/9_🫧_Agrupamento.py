import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pages.util.plot_pages_util import read_titanic_df
from sklearn.base import TransformerMixin
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler

color_scale = ['#00ccff','#cc00ff','#ffcc00','#0066bb','#6600bb','#bb0066','#bb6600','#ff0066','#66ff66','#ee0503']
algorithm = 'KMeans'
algorithms = ['KMeans', 'DBSCAN', 'Hierarquico (Agglomerative)']
n_clusters = 3
eps = 0.5
min_samples = 5
cluster_variants = {
    'cluster': ('Cluster Sem Normalização', None),
    'cluster_standard': ('Cluster Com Normalização Padrão', StandardScaler),
    'cluster_minmax': ('Cluster Com Normalização MinMax', MinMaxScaler),
}
clustering_cols_opts = ['idade','tarifa','classe','sexo','sobreviveu']
clustering_cols = clustering_cols_opts.copy()

def build_page():
    build_header()
    build_body()

def build_header():
    st.write('<h1>Agrupamento (<i>Clustering</i>) com a base do Titanic</h1>', unsafe_allow_html=True)
    st.write('''<i>Ao utilizar apenas as colunas de idade e tarifa na clusterização com 3 grupos, 
a normalização padrão (que considera o desvio padrão dos valores em torno da média) pareceu 
fornecer grupos mais coesos (https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html). 
Veja os grupos formados e observe que sem a normalização, o KMeans considerou muito fortemente a tarifa, 
enquanto que com a normalizaç0ão minmax, o resultado considera muito fortemente a idade. Por outro lado,
a normalização padrão pareceu levar em consideração ambas colunas com pesos equivalentes.
</i>''', unsafe_allow_html=True)
    st.write('''<i>A página agora permite também escolher o algoritmo (KMeans, DBSCAN ou Hierárquico) e
possui gráfico de cotovelo (inertia) e coeficiente de silhouette para complementar a avaliação.
</i>''', unsafe_allow_html=True)

def build_body():
    global algorithm, n_clusters, clustering_cols, eps, min_samples
    algorithm = st.selectbox('Algoritmo', options=algorithms, key='algoritmo')
    c1, c2 = st.columns(2)
    clustering_cols = c1.multiselect('Colunas', options=clustering_cols_opts,  default=clustering_cols_opts[0:2])
    if len(clustering_cols) < 2:
        st.error('É preciso selecionar pelo menos 2 colunas.')
        return
    if algorithm == 'DBSCAN':
        eps = c2.slider('EPS', min_value=0.1, max_value=2.0, value=0.5)
        min_samples = c2.slider('Amostras mínimas *(MinPts)*', min_value=2, max_value=10, value=5)
    else:
        n_clusters = c2.slider('Quantidade de Clusters', min_value=2, max_value=10, value=3)
    dfs, df_enc = create_dfs()
    for df, title, desc in dfs.values():
        plot_dataframe(df, title, desc)
    if algorithm == 'KMeans':
        plot_elbow(df_enc)
    scores = compute_scores(df_enc, dfs['df_clusters'][0])
    for col, (name, _) in cluster_variants.items():
        plot_cluster(dfs['df_clusters'][0], col, name, scores.get(col))

def create_dfs():
    cols = ['id','idade','tarifa','sobreviveu_val','classe_val','sexo']
    df_raw = create_df_raw(cols)
    df_clean = df_raw.dropna()
    df_enc = create_df_encoded(df_clean)
    df_clusters = create_df_clusters(df_enc)
    return {
        'df_raw': (df_raw, 'Dataframe Original', 'Dataframe original do Titanic, com um subconjunto de colunas utilizados para o agrupamento (clusterização).'),
        'df_clean': (df_clean, 'Dataframe Sem Nulos', 'Dataframe após a remoção dos registros que possuíam valores nulos nas colunas selecionadas.'),
        'df_encoded': (df_enc, 'Dataframe Codificado', 'Dataframe após a códificação (encoding) das colunas categóricas, a fim de ser utilizado pelo algoritmo de clusterização selecionado (KMeans, DBSCAN ou Hierárquico).'),
        'df_clusters': (df_clusters, 'Clusterização do Dataframe Codificado e com Normalizações', f'Dataframe após a execução do algoritmo {algorithm} sob diferentes normalizações nas colunas de idade e tarifa, a fim de mitigar o impacto de diferentes escalas de medida na clusterização.'),
    }, df_enc
    
def create_df_raw(cols: list[str]):
    df_raw = read_titanic_df()
    df_raw = df_raw[cols].copy()
    df_raw.rename(columns={'classe_val':'classe',
                           'sobreviveu_val':'sobreviveu'}, inplace=True)
    return df_raw

def create_df_encoded(df: pd.DataFrame) -> pd.DataFrame:
    df_enc = df.copy()
    lenc = LabelEncoder()
    df_enc['sexo'] = lenc.fit_transform(df_enc['sexo'])
    return df_enc

def create_df_clusters(df_enc:pd.DataFrame) -> pd.DataFrame:
    #Colunas de id não devem ser utilizas para a clusterização, uma vez que não possuem sentido semântico.
    #Caso sejam incluídas, elas irão interferir nos elementos incluídos em cada cluster.
    df_clusters = df_enc.copy()
    df_clusters['cluster'] = clusterize(df_enc)
    df_clusters['cluster_standard'] = clusterize(df_enc, StandardScaler())
    df_clusters['cluster_minmax'] = clusterize(df_enc, MinMaxScaler())
    return df_clusters

def clusterize(df: pd.DataFrame, scaler:TransformerMixin=None) -> pd.DataFrame:
    df_result = df[clustering_cols].copy()
    if scaler is not None:
        df_result = scale(df_result, scaler)
    X = df_result.values
    # O random_state seta o valor inicial do centroid dos clusters.
    # Ele é configurado para que os mesmos valores sejam atribuidos aos clusters a cada execução.
    # Teste com o valor 4294967295 e 12345 e veja que, apesar de todos os registros permanecerem 
    # nos mesmos clusters, o número atribuído a cada cluster é alterado.
    if algorithm == 'DBSCAN':
        # Referência: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html
        clusterer = DBSCAN(eps=eps, min_samples=min_samples)
    elif algorithm == 'Hierarquico (Agglomerative)':
        # Referência: https://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html
        clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    else:
        clusterer = KMeans(n_clusters=n_clusters, n_init=10, random_state=4294967295)
    return clusterer.fit_predict(X)

def scaled_variant(df: pd.DataFrame, scaler_class):
    df_variant = df[clustering_cols].copy()
    if scaler_class is not None:
        df_variant = scale(df_variant, scaler_class())
    return df_variant.values

def compute_scores(df_enc: pd.DataFrame, df_clusters: pd.DataFrame) -> dict:
    # silhouette é calculado sobre os dados escalados de cada normalização
    # (notar que os scores das variantes não são comparáveis entre si).
    # DBSCAN fica sem silhouette: com pontos de ruído, o score é enganoso.
    if algorithm == 'DBSCAN':
        return {}
    scores = {}
    for col, (_, scaler_class) in cluster_variants.items():
        labels = df_clusters[col].values
        if 1 < len(set(labels)) < len(labels):
            scores[col] = silhouette_score(scaled_variant(df_enc, scaler_class), labels)
    return scores

def plot_elbow(df_enc: pd.DataFrame):
    ks = list(range(2, 11))
    fig = go.Figure()
    for name, (_, scaler_class) in cluster_variants.items():
        X = scaled_variant(df_enc, scaler_class)
        inertias = [KMeans(n_clusters=k, n_init=10, random_state=4294967295).fit(X).inertia_ for k in ks]
        fig.add_trace(go.Scatter(x=ks, y=inertias, mode='lines+markers', name=name))
    fig.update_layout(title='Cotovelo — Inertia por Quantidade de Clusters',
                      xaxis_title='Quantidade de Clusters (k)', yaxis_title='Inertia')
    st.plotly_chart(fig, use_container_width=True)

def scale(df:pd.DataFrame, scaler:TransformerMixin):
    scaling_cols = [x for x in ['idade','tarifa'] if x in clustering_cols]
    for c in scaling_cols:
        vals = df[[c]].values
        df[c] = scaler.fit_transform(vals)
    return df

def plot_dataframe(df, title, desc):
    with st.expander(title):
        st.write(f'<i>{desc}</i>', unsafe_allow_html=True)
        c1, _, c2 = st.columns([.5,.05,.45])
        c1.write('<h3>Dados*</h3>', unsafe_allow_html=True)
        c1.dataframe(df, use_container_width=True)
        c2.write('<h3>Descrição</h3>', unsafe_allow_html=True)
        c2.dataframe(df.describe())

def plot_cluster(df:pd.DataFrame, cluster_col:str, cluster_name:str, score:float=None):
    df.sort_values(by=[cluster_col,'id'], inplace=True)
    df[cluster_col] = df[cluster_col].apply(lambda x: 'Ruído' if x == -1 else f'Cluster {x}')
    df_cluster_desc = df[['id',cluster_col]].copy().groupby(by=cluster_col).count()
    df_cluster_desc.rename(columns={'id':'qtd'}, inplace=True)
    expander = st.expander(cluster_name)
    if score is not None:
        expander.write(f'Coeficiente de silhouette: {score:.4f}', unsafe_allow_html=True)
    expander.dataframe(df_cluster_desc)
    cols = expander.columns(len(clustering_cols))
    for c1 in clustering_cols:
        for cidx, c2 in enumerate(clustering_cols):
            fig = px.scatter(df, x=c1, y=c2, color=cluster_col, color_discrete_sequence=color_scale)
            cols[cidx].plotly_chart(fig, use_container_width=True)

build_page()