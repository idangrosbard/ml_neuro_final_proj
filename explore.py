import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import plotly.express as px

def plot_pca(X: pd.DataFrame, y: pd.Series):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # dummy coding
    y = pd.get_dummies(y).iloc[:, 0]
    df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df['y'] = y.values
    fig = px.scatter(df, x='PC1', y='PC2', color='y')
    fig.show()

def plot_pca_variance(X: pd.DataFrame):
    n_feats = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_feats)
    pca.fit(X)
    px.bar(x=range(1, n_feats + 1), y=pca.explained_variance_ratio_).show()

def plot_tsne(X: pd.DataFrame, y: pd.Series):
    tsne = TSNE(n_components=2)
    X_tsne = tsne.fit_transform(X)
    # dummy coding
    y = pd.get_dummies(y).iloc[:, 0]
    df = pd.DataFrame(X_tsne, columns=['tSNE1', 'tSNE2'])
    df['y'] = y.values
    fig = px.scatter(df, x='tSNE1', y='tSNE2', color='y')
    fig.show()