import pandas as pd
import numpy as np
from sklearn.decomposition import PCA

import plotly.express as px

def plot_pca(X, y):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df['y'] = y
    fig = px.scatter(df, x='PC1', y='PC2', color='y')
    fig.show()

def plot_pca_variance(X: pd.DataFrame):
    n_feats = min(X.shape[0], X.shape[1])
    pca = PCA(n_components=n_feats)
    pca.fit(X)
    px.bar(x=range(1, n_feats + 1), y=pca.explained_variance_ratio_).show()