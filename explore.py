import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

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

def plot_cor_mat(X: pd.DataFrame):
    # Choose only numeric columns
    X = X.select_dtypes(include=['number'])
    X.dropna(axis=1, inplace=True)
    corr_mat = X.corr()
    fig = px.imshow(corr_mat)
    fig.show()

def plot_norm_pca_variance(X: pd.DataFrame, cummulative: bool = False):
    X = X.select_dtypes(include=['number'])
    scaler = StandardScaler()
    scaler.fit(X)
    normed = scaler.transform(X)
    normed = pd.DataFrame(normed, columns=X.columns)

    n_feats = min(normed.shape[0], normed.shape[1])
    pca = PCA(n_components=n_feats)
    pca.fit(normed)
    y = pca.explained_variance_ratio_
    if cummulative:
        for i in range(1, len(y)):
            y[i] += y[i - 1]
    px.line(x=range(1, n_feats + 1), y=y).show()

def plot_pca_feature_correlation(X: pd.DataFrame):
    X = X.select_dtypes(include=['number'])
    scaler = StandardScaler()
    scaler.fit(X)
    normed = scaler.transform(X)
    normed = pd.DataFrame(normed, columns=X.columns)
    
    n_feats = min(normed.shape[0], normed.shape[1])
    pca = PCA(n_components=n_feats)
    pca.fit(normed)

    # calculate the correlation between each feature and each principal component
    feat_norms = np.linalg.norm(normed.to_numpy(), 2.0, axis=1)[:, np.newaxis]
    row_normed = normed.to_numpy() / feat_norms
    corr_mat = row_normed @ pca.components_.transpose()
    print(corr_mat.shape)
    corr_mat = pd.DataFrame(corr_mat, columns=X.columns)
    
    px.imshow(corr_mat).show()
    




    
    # feat_norms = np.linalg.norm(normed.to_numpy(), 2.0, axis=1)[:, np.newaxis]

    # row_normed = normed.to_numpy() / feat_norms

    # print(pca.components_.shape)
    # print(row_normed.shape)

    # corr_mat = pca.components_ @ row_normed.transpose()

    # corr_mat = pd.DataFrame(corr_mat, columns=X.columns)

    # px.imshow(corr_mat).show()


    
    