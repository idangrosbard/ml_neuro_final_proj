import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from typing import List

import plotly.express as px

def feat_pca_projection(X: pd.DataFrame, n_feats: int = 2) -> List[List[str]]:
    X = X.select_dtypes(include=['number'])
    scaler = StandardScaler()
    scaler.fit(X)
    normed = scaler.transform(X)
    normed[np.isnan(normed)] = 0
    
    normed = pd.DataFrame(normed, columns=X.columns)
    
    pca = PCA(n_components=n_feats)
    pca.fit(normed)

    # calculate the correlation between each feature and each principal component
    projected = pca.transform(normed)

    corr_mat = np.corrcoef(normed.to_numpy().transpose(), projected.transpose())
    corr_mat = corr_mat[:normed.shape[1], normed.shape[1]:].transpose()

    # corr_mat = projected_normed.transpose() @ feat_normed
    explained_var_mat = corr_mat ** 2
    
    # Choose highest corr PC
    pc = explained_var_mat.max(axis=0)[:, np.newaxis]
    pc_idx = explained_var_mat.argmax(axis=0)
    
    pc_feats = [[] for _ in range(n_feats)]
    for curr_pc, col in zip(pc_idx, normed.columns):
        pc_feats[curr_pc].append(col)

    corr_mat = pd.DataFrame((explained_var_mat >= pc.transpose()).astype(int) * explained_var_mat, index=[i+1 for i in range(n_feats)], columns=X.columns).transpose()
    
    corr_mat = corr_mat.sort_values(by=[i+1 for i in range(n_feats)], ascending=False).transpose()

    explained_order = corr_mat.columns.to_list()

    explained_variance = pd.DataFrame(explained_var_mat, index=[i+1 for i in range(n_feats)], columns=X.columns)[explained_order]
    
    px.imshow(explained_variance).show()
    px.imshow(corr_mat).show()
    return pc_feats


def plot_pc_feats_freq(pc_feats: List[str], pc_idx: int) -> None:

    df = {'word': [], 'sw_idx': [], 'count': []}
    curr_pc = pc_feats[pc_idx]
    total_feats = len(curr_pc)
    for feat in curr_pc:
        feat = feat.replace('FS_', '')
        subwords = feat.split('_')
        for j, sw in enumerate(subwords):
            df['word'].append(sw)
            df['sw_idx'].append(j)
            df['count'].append(1)
    
    summary = pd.DataFrame(df)
    summary = summary.groupby(['word', 'sw_idx']).count()
    summary['freq'] = summary['count'] / total_feats
    
    summary = summary.sort_values(by='freq', ascending=False).reset_index()
    px.bar(summary, y='freq', x='word', color='sw_idx').show()


def pca_reduce(X: pd.DataFrame, n_feats: int = 2) -> np.ndarray:
    X = X.select_dtypes(include=['number'])
    scaler = StandardScaler()
    scaler.fit(X)
    normed = scaler.transform(X)
    normed[np.isnan(normed)] = 0
    
    normed = pd.DataFrame(normed, columns=X.columns)
    
    pca = PCA(n_components=n_feats)
    pca.fit(normed)

    # calculate the correlation between each feature and each principal component
    projected = pca.transform(normed)

    return projected