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
    pc_total_feats = [len(pc) for pc in pc_feats]
    pc_split_feats = [[] for _ in range(len(pc_feats))]
    for i, pc in enumerate(pc_feats):    
        temp = []
        for feat in pc:
            feat = feat.replace('FS_', '')
            subwords = feat.split('_')
            for sw in subwords:
                temp.append(sw)
        pc_split_feats[i] = temp

    # plot word frequency
    pc1 = pc_split_feats[pc_idx]
    word_freq = {}
    for word in pc1:
        if word in word_freq:
            word_freq[word] += 1
        else:
            word_freq[word] = 1

    word_freq = pd.DataFrame.from_dict(word_freq, orient='index', columns=['count'])
    word_freq['perc'] = word_freq['count'] / pc_total_feats[pc_idx]
    word_freq = word_freq.sort_values(by='perc', ascending=False)
    px.bar(word_freq['perc']).show()


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