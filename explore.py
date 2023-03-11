import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from collections import Counter
import matplotlib.pyplot as plt
import plotly.express as px

def plot_pca(X: pd.DataFrame, y: pd.Series, df_name: str):
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    # dummy coding
    y = pd.get_dummies(y).iloc[:, 0]
    df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])
    df['y'] = y.values
    fig = px.scatter(df, x='PC1', y='PC2', color='y', title= f'{df_name} ')
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
    


def show_data_balance(data):

    data_keys = Counter(data).keys()  # equals to list(set(words))
    data_values = Counter(data).values()  # counts the elements' frequency

    for k,v in zip(data_keys,data_values):
        percentage = np.round(v/len(data)*100, decimals=2)
        print (f'{percentage}% {k}')
    # feat_norms = np.linalg.norm(normed.to_numpy(), 2.0, axis=1)[:, np.newaxis]

    # row_normed = normed.to_numpy() / feat_norms

    # print(pca.components_.shape)
    # print(row_normed.shape)

    # corr_mat = pca.components_ @ row_normed.transpose()

    # corr_mat = pd.DataFrame(corr_mat, columns=X.columns)

    # px.imshow(corr_mat).show()



COLORS = "red", "green", "blue"
PARAM_COLUMNS = [
    "param_max_depth", "param_n_estimators", "mean_train_score",
    "mean_test_score", "std_train_score", "std_test_score"
]
PARAM_COLUMN_NAMES = [
    "Max Depth", "Number of Estimators", "Mean Train Score", "Mean Test Score",
    "Train Score STD", "Test Score STD"
]


def identify_leakage(df, col):
    """Identify leakage in a dataframe, i.e. features that are equal (or highly correlated) to target
    Args:
        df (pd.DataFrame): The dataframe to check.
        col (str): The column to check for leakage.
    """
    # Add dummy coding for Gender
    df['F'] = (df[col] == 'F')

    # calculate correlation matrix
    corr = df.corr()
    px.bar(corr['F'].sort_values(ascending=False)).show()


def explore_categorical(df):
    for col in df.columns:
        num_uniques = len(set(df[col]))
        if num_uniques >10:
            print( f'{col}: {num_uniques}')
        else:
            print (f'{col}: {set(df[col])}')

def plot_search_results(model_searcher, N_ESTIMATORS):
    data = pd.DataFrame(model_searcher.cv_results_)[PARAM_COLUMNS].copy()
    data.columns = PARAM_COLUMN_NAMES
    param_fig, param_ax = plt.subplots(figsize=(10, 5))
    param_ax.set_title(
        "GridSearchCV Test-set Accuracy by Number of Estimators and Max. Depth"
    )

    for i, (key, grp) in enumerate(data.groupby(["Max Depth"])):
        grp.plot(kind="line",
                 x="Number of Estimators",
                 y="Mean Test Score",
                 color=COLORS[i],
                 label=str(key),
                 ax=param_ax)
        score_mean = grp["Mean Test Score"]
        score_std = grp["Test Score STD"]
        score_lower_limit = score_mean - score_std
        score_upper_limit = score_mean + score_std
        param_ax.fill_between(N_ESTIMATORS,
                              score_lower_limit,
                              score_upper_limit,
                              color=COLORS[i],
                              alpha=0.1)
    param_ax.set_ylabel("Accuracy")
    param_ax.scatter(model_searcher.best_params_["n_estimators"],
                     model_searcher.best_score_,
                     marker="*",
                     color="black",
                     s=150,
                     label="Selected Model")
    _ = param_ax.legend()
    
    