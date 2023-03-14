import pandas as pd
from typing import List

import plotly.express as px

from reduce_dim import pca_reduce
from preprocess import preprocess
from train import train_model

from tqdm.notebook import tqdm


def xgboost_pc_progression(data : pd.DataFrame, n_pcs: List[int]):
    pcs_df = {'pc': [], 'importance': [], 'n_feats': []}
    max_pc = max(n_pcs)
    test_accs = []
    
    for n_pc in tqdm(n_pcs):
        copy = data.copy()
        importances, _ = xgboost_pc(copy, n_pc)
        for feature, importance in importances.items():
            pcs_df['pc'].append(feature)
            pcs_df['importance'].append(importance)
            pcs_df['n_feats'].append(n_pc)
        
        for i in range(n_pc, max_pc):
            pcs_df['pc'].append(f'PC{i}')
            pcs_df['importance'].append(0)
            pcs_df['n_feats'].append(n_pc)
    pcs_df = pd.DataFrame(pcs_df)

    px.line(pcs_df, x='n_feats', y='importance', color='pc', log_x=True).show()


def xgboost_pc(data : pd.DataFrame, n_pc: int):
    y = (data['Gender'] == 'F').astype(int)
    X = data.drop(['Gender'], axis=1)

    projected = pca_reduce(X, n_pc)
    X_train, X_test, y_train, y_test = preprocess(pd.DataFrame(projected), y)

    train_acc, test_acc, clf = train_model(X_train, X_test, y_train, y_test, 'xgboost-explained', verbose=False)
    
    clf.get_booster().feature_names = [f'PC{i}' for i in range(n_pc)]
    importances = clf.get_booster().get_score(importance_type='gain')
    
    return importances, test_acc
    
    
