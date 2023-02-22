import pandas as pd
import plotly.express as px
from kfold_train import k_fold_train_model
from typing import List


def model_compare(dfs: List[pd.DataFrame], data_label: List[str], k: int, models: list):
    results = {'model': [], 'data': [], 'train_acc': [], 'test_acc': []}
    for df, label in zip(dfs, data_label):
        y = (df['Gender'] == 'F').astype(int)
        X = df.drop(['Gender'], axis=1)
        # print(X)
        print(df.groupby('Gender').count())

        for model in models:
            train, test, _ = k_fold_train_model(X, y, k, model)
            results['model'].append(model)
            results['data'].append(label)
            results['train_acc'].append(train)
            results['test_acc'].append(test)
    
    results = pd.DataFrame(results)
    px.bar(results, x='model', y='test_acc', color='data', barmode='group').show()