import pandas as pd
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from train import train_model

def k_fold_train_model(X: pd.DataFrame, y: pd.DataFrame, k: int, model: str):
    all_train_acc = []
    all_test_acc = []

    # Choose only numeric columns
    X = X.select_dtypes(include=['number'])

    # Split to k folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)
    for train_index, test_index in kf.split(X):
        X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        y_train, y_test = y.iloc[train_index], y.iloc[test_index]

        scaler = StandardScaler()
        scaler.fit(X_train)
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)

        # Replace NaN with 0
        X_train[np.isnan(X_train)] = 0
        X_test[np.isnan(X_test)] = 0

        train_acc, test_acc = train_model(X_train, X_test, y_train, y_test, model)
        
        all_train_acc.append(train_acc)
        all_test_acc.append(test_acc)

    print(f'Model: {model}, Average training accuracy: {np.mean(all_train_acc)}, Average test accuracy: {np.mean(all_test_acc)}')
    