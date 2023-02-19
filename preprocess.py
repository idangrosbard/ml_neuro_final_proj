import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def preprocess(X: pd.DataFrame, y: pd.Series):
    # Choose only numeric columns
    X = X.select_dtypes(include=['number'])

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # print only string columns
    print(X_test.select_dtypes(include=['object']).columns)

    # Scale the data
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    # Replace NaN with 0
    X_train[np.isnan(X_train)] = 0
    X_test[np.isnan(X_test)] = 0

    return X_train, X_test, y_train, y_test