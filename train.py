import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple

from sklearn.metrics import accuracy_score

def train_model(X_train, X_test, y_train, y_test, model) -> Tuple[float, float]:
    # Fit the model
    if model == 'rbf':
        clf = SVC(kernel='rbf')
    if model == 'xgboost':
        clf = GradientBoostingClassifier()
    if model == 'random forrest':
        clf = RandomForestClassifier()
    if model == 'tree':
        clf = DecisionTreeClassifier()
    
    clf.fit(X_train, y_train)
    # Calculate training accuracy
    y_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)
    y_pred = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred)
    
    print(f'Model: {model}, Training accuracy: {train_acc}, Test accuracy: {test_acc}')

    return train_acc, test_acc
    