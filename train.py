import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple

from xgboost import XGBClassifier, plot_tree, to_graphviz, plot_importance
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score

def train_model(X_train, X_test, y_train, y_test, model) -> Tuple[float, float, object]:
    # Fit the model
    if model == 'rbf':
        clf = SVC(kernel='rbf')
    if model == 'xgboost':
        clf = GradientBoostingClassifier()
    if model == 'xgboost-explained':
        clf = XGBClassifier()
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

    if model == 'xgboost-explained':
        print(clf.get_num_boosting_rounds())
        # for i in range(clf.get_num_boosting_rounds()):
        #     plot_tree(clf, num_trees=i, rankdir='LR')
        #     # to_graphviz(clf, num_trees=0, rankdir='LR').render()
        #     plt.gcf().set_size_inches(100, 20)
        #     plt.show()
        plot_importance(clf)
        plt.gcf().set_size_inches(20, 100)
        plt.show()

    return train_acc, test_acc, clf
    