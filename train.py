import numpy as np
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from typing import Tuple

from xgboost import XGBClassifier, plot_tree, to_graphviz, plot_importance
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

def train_model(X_train, X_test, y_train, y_test, model, verbose:bool = True) -> Tuple[float, float, object]:
    # Fit the model
    if model == 'rbf':
        clf = SVC(kernel='rbf', random_state=42)
    if model == 'xgboost':
        clf = GradientBoostingClassifier(random_state=42)
    if model == 'xgboost-explained':
        clf = XGBClassifier(n_estimators=80, max_depth=2, random_state=42)
    if model == 'random forrest':
        clf = RandomForestClassifier(random_state=42)
    if model == 'tree':
        clf = DecisionTreeClassifier(random_state=42)
    
    clf.fit(X_train, y_train)
    # Calculate training accuracy
    y_pred = clf.predict(X_train)
    train_acc = accuracy_score(y_train, y_pred)
    y_pred_test = clf.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)


    if verbose:
        print(f'Model: {model}, Training accuracy: {np.round(train_acc, decimals =4)}, '
              f'Test accuracy: {np.round(test_acc, decimals =4)}')

    if verbose:
        if model == 'xgboost-explained':
            print(clf.get_num_boosting_rounds())
            # for i in range(clf.get_num_boosting_rounds()):
            #     plot_tree(clf, num_trees=i, rankdir='LR')
            #     # to_graphviz(clf, num_trees=0, rankdir='LR').render()
            #     plt.gcf().set_size_inches(100, 20)
            #     plt.show()
            plot_importance(clf, importance_type='gain')
            plt.gcf().set_size_inches(20, 100)
            plt.show()

    return train_acc, test_acc, clf
    