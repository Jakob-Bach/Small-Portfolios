"""Prediction

Prediction-model-based portfolio evaluation (instead of just VBS-based one).
"""

from typing import Dict
import warnings

import pandas as pd
import sklearn.impute
import sklearn.metrics
import sklearn.model_selection
import sklearn.tree


# For "runtimes" of a portfolio and corresponding instance "feature", train prediction model
# and evaluate with cross-validation. Compute MCC as well as objective value for the recommended
# solvers. Return dict with metrics' names and values, averaged over CV folds.
def predict_and_evaluate(runtimes: pd.DataFrame, features: pd.DataFrame) -> Dict[str, float]:
    # Filter warnings from stratified split and from MCC computation:
    warnings.filterwarnings(action='ignore', message='The least populated class in y has only')
    warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
    model = sklearn.tree.DecisionTreeClassifier(random_state=25)
    imputer = sklearn.impute.SimpleImputer(strategy='mean')
    splitter = sklearn.model_selection.StratifiedKFold(n_splits=10, shuffle=True, random_state=25)
    # Find fastest solver for each row (instance), but use positions instead of solver names as
    # class labels (since we use position-based indexing later):
    y = runtimes.idxmin(axis='columns').replace(runtimes.columns, range(len(runtimes.columns)))
    results = []
    for train_idx, test_idx in splitter.split(X=features, y=y):
        X_train = features.iloc[train_idx]
        y_train = y.iloc[train_idx]
        runtimes_train = runtimes.iloc[train_idx]
        X_test = features.iloc[test_idx]
        y_test = y.iloc[test_idx]
        runtimes_test = runtimes.iloc[test_idx]
        X_train = pd.DataFrame(imputer.fit_transform(X=X_train), columns=list(X_train))
        X_test = pd.DataFrame(imputer.transform(X=X_test), columns=list(X_test))
        model.fit(X=X_train, y=y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        result = {}
        result['train_mcc'] = sklearn.metrics.matthews_corrcoef(y_true=y_train, y_pred=pred_train)
        result['test_mcc'] = sklearn.metrics.matthews_corrcoef(y_true=y_test, y_pred=pred_test)
        # To compute objective value, we need to extract runtime of predicted solver for each
        # instance; as "runtimes.values" is "ndarray" (not "DataFrame"), following syntax works:
        result['train_objective'] = runtimes_train.values[range(len(train_idx)), pred_train].mean()
        result['test_objective'] = runtimes_test.values[range(len(test_idx)), pred_test].mean()
        result['train_vbs'] = runtimes_train.min(axis='columns').mean()
        result['test_vbs'] = runtimes_test.min(axis='columns').mean()
        result['train_vws'] = runtimes_train.max(axis='columns').mean()
        result['test_vws'] = runtimes_test.max(axis='columns').mean()
        results.append(result)
    return pd.DataFrame(results).mean().to_dict()  # average over folds
