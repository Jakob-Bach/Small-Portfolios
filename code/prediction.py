"""Prediction

Prediction-model-based portfolio evaluation, using the prediction model to make instance-specific
solver recommendations within a portfolio.
"""

import warnings

import pandas as pd
import sklearn.ensemble
import sklearn.impute
import sklearn.metrics


MODELS = {'Random forest': sklearn.ensemble.RandomForestClassifier}


# For train/test split of "runtimes" of a portfolio and corresponding instance "features", train
# prediction models and evaluate. Compute MCC as well as objective value for the recommended
# solvers. Return data frame with evaluation metrics, including feature importances.
def predict_and_evaluate(runtimes_train: pd.DataFrame, runtimes_test: pd.DataFrame,
                         features_train: pd.DataFrame, features_test: pd.DataFrame) -> pd.DataFrame:
    # Replace missing values with out-of-range value (sorry, a bit hacky, but easier than
    # considering the natural (domain-specific) range of each feature):
    impute_value = min(features_train.min().min(), features_test.min().min()) - 1
    imputer = sklearn.impute.SimpleImputer(strategy='constant', fill_value=impute_value)
    X_train = pd.DataFrame(imputer.fit_transform(X=features_train), columns=list(features_train))
    X_test = pd.DataFrame(imputer.transform(X=features_test), columns=list(features_test))
    # Find fastest solver for each row (instance), but use positions instead of solver names as
    # class labels (since we use position-based indexing later):
    y_train = runtimes_train.idxmin(axis='columns').replace(
        runtimes_train.columns, range(len(runtimes_train.columns)))
    y_test = runtimes_test.idxmin(axis='columns').replace(
        runtimes_test.columns, range(len(runtimes_test.columns)))
    results = []
    feature_importances = []
    for model_name in MODELS:
        model = MODELS[model_name](random_state=25, n_estimators=100)
        model.fit(X=X_train, y=y_train)
        pred_train = model.predict(X_train)
        pred_test = model.predict(X_test)
        result = {'model': model_name}
        with warnings.catch_warnings():
            # Filter warnings which occur if there only is one class in true or pred:
            warnings.filterwarnings(action='ignore', message='invalid value encountered in double_scalars')
            result['train_pred_mcc'] = sklearn.metrics.matthews_corrcoef(y_true=y_train, y_pred=pred_train)
            result['test_pred_mcc'] = sklearn.metrics.matthews_corrcoef(y_true=y_test, y_pred=pred_test)
        # To compute objective value, we need to extract runtime of predicted solver for each
        # instance; as "runtimes.values" is "ndarray" (not "DataFrame"), following syntax works:
        result['train_pred_objective'] = runtimes_train.values[range(len(runtimes_train)), pred_train].mean()
        result['test_pred_objective'] = runtimes_test.values[range(len(runtimes_test)), pred_test].mean()
        results.append(result)
        feature_importances.append(model.feature_importances_)
    results = pd.DataFrame(results)
    feature_importances = pd.DataFrame(feature_importances, columns=['imp.' + x for x in features_train.columns])
    results = pd.concat([results, feature_importances], axis='columns')
    return results
