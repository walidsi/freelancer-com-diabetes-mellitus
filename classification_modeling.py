# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score, fbeta_score
from sklearn.base import BaseEstimator
from time import time
from IPython.display import display
# Allows the use of display() for DataFrames
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, StratifiedKFold
from sklearn.model_selection._search import BaseSearchCV

import pandas as pd
import numpy as np
# Import supplementary visualization code visuals.py
import visuals as vs

_g_beta = 1


def set_beta_f_score(beta: float):
    _g_beta = beta


def calculate_naive_evaluation_mertics(target: pd.Series):
    """TP = np.sum(target) # Counting the ones as this is the naive case.
    FP = target.count() - TP # Specific to the naive case

    TN = 0 # No predicted negatives in the naive case
    FN = 0 # No predicted negatives in the naive case

    Args:
        target (pd.Series): target column

    Returns:
        naive_accuracy, naive_precision, naive_recall, naive_fscore
    """

    # TODO: Calculate accuracy, precision and recall
    naive_accuracy = (np.sum(target) + 0) / target.count()
    naive_recall = np.sum(target) / np.sum(target)
    naive_precision = np.sum(target) / target.count()

    # TODO: Calculate F-score using the formula above for beta = 0.5 and correct values for precision and recall.
    naive_fscore = ((1.0 + 0.5*0.5) * naive_precision * naive_recall) / \
        (((0.5*0.5) * naive_precision) + naive_recall)

    # Print the results
    print("Naive Predictor: [Accuracy score: {:.4f}, F-score: {:.4f}]".format(naive_accuracy, naive_fscore))

    return naive_accuracy, naive_precision, naive_recall, naive_fscore


def train_predict(learner, sample_size, X_train, y_train, X_test, y_test):
    '''
    inputs:
       - learner: the learning algorithm to be trained and predicted on
       - sample_size: the size of samples (number) to be drawn from training set
       - X_train: features training set
       - y_train: income training set
       - X_test: features testing set
       - y_test: income testing set
    '''

    results = {}

    # TODO: Fit the learner to the training data using slicing with 'sample_size' using .fit(training_features[:], training_labels[:])
    start = time()  # Get start time
    learner.fit(X_train[:sample_size], y_train[:sample_size])
    end = time()  # Get end time

    # TODO: Calculate the training time
    results['train_time'] = end - start

    training_set_size_to_predict = len(y_test)  # or 300

    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:training_set_size_to_predict])
    end = time()  # Get end time

    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start

    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:training_set_size_to_predict], predictions_train)

    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(y_train[:training_set_size_to_predict], predictions_train, beta=_g_beta)

    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=_g_beta)

    # Success
    print("{} trained on {} samples.".format(learner.__class__.__name__, sample_size))

    # Return the results
    return results


def train_and_compare_models(models: list, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame,
                             y_test: pd.Series, naive_accuracy: float, naive_fscore: float):
    """train, evaluate and compare models

    Args:
        models (list): list of models to train
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training labels
        X_test (pd.DataFrame): testing features
        y_test (pd.Series): testing labels
        naive_accuracy (float): naive accuracy score
        naive_fscore (float): naive fscore score
    """

    # TODO: Calculate the number of samples for 1%, 10%, and 100% of the training data
    # HINT: samples_100 is the entire training set i.e. len(y_train)
    # HINT: samples_10 is 10% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
    # HINT: samples_1 is 1% of samples_100 (ensure to set the count of the values to be `int` and not `float`)
    samples_100 = len(y_train)
    samples_10 = int(0.1 * samples_100)
    samples_1 = int(0.01 * samples_100)

    # Collect results on the learners
    results = {}
    for clf in models:
        clf_name = clf.__class__.__name__
        results[clf_name] = {}
        for i, samples in enumerate([samples_1, samples_10, samples_100]):
            results[clf_name][i] = train_predict(clf, samples, X_train, y_train, X_test, y_test)

    # Run metrics visualization for the three supervised learning models chosen
    vs.evaluate(results, naive_accuracy, naive_fscore)


def optimize_model(model: BaseEstimator,
                   parameters: dict,
                   X_train: pd.DataFrame,
                   y_train: pd.Series,
                   X_test: pd.DataFrame,
                   y_test: pd.Series,
                   scorer,
                   randomized: bool = False) -> BaseSearchCV:
    """Optimize the model using GridSearchCV or RandomizedSearchCV

    Args:
        model (sklearn.base.BaseEstimator): model to optimize
        parameters (_type_): _description_
        X_train (pd.DataFrame): training features
        y_train (pd.Series): training labels
        X_test (pd.DataFrame): testing features
        y_test (pd.Series): testing labels

    Returns:
        GridSearchCV: Grid search object with all scores, parameters, and best parameters and best estimator
    """

    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
    if randomized:
        grid_obj = RandomizedSearchCV(model,
                                      param_distributions=parameters,
                                      scoring=scorer,
                                      cv=StratifiedKFold(),
                                      random_state=0,
                                      n_jobs=-1)
    else:
        grid_obj = GridSearchCV(model, param_grid=parameters, scoring=scorer, cv=StratifiedKFold(), n_jobs=-1)

    # TODO: Fit the grid search object to the training data and find the optimal parameters using fit()
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the estimator
    # print(type(grid_fit.best_estimator_))
    best_clf = grid_fit.best_estimator_

    # Make predictions using the unoptimized and model
    predictions = (model.fit(X_train, y_train)).predict(X_test)
    best_predictions = best_clf.predict(X_test)

    # Report the before-and-afterscores
    print("Unoptimized model\n------")
    print("Accuracy score on testing data: {:.4f}".format(accuracy_score(y_test, predictions)))
    print("F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=_g_beta)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(accuracy_score(y_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(fbeta_score(y_test, best_predictions, beta=_g_beta)))

    return grid_fit
