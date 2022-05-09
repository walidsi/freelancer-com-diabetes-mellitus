# TODO: Import two metrics from sklearn - fbeta_score and accuracy_score
from sklearn.metrics import accuracy_score, fbeta_score
from time import time
from IPython.display import display
# Allows the use of display() for DataFrames
from sklearn.model_selection import GridSearchCV

# Import supplementary visualization code visuals.py
import visuals as vs


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

    # TODO: Get the predictions on the test set(X_test),
    #       then get predictions on the first 300 training samples(X_train) using .predict()
    start = time()  # Get start time
    predictions_test = learner.predict(X_test)
    predictions_train = learner.predict(X_train[:300])
    end = time()  # Get end time

    # TODO: Calculate the total prediction time
    results['pred_time'] = end - start

    # TODO: Compute accuracy on the first 300 training samples which is y_train[:300]
    results['acc_train'] = accuracy_score(y_train[:300], predictions_train)

    # TODO: Compute accuracy on test set using accuracy_score()
    results['acc_test'] = accuracy_score(y_test, predictions_test)

    # TODO: Compute F-score on the the first 300 training samples using fbeta_score()
    results['f_train'] = fbeta_score(
        y_train[:300], predictions_train, beta=1)

    # TODO: Compute F-score on the test set which is y_test
    results['f_test'] = fbeta_score(y_test, predictions_test, beta=1)

    # Success
    print("{} trained on {} samples.".format(
        learner.__class__.__name__, sample_size))

    # Return the results
    return results


def train_and_compare_models(models, X_train, y_train, X_test, y_test, naive_accuracy, naive_fscore):
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
            results[clf_name][i] = train_predict(
                clf, samples, X_train, y_train, X_test, y_test)

    # Run metrics visualization for the three supervised learning models chosen
    vs.evaluate(results, naive_accuracy, naive_fscore)


def optimize_model(model, parameters, X_train, y_train, X_test, y_test) -> GridSearchCV:
    from sklearn.metrics import accuracy_score, fbeta_score

    # TODO: Import 'GridSearchCV', 'make_scorer', and any other necessary libraries
    from sklearn.model_selection import GridSearchCV
    from sklearn.metrics import make_scorer

    # TODO: Make an fbeta_score scoring object using make_scorer()
    scorer = make_scorer(fbeta_score, beta=0.5)

    # TODO: Perform grid search on the classifier using 'scorer' as the scoring method using GridSearchCV()
    grid_obj = GridSearchCV(model, param_grid=parameters, scoring=scorer)

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
    print("Accuracy score on testing data: {:.4f}".format(
        accuracy_score(y_test, predictions)))
    print(
        "F-score on testing data: {:.4f}".format(fbeta_score(y_test, predictions, beta=0.5)))
    print("\nOptimized Model\n------")
    print("Final accuracy score on the testing data: {:.4f}".format(
        accuracy_score(y_test, best_predictions)))
    print("Final F-score on the testing data: {:.4f}".format(
        fbeta_score(y_test, best_predictions, beta=0.5)))

    return grid_fit
