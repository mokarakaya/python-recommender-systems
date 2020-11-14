import numpy as np
from joblib import Parallel, delayed

FLOAT_MAX = np.finfo(np.float32).max


def _get_precision_recall(recommendations, targets):
    num_hit = len(set(recommendations).intersection(set(targets)))

    return float(num_hit) / len(recommendations), float(num_hit) / len(targets)


def __run_parallel(arg):
    np.random.seed()  # numpy needs this to produce expected results with joblib
    user_id, row, model, train, k = arg
    if not len(row.indices):
        return

    return _get_scores(user_id, row, model, train, k)


def _evaluate_parallel(model, test, train, k, test_percentage):
    input_full = [(user_id, row, model, train, k)
                  for user_id, row in enumerate(test) if test_percentage > np.random.random()]
    results_par = Parallel(n_jobs=-1, verbose=0, backend="multiprocessing", max_nbytes='10M')(
        map(delayed(__run_parallel), input_full))
    precision = [x[0] for x in results_par if x is not None]
    recall = [x[1] for x in results_par if x is not None]
    coverage = np.array(([x[2] for x in results_par if x is not None])).flatten()

    return precision, recall, coverage


def _get_scores(user_id, row, model, train, k):
    predictions = -model.predict(user_id)
    if train is not None:
        rated = train[user_id].indices
        predictions[rated] = FLOAT_MAX

    predictions = predictions.argsort()
    recommendations = predictions[:k]
    targets = row.indices

    user_precision, user_recall = _get_precision_recall(recommendations, targets)
    return user_precision, user_recall, recommendations


def _evaluate_seq(model, test, train, k, test_percentage):
    precision = []
    recall = []
    coverage = set()

    for user_id, row in enumerate(test):
        if not len(row.indices):
            continue
        if np.random.random() > test_percentage:
            continue
        user_precision, user_recall, recommendations = _get_scores(user_id, row, model, train, k)
        precision.append(user_precision)
        recall.append(user_recall)
        coverage.update(recommendations)
    return precision, recall, coverage


def evaluate(model, test, train=None, k=10, parallel=False, test_percentage=0.01):
    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    precision, recall, coverage = _evaluate_parallel(model, test, train, k, test_percentage) if parallel \
        else _evaluate_seq(model, test, train, k, test_percentage)

    precision = np.average(precision)
    recall = np.average(recall)
    coverage = len(set(coverage))

    results = {'precision': precision, 'recall': recall, 'coverage': coverage}
    return results
