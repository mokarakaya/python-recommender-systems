import numpy as np

import scipy.stats as st


FLOAT_MAX = np.finfo(np.float32).max


def _get_precision_recall(recommendations, targets):

    num_hit = len(set(recommendations).intersection(set(targets)))

    return float(num_hit) / len(recommendations), float(num_hit) / len(targets)


def evaluate(model, test, train=None, k=10):

    test = test.tocsr()

    if train is not None:
        train = train.tocsr()

    precision = []
    recall = []
    coverage = set()

    for user_id, row in enumerate(test):

        if not len(row.indices):
            continue

        predictions = -model.predict(user_id)

        if train is not None:
            rated = train[user_id].indices
            predictions[rated] = FLOAT_MAX

        predictions = predictions.argsort()
        recommendations = predictions[:k]
        coverage.update(recommendations)
        targets = row.indices

        user_precision, user_recall = _get_precision_recall(recommendations, targets)
        precision.append(user_precision)
        recall.append(user_recall)

    precision = np.average(np.array(precision).squeeze())
    recall = np.average(np.array(recall).squeeze())
    results = {'precision': precision, 'recall': recall, 'coverage': len(coverage)}
    return results

