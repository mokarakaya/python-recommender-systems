from collaborative_filtering.knncf import KnnCf
import pandas as pd
from evaluation.util import train_test_split
from evaluation.evaluation_metrics import evaluate
from scipy import sparse
from collaborative_filtering.lightfm_model import LightFMModel
import datetime


def get_models(df, user_key, item_key, rating_key):
    numberOfUsers = df[user_key].max()
    numberOfItems = df[item_key].max()
    knncf = KnnCf(df[user_key].max(), df[item_key].max(), user_key, item_key, rating_key)
    lightfm = LightFMModel(numberOfUsers, numberOfItems, user_key, item_key, rating_key)
    return {'knncf': knncf, 'LightFM': lightfm}
    # return {'testing': lightfm}

def evaluate_cf_models(df, user_key, item_key, rating_key):

    train_df, test_df = train_test_split(df)
    test = sparse.coo_matrix((test_df[rating_key], (test_df[user_key], test_df[item_key])))
    train = sparse.coo_matrix((train_df[rating_key], (train_df[user_key], train_df[item_key])))

    models = get_models(df, user_key, item_key, rating_key)
    for key, model in models.items():
        print(key)
        begin_time = datetime.datetime.now()
        model.fit(train_df)
        print('fit duration:', datetime.datetime.now() - begin_time)
        begin_time = datetime.datetime.now()
        results = evaluate(model, test, train=train)
        print('evaluate duration:', datetime.datetime.now() - begin_time)
        print(results)
        print()
