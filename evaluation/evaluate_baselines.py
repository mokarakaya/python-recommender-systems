from collaborative_filtering.knncf import KnnCf
import pandas as pd
from evaluation.util import train_test_split
from evaluation.evaluation_metrics import evaluate
from scipy import sparse
import datetime
from baselines.random_model import RandomModel
from baselines.popularity_based_model import PopularityBasedModel
from baselines.avergate_rating_based_model import AverageRatingBasedModel

def get_models(df, user_key, item_key, rating_key):
    number_of_users = df[user_key].max()
    number_of_items = df[item_key].max()
    random_model = RandomModel(number_of_users, number_of_items)
    popularity_model = PopularityBasedModel(number_of_users, number_of_items, item_key)
    average_rating_model = AverageRatingBasedModel(number_of_users, number_of_items, item_key, rating_key)
    return {'random_model': random_model,
            'popularity_model': popularity_model,
            'average_rating_model': average_rating_model}


def evaluate_baseline_models(df, user_key, item_key, rating_key):

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

