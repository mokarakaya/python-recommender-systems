from collaborative_filtering.knncf import KnnCf
import pandas as pd
from evaluation.util import train_test_split
from evaluation.evaluation_metrics import evaluate
from scipy import sparse
import datetime
from baselines.random_model import RandomModel
from baselines.popularity_based_model import PopularityBasedModel
from baselines.avergate_rating_based_model import AverageRatingBasedModel

def get_models(df):
    number_of_users = df['userId'].max()
    number_of_items = df['itemId'].max()
    random_model = RandomModel(number_of_users, number_of_items)
    popularity_model = PopularityBasedModel(number_of_users, number_of_items)
    average_rating_model = AverageRatingBasedModel(number_of_users, number_of_items)
    return {'random_model': random_model,
            'popularity_model': popularity_model,
            'average_rating_model': average_rating_model}


def evaluate_baseline_models():
    df = pd.read_csv('datasets/movielens100K/u.data', delimiter='\t')
    df.columns = ['userId', 'itemId', 'ratings', 'timestamp']
    df = df.drop(columns=['timestamp'])

    train_df, test_df = train_test_split(df)
    test = sparse.coo_matrix((test_df['ratings'], (test_df['userId'], test_df['itemId'])))
    train = sparse.coo_matrix((train_df['ratings'], (train_df['userId'], train_df['itemId'])))

    models = get_models(df)
    for key, model in models.items():
        begin_time = datetime.datetime.now()
        model.fit(train_df)
        results = evaluate(model, test, train=train)
        print(key)
        print(results)
        print(datetime.datetime.now() - begin_time)
        print()
