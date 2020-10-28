from collaborative_filtering.knncf import KnnCf
import pandas as pd
from evaluation.util import train_test_split
from evaluation.evaluation_metrics import evaluate
from scipy import sparse
from collaborative_filtering.lightfm_model import LightFMModel
import datetime


def get_models(df):
    numberOfUsers = df['userId'].max()
    numberOfItems = df['itemId'].max()
    knncf = KnnCf(df['userId'].max(), df['itemId'].max())
    lightfm = LightFMModel(numberOfUsers, numberOfItems)
    return {'knncf': knncf, 'LightFM': lightfm}
    # return {'testing': lightfm}

def evaluate_cf_models():
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
