from collaborative_filtering.knncf import KnnCf
import pandas as pd
from evaluation.util import train_test_split
from evaluation.evaluation_metrics import evaluate
from scipy import sparse
from content_based.knn.knncbf import KnnCbf
import datetime
import numpy as np

def get_models(df):
    number_of_users = df['userId'].max()
    number_of_items = df['itemId'].max()
    knncbf = KnnCbf(number_of_users, number_of_items)
    return {'knncbf': knncbf}

def evaluate_cbf_models():
    df_items = pd.read_csv('datasets/movielens100K/u.item', delimiter='|', encoding='latin1', header=None)
    df_genre = pd.read_csv('datasets/movielens100K/u.genre', delimiter='|', encoding='latin1', header=None)
    df_genre.columns = ['genre', 'id']
    genres = df_genre['genre'].values
    df_items_columns = np.append(np.array(['itemId']), genres)
    cols = [1, 2, 3, 4]
    df_items = df_items.drop(df_items.columns[cols], axis=1)
    df_items.columns = df_items_columns
    df_items = pd.DataFrame(np.zeros((1, 20)), columns=df_items_columns).append(df_items)

    df_items = df_items.set_index('itemId')
    df = pd.read_csv('datasets/movielens100K/u.data', delimiter='\t')
    df.columns = ['userId', 'itemId', 'ratings', 'timestamp']
    df = df.drop(columns=['timestamp'])


    train_df, test_df = train_test_split(df)
    test = sparse.coo_matrix((test_df['ratings'], (test_df['userId'], test_df['itemId'])))
    train = sparse.coo_matrix((train_df['ratings'], (train_df['userId'], train_df['itemId'])))

    models = get_models(df)
    for key, model in models.items():
        begin_time = datetime.datetime.now()
        model.fit(train_df, df_items)
        results = evaluate(model, test, train=train)
        print(key)
        print(results)
        print(datetime.datetime.now() - begin_time)
        print()
