from evaluation.util import train_test_split
from evaluation.evaluation_metrics import evaluate
from scipy import sparse
from hybrid.knncf_knncbf import KnnCfKnnCbf
import datetime

def get_models(df, user_key, item_key, rating_key):
    number_of_users = df[user_key].max()
    number_of_items = df[item_key].max()
    knncfknncbf = KnnCfKnnCbf(number_of_users, number_of_items, user_key, item_key, rating_key)
    return {'movielens100K': {'knncfknncbf': knncfknncbf},
            'theMovies': {}}

def evaluate_hybrid_models(df, user_key, item_key, rating_key, df_items, dataset):
    train_df, test_df = train_test_split(df)
    test = sparse.coo_matrix((test_df[rating_key], (test_df[user_key], test_df[item_key])))
    train = sparse.coo_matrix((train_df[rating_key], (train_df[user_key], train_df[item_key])))

    models = get_models(df, user_key, item_key, rating_key)
    models = models[dataset]
    for key, model in models.items():
        begin_time = datetime.datetime.now()
        model.fit(train_df, df_items)
        results = evaluate(model, test, train=train)
        print(key)
        print(results)
        print('duration:', datetime.datetime.now() - begin_time)
        print()
