from builtins import str

"""
Available at:
https://grouplens.org/datasets/movielens/100k/

"""
import pandas as pd


class Movielens100KReader:

    def __init__(self):
        self.path = '~/develop/datasets/experiment/movielens100K/'

    def read_dataset(self):
        user_key = 'userId'
        item_key = 'itemId'
        rating_key = 'ratings'
        df = pd.read_csv('{}{}'.format(self.path, 'u.data'), delimiter='\t')
        df.columns = [user_key, item_key, rating_key, 'timestamp']
        df = df.drop(columns=['timestamp'])
        return df, user_key, item_key, rating_key
