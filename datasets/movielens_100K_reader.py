import pandas as pd
import numpy as np

"""
Available at:
https://grouplens.org/datasets/movielens/100k/

"""
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
        df_items = self.read_content()
        return df, user_key, item_key, rating_key, df_items

    def read_content(self):
        df_items = pd.read_csv('{}{}'.format(self.path, 'u.item'), delimiter='|', encoding='latin1', header=None)
        df_genre = pd.read_csv('{}{}'.format(self.path, 'u.genre'), delimiter='|', encoding='latin1', header=None)
        df_genre.columns = ['genre', 'id']
        genres = df_genre['genre'].values
        df_items_columns = np.append(np.array(['itemId']), genres)
        cols = [1, 2, 3, 4]
        df_items = df_items.drop(df_items.columns[cols], axis=1)
        df_items.columns = df_items_columns
        df_items = pd.DataFrame(np.zeros((1, 20)), columns=df_items_columns).append(df_items)

        df_items = df_items.set_index('itemId')
        return df_items