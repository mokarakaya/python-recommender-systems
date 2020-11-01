import pandas as pd
"""
https://www.kaggle.com/rounakbanik/the-movies-dataset
"""

class TheMoviesReader:

    def __init__(self):
        self.path = '~/develop/datasets/experiment/the-movies-dataset/'

    def read_dataset(self, is_sample=True):
        user_key = 'userId'
        item_key = 'movieId'
        rating_key = 'rating'
        ratings_file = 'ratings_small.csv' if is_sample else 'ratings.csv'
        df = pd.read_csv('{}{}'.format(self.path, ratings_file))
        return df, user_key, item_key, rating_key
