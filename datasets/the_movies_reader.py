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
        df_items = self.read_content(df)
        return df, user_key, item_key, rating_key, df_items

    def read_content(self, df_ratings):
        df_links = pd.read_csv('{}{}'.format(self.path, 'links.csv'))
        df_movies = pd.read_csv('{}{}'.format(self.path, 'movies_metadata.csv'))
        df_movies = df_movies[df_movies['id'].str.isnumeric()]
        df_movies['imdb_id'] = df_movies['imdb_id'].astype(str)
        df_movies['imdb_id_num'] = df_movies['imdb_id'].str.replace('tt', '')
        df_movies = df_movies[df_movies['imdb_id_num'].str.isnumeric()]
        df_movies['imdb_id_num'] = df_movies['imdb_id_num'].astype(int)
        df_sum = pd.merge(df_movies, df_links, how='inner', left_on='imdb_id_num', right_on='imdbId')
        df_sum = df_sum[['movieId', 'overview']]
        df_sum = df_sum.drop_duplicates()
        df_sum.set_index(['movieId'], inplace=True)
        movie_ids = df_ratings['movieId'].unique()
        df_items = df_sum[df_sum.index.isin(movie_ids)]
        return df_items
