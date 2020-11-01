import numpy as np


class AverageRatingBasedModel(object):

    def __init__(self, number_of_users, number_of_items, item_key, rating_key):
        self.number_of_users = number_of_users
        self.number_of_items = number_of_items
        self.item_key = item_key
        self.rating_key = rating_key
        self.train = None
        self.similarities = None

    def fit(self, train_df):
        item_ids = train_df[self.item_key]
        ratings = train_df[self.rating_key]

        counts = np.bincount(item_ids)
        counts += 1
        sums = np.bincount(item_ids, weights=ratings)
        self.averages = sums / counts

    def predict(self, user_ids, item_ids=None):
        return self.averages
