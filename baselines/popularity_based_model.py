import numpy as np
import sklearn.preprocessing as pp


class PopularityBasedModel(object):

    def __init__(self, number_of_users, number_of_items, item_key):
        self.number_of_users = number_of_users
        self.number_of_items = number_of_items
        self.item_key = item_key
        self.train = None
        self.similarities = None

    def fit(self, train_df):
        self.popularity = np.bincount(train_df[self.item_key])

    def predict(self, user_ids, item_ids=None):
        return pp.minmax_scale(self.popularity, feature_range=(1, 5))
