from scipy import sparse
import numpy as np
from lightfm import LightFM


class LightFMModel:
    def __init__(self, number_of_users, number_of_items, user_key, item_key, rating_key):
        self.number_of_users = number_of_users
        self.number_of_items = number_of_items
        self.user_key = user_key
        self.item_key = item_key
        self.rating_key = rating_key
        self.model = LightFM(loss='warp')
        self.train = None

    def fit(self, train_df):
        self.train = sparse.coo_matrix(
            (train_df[self.rating_key], (train_df[self.user_key], train_df[self.item_key])),
            shape=(self.number_of_users+1, self.number_of_items+1)).tocsr()
        self.model.fit(self.train, epochs=50, num_threads=2)

    def predict(self, user_id):
        return self.model.predict(user_id, np.arange(self.number_of_items+1))