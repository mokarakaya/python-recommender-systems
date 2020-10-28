from scipy import sparse
import numpy as np
from lightfm import LightFM


class LightFMModel:
    def __init__(self, number_of_users, number_of_items):
        self.number_of_users = number_of_users
        self.number_of_items = number_of_items
        self.model = LightFM(loss='warp')
        self.train = None

    def fit(self, train):
        self.train = sparse.coo_matrix(
            (train['ratings'], (train['userId'], train['itemId'])),
            shape=(self.number_of_users+1, self.number_of_items+1))\
            .tocsr()
        self.model.fit(self.train, epochs=25, num_threads=2)

    def predict(self, user_id):
        return self.model.predict(user_id, np.arange(self.number_of_items+1))