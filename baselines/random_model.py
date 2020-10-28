import numpy as np


class RandomModel:

    def __init__(self, number_of_users, number_of_items):
        self.number_of_users = number_of_users
        self.number_of_items = number_of_items
        self.train = None
        self.similarities = None

    def fit(self, train):
        pass

    def predict(self, user_ids, item_ids=None):
        predictions = np.random.rand(self.number_of_items+1)
        predictions[self.number_of_items - 1] = 0
        return predictions
