from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


class KnnCf:
    def __init__(self, number_of_users, number_of_items):
        self.number_of_users = number_of_users
        self.number_of_items = number_of_items
        self.train = None
        self.similarities = None

    def fit(self, train_df):
        self.train = sparse.coo_matrix(
            (train_df['ratings'], (train_df['itemId'], train_df['userId'])),
            shape=(self.number_of_items+1, self.number_of_users+1)).tocsr()
        self.similarities = cosine_similarity(self.train, dense_output=False)

    def predict(self, user_id, method='dot'):
        if method == 'dot':
            items = self.train.transpose()[user_id].transpose().todense()
            prediction = self.similarities.dot(items)
            return np.asarray(prediction.reshape(1, -1))[0]
        else:
            _, items, _ = sparse.find(self.train.transpose()[user_id])
            prediction = self.similarities[items].mean(axis=0)
            return np.asarray(prediction)[0]
