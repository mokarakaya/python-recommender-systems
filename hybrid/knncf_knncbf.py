from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer


class KnnCfKnnCbf:
    def __init__(self, number_of_users, number_of_items):
        self.number_of_users = number_of_users
        self.number_of_items = number_of_items
        self.train = None
        self.cf_similarities = None
        self.cbf_similarities = None
        self.items = None

    def fit(self, df_train, df_items):
        self.train = sparse.coo_matrix(
            (df_train['ratings'], (df_train['itemId'], df_train['userId'])),
            shape=(self.number_of_items+1, self.number_of_users+1)).tocsr()
        self.items = sparse.coo_matrix(df_items)

        self.cbf_similarities = cosine_similarity(self.items, dense_output=False)
        self.train = sparse.coo_matrix(
            (df_train['ratings'], (df_train['itemId'], df_train['userId'])),
            shape=(self.number_of_items+1, self.number_of_users+1)).tocsr()
        self.cf_similarities = cosine_similarity(self.train, dense_output=False)

    def predict(self, user_id):
        items = self.train.transpose()[user_id].transpose().todense()
        prediction_cf = self.cf_similarities.dot(items)
        prediction_cf = np.asarray(prediction_cf.reshape(1, -1))[0]
        prediction_cbf = self.cbf_similarities.dot(items)
        prediction_cbf = np.asarray(prediction_cbf.reshape(1, -1))[0]
        return np.add(prediction_cf, prediction_cbf) / 2
