from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer

class KnnCbf:
    def __init__(self, number_of_users, number_of_items, user_key, item_key, rating_key):
        self.number_of_users = number_of_users
        self.number_of_items = number_of_items
        self.user_key = user_key
        self.item_key = item_key
        self.rating_key = rating_key
        self.train = None
        self.similarities = None
        self.items = None

    def fit(self, df_train, df_items):
        self.train = sparse.coo_matrix(
            (df_train[self.rating_key], (df_train[self.item_key], df_train[self.user_key])),
            shape=(self.number_of_items+1, self.number_of_users+1)).tocsr()
        self.items = sparse.coo_matrix(df_items)
        # transformer = TfidfTransformer(smooth_idf=True, norm='l2')
        # tfidf = transformer.fit_transform(self.items).toarray()

        self.similarities = cosine_similarity(self.items, dense_output=False)

    def predict(self, user_id, method='dot'):
        if method == 'dot':
            items = self.train.transpose()[user_id].transpose().todense()
            prediction = self.similarities.dot(items)
            return np.asarray(prediction.reshape(1, -1))[0]
        else:
            _, items, _ = sparse.find(self.train.transpose()[user_id])
            prediction = self.similarities[items].mean(axis=0)
            return np.asarray(prediction)[0]
