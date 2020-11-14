from scipy.sparse.linalg import svds
from scipy import sparse


class SVD:
    def __init__(self, number_of_users, number_of_items, user_key, item_key, rating_key):
        self.number_of_users = number_of_users
        self.number_of_items = number_of_items
        self.user_key = user_key
        self.item_key = item_key
        self.rating_key = rating_key
        self.train = None
        self.similarities = None
        self.item_vector = None
        self.user_vector = None

    def fit(self, train_df, k=20):
        self.train = sparse.coo_matrix((train_df[self.rating_key], (train_df[self.item_key], train_df[self.user_key])),
                                  shape=(self.number_of_items + 1, self.number_of_users + 1), dtype=float).tocsr()
        self.item_vector, _, self.user_vector = svds(self.train, k=k)
        self.item_vector = self.item_vector.T
        self.user_vector = self.user_vector.T

    def predict(self, user_id):
        return self.user_vector[user_id].dot(self.item_vector)
