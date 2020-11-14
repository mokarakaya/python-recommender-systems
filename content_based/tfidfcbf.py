from scipy import sparse
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import  sys
class TfIdfCbf:
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
        content = df_items['overview'].reindex(range(df_items['overview'].index.max() + 1))
        content = content.fillna('')
        self.train = sparse.coo_matrix(
            (df_train[self.rating_key], (df_train[self.item_key], df_train[self.user_key])),
            shape=(self.number_of_items+1, self.number_of_users+1)).tocsr()
        vectorizer = TfidfVectorizer(stop_words='english')
        tfidf_vectors = vectorizer.fit_transform(content)
        self.similarities = cosine_similarity(tfidf_vectors, dense_output=False)

    def predict(self, user_id, method='sum'):
        if method == 'dot':
            items = self.train.transpose()[user_id].transpose().todense()
            prediction = self.similarities.dot(items)
            return np.asarray(prediction.reshape(1, -1))[0]
        else:
            _, items, _ = sparse.find(self.train.transpose()[user_id])
            temp_sim = self.similarities[items]
            # to get temp_sim.mean(axis=0) without nonzero error
            prediction = temp_sim.sum(axis=0) / (temp_sim.getnnz(axis=0) + sys.float_info.epsilon)
            return np.asarray(prediction)[0]
