from datasets.movielens_100K_reader import Movielens100KReader
from datasets.the_movies_reader import TheMoviesReader

datasets = {'movielens100K': Movielens100KReader(),
            'theMovies': TheMoviesReader()}


def read_dataset(dataset):
    reader = datasets[dataset]
    return reader.read_dataset()
