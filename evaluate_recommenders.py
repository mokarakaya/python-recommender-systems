import sys, os

sys.path.append(os.getcwd())
from evaluation.evaluate_cf import evaluate_cf_models
from evaluation.evaluate_cbf import evaluate_cbf_models
from evaluation.evaluate_hybrid import evaluate_hybrid_models
from evaluation.evaluate_baselines import evaluate_baseline_models
import pandas as pd
from datasets.data_reader import read_dataset

# dataset = 'movielens100K'
dataset = 'theMovies'

df, user_key, item_key, rating_key = read_dataset(dataset)
evaluate_baseline_models(df, user_key, item_key, rating_key)
evaluate_cf_models(df, user_key, item_key, rating_key)
# evaluate_cbf_models()
# evaluate_hybrid_models()
