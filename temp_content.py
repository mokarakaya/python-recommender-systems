import pandas as pd
import scipy.sparse as sp
import numpy as np
# df_ratings = pd.read_csv('~/develop/datasets/experiment/the-movies-dataset/ratings.csv')
df = pd.read_csv('~/develop/datasets/experiment/the-movies-dataset/movies_metadata.csv')
#print(len(df_ratings['movieId'].unique())) 45115
#print(df_ratings['movieId'].max()) 176275
print(len(df))

df = df[df['id'].str.isnumeric()]
df['id'] = df['id'].astype(int)
print(len(df['id'].unique()))
print(df['id'].max())
