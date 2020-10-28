import numpy as np


def train_test_split(df, test_size=0.2):
    msk = np.random.rand(len(df)) > test_size
    train = df[msk]
    test = df[~msk]
    return train, test