import os
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold

import rampwf as rw

problem_title = 'Flight Chronicles - Predicting Flight ratings'

labels = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Predictions = rw.prediction_types.make_multiclass(labels)

workflow = rw.workflows.Estimator()


score_types = [
    rw.score_types.BalancedAccuracy(
        name="bal_acc", precision=3, adjusted=False
    ),
    rw.score_types.Accuracy(name="acc", precision=3),
]


def get_data(path=".", split="train"):
    data_df = pd.read_csv(os.path.join(path, "data", split + ".csv"))
    y = np.array(data_df["rating"].astype("int8"))
    X = data_df.drop(columns=["rating"])
    return X, y


def get_test_data(path="."):
    return get_data(path, "test")


def get_train_data(path="."):
    return get_data(path, "train")


def get_cv(X, y):
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    return cv.split(X, y)
