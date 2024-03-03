# from sklearn.base import BaseEstimator
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier


# class Classifier(BaseEstimator):
#     def __init__(self):
#         self.categorical_cols = ['header',
#                                  'author',
#                                  'date',
#                                  'place',
#                                  'content',
#                                  'aircraft',
#                                  'traveller_type',
#                                  'seat_type',
#                                  'route',
#                                  'date_flown',
#                                  'recommended',
#                                  'trip_verified']
#         self.categorical_transformer = make_pipeline(LabelEncoder())
#         self.preprocessor = make_column_transformer((
#             self.categorical_transformer,
#             self.categorical_cols))
#         self.model = RandomForestClassifier(n_estimators=100, random_state=1)
#         self.pipe = make_pipeline(self.preprocessor, self.model)

#     def fit(self, X, y):
#         self.pipe.fit(X, y)

#     def predict(self, X):
#         return self.pipe.predict(X)

#     def predict_proba(self, X):
#         return self.pipe.predict_proba(X)


def get_estimator():
    categorical_cols = ['header',
                        'author',
                        'date',
                        'place',
                        'content',
                        'aircraft',
                        'traveller_type',
                        'seat_type',
                        'route',
                        'date_flown',
                        'recommended',
                        'trip_verified']
    categorical_transformer = make_pipeline(LabelEncoder())
    preprocessor = make_column_transformer((
                    categorical_transformer,
                    categorical_cols))
    model = RandomForestClassifier(n_estimators=100, random_state=1)
    pipe = make_pipeline(preprocessor, model)

    return pipe
