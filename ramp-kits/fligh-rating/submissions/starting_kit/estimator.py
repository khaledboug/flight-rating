from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.compose import make_column_transformer
from sklearn.ensemble import RandomForestClassifier


def get_estimator():
    # preprocessing
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
    categorical_transformer = make_pipeline(OrdinalEncoder(
        handle_unknown='use_encoded_value', unknown_value=-1))
    preprocessor = make_column_transformer((
                    categorical_transformer,
                    categorical_cols))

    # model
    model = RandomForestClassifier(n_estimators=100, random_state=1)

    pipe = make_pipeline(preprocessor, model)

    return pipe
