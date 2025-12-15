import numpy as np
from catboost import CatBoostRegressor
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline

from models.base.preprocessor import CarColumnProcessor


class BaseModel:
    def __init__(self):
        preproc = CarColumnProcessor()

        base_reg = Pipeline([
            ("preprocess", preproc),
            ("regressor", CatBoostRegressor(
                loss_function="RMSE",
                eval_metric="RMSE",
                iterations=20000,
                learning_rate=0.01,
                depth=8,
                l2_leaf_reg=5.0,
                random_seed=42,
                allow_writing_files=False,
                verbose=200,
                od_type="Iter",
                od_wait=300,
                thread_count=-1
            )),
        ])

        self.model = TransformedTargetRegressor(
            regressor=base_reg,
            func=np.log1p,
            inverse_func=np.expm1
        )

    def fit(self, X, y):
        preproc = self.model.regressor.named_steps["preprocess"]
        preproc.fit(X, y)
        cat_idx = [preproc.feature_names_.index(c) for c in preproc.cat_cols_]
        self.model.regressor.named_steps["regressor"].set_params(cat_features=cat_idx)
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)