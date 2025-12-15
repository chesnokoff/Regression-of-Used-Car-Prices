import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin


class CarColumnProcessor(BaseEstimator, TransformerMixin):
    def __init__(self, drop_cols=("id",)):
        self.drop_cols = tuple(drop_cols)

    def fit(self, X, y=None):
        X = X.copy()
        self.drop_cols_ = [c for c in self.drop_cols if c in X.columns]
        X = X.drop(columns=self.drop_cols_, errors="ignore")

        X = self._add_features(X)

        self.num_cols_ = X.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = X.select_dtypes(exclude=[np.number]).columns.tolist()

        self.num_medians_ = X[self.num_cols_].median(numeric_only=True)
        self.feature_names_ = self.num_cols_ + self.cat_cols_
        return self

    def _add_features(self, X: pd.DataFrame):
        X = X.copy()

        yr = pd.to_numeric(X.get("model_year"), errors="coerce")
        X["age"] = (2025 - yr).clip(lower=0)

        mil = pd.to_numeric(X.get("milage"), errors="coerce")
        X["milage_per_year"] = mil / (X["age"].fillna(0) + 1)
        X["log_milage"] = np.log1p(mil.clip(lower=0))

        X["brand_model"] = X["brand"].astype("string") + "_" + X["model"].astype("string")

        X["same_color"] = (X["ext_col"].astype("string").fillna("MISSING") == X["int_col"].astype("string").fillna("MISSING")).astype(int)

        for col in ["brand", "model", "brand_model", "ext_col", "int_col", "transmission", "fuel_type"]:
            if col in X.columns:
                s = X[col].astype("string").fillna("MISSING")
                vc = s.value_counts()
                X[f"{col}_freq"] = s.map(vc).astype(float)
            else:
                X[f"{col}_freq"] = 0.0

        return X

    def transform(self, X):
        X = X.copy()
        X = X.drop(columns=self.drop_cols, errors="ignore")

        X = self._add_features(X)

        for c in self.feature_names_:
            if c not in X.columns:
                X[c] = np.nan

        if self.num_cols_:
            X[self.num_cols_] = X[self.num_cols_].apply(pd.to_numeric, errors="coerce")
            X[self.num_cols_] = X[self.num_cols_].fillna(self.num_medians_)

        for c in self.cat_cols_:
            X[c] = X[c].astype("string").fillna("MISSING")

        return X[self.feature_names_]
