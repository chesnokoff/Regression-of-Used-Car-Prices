import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


class Model:
    def __init__(self):
        numeric = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
        ])

        categorical = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("enc", OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1
            )),
        ])

        preproc = ColumnTransformer([
            ("num", numeric, make_column_selector(dtype_include=np.number)),
            ("cat", categorical, make_column_selector(dtype_exclude=np.number)),
        ])

        self.model = Pipeline([
            ("preprocess", preproc),
            ("regressor", HistGradientBoostingRegressor(
                loss="squared_error",
                learning_rate=0.05,
                max_depth=8,
                max_iter=2000,
                l2_regularization=0.1,
                random_state=42,
                early_stopping=True,
            )),
        ])

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)


def main():
    train = pd.read_csv("data/train.csv")
    test = pd.read_csv("data/test.csv")
    sample_sub = pd.read_csv("data/sample_submission.csv")

    id_col = "id"
    target_col = "price"

    feature_cols = [c for c in train.columns if c not in [id_col, target_col]]

    X_train = train[feature_cols]
    y_train = train[target_col]

    X_test = test[feature_cols]

    model = Model().fit(X_train, y_train)
    preds = model.predict(X_test)

    submission = sample_sub.copy()
    submission[target_col] = preds
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
