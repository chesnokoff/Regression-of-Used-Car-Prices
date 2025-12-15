import pandas as pd
from models.base.basemodel import BaseModel
from models.advanced.AdvancedModel import AdvancedModel

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

    model = BaseModel().fit(X_train, y_train)
    preds = model.predict(X_test)

    submission = sample_sub.copy()
    submission[target_col] = preds
    submission.to_csv("submission.csv", index=False)
    print("Saved submission.csv")


if __name__ == "__main__":
    main()
