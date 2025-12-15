import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor, early_stopping
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor


class AdvancedModel:
    def __init__(self, n_splits=5, drop_cols=("id", "price")):
        self.n_splits = n_splits
        self.drop_cols = tuple(drop_cols)

        self.feature_cols_ = []
        self.num_cols_ = []
        self.cat_cols_ = []
        self.kf_ = None
        self.fold_models_ = []
        self.rmse_lgbm_ = None
        self.rmse_xgb_ = None
        self.rmse_blend_ = None
        self.blend_weights_ = None
        self.fitted_ = False

    def _prepare_feature_lists(self, X):
        X_local = X.drop(columns=self.drop_cols, errors="ignore")
        self.num_cols_ = X_local.select_dtypes(include=[np.number]).columns.tolist()
        self.cat_cols_ = [col for col in X_local.columns if col not in self.num_cols_]
        self.feature_cols_ = self.num_cols_ + self.cat_cols_

    def _build_preprocessor(self):
        numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
        categorical = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ),
            ]
        )
        return ColumnTransformer(
            [
                ("num", numeric, self.num_cols_),
                ("cat", categorical, self.cat_cols_),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )

    def _split_features(self, X):
        return X.drop(columns=self.drop_cols, errors="ignore").reindex(
            columns=self.feature_cols_
        )

    def fit(self, X, y):
        X, y = X.copy(), pd.Series(y).reset_index(drop=True)
        self._prepare_feature_lists(X)
        X = self._split_features(X).reset_index(drop=True)

        lgbm_oof = np.zeros(len(X))
        xgb_oof = np.zeros(len(X))
        self.kf_ = KFold(n_splits=self.n_splits, shuffle=True, random_state=42)
        self.fold_models_ = []

        for fold, (train_idx, val_idx) in enumerate(self.kf_.split(X, y)):
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            preprocessor = self._build_preprocessor()
            X_train_proc = preprocessor.fit_transform(X_train)
            X_val_proc = preprocessor.transform(X_val)

            lgbm_model = LGBMRegressor(
                num_leaves=426,
                max_depth=20,
                learning_rate=0.011353178352988012,
                n_estimators=10000,
                metric="rmse",
                subsample=0.5772552201954328,
                colsample_bytree=0.9164865430101521,
                reg_alpha=1.48699088003429e-06,
                reg_lambda=0.41539458543414265,
                min_data_in_leaf=73,
                feature_fraction=0.751673655170548,
                bagging_fraction=0.5120415391590843,
                bagging_freq=2,
                random_state=42,
                min_child_weight=0.017236362383443497,
                cat_smooth=54.81317407769262,
                verbose=-1,
            )
            lgbm_model.fit(
                X_train_proc,
                y_train,
                eval_set=[(X_val_proc, y_val)],
                eval_metric="rmse",
                callbacks=[early_stopping(200, verbose=False)],
            )
            lgb_val_pred = lgbm_model.predict(X_val_proc)
            lgbm_oof[val_idx] = lgb_val_pred

            xgb_model = XGBRegressor(
                reg_lambda=0.03880258557285165,
                reg_alpha=0.02129832295514386,
                colsample_bytree=0.4,
                subsample=0.7,
                learning_rate=0.014,
                max_depth=17,
                random_state=42,
                min_child_weight=85,
                n_estimators=10000,
                eval_metric="rmse",
            )
            xgb_model.fit(X_train_proc, y_train, eval_set=[(X_val_proc, y_val)], verbose=False)
            xgb_val_pred = xgb_model.predict(X_val_proc)
            xgb_oof[val_idx] = xgb_val_pred

            self.fold_models_.append({"preprocessor": preprocessor, "lgbm": lgbm_model, "xgb": xgb_model})

            fold_rmse_lgbm = np.sqrt(mean_squared_error(y_val, lgb_val_pred))
            fold_rmse_xgb = np.sqrt(mean_squared_error(y_val, xgb_val_pred))
            print(
                f"Fold {fold + 1}/{self.n_splits} - LGBM RMSE: {fold_rmse_lgbm:.4f}, "
                f"XGB RMSE: {fold_rmse_xgb:.4f}"
            )

        self.rmse_lgbm_ = float(np.sqrt(mean_squared_error(y, lgbm_oof)))
        self.rmse_xgb_ = float(np.sqrt(mean_squared_error(y, xgb_oof)))
        inv_rmse_lgbm = 1.0 / self.rmse_lgbm_ if self.rmse_lgbm_ > 0 else 0.0
        inv_rmse_xgb = 1.0 / self.rmse_xgb_ if self.rmse_xgb_ > 0 else 0.0
        total_weight = inv_rmse_lgbm + inv_rmse_xgb
        lgbm_weight = inv_rmse_lgbm / total_weight if total_weight else 0.5
        xgb_weight = inv_rmse_xgb / total_weight if total_weight else 0.5
        self.blend_weights_ = (lgbm_weight, xgb_weight)

        blended_oof = lgbm_oof * lgbm_weight + xgb_oof * xgb_weight
        self.rmse_blend_ = float(np.sqrt(mean_squared_error(y, blended_oof)))
        print(
            f"OOF RMSE -> LGBM: {self.rmse_lgbm_:.4f}, XGB: {self.rmse_xgb_:.4f}, "
            f"Blend: {self.rmse_blend_:.4f}"
        )
        print(f"Blend weights -> LGBM: {lgbm_weight:.3f}, XGB: {xgb_weight:.3f}")

        return self

    def predict(self, X):
        X_proc = self._split_features(X)
        lgbm_preds = []
        xgb_preds = []

        for fold_model in self.fold_models_:
            Xt = fold_model["preprocessor"].transform(X_proc)
            lgbm_preds.append(fold_model["lgbm"].predict(Xt))
            xgb_preds.append(fold_model["xgb"].predict(Xt))

        lgbm_mean = np.mean(lgbm_preds, axis=0)
        xgb_mean = np.mean(xgb_preds, axis=0)
        lgbm_weight, xgb_weight = self.blend_weights_ or (0.5, 0.5)
        return lgbm_mean * lgbm_weight + xgb_mean * xgb_weight
