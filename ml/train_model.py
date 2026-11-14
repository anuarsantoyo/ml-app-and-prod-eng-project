"""Train a house-price model on the Ames Housing dataset.

Pipeline: load -> preprocess (impute / encode / scale, log-transform target) ->
split -> compare 3 models with hyperparameter tuning -> save the best to joblib.

Run with the ML interpreter:   /usr/bin/python3 ml/train_model.py
"""
import numpy as np
import pandas as pd
import joblib

from sklearn.datasets import fetch_openml
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
TEST_SIZE = 0.2


def load_ames():
    bunch = fetch_openml("house_prices", as_frame=True, parser="auto")
    X = bunch.data
    y = bunch.target
    if "Id" in X.columns:
        X = X.drop(columns=["Id"])
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    return X, y


def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def main():
    print("Loading Ames dataset...")
    X, y = load_ames()
    print(f"  rows: {X.shape[0]}, features: {X.shape[1]}")
    print(f"  target mean: ${y.mean():,.0f}   skew: {y.skew():.2f} (right-skewed -> log transform)")

    # --- split feature types ---
    num_cols = X.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["category", "object"]).columns.tolist()
    print(f"  numeric features: {len(num_cols)}, categorical: {len(cat_cols)}")

    # --- preprocessor ---
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), num_cols),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), cat_cols),
    ])

    def make_model(regressor):
        # log1p transform on the target, inverse expm1 -> predict() returns dollars
        return TransformedTargetRegressor(
            regressor=Pipeline([("pre", preprocessor), ("model", regressor)]),
            func=np.log1p, inverse_func=np.expm1,
        )

    # --- candidate models + their tuning grids ---
    candidates = [
        (
            "LinearRegression",
            make_model(LinearRegression()),
            None,  # no hyperparameters to tune
        ),
        (
            "RandomForest",
            make_model(RandomForestRegressor(random_state=RANDOM_STATE)),
            {
                "regressor__model__n_estimators": [100, 200, 400],
                "regressor__model__max_depth": [None, 10, 20, 30],
                "regressor__model__min_samples_leaf": [1, 2, 4],
            },
        ),
        (
            "GradientBoosting",
            make_model(GradientBoostingRegressor(random_state=RANDOM_STATE)),
            {
                "regressor__model__n_estimators": [100, 200, 300],
                "regressor__model__learning_rate": [0.05, 0.1, 0.15],
                "regressor__model__max_depth": [3, 4, 5],
            },
        ),
    ]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    results = []
    for name, model, grid in candidates:
        print(f"\n=== {name} ===")
        if grid is None:
            model.fit(X_train, y_train)
        else:
            search = RandomizedSearchCV(
                model, grid, n_iter=12, cv=5,
                scoring="neg_root_mean_squared_error",
                random_state=RANDOM_STATE, n_jobs=-1,
            )
            search.fit(X_train, y_train)
            model = search.best_estimator_
            print(f"  best params: {search.best_params_}")
            print(f"  best CV RMSE: ${-search.best_score_:,.0f}")

        y_pred = model.predict(X_test)
        r = {
            "name": name,
            "model": model,
            "rmse": rmse(y_test, y_pred),
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "r2": float(r2_score(y_test, y_pred)),
        }
        results.append(r)
        print(f"  test RMSE: ${r['rmse']:,.0f}   MAE: ${r['mae']:,.0f}   R2: {r['r2']:.4f}")

    # --- pick the best by RMSE ---
    best = min(results, key=lambda r: r["rmse"])
    print(f"\n=== BEST: {best['name']} (RMSE ${best['rmse']:,.0f}) ===")

    artifact = {
        "version": "1.0.0",
        "model_name": best["name"],
        "model": best["model"],
        "metrics": {"rmse": best["rmse"], "mae": best["mae"], "r2": best["r2"]},
        "numeric_features": num_cols,
        "categorical_features": cat_cols,
        "trained_on": "ames_housing",
        "target": "sale_price",
    }
    out = "house_price_model.joblib"
    joblib.dump(artifact, out)
    print(f"\nSaved {out}")
    print(f"  metrics: RMSE ${best['rmse']:,.0f}, MAE ${best['mae']:,.0f}, R2 {best['r2']:.4f}")


if __name__ == "__main__":
    main()
