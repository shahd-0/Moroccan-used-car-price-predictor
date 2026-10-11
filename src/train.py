"""Train and compare the models, then save the best one.

Usage (from the repo root):
    python src/train.py

Outputs:
    models/model.pkl      trained pipeline + fair-range info + category lists (used by predict.py)
    docs/results.md       metrics table for the README
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeRegressor

from preprocessing import (CATEGORICAL_FEATURES, FEATURES, NUMERIC_FEATURES,
                           TARGET, clean_listings)

ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT / "data" / "cars_dataframe.csv"
MODEL_PATH = ROOT / "models" / "model.pkl"
RESULTS_PATH = ROOT / "docs" / "results.md"

# The fair range covers the middle 80% of how far real prices were from predictions
FAIR_LOW_Q, FAIR_HIGH_Q = 0.10, 0.90


def build_model(regressor):
    """Preprocessing + regressor. The price is modelled on a log scale, which
    stops a few very expensive cars from dominating the training."""
    preprocess = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=20), CATEGORICAL_FEATURES),
        ("num", StandardScaler(), NUMERIC_FEATURES),
    ])
    pipe = Pipeline([("preprocess", preprocess), ("regressor", regressor)])
    return TransformedTargetRegressor(pipe, func=np.log1p, inverse_func=np.expm1)


def evaluate(y_true, y_pred):
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "Median % error": float(np.median(np.abs(y_pred - y_true) / y_true) * 100),
    }


def main():
    raw = pd.read_csv(DATA_PATH)
    df = clean_listings(raw)
    print(f"Loaded {len(raw):,} listings, {len(df):,} left after cleaning")

    X, y = df[FEATURES], df[TARGET]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    candidates = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(min_samples_leaf=5, random_state=42),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, min_samples_leaf=2, n_jobs=-1, random_state=42),
    }

    results, fitted = {}, {}
    for name, regressor in candidates.items():
        start = time.time()
        model = build_model(regressor).fit(X_train, y_train)
        results[name] = evaluate(y_test, model.predict(X_test))
        fitted[name] = model
        print(f"{name:18s} MAE {results[name]['MAE']:>9,.0f}  R2 {results[name]['R2']:.3f}"
              f"  ({time.time() - start:.0f}s)")

    best_name = min(results, key=lambda n: results[n]["MAE"])
    best = fitted[best_name]
    print(f"\nBest model by MAE: {best_name}")

    # Fair price range from the best model's errors on the test set
    ratio = y_test.to_numpy() / best.predict(X_test)
    ratio_low, ratio_high = np.quantile(ratio, [FAIR_LOW_Q, FAIR_HIGH_Q])
    print(f"Fair range = predicted x [{ratio_low:.2f}, {ratio_high:.2f}]")

    bundle = {
        "model": best,
        "model_name": best_name,
        "ratio_low": float(ratio_low),
        "ratio_high": float(ratio_high),
        "metrics": results[best_name],
        "choices": {
            "brands": sorted(df["Brand"].unique()),
            "models_by_brand": {b: sorted(g["Model"].unique()) for b, g in df.groupby("Brand")},
            "fuels": sorted(df["Fuel"].unique()),
            "gearboxes": sorted(df["Gearbox"].unique()),
            "locations": sorted(df["Location"].unique()),
            "conditions": sorted(df["Condition"].unique()),
            "origins": sorted(df["Origin"].unique()),
        },
    }
    MODEL_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(bundle, MODEL_PATH, compress=3)
    print(f"Saved {MODEL_PATH.relative_to(ROOT)}")

    # Markdown table for the README
    lines = ["| Model | MAE (MAD) | RMSE (MAD) | R² | Median % error |", "|---|---|---|---|---|"]
    for name, m in results.items():
        lines.append(f"| {name} | {m['MAE']:,.0f} | {m['RMSE']:,.0f} | {m['R2']:.3f} | {m['Median % error']:.1f}% |")
    lines += ["", f"Best model: **{best_name}**. Fair range = predicted price x "
              f"[{ratio_low:.2f}, {ratio_high:.2f}] (covers the middle 80% of test listings).",
              f"Rows used: {len(df):,} (train {len(X_train):,} / test {len(X_test):,})."]
    RESULTS_PATH.parent.mkdir(exist_ok=True)
    RESULTS_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Saved {RESULTS_PATH.relative_to(ROOT)}")
    print(json.dumps({k: round(v, 3) for k, v in results[best_name].items()}))


if __name__ == "__main__":
    main()
