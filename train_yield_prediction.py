"""
train_yield_prediction.py

Trains a Random Forest regressor that predicts crop yield (tonnes/hectare)
given state, season, crop, cultivated area and year.

District_Name is deliberately dropped from the feature set (645 unique
values, and a farmer's live "location" input can't be reliably matched
to this historical district taxonomy via geocoding) — State_Name captures
the regional signal in a way that generalizes to new inputs.

Run:
    python -m models.train_yield_prediction
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import config

CATEGORICAL_FEATURES = ["State_Name", "Season", "Crop"]
NUMERIC_FEATURES = ["Crop_Year", "Area"]
TARGET_COLUMN = "Yield_tonnes_per_hectare"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.YIELD_CSV)
    needed = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [TARGET_COLUMN])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in yield CSV: {missing}")
    # Drop rows with non-positive / absurd yield outliers (defensive cleaning)
    df = df[(df[TARGET_COLUMN] > 0) & (df["Area"] > 0)].copy()
    return df


def build_pipeline() -> Pipeline:
    # Categorical branch: impute missing category with the most frequent value, then one-hot encode.
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    # Numeric branch: impute missing numbers with the column median.
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=18,
        min_samples_leaf=5,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    pipeline = Pipeline(steps=[("preprocess", preprocessor), ("model", model)])
    return pipeline


def train():
    print("=" * 70)
    print("TRAINING: Crop Yield Prediction Model (Random Forest Regressor)")
    print("=" * 70)

    df = load_data()
    print(f"Loaded {len(df)} usable rows across {df['Crop'].nunique()} crops, "
          f"{df['State_Name'].nunique()} states, {df['Season'].nunique()} seasons")

    # --- Preprocessing / imputation validation ---
    # Source data has 0 nulls; briefly validate the imputers against a
    # synthetically-dirtied copy so the pipeline's missing-data handling is
    # actually exercised, not just declared.
    rng = np.random.RandomState(config.RANDOM_STATE)
    dirty = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        mask = rng.rand(len(dirty)) < 0.03
        dirty.loc[mask, col] = np.nan
    print(f"[Preprocessing check] Injected missing values into "
          f"{dirty.isna().sum().sum()} cells to validate imputers before training.")

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=config.RANDOM_STATE
    )

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nMAE  : {mae:.4f} t/ha")
    print(f"RMSE : {rmse:.4f} t/ha")
    print(f"R^2  : {r2:.4f}")

    # Per-crop error breakdown
    results = X_test.copy()
    results["actual"] = y_test.values
    results["predicted"] = y_pred
    results["abs_error"] = (results["actual"] - results["predicted"]).abs()
    print("\nMean Absolute Error by crop:")
    print(results.groupby("Crop")["abs_error"].mean().sort_values())

    joblib.dump(pipeline, config.YIELD_MODEL_PATH)
    print(f"\nSaved full pipeline (preprocessor + model) -> {config.YIELD_MODEL_PATH}")

    return pipeline, r2


if __name__ == "__main__":
    train()
