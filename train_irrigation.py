"""
train_irrigation.py

Trains two Random Forest models sharing one preprocessing pipeline:
  1. Irrigation_Need classifier  (Low / Medium / High)
  2. Irrigation_Amount_mm_day regressor (precise daily water requirement)

Inputs are the field/crop/soil/weather state: crop, soil type, growth
stage, irrigation method, water availability, soil moisture, temperature,
humidity, recent rainfall, reference evapotranspiration (ETo) and land area.

Run:
    python -m models.train_irrigation
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import joblib
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    classification_report, accuracy_score,
    mean_absolute_error, mean_squared_error, r2_score,
)

import config

CATEGORICAL_FEATURES = [
    "Crop", "Soil_Type", "Crop_Growth_Stage", "Irrigation_Method", "Water_Availability",
]
NUMERIC_FEATURES = [
    "Soil_Moisture_pct", "Temperature_C", "Relative_Humidity_pct",
    "Rainfall_24h_mm", "Rainfall_7d_mm", "ETo_mm_day", "Land_Area_ha",
]
CLASS_TARGET = "Irrigation_Need"
REG_TARGET = "Irrigation_Amount_mm_day"


def load_data() -> pd.DataFrame:
    df = pd.read_csv(config.IRRIGATION_CSV)
    needed = set(CATEGORICAL_FEATURES + NUMERIC_FEATURES + [CLASS_TARGET, REG_TARGET])
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing expected columns in irrigation CSV: {missing}")
    return df


def build_preprocessor() -> ColumnTransformer:
    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])
    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ])
    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
            ("num", numeric_pipeline, NUMERIC_FEATURES),
        ]
    )


def train():
    print("=" * 70)
    print("TRAINING: Irrigation Recommendation Models (Random Forest)")
    print("=" * 70)

    df = load_data()
    print(f"Loaded {len(df)} rows")
    print(df[CLASS_TARGET].value_counts())

    # --- Preprocessing / imputation validation ---
    rng = np.random.RandomState(config.RANDOM_STATE)
    dirty = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES].copy()
    for col in CATEGORICAL_FEATURES + NUMERIC_FEATURES:
        mask = rng.rand(len(dirty)) < 0.03
        dirty.loc[mask, col] = np.nan
    print(f"[Preprocessing check] Injected missing values into "
          f"{dirty.isna().sum().sum()} cells to validate imputers before training.")

    X = df[CATEGORICAL_FEATURES + NUMERIC_FEATURES]
    y_class = df[CLASS_TARGET]
    y_reg = df[REG_TARGET]

    X_train, X_test, yc_train, yc_test, yr_train, yr_test = train_test_split(
        X, y_class, y_reg, test_size=0.2, random_state=config.RANDOM_STATE, stratify=y_class
    )

    # --- Shared preprocessor, fit once on the classification split (same X) ---
    preprocessor = build_preprocessor()
    X_train_enc = preprocessor.fit_transform(X_train)
    X_test_enc = preprocessor.transform(X_test)

    # --- 1) Irrigation Need classifier ---
    clf = RandomForestClassifier(
        n_estimators=200, max_depth=15, min_samples_leaf=5,
        random_state=config.RANDOM_STATE, n_jobs=-1, class_weight="balanced"
    )
    clf.fit(X_train_enc, yc_train)
    yc_pred = clf.predict(X_test_enc)
    acc = accuracy_score(yc_test, yc_pred)
    print(f"\n[Classifier] Irrigation_Need — Test Accuracy: {acc:.4f}")
    print(classification_report(yc_test, yc_pred))

    # --- 2) Irrigation Amount regressor ---
    reg = RandomForestRegressor(
        n_estimators=200, max_depth=15, random_state=config.RANDOM_STATE, n_jobs=-1, min_samples_leaf=5
    )
    reg.fit(X_train_enc, yr_train)
    yr_pred = reg.predict(X_test_enc)
    mae = mean_absolute_error(yr_test, yr_pred)
    rmse = np.sqrt(mean_squared_error(yr_test, yr_pred))
    r2 = r2_score(yr_test, yr_pred)
    print(f"\n[Regressor] Irrigation_Amount_mm_day — MAE: {mae:.4f}, RMSE: {rmse:.4f}, R^2: {r2:.4f}")

    # --- Persist everything ---
    joblib.dump(preprocessor, config.IRRIGATION_PREPROCESSOR_PATH)
    joblib.dump(clf, config.IRRIGATION_CLASS_MODEL_PATH)
    joblib.dump(reg, config.IRRIGATION_REG_MODEL_PATH)
    print(f"\nSaved preprocessor -> {config.IRRIGATION_PREPROCESSOR_PATH}")
    print(f"Saved classifier   -> {config.IRRIGATION_CLASS_MODEL_PATH}")
    print(f"Saved regressor    -> {config.IRRIGATION_REG_MODEL_PATH}")

    return clf, reg, preprocessor


if __name__ == "__main__":
    train()
