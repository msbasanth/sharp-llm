"""
Retrain HVSS ML models with current scikit-learn version.

The original models were trained with sklearn 1.2.1 and fail to load on 1.8+.
This script retrains them from the same TrainingData.xlsx using the same
architecture (MLPRegressor) to produce compatible .pkl files.

Usage:
    python scripts/retrain_hvss_models.py
"""

import os
import pickle

import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score

TRAINING_DATA = "src/insights/hvss-calculator-lab-main/TrainingData.xlsx"
OUTPUT_DIR = "src/insights/hvss-calculator-lab-main/models"


def train_model(X: np.ndarray, y: np.ndarray, name: str) -> Pipeline:
    """Train a StandardScaler + MLPRegressor pipeline for HVSS scoring."""
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPRegressor(
            hidden_layer_sizes=(256, 128),
            activation="relu",
            solver="adam",
            alpha=0.001,
            max_iter=5000,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1,
        )),
    ])
    model.fit(X, y)

    # Report cross-validation score
    scores = cross_val_score(model, X, y, cv=5, scoring="r2")
    print(f"  {name}: R²={scores.mean():.4f} (±{scores.std():.4f}), "
          f"samples={len(X)}")
    return model


def main():
    print(f"Loading training data from: {TRAINING_DATA}")
    xl = pd.ExcelFile(TRAINING_DATA)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 1. Exploitability model: [AV, EAC, PR, UI] → Score
    print("\n[1/5] Training exploitability_model...")
    df = xl.parse("Exploitability").dropna()
    X = df[["AV", "EAC", "PR", "UI"]].values
    y = df["Score"].values
    model = train_model(X, y, "exploitability")
    with open(os.path.join(OUTPUT_DIR, "exploitability_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # 2. XCIA model: [AV, EAC, PR, UI, C, I, A] → Score
    print("\n[2/5] Training xcia_model...")
    df = xl.parse("XCIA").dropna()
    X = df[["AV", "EAC", "PR", "UI", "C", "I", "A"]].values
    y = df["Score"].values
    model = train_model(X, y, "xcia")
    with open(os.path.join(OUTPUT_DIR, "xcia_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # 3. XPS model: [AV, EAC, PR, UI, XPS] → Score
    print("\n[3/5] Training xps_model...")
    df = xl.parse("XPS").dropna()
    X = df[["AV", "EAC", "PR", "UI", "XPS"]].values
    y = df["Score"].values
    model = train_model(X, y, "xps")
    with open(os.path.join(OUTPUT_DIR, "xps_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # 4. XSD model: [AV, EAC, PR, UI, XSD] → Score
    print("\n[4/5] Training xsd_model...")
    df = xl.parse("XSD").dropna()
    X = df[["AV", "EAC", "PR", "UI", "XSD"]].values
    y = df["Score"].values
    model = train_model(X, y, "xsd")
    with open(os.path.join(OUTPUT_DIR, "xsd_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    # 5. XHB model: [AV, EAC, PR, UI, XHB] → Score
    print("\n[5/5] Training xhb_model...")
    df = xl.parse("XHB").dropna()
    X = df[["AV", "EAC", "PR", "UI", "XHB"]].values
    y = df["Score"].values
    model = train_model(X, y, "xhb")
    with open(os.path.join(OUTPUT_DIR, "xhb_model.pkl"), "wb") as f:
        pickle.dump(model, f)

    print(f"\n✓ All 5 models retrained and saved to: {OUTPUT_DIR}/")

    # Verify loading
    print("\nVerifying models load correctly...")
    for fname in ["exploitability_model.pkl", "xcia_model.pkl",
                  "xps_model.pkl", "xsd_model.pkl", "xhb_model.pkl"]:
        path = os.path.join(OUTPUT_DIR, fname)
        with open(path, "rb") as f:
            m = pickle.load(f)
        print(f"  ✓ {fname}: {type(m).__name__}")

    # Quick smoke test
    print("\nSmoke test predictions:")
    with open(os.path.join(OUTPUT_DIR, "exploitability_model.pkl"), "rb") as f:
        exp_model = pickle.load(f)
    # Network, Low complexity, No privileges, No UI → high exploitability
    pred = exp_model.predict([[1, 2, 1, 1]])[0]
    print(f"  Exploitability [AV=Network, EAC=Low, PR=None, UI=None]: {pred:.1f}")
    # Physical, Extreme, High privileges, Required → low exploitability
    pred = exp_model.predict([[4, 6, 3, 2]])[0]
    print(f"  Exploitability [AV=Physical, EAC=Extreme, PR=High, UI=Required]: {pred:.1f}")


if __name__ == "__main__":
    main()
