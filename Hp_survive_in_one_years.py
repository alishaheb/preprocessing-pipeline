#!/usr/bin/env python3
"""
Logistic Regression (OOP + Design Patterns) with full metrics.

- Strategy + Factory patterns for preprocessing (imputer, scaler, encoder).
- ColumnTransformer handles numeric + categorical.
- Saves holdout metrics, 5-fold CV metrics, confusion matrix, classification report,
  ROC and PR curves, and the trained model.

Usage:
  python logreg_oop_pipeline.py \
    --csv /mnt/data/Hp_Preprocessed_data.csv \
    --target <your_target_col> \
    --outdir ./artifacts \
    --test-size 0.2 \
    --random-state 42
If --target is omitted, the script will try to infer a binary target.
"""

import os
import json
import argparse
from typing import List, Optional, Dict, Any, Protocol

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import joblib


# -----------------------------
# Strategy Interfaces & Concrete Strategies
# -----------------------------
class ScalerStrategy(Protocol):
    def create(self):
        ...

class StandardScalerStrategy:
    def create(self):
        return StandardScaler(with_mean=True, with_std=True)

class NoScalerStrategy:
    def create(self):
        return FunctionTransformer(lambda x: x, validate=False)


class ImputerStrategy(Protocol):
    def numeric(self):
        ...
    def categorical(self):
        ...

class SimpleImputerStrategy:
    def __init__(self, num_strategy: str = "median", cat_strategy: str = "most_frequent"):
        self.num_strategy = num_strategy
        self.cat_strategy = cat_strategy

    def numeric(self):
        return SimpleImputer(strategy=self.num_strategy)

    def categorical(self):
        return SimpleImputer(strategy=self.cat_strategy)


class EncoderStrategy(Protocol):
    def create(self):
        ...

class OneHotEncoderStrategy:
    def __init__(self, handle_unknown="ignore"):
        self.handle_unknown = handle_unknown

    def create(self):
        # Handle both old and new sklearn APIs
        try:
            return OneHotEncoder(handle_unknown=self.handle_unknown, sparse_output=False)
        except TypeError:
            return OneHotEncoder(handle_unknown=self.handle_unknown, sparse=False)


# -----------------------------
# Factory
# -----------------------------
class PreprocessingFactory:
    @staticmethod
    def make_scaler(name: str) -> ScalerStrategy:
        name = (name or "standard").lower()
        if name == "none":
            return NoScalerStrategy()
        return StandardScalerStrategy()

    @staticmethod
    def make_imputer(num: str = "median", cat: str = "most_frequent") -> ImputerStrategy:
        return SimpleImputerStrategy(num, cat)

    @staticmethod
    def make_encoder(name: str = "onehot") -> EncoderStrategy:
        return OneHotEncoderStrategy()


# -----------------------------
# Preprocessing Pipeline
# -----------------------------
class PreprocessingPipeline:
    def __init__(
        self,
        numeric_cols: List[str],
        categorical_cols: List[str],
        scaler: ScalerStrategy,
        imputer: ImputerStrategy,
        encoder: EncoderStrategy,
    ):
        self.numeric_cols = numeric_cols
        self.categorical_cols = categorical_cols
        self.scaler = scaler
        self.imputer = imputer
        self.encoder = encoder
        self.column_transformer: Optional[ColumnTransformer] = None

    def build(self) -> ColumnTransformer:
        num_pipe = Pipeline(steps=[
            ("imputer", self.imputer.numeric()),
            ("scaler", self.scaler.create()),
        ])
        cat_pipe = Pipeline(steps=[
            ("imputer", self.imputer.categorical()),
            ("encoder", self.encoder.create()),
        ])
        self.column_transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric_cols),
                ("cat", cat_pipe, self.categorical_cols),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        return self.column_transformer


# -----------------------------
# Trainer / Evaluator
# -----------------------------
class LogisticRegressionTrainer:
    def __init__(self, preproc: PreprocessingPipeline, random_state: int = 42):
        self.preproc = preproc
        self.random_state = random_state
        self.model: Optional[Pipeline] = None

    def build(self) -> Pipeline:
        pre = self.preproc.build()
        clf = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            solver="lbfgs",
            random_state=self.random_state
        )
        self.model = Pipeline(steps=[("preprocess", pre), ("clf", clf)])
        return self.model

    def fit(self, X: pd.DataFrame, y: pd.Series):
        if self.model is None:
            self.build()
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


# -----------------------------
# Helpers
# -----------------------------
COMMON_TARGETS = [
    "target", "label", "class", "y",
    "churn", "default", "is_fraud", "fraud",
    "diabetes", "outcome", "Outcome", "diagnosis", "has_disease"
]

def infer_binary_target(df: pd.DataFrame) -> Optional[str]:
    cols = df.columns.tolist()
    # common names
    for c in cols:
        if c in COMMON_TARGETS or c.lower() in [t.lower() for t in COMMON_TARGETS]:
            uniq = pd.unique(df[c].dropna())
            if len(uniq) == 2 or set(uniq).issubset({0, 1, "0", "1", True, False}):
                return c
    # last column heuristic
    last = cols[-1]
    uniq = pd.unique(df[last].dropna())
    if len(uniq) == 2 or set(uniq).issubset({0, 1, "0", "1", True, False}):
        return last
    # any column with exactly 2 unique vals
    for c in cols:
        uniq = pd.unique(df[c].dropna())
        if len(uniq) == 2:
            return c
    return None


def to_binary(y: pd.Series, notes: Dict[str, Any]) -> pd.Series:
    y2 = y.copy()
    if y2.dtype == bool:
        return y2.astype(int)

    mapping = {
        "yes": 1, "no": 0,
        "true": 1, "false": 0,
        "positive": 1, "negative": 0,
        "pos": 1, "neg": 0
    }
    if y2.dtype == object:
        y2 = y2.astype(str).str.strip().str.lower().map(lambda v: mapping.get(v, v))

    try:
        y2 = y2.astype(int)
    except Exception:
        uniq = sorted(pd.unique(y2.dropna()).tolist())
        encoder = {val: idx for idx, val in enumerate(uniq)}
        y2 = y2.map(encoder)
        if len(uniq) > 2:
            # reduce to binary using top-2 frequent classes
            counts = y2.value_counts()
            top_two = counts.index[:2].tolist()
            y2 = y2.apply(lambda v: 1 if v == top_two[0] else 0)
            notes["note_multiclass_coerced"] = (
                "Target had >2 classes; coerced to binary via the two most frequent classes."
            )
    return y2


def detect_column_types(X: pd.DataFrame):
    numeric_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return numeric_cols, categorical_cols


def plot_roc(y_true, y_score, out_path) -> bool:
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        if len(fpr) == 0:
            return False
        plt.figure()
        plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (Logistic Regression)")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False


def plot_pr(y_true, y_score, out_path) -> bool:
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        if len(prec) == 0:
            return False
        plt.figure()
        plt.plot(rec, prec)
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision-Recall Curve (Logistic Regression)")
        plt.savefig(out_path, bbox_inches="tight")
        plt.close()
        return True
    except Exception:
        return False


# -----------------------------
# Main
# -----------------------------
def main():
    parser = argparse.ArgumentParser(description="Logistic Regression (OOP + Design Patterns)")
    parser.add_argument("--csv", type=str, default="Hp_treatment.csv",
                        help="Path to input CSV")
    parser.add_argument("--target", type=str, default=None,
                        help="Name of the target column (binary). If omitted, will try to infer.")
    parser.add_argument("--outdir", type=str, default="./artifacts",
                        help="Directory to save outputs")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test split size")
    parser.add_argument("--random-state", type=int, default=42, help="Random state")
    parser.add_argument("--scaler", type=str, default="standard", choices=["standard", "none"],
                        help="Scaling strategy for numeric features")
    parser.add_argument("--num-impute", type=str, default="median",
                        choices=["mean", "median", "most_frequent", "constant"],
                        help="Numeric imputation strategy")
    parser.add_argument("--cat-impute", type=str, default="most_frequent",
                        choices=["most_frequent", "constant"],
                        help="Categorical imputation strategy")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load data
    if not os.path.exists(args.csv):
        raise FileNotFoundError(f"CSV not found: {args.csv}")
    df = pd.read_csv(args.csv)

    notes: Dict[str, Any] = {}
    # Determine target
    target_col = args.target or infer_binary_target(df)
    if target_col is None:
        target_col = df.columns[-1]
        notes["warning"] = (
            f"Could not confidently infer a binary target. Defaulted to last column '{target_col}'."
        )
    else:
        notes["target_detected"] = f"Using target: '{target_col}'"

    y_raw = df[target_col]
    X_raw = df.drop(columns=[target_col])
    y = to_binary(y_raw, notes)

    # Column typing
    numeric_cols, categorical_cols = detect_column_types(X_raw)
    notes["numeric_cols"] = numeric_cols
    notes["categorical_cols"] = categorical_cols

    # Build preprocessing via Factory/Strategy
    scaler = PreprocessingFactory.make_scaler(args.scaler)
    imputer = PreprocessingFactory.make_imputer(num=args.num_impute, cat=args.cat_impute)
    encoder = PreprocessingFactory.make_encoder("onehot")
    preproc = PreprocessingPipeline(numeric_cols, categorical_cols, scaler, imputer, encoder)

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X_raw, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    # Train
    trainer = LogisticRegressionTrainer(preproc=preproc, random_state=args.random_state)
    model = trainer.build()
    model.fit(X_train, y_train)

    # Inference
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)

    # Holdout metrics
    metrics: Dict[str, Any] = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
    }

    # AUCs only if valid
    if len(np.unique(y_test)) == 2:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_test, y_proba))
        except Exception:
            pass
        try:
            metrics["average_precision"] = float(average_precision_score(y_test, y_proba))
        except Exception:
            pass

    # Confusion matrix & classification report
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)

    # Cross-validation
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=args.random_state)
    cv_scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }
    cv_results = cross_validate(model, X_raw, y, scoring=cv_scoring, cv=cv, n_jobs=None, return_train_score=False)
    cv_summary = {k.replace("test_", "cv_mean_"): float(np.nanmean(v))
                  for k, v in cv_results.items() if k.startswith("test_")}

    # Save artifacts
    pd.DataFrame([metrics]).to_csv(os.path.join(args.outdir, "logreg_metrics_holdout.csv"), index=False)
    pd.DataFrame([cv_summary]).to_csv(os.path.join(args.outdir, "logreg_metrics_cv.csv"), index=False)
    pd.DataFrame(cm, columns=["Pred 0", "Pred 1"], index=["True 0", "True 1"]) \
        .to_csv(os.path.join(args.outdir, "confusion_matrix.csv"))
    pd.DataFrame(report).T.to_csv(os.path.join(args.outdir, "classification_report.csv"))

    # Curves
    roc_ok = plot_roc(y_test, y_proba, os.path.join(args.outdir, "roc_curve.png"))
    pr_ok = plot_pr(y_test, y_proba, os.path.join(args.outdir, "pr_curve.png"))
    notes["roc_curve_saved"] = bool(roc_ok)
    notes["pr_curve_saved"] = bool(pr_ok)

    # Model
    joblib.dump(model, os.path.join(args.outdir, "logreg_model.pkl"))

    # Summary JSON
    summary = {
        "csv": args.csv,
        "rows": int(len(df)),
        "features": int(X_raw.shape[1]),
        "numeric_features": int(len(numeric_cols)),
        "categorical_features": int(len(categorical_cols)),
        "target_col": target_col,
        "holdout_metrics": metrics,
        "cv_summary": cv_summary,
        "notes": notes,
    }
    with open(os.path.join(args.outdir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
