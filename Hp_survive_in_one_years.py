#!/usr/bin/env python3
"""
Minimal OOP Logistic Regression with CLI.

- Clean preprocessing (impute+scale numeric, impute+onehot categorical)
- Full metrics (accuracy, precision, recall, F1, ROC-AUC, PR-AUC)
- Confusion matrix, classification report
- 5-fold CV means
- Saves artifacts to --outdir (CSVs, curves, model)

Usage:
  python logreg_cli.py --csv /path/to/data.csv --target <label_col> --outdir ./artifacts --scale
"""

import os, json, argparse
from typing import List, Tuple, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, confusion_matrix, classification_report,
                             roc_curve, precision_recall_curve, average_precision_score)
import joblib


# -------- helpers --------
COMMON_TARGETS = ["target","label","class","y","churn","default","is_fraud","fraud",
                  "diabetes","outcome","Outcome","diagnosis","has_disease"]

def infer_binary_target(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        if c in COMMON_TARGETS or c.lower() in [t.lower() for t in COMMON_TARGETS]:
            u = pd.unique(df[c].dropna())
            if len(u) == 2 or set(u).issubset({0,1,"0","1",True,False}):
                return c
    last = df.columns[-1]
    u = pd.unique(df[last].dropna())
    if len(u) == 2 or set(u).issubset({0,1,"0","1",True,False}):
        return last
    for c in df.columns:
        if len(pd.unique(df[c].dropna())) == 2:
            return c
    return None

def to_binary(y: pd.Series, notes: Dict[str,Any]) -> pd.Series:
    if y.dtype == bool:
        return y.astype(int)
    mapping = {"yes":1,"no":0,"true":1,"false":0,"positive":1,"negative":0,"pos":1,"neg":0}
    if y.dtype == object:
        y = y.astype(str).str.strip().str.lower().map(lambda v: mapping.get(v,v))
    try:
        return y.astype(int)
    except Exception:
        uniq = sorted(pd.unique(y.dropna()))
        enc = {v:i for i,v in enumerate(uniq)}
        y2 = y.map(enc)
        if len(uniq) > 2:
            counts = y2.value_counts()
            top2 = counts.index[:2].tolist()
            y2 = y2.apply(lambda v: 1 if v == top2[0] else 0)
            notes["note_multiclass_coerced"] = "Target had >2 classes; coerced to two most frequent."
        return y2

def detect_columns(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat

def onehot_compat():
    # Support sklearn old/new API (sparse vs sparse_output)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


# -------- minimal OOP --------
class SimplePreprocessor:
    """Build ColumnTransformer for numeric + categorical features."""
    def __init__(self, numeric: List[str], categorical: List[str],
                 num_impute="median", cat_impute="most_frequent", scale=True):
        self.numeric = numeric
        self.categorical = categorical
        self.num_impute = num_impute
        self.cat_impute = cat_impute
        self.scale = scale

    def build(self) -> ColumnTransformer:
        num_steps = [("imputer", SimpleImputer(strategy=self.num_impute))]
        if self.scale:
            num_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
        else:
            num_steps.append(("noop", FunctionTransformer(lambda x: x, validate=False)))

        cat_steps = [("imputer", SimpleImputer(strategy=self.cat_impute)),
                     ("encoder", onehot_compat())]

        return ColumnTransformer(
            transformers=[
                ("num", Pipeline(num_steps), self.numeric),
                ("cat", Pipeline(cat_steps), self.categorical),
            ],
            remainder="drop",
            verbose_feature_names_out=False
        )

class LogRegExperiment:
    """Train/evaluate Logistic Regression and save outputs."""
    def __init__(self, preproc: ColumnTransformer, random_state: int = 42):
        self.model = Pipeline([
            ("preprocess", preproc),
            ("clf", LogisticRegression(max_iter=500, class_weight="balanced",
                                       solver="lbfgs", random_state=random_state))
        ])

    def fit(self, X: pd.DataFrame, y: pd.Series):
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame):
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame):
        return self.model.predict_proba(X)[:,1]

    def evaluate(self, X_test, y_test) -> Dict[str, float]:
        y_pred = self.predict(X_test)
        out = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "precision": float(precision_score(y_test, y_pred, zero_division=0)),
            "recall": float(recall_score(y_test, y_pred, zero_division=0)),
            "f1": float(f1_score(y_test, y_pred, zero_division=0)),
        }
        if len(np.unique(y_test)) == 2:
            y_proba = self.predict_proba(X_test)
            try: out["roc_auc"] = float(roc_auc_score(y_test, y_proba))
            except: out["roc_auc"] = float("nan")
            try: out["average_precision"] = float(average_precision_score(y_test, y_proba))
            except: out["average_precision"] = float("nan")
        else:
            out["roc_auc"] = float("nan")
            out["average_precision"] = float("nan")
        return out

    def cv_scores(self, X, y, random_state=42) -> Dict[str,float]:
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        scoring = {"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}
        res = cross_validate(self.model, X, y, scoring=scoring, cv=cv, n_jobs=None, return_train_score=False)
        return {f"cv_mean_{k[5:]}": float(np.nanmean(v)) for k,v in res.items() if k.startswith("test_")}

    def save_curves(self, X_test, y_test, outdir: str) -> Dict[str,bool]:
        os.makedirs(outdir, exist_ok=True)
        ok = {"roc": False, "pr": False}
        if len(np.unique(y_test)) != 2:
            return ok
        y_proba = self.predict_proba(X_test)
        # ROC
        try:
            fpr, tpr, _ = roc_curve(y_test, y_proba)
            plt.figure(); plt.plot(fpr, tpr)
            plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
            plt.title("ROC Curve (LogReg)"); plt.savefig(os.path.join(outdir,"roc_curve.png"), bbox_inches="tight")
            plt.close(); ok["roc"] = True
        except: pass
        # PR
        try:
            prec, rec, _ = precision_recall_curve(y_test, y_proba)
            plt.figure(); plt.plot(rec, prec)
            plt.xlabel("Recall"); plt.ylabel("Precision")
            plt.title("Precision-Recall (LogReg)"); plt.savefig(os.path.join(outdir,"pr_curve.png"), bbox_inches="tight")
            plt.close(); ok["pr"] = True
        except: pass
        return ok


# -------- CLI --------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, required=True, help="Path to CSV")
    ap.add_argument("--target", type=str, default=None, help="Binary target column (optional)")
    ap.add_argument("--outdir", type=str, default="./artifacts", help="Where to save outputs")
    ap.add_argument("--test-size", type=float, default=0.2)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--scale", action="store_true", help="Scale numeric features (StandardScaler)")
    ap.add_argument("--num-impute", type=str, default="median", choices=["mean","median","most_frequent","constant"])
    ap.add_argument("--cat-impute", type=str, default="most_frequent", choices=["most_frequent","constant"])
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    df = pd.read_csv(args.csv)

    notes: Dict[str,Any] = {}
    target = args.target or infer_binary_target(df)
    if target is None:
        target = df.columns[-1]
        notes["warning"] = f"Could not infer target; defaulted to last column '{target}'."
    else:
        notes["target"] = target

    y_raw = df[target]
    X_raw = df.drop(columns=[target])
    y = to_binary(y_raw, notes)
    num_cols, cat_cols = detect_columns(X_raw)

    preproc = SimplePreprocessor(num_cols, cat_cols,
                                 num_impute=args.num_impute,
                                 cat_impute=args.cat_impute,
                                 scale=args.scale).build()

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=args.test_size, random_state=args.random_state, stratify=y
    )

    exp = LogRegExperiment(preproc, random_state=args.random_state)
    exp.fit(X_tr, y_tr)

    # Holdout metrics
    holdout = exp.evaluate(X_te, y_te)
    # CV means
    cv_means = exp.cv_scores(X_raw, y, random_state=args.random_state)
    # Confusion matrix + report
    y_pred = exp.predict(X_te)
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, zero_division=0, output_dict=True)

    # Save everything
    pd.DataFrame([holdout]).to_csv(os.path.join(args.outdir, "logreg_metrics_holdout.csv"), index=False)
    pd.DataFrame([cv_means]).to_csv(os.path.join(args.outdir, "logreg_metrics_cv.csv"), index=False)
    pd.DataFrame(cm, columns=["Pred 0","Pred 1"], index=["True 0","True 1"]) \
        .to_csv(os.path.join(args.outdir, "confusion_matrix.csv"))
    pd.DataFrame(report).T.to_csv(os.path.join(args.outdir, "classification_report.csv"))

    # Curves + model + summary
    curves_ok = exp.save_curves(X_te, y_te, args.outdir)
    joblib.dump(exp.model, os.path.join(args.outdir, "logreg_model.pkl"))

    summary = {
        "rows": int(len(df)),
        "features": int(X_raw.shape[1]),
        "numeric_features": int(len(num_cols)),
        "categorical_features": int(len(cat_cols)),
        "target": target,
        "holdout": holdout,
        "cv_means": cv_means,
        "curves": curves_ok,
        "notes": notes
    }
    with open(os.path.join(args.outdir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Also print key metrics to console (nice for CLI)
    print("\n=== Holdout Metrics ===")
    for k in ["accuracy","precision","recall","f1","roc_auc","average_precision"]:
        print(f"{k:>18}: {summary['holdout'].get(k, float('nan')):.4f}")
    print("\n=== CV Means (5-fold) ===")
    for k,v in summary["cv_means"].items():
        print(f"{k:>18}: {v:.4f}")
    print(f"\nArtifacts saved to: {os.path.abspath(args.outdir)}\n")


if __name__ == "__main__":
    main()
