#!/usr/bin/env python3
"""
CSV-only Logistic Regression (no CLI)

Edit the CONFIG section below and run:
    python logreg_no_cli.py
"""

# ======================
# CONFIG (edit these)
# ======================
CSV_PATH     = "/mnt/data/Hp_Preprocessed_data.csv"  # path to your CSV
TARGET_COL   = None       # e.g., "patient_rural_urban_URBAN" or None to infer
OUTDIR       = "./artifacts"   # output folder
SCALE_NUM    = True       # StandardScaler on numeric features?
TEST_SIZE    = 0.20
RANDOM_STATE = 42
N_SPLITS_CV  = 5

# ======================
# CODE (no CLI below)
# ======================
import os, json
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve, precision_recall_curve, average_precision_score
)
import joblib


# ---------- helpers ----------
COMMON_TARGETS = ["target","label","class","y","churn","default","is_fraud","fraud",
                  "diabetes","outcome","Outcome","diagnosis","has_disease"]

def infer_binary_target(df: pd.DataFrame) -> Optional[str]:
    # 1) common names
    for c in df.columns:
        if c in COMMON_TARGETS or c.lower() in [t.lower() for t in COMMON_TARGETS]:
            u = pd.unique(df[c].dropna())
            if len(u) == 2 or set(u).issubset({0,1,"0","1",True,False}):
                return c
    # 2) last column if it looks binary
    last = df.columns[-1]
    u = pd.unique(df[last].dropna())
    if len(u) == 2 or set(u).issubset({0,1,"0","1",True,False}):
        return last
    # 3) any exact-2-unique column
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

def split_num_cat(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    num = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
    cat = [c for c in X.columns if c not in num]
    return num, cat

def onehot_compat():
    # Support older/newer sklearn versions (sparse vs sparse_output)
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)

def build_preprocessor(num_cols: List[str], cat_cols: List[str], scale: bool) -> ColumnTransformer:
    num_steps = [("imputer", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scaler", StandardScaler(with_mean=True, with_std=True)))
    else:
        num_steps.append(("noop", FunctionTransformer(lambda x: x, validate=False)))

    cat_steps = [("imputer", SimpleImputer(strategy="most_frequent")),
                 ("encoder", onehot_compat())]

    return ColumnTransformer(
        transformers=[
            ("num", Pipeline(num_steps), num_cols),
            ("cat", Pipeline(cat_steps), cat_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

def plot_curves(y_true, y_score, outdir) -> Dict[str,bool]:
    ok = {"roc": False, "pr": False}
    if len(np.unique(y_true)) != 2:
        return ok
    # ROC
    try:
        fpr, tpr, _ = roc_curve(y_true, y_score)
        plt.figure(); plt.plot(fpr, tpr)
        plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
        plt.title("ROC Curve (LogReg)"); plt.savefig(os.path.join(outdir,"roc_curve.png"), bbox_inches="tight")
        plt.close(); ok["roc"] = True
    except Exception:
        pass
    # PR
    try:
        prec, rec, _ = precision_recall_curve(y_true, y_score)
        plt.figure(); plt.plot(rec, prec)
        plt.xlabel("Recall"); plt.ylabel("Precision")
        plt.title("Precision-Recall (LogReg)"); plt.savefig(os.path.join(outdir,"pr_curve.png"), bbox_inches="tight")
        plt.close(); ok["pr"] = True
    except Exception:
        pass
    return ok


def run_logreg(
    csv_path: str = CSV_PATH,
    target_col: Optional[str] = TARGET_COL,
    outdir: str = OUTDIR,
    scale_num: bool = SCALE_NUM,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
    n_splits_cv: int = N_SPLITS_CV
) -> Dict[str, Any]:
    """Run end-to-end logistic regression on a CSV and save artifacts."""
    os.makedirs(outdir, exist_ok=True)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    df = pd.read_csv(csv_path)

    # Determine target
    notes: Dict[str,Any] = {}
    target = target_col or infer_binary_target(df)
    if target is None:
        target = df.columns[-1]
        notes["warning"] = f"Could not infer target; defaulted to last column '{target}'."
    else:
        notes["target"] = target

    y_raw = df[target]
    X_raw = df.drop(columns=[target])
    y = to_binary(y_raw, notes)

    # Columns & preprocessor
    num_cols, cat_cols = split_num_cat(X_raw)
    preproc = build_preprocessor(num_cols, cat_cols, scale=scale_num)

    # Model
    model = Pipeline([
        ("preprocess", preproc),
        ("clf", LogisticRegression(max_iter=500, class_weight="balanced",
                                   solver="lbfgs", random_state=random_state))
    ])

    # Split & train
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_raw, y, test_size=test_size, random_state=random_state, stratify=y
    )
    model.fit(X_tr, y_tr)

    # Predictions & metrics
    y_pred = model.predict(X_te)
    metrics = {
        "accuracy": float(accuracy_score(y_te, y_pred)),
        "precision": float(precision_score(y_te, y_pred, zero_division=0)),
        "recall": float(recall_score(y_te, y_pred, zero_division=0)),
        "f1": float(f1_score(y_te, y_pred, zero_division=0)),
        "roc_auc": float("nan"),
        "average_precision": float("nan"),
    }

    curves = {"roc": False, "pr": False}
    if len(np.unique(y_te)) == 2:
        try:
            y_proba = model.predict_proba(X_te)[:,1]
            metrics["roc_auc"] = float(roc_auc_score(y_te, y_proba))
            metrics["average_precision"] = float(average_precision_score(y_te, y_proba))
            curves = plot_curves(y_te, y_proba, outdir)
        except Exception:
            pass

    # Confusion matrix & report
    cm = confusion_matrix(y_te, y_pred)
    report = classification_report(y_te, y_pred, zero_division=0, output_dict=True)

    # 5-fold CV
    cv = StratifiedKFold(n_splits=n_splits_cv, shuffle=True, random_state=random_state)
    scoring = {"accuracy":"accuracy","precision":"precision","recall":"recall","f1":"f1","roc_auc":"roc_auc"}
    cv_res = cross_validate(model, X_raw, y, scoring=scoring, cv=cv, n_jobs=None, return_train_score=False)
    cv_means = {f"cv_mean_{k[5:]}": float(np.nanmean(v)) for k,v in cv_res.items() if k.startswith("test_")}

    # Save artifacts
    pd.DataFrame([metrics]).to_csv(os.path.join(outdir, "logreg_metrics_holdout.csv"), index=False)
    pd.DataFrame([cv_means]).to_csv(os.path.join(outdir, "logreg_metrics_cv.csv"), index=False)
    pd.DataFrame(cm, columns=["Pred 0","Pred 1"], index=["True 0","True 1"])) \
        .to_csv(os.path.join(outdir, "confusion_matrix.csv"))
    pd.DataFrame(report).T.to_csv(os.path.join(outdir, "classification_report.csv"))
    joblib.dump(model, os.path.join(outdir, "logreg_model.pkl"))

    summary = {
        "csv": csv_path,
        "rows": int(len(df)),
        "features": int(X_raw.shape[1]),
        "numeric_features": int(len(num_cols)),
        "categorical_features": int(len(cat_cols)),
        "target": target,
        "holdout": metrics,
        "cv_means": cv_means,
        "curves_saved": curves,
        "notes": notes
    }
    with open(os.path.join(outdir, "run_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # Optional: quick print to console for convenience
    print("\n=== Holdout Metrics ===")
    for k in ["accuracy","precision","recall","f1","roc_auc","average_precision"]:
        v = summary["holdout"].get(k, float("nan"))
        print(f"{k:>18}: {v:.4f}" if isinstance(v,(int,float)) else f"{k:>18}: {v}")
    print("\n=== CV Means ({}-fold) ===".format(n_splits_cv))
    for k,v in summary["cv_means"].items():
        print(f"{k:>18}: {v:.4f}")
    print(f"\nArtifacts saved to: {os.path.abspath(outdir)}\n")

    return summary


# Run immediately when the file is executed
if __name__ == "__main__":
    run_logreg()
