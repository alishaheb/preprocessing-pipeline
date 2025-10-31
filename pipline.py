# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse
import os
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Classification metrics
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    r2_score, mean_absolute_error, mean_squared_error
)

# Models (classification)
from sklearn.linear_model import LogisticRegression, Ridge, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, SVR
from sklearn.neural_network import MLPClassifier, MLPRegressor

# Warnings off to keep CLI clean
import warnings
warnings.filterwarnings("ignore")


# ---------------------- Utilities ----------------------

ID_LIKE_COLUMNS = {"id", "ID", "Id", "index", "Index", "PassengerId", "Name", "Ticket", "Cabin"}

def is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def infer_task_from_target(y: pd.Series, multiclass_threshold: int = 3) -> Tuple[str, np.ndarray, Dict]:
    """
    Return (task, y_encoded, mapping_info)
    task in {'classification', 'regression'}
    For classification, string labels are encoded to integers.
    """
    mapping = {}
    y_clean = y.dropna()
    # Try numeric first
    if is_numeric_series(y):
        n_unique = int(y_clean.nunique())
        # Heuristic: many unique numeric values -> regression
        if n_unique >= multiclass_threshold and n_unique > 10:
            return "regression", y.to_numpy(), mapping
        elif n_unique == 2:
            # binary classification
            # Ensure labels are 0/1
            uniq = sorted(y_clean.unique())
            mapping = {uniq[0]: 0, uniq[1]: 1}
            return "classification", y.map(mapping).to_numpy(), {"classes_": uniq, "mapping": mapping}
        else:
            # few unique numeric categories -> classification
            uniq = sorted(y_clean.unique())
            mapping = {val: i for i, val in enumerate(uniq)}
            return "classification", y.map(mapping).to_numpy(), {"classes_": uniq, "mapping": mapping}
    else:
        # Non-numeric: definitely classification; encode
        uniq = y_clean.astype(str).unique()
        uniq_sorted = sorted(uniq, key=lambda v: str(v))
        mapping = {val: i for i, val in enumerate(uniq_sorted)}
        return "classification", y.astype(str).map(mapping).to_numpy(), {"classes_": uniq_sorted, "mapping": mapping}


def drop_id_like(df: pd.DataFrame, extra_drops: Optional[List[str]] = None) -> pd.DataFrame:
    cols = set(df.columns)
    drops = [c for c in cols if c in ID_LIKE_COLUMNS]
    if extra_drops:
        drops.extend([c for c in extra_drops if c in cols])
    if drops:
        return df.drop(columns=list(set(drops)))
    return df


# ---------------------- Preprocessor ----------------------

class DataPreprocessor:
    """
    - Numeric: median impute -> optional StandardScaler
    - Categorical: most_frequent impute -> OneHotEncoder(handle_unknown='infrequent_if_exist')
      with max_categories guard to avoid exploding dimensionality on high-cardinality features.
    Produces dense output for models that need dense (e.g., GaussianNB, MLP).
    """
    def __init__(
        self,
        target: str,
        scale_numeric: bool = True,
        max_categories: int = 40,  # tighter cap keeps features manageable on generic datasets
    ):
        self.target = target
        self.scale_numeric = scale_numeric
        self.max_categories = max_categories
        self.categorical: Optional[List[str]] = None
        self.numeric: Optional[List[str]] = None
        self._transformer: Optional[ColumnTransformer] = None

    def _infer_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        features = [c for c in df.columns if c != self.target]
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in features if c not in numeric]
        return categorical, numeric

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' not found. Columns: {list(df.columns)}")

        cats, nums = self._infer_feature_types(df)
        self.categorical, self.numeric = cats, nums

        num_steps = [("imputer", SimpleImputer(strategy="median"))]
        if self.scale_numeric:
            num_steps.append(("scaler", StandardScaler()))
        num_pipe = Pipeline(num_steps)

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(
                handle_unknown="infrequent_if_exist",
                max_categories=self.max_categories,
                sparse=False))
        ])

        self._transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric),
                ("cat", cat_pipe, self.categorical),
            ],
            remainder="drop",
            sparse_threshold=0.0,  # force dense ndarray
        )
        return self

    @property
    def transformer(self) -> ColumnTransformer:
        if self._transformer is None:
            raise RuntimeError("Call fit(df) before accessing transformer.")
        return self._transformer


# ---------------------- Model Factories ----------------------

class ClassificationFactory:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def build(self, n_classes: int) -> Dict[str, object]:
        models = {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=self.random_state),
            "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=self.random_state),
            "MLP": MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=400, random_state=self.random_state),
        }
        # GaussianNB works best when features are scaled-ish and dense; keep it as a quick baseline
        models["NaiveBayes"] = GaussianNB()
        return models


class RegressionFactory:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def build(self) -> Dict[str, object]:
        return {
            "LinearRegression": LinearRegression(),
            "Ridge": Ridge(random_state=self.random_state),
            "RandomForestRegressor": RandomForestRegressor(n_estimators=400, random_state=self.random_state),
            "SVR_RBF": SVR(kernel="rbf"),
            "MLPRegressor": MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=400, random_state=self.random_state),
            "GBR": GradientBoostingRegressor(random_state=self.random_state),
        }


# ---------------------- Evaluators ----------------------

class Evaluator:
    @staticmethod
    def _safe_auc(estimator: Pipeline, X: pd.DataFrame, y: np.ndarray, n_classes: int) -> float:
        # Works for binary and multiclass (OVR)
        try:
            proba = estimator.predict_proba(X)
            if proba.ndim == 1 or proba.shape[1] == 1:  # rare case
                return np.nan
            if n_classes == 2:
                return roc_auc_score(y, proba[:, 1])
            else:
                return roc_auc_score(y, proba, multi_class="ovr")
        except Exception:
            try:
                dec = estimator.decision_function(X)
                if n_classes == 2:
                    return roc_auc_score(y, dec)
                else:
                    # decision_function can be (n_samples, n_classes)
                    return roc_auc_score(y, dec, multi_class="ovr")
            except Exception:
                return np.nan

    def eval_classification(self, model: Pipeline, X_tr, y_tr, X_te, y_te) -> Dict[str, float]:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        n_classes = int(np.unique(y_te).size)
        metrics = {
            "accuracy": accuracy_score(y_te, y_pred),
            "precision_macro": precision_score(y_te, y_pred, average="macro", zero_division=0),
            "recall_macro": recall_score(y_te, y_pred, average="macro", zero_division=0),
            "f1_macro": f1_score(y_te, y_pred, average="macro", zero_division=0),
            "roc_auc": self._safe_auc(model, X_te, y_te, n_classes=n_classes),
        }
        # For binary, also add f1 (binary) for convenience
        if n_classes == 2:
            metrics["f1_binary"] = f1_score(y_te, y_pred, zero_division=0)
        return metrics

    def eval_regression(self, model: Pipeline, X_tr, y_tr, X_te, y_te) -> Dict[str, float]:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        return {
            "r2": r2_score(y_te, y_pred),
            "mae": mean_absolute_error(y_te, y_pred),
            "rmse": mean_squared_error(y_te, y_pred, squared=False),
        }


# ---------------------- Orchestrator ----------------------

class AutoMLPipeline:
    def __init__(
        self,
        csv_path: str,
        target: str,
        task: str = "auto",  # 'auto' | 'classification' | 'regression'
        test_size: float = 0.25,
        random_state: int = 42,
        scale_numeric: bool = True,
        max_categories: int = 40,
        drop_cols: Optional[List[str]] = None,
    ):
        self.csv_path = csv_path
        self.target = target
        self.task = task
        self.test_size = test_size
        self.random_state = random_state
        self.scale_numeric = scale_numeric
        self.max_categories = max_categories
        self.drop_cols = drop_cols or []

        self.df: Optional[pd.DataFrame] = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.task_inferred: Optional[str] = None
        self.target_mapping: Dict = {}

        self.preprocessor: Optional[DataPreprocessor] = None
        self.clf_factory = ClassificationFactory(random_state=random_state)
        self.reg_factory = RegressionFactory(random_state=random_state)
        self.evaluator = Evaluator()

    def load(self) -> "AutoMLPipeline":
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.csv_path}")
        self.df = pd.read_csv(self.csv_path)

        # accept case-insensitive target
        if self.target not in self.df.columns:
            matches = [c for c in self.df.columns if c.lower() == self.target.lower()]
            if matches:
                self.target = matches[0]
            else:
                raise ValueError(f"Target '{self.target}' not found. Columns: {list(self.df.columns)}")

        # drop id-like columns and user-specified
        self.df = drop_id_like(self.df, extra_drops=self.drop_cols)

        # drop rows with missing target
        self.df = self.df.dropna(subset=[self.target])

        # task inference / encoding
        raw_y = self.df[self.target]
        if self.task == "auto":
            task, y_enc, info = infer_task_from_target(raw_y)
        else:
            # If user forced a task, only encode for classification
            if self.task == "classification":
                _, y_enc, info = infer_task_from_target(raw_y)  # reuse encoding logic
                task = "classification"
            elif self.task == "regression":
                # ensure numeric
                if not is_numeric_series(raw_y):
                    y_enc = pd.to_numeric(raw_y, errors="coerce")
                else:
                    y_enc = raw_y.to_numpy()
                if np.isnan(y_enc).any():
                    raise ValueError("Target contains non-numeric values; cannot run regression.")
                info = {}
                task = "regression"
            else:
                raise ValueError("task must be one of {'auto','classification','regression'}")

        self.task_inferred = task
        self.target_mapping = info

        X = self.df.drop(columns=[self.target])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_enc, test_size=self.test_size, random_state=self.random_state,
            stratify=y_enc if task == "classification" else None
        )
        return self

    def build_preprocessor(self) -> "AutoMLPipeline":
        if self.df is None:
            raise RuntimeError("Call load() first.")
        self.preprocessor = DataPreprocessor(
            target=self.target,
            scale_numeric=self.scale_numeric,
            max_categories=self.max_categories
        ).fit(self.df)
        return self

    def run_all(self) -> pd.DataFrame:
        if self.preprocessor is None or self.X_train is None:
            raise RuntimeError("Call load().build_preprocessor() first.")

        results = []
        if self.task_inferred == "classification":
            n_classes = int(np.unique(self.y_train).size)
            for name, clf in self.clf_factory.build(n_classes=n_classes).items():
                pipe = Pipeline([("prep", self.preprocessor.transformer), ("clf", clf)])
                metrics = self.evaluator.eval_classification(pipe, self.X_train, self.y_train, self.X_test, self.y_test)
                results.append({"model": name, **metrics})
            order_key = "f1_macro" if n_classes > 2 else "f1_binary"
            order_key = order_key if order_key in results[0] else "f1_macro"
            df = pd.DataFrame(results).sort_values(order_key, ascending=False).reset_index(drop=True)

        else:  # regression
            for name, reg in self.reg_factory.build().items():
                pipe = Pipeline([("prep", self.preprocessor.transformer), ("reg", reg)])
                metrics = self.evaluator.eval_regression(pipe, self.X_train, self.y_train, self.X_test, self.y_test)
                results.append({"model": name, **metrics})
            df = pd.DataFrame(results).sort_values("rmse", ascending=True).reset_index(drop=True)

        return df


# ---------------------- CLI ----------------------

def parse_args():
    p = argparse.ArgumentParser(description="Auto ML pipeline for generic tabular CSVs")
    p.add_argument("--csv", type=str, required=True, help="Path to CSV")
    p.add_argument("--target", type=str, required=True, help="Target column name (case-insensitive accepted)")
    p.add_argument("--task", type=str, default="auto", choices=["auto", "classification", "regression"],
                   help="Task type. 'auto' will infer from the target.")
    p.add_argument("--test_size", type=float, default=0.25, help="Hold-out size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--no_scale", action="store_true", help="Disable numeric StandardScaler")
    p.add_argument("--max_categories", type=int, default=40, help="Cap for OneHotEncoder to avoid blow-up")
    p.add_argument("--drop_cols", type=str, default="", help="Comma-separated list of columns to drop")
    p.add_argument("--out", type=str, default="ml_results.csv", help="Where to save results CSV")
    return p.parse_args()


def main():
    args = parse_args()
    drop_cols = [c.strip() for c in args.drop_cols.split(",") if c.strip()] if args.drop_cols else []

    pipe = (
        AutoMLPipeline(
            csv_path=args.csv,
            target=args.target,
            task=args.task,
            test_size=args.test_size,
            random_state=args.seed,
            scale_numeric=(not args.no_scale),
            max_categories=args.max_categories,
            drop_cols=drop_cols
        )
        .load()
        .build_preprocessor()
    )
    results = pipe.run_all()

    print("\n=== Task:", pipe.task_inferred, "===")
    if pipe.task_inferred == "classification" and pipe.target_mapping:
        classes = pipe.target_mapping.get("classes_", None)
        if classes is not None:
            print("Class mapping:", {i: c for i, c in enumerate(classes)})

    print("\n=== Results (hold-out set) ===")
    print(results.to_string(index=False))
    results.to_csv(args.out, index=False)
    print(f"\nSaved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()
