#!/usr/bin/env python3
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
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


# ---------- Preprocessing ----------

class DataPreprocessor:
    """
    - Numeric: median impute -> StandardScaler
    - Categorical: most_frequent impute -> OneHotEncoder
    Force dense output so models like GaussianNB & MLP are happy.
    """
    def __init__(
        self,
        target: str,
        categorical: Optional[List[str]] = None,
        numeric: Optional[List[str]] = None
    ):
        self.target = target
        self.categorical = categorical
        self.numeric = numeric
        self._transformer: Optional[ColumnTransformer] = None

    def _infer_feature_types(self, df: pd.DataFrame) -> Tuple[List[str], List[str]]:
        features = [c for c in df.columns if c != self.target]
        numeric = [c for c in features if pd.api.types.is_numeric_dtype(df[c])]
        categorical = [c for c in features if c not in numeric]
        return categorical, numeric

    def fit(self, df: pd.DataFrame) -> "DataPreprocessor":
        if self.target not in df.columns:
            raise ValueError(f"Target '{self.target}' not found. Columns: {list(df.columns)}")

        if self.categorical is None or self.numeric is None:
            cats, nums = self._infer_feature_types(df)
            if self.categorical is None:
                self.categorical = cats
            if self.numeric is None:
                self.numeric = nums

        num_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ])

        cat_pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ohe", OneHotEncoder(handle_unknown="ignore", sparse=False))
        ])

        # Force dense output (sparse_threshold=0 ensures np.ndarray output)
        self._transformer = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric),
                ("cat", cat_pipe, self.categorical),
            ],
            remainder="drop",
            sparse_threshold=0.0,
        )
        return self

    @property
    def transformer(self) -> ColumnTransformer:
        if self._transformer is None:
            raise RuntimeError("Call fit(df) before accessing transformer.")
        return self._transformer


# ---------- Models ----------

class ModelFactory:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state

    def build(self) -> Dict[str, object]:
        return {
            "LogisticRegression": LogisticRegression(max_iter=1000, random_state=self.random_state),
            "RandomForest": RandomForestClassifier(n_estimators=300, random_state=self.random_state),
            "SVM_RBF": SVC(kernel="rbf", probability=True, random_state=self.random_state),
            "MLP": MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=300, random_state=self.random_state),
            "NaiveBayes": GaussianNB(),
        }


# ---------- Evaluation ----------

class Evaluator:
    def __init__(self):
        pass

    @staticmethod
    def _safe_auc(estimator: Pipeline, X: pd.DataFrame, y: pd.Series) -> float:
        try:
            proba = estimator.predict_proba(X)[:, 1]
            return roc_auc_score(y, proba)
        except Exception:
            try:
                dec = estimator.decision_function(X)
                return roc_auc_score(y, dec)
            except Exception:
                return np.nan

    def evaluate(self, model: Pipeline, X_tr, y_tr, X_te, y_te) -> Dict[str, float]:
        model.fit(X_tr, y_tr)
        y_pred = model.predict(X_te)
        return {
            "accuracy": accuracy_score(y_te, y_pred),
            "precision": precision_score(y_te, y_pred, zero_division=0),
            "recall": recall_score(y_te, y_pred, zero_division=0),
            "f1": f1_score(y_te, y_pred, zero_division=0),
            "roc_auc": self._safe_auc(model, X_te, y_te),
        }


# ---------- Orchestrator ----------

class MLPipeline:
    def __init__(
        self,
        csv_path: str,
        target: str = "Survived",
        test_size: float = 0.25,
        random_state: int = 42,
    ):
        self.csv_path = csv_path
        self.target = target
        self.test_size = test_size
        self.random_state = random_state

        self.df: Optional[pd.DataFrame] = None
        self.X_train = self.X_test = None
        self.y_train = self.y_test = None
        self.preprocessor: Optional[DataPreprocessor] = None
        self.factory = ModelFactory(random_state=random_state)
        self.evaluator = Evaluator()

    def load(self) -> "MLPipeline":
        if not os.path.exists(self.csv_path):
            raise FileNotFoundError(self.csv_path)
        self.df = pd.read_csv(self.csv_path)

        # accept case-insensitive target
        if self.target not in self.df.columns:
            matches = [c for c in self.df.columns if c.lower() == self.target.lower()]
            if matches:
                self.target = matches[0]
            else:
                raise ValueError(f"Target '{self.target}' not found. Columns: {list(self.df.columns)}")

        # remove obvious IDs if present
        for c in ["PassengerId", "Name", "Ticket", "Cabin", "id", "ID"]:
            if c in self.df.columns and c != self.target:
                # Keep Ticket/Cabin if you know they help (feature engineering),
                # but by default they’re too messy for a quick baseline.
                self.df.drop(columns=[c], inplace=True, errors="ignore")

        # drop rows with missing target
        self.df = self.df.dropna(subset=[self.target])

        # y to 0/1 if binary strings like "yes/no"
        y = self.df[self.target]
        if not pd.api.types.is_numeric_dtype(y):
            uniq = y.dropna().unique()
            if len(uniq) == 2:
                y = y.map({uniq[0]: 0, uniq[1]: 1}).astype(int)
            else:
                y = pd.to_numeric(y, errors="coerce").astype(int)

        X = self.df.drop(columns=[self.target])
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=self.test_size, stratify=y, random_state=self.random_state
        )
        return self

    def build_preprocessor(self) -> "MLPipeline":
        if self.df is None:
            raise RuntimeError("Call load() first.")
        self.preprocessor = DataPreprocessor(target=self.target).fit(self.df)
        return self

    def run_all(self) -> pd.DataFrame:
        if self.preprocessor is None or self.X_train is None:
            raise RuntimeError("Call load().build_preprocessor() first.")

        results = []
        for name, clf in self.factory.build().items():
            pipe = Pipeline([("prep", self.preprocessor.transformer), ("clf", clf)])
            metrics = self.evaluator.evaluate(pipe, self.X_train, self.y_train, self.X_test, self.y_test)
            results.append({"model": name, **metrics})

        df = pd.DataFrame(results).sort_values("f1", ascending=False).reset_index(drop=True)
        return df


# ---------- CLI ----------

def parse_args():
    p = argparse.ArgumentParser(description="OOP ML pipeline: preprocess + 5 models")
    p.add_argument("--csv", type=str, default="car_prices.csv", help="Path to CSV")
    p.add_argument("--target", type=str, default="sellingprice", help="Target column")
    p.add_argument("--test_size", type=float, default=0.25, help="Hold-out size")
    p.add_argument("--seed", type=int, default=42, help="Random seed")
    p.add_argument("--out", type=str, default="ml_results.csv", help="Where to save results CSV")
    return p.parse_args()


def main():
    args = parse_args()
    pipe = (
        MLPipeline(csv_path=args.csv, target=args.target, test_size=args.test_size, random_state=args.seed)
        .load()
        .build_preprocessor()
    )
    results = pipe.run_all()
    print("\n=== Results (hold-out set) ===")
    print(results.to_string(index=False))
    results.to_csv(args.out, index=False)
    print(f"\nSaved: {os.path.abspath(args.out)}")


if __name__ == "__main__":
    main()

