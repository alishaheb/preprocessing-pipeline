#!/usr/bin/env python3
"""
OOP ML Pipeline for `drop_out.csv`
- Preprocess (impute + optional scale)
- Train 5 models (LogReg, RF, GNB, MLP, SVM)
- Stratified 5-Fold CV evaluation (acc, precision, recall, f1_weighted)
- Saves reports to ./reports
"""

from __future__ import annotations
import os
import argparse
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
# Models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC


# ---------- Utilities ----------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().replace("\t", "").replace("  ", " ") for c in df.columns]
    return df


# ---------- Data Layer ----------
@dataclass
class DataConfig:
    csv_path: str
    target_col: str = "Target"
    sep: str = ";"
    na_values: Tuple[str, ...] = ("", "NA", "NaN", "nan", None)


class DatasetLoader:
    def __init__(self, cfg: DataConfig) -> None:
        self.cfg = cfg

    def load(self) -> Tuple[pd.DataFrame, pd.Series]:
        if not os.path.exists(self.cfg.csv_path):
            raise FileNotFoundError(f"CSV not found: {self.cfg.csv_path}")

        df = pd.read_csv(self.cfg.csv_path, sep=self.cfg.sep, na_values=list(self.cfg.na_values))
        df = clean_columns(df)

        # Drop high-missing columns (>70%)
        threshold = 0.7
        missing_ratio = df.isnull().mean()
        cols_to_drop = missing_ratio[missing_ratio > threshold].index.tolist()
        if self.cfg.target_col in cols_to_drop:
            cols_to_drop.remove(self.cfg.target_col)
        if cols_to_drop:
            print(f"Dropping columns with >{int(threshold*100)}% missing: {cols_to_drop}")
            df.drop(columns=cols_to_drop, inplace=True)

        if self.cfg.target_col not in df.columns:
            raise KeyError(f"Target column '{self.cfg.target_col}' not found. "
                           f"Available: {list(df.columns)}")

        y = df[self.cfg.target_col]
        X = df.drop(columns=[self.cfg.target_col])

        return X, y


# ---------- Preprocessing ----------
class PreprocessorFactory:
    def __init__(self, numeric_features: List[str]) -> None:
        self.numeric_features = numeric_features

    def make_no_scale(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", SimpleImputer(strategy="median"), self.numeric_features),
            ],
            remainder="drop",
            n_jobs=None,
        )

    def make_scale(self) -> ColumnTransformer:
        return ColumnTransformer(
            transformers=[
                ("num", Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]), self.numeric_features),
            ],
            remainder="drop",
            n_jobs=None,
        )


# ---------- Models ----------
class ModelFactory:
    def __init__(self, pre_no_scale: ColumnTransformer, pre_scale: ColumnTransformer) -> None:
        self.pre_no_scale = pre_no_scale
        self.pre_scale = pre_scale

    def build_all(self) -> Dict[str, Pipeline]:
        models: Dict[str, Pipeline] = {}

        models["LogisticRegression"] = Pipeline([
            ("prep", self.pre_scale),
            ("clf", LogisticRegression(max_iter=1000, n_jobs=None, multi_class="auto"))
        ])

        models["RandomForest"] = Pipeline([
            ("prep", self.pre_no_scale),
            ("clf", RandomForestClassifier(
                n_estimators=300, random_state=42, n_jobs=-1
            ))
        ])

        models["GaussianNB"] = Pipeline([
            ("prep", self.pre_no_scale),
            ("clf", GaussianNB())
        ])

        models["MLP"] = Pipeline([
            ("prep", self.pre_scale),
            ("clf", MLPClassifier(
                hidden_layer_sizes=(128, 64),
                activation="relu",
                solver="adam",
                max_iter=200,
                random_state=42
            ))
        ])

        models["SVM"] = Pipeline([
            ("prep", self.pre_scale),
            ("clf", SVC(kernel="rbf", C=1.0, gamma="scale"))
        ])

        return models


# ---------- Evaluation ----------
@dataclass
class CVConfig:
    n_splits: int = 5
    shuffle: bool = True
    random_state: int = 42


class Evaluator:
    def __init__(self, cv_cfg: CVConfig) -> None:
        self.cv_cfg = cv_cfg
        self.cv = StratifiedKFold(
            n_splits=cv_cfg.n_splits,
            shuffle=cv_cfg.shuffle,
            random_state=cv_cfg.random_state
        )
        self.scoring = {
            "accuracy": "accuracy",
            "precision_weighted": "precision_weighted",
            "recall_weighted": "recall_weighted",
            "f1_weighted": "f1_weighted",
        }

    def evaluate(self, name: str, pipe: Pipeline, X: pd.DataFrame, y_enc: np.ndarray) -> pd.DataFrame:
        out = cross_validate(
            pipe, X, y_enc, cv=self.cv, scoring=self.scoring, n_jobs=-1, error_score="raise"
        )
        summary = {
            "model": name,
            "cv_splits": self.cv_cfg.n_splits,
            "accuracy_mean": np.mean(out["test_accuracy"]),
            "accuracy_std": np.std(out["test_accuracy"]),
            "precision_w_mean": np.mean(out["test_precision_weighted"]),
            "precision_w_std": np.std(out["test_precision_weighted"]),
            "recall_w_mean": np.mean(out["test_recall_weighted"]),
            "recall_w_std": np.std(out["test_recall_weighted"]),
            "f1_w_mean": np.mean(out["test_f1_weighted"]),
            "f1_w_std": np.std(out["test_f1_weighted"]),
            "fit_time_mean": np.mean(out["fit_time"]),
            "score_time_mean": np.mean(out["score_time"]),
        }
        return pd.DataFrame([summary])


# ---------- Orchestration ----------
class MLPipeline:
    def __init__(self, data_cfg: DataConfig, cv_cfg: CVConfig, reports_dir: str = "./reports") -> None:
        self.data_cfg = data_cfg
        self.cv_cfg = cv_cfg
        self.reports_dir = reports_dir
        ensure_dir(self.reports_dir)
        self.le: LabelEncoder | None = None
        self.models: Dict[str, Pipeline] = {}

    def run(self) -> pd.DataFrame:
        X, y = DatasetLoader(self.data_cfg).load()
        self.le = LabelEncoder()
        y_enc = self.le.fit_transform(y)

        num_features = X.columns.tolist()
        pre_factory = PreprocessorFactory(num_features)
        pre_no_scale = pre_factory.make_no_scale()
        pre_scale = pre_factory.make_scale()

        self.models = ModelFactory(pre_no_scale, pre_scale).build_all()
        evaluator = Evaluator(self.cv_cfg)

        all_rows: List[pd.DataFrame] = []
        for name, pipe in self.models.items():
            print(f"\u25B6 Training & evaluating: {name}")
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                res_df = evaluator.evaluate(name, pipe, X, y_enc)
            all_rows.append(res_df)
            out_path = os.path.join(self.reports_dir, f"{name}_cv_report.csv")
            res_df.to_csv(out_path, index=False)
            print(f"   Saved: {out_path}")

        summary_df = pd.concat(all_rows, ignore_index=True).sort_values("f1_w_mean", ascending=False)
        summary_path = os.path.join(self.reports_dir, "summary_ranked.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n✅ Summary saved: {summary_path}\n")
        print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
        return summary_df


# ---------- CLI ----------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="OOP ML pipeline on drop_out.csv")
    p.add_argument("--csv", type=str, default="drop_out.csv",
                   help="Path to dataset CSV (semicolon-separated).")
    p.add_argument("--target", type=str, default="Target", help="Target column name.")
    p.add_argument("--splits", type=int, default=5, help="Number of CV folds.")
    p.add_argument("--reports", type=str, default="./reports", help="Directory to save reports.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    data_cfg = DataConfig(csv_path=args.csv, target_col=args.target)
    cv_cfg = CVConfig(n_splits=args.splits)
    pipeline = MLPipeline(data_cfg, cv_cfg, reports_dir=args.reports)
    pipeline.run()


if __name__ == "__main__":
    main()