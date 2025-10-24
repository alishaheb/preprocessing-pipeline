# main.py
from __future__ import annotations
import os
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Any
import warnings

# --- third-party ---
import pandas as pd
import numpy as np

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    r2_score, mean_absolute_error, mean_squared_error
)
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


# =========================
# Config
# =========================
@dataclass
class CSVReadConfig:
    sep: str = ","
    header: Optional[int] = 0
    index_col: Optional[int | str] = None
    parse_dates: Optional[List[str]] = None
    na_values: Optional[List[str]] = field(default_factory=lambda: ["", "NA", "NaN", "N/A", "null", "None"])


@dataclass
class CleanConfig:
    drop_duplicates: bool = True
    drop_all_nan_cols: bool = True
    strip_whitespace: bool = True
    drop_id_like_columns: bool = True           # columns with all-unique values (very likely IDs)
    id_uniqueness_threshold: float = 0.98       # >= 98% unique → drop as ID
    target_map: Dict[str, str] = field(default_factory=dict)  # filename -> target column (optional override)
    force_task: Optional[str] = None            # "classification" | "regression" | None (auto)


@dataclass
class ModelConfig:
    random_state: int = 42
    test_size: float = 0.2
    dt_max_depth: Optional[int] = None
    dt_min_samples_split: int = 2
    dt_min_samples_leaf: int = 1


# =========================
# Data utilities
# =========================
class DataLoader:
    def __init__(self, read_cfg: CSVReadConfig):
        self.read_cfg = read_cfg

    def load_csv(self, path: str) -> pd.DataFrame:
        if not os.path.exists(path):
            raise FileNotFoundError(f"CSV not found: {path}")
        return pd.read_csv(
            path,
            sep=self.read_cfg.sep,
            header=self.read_cfg.header,
            index_col=self.read_cfg.index_col,
            parse_dates=self.read_cfg.parse_dates,
            na_values=self.read_cfg.na_values,
        )


class TabularCleaner:
    def __init__(self, cfg: CleanConfig):
        self.cfg = cfg

    # distinct functions (called from clean())
    def _strip_whitespace(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.strip_whitespace:
            return df
        for col in df.columns:
            if pd.api.types.is_object_dtype(df[col]) or pd.api.types.is_string_dtype(df[col]):
                df[col] = df[col].astype("string").str.strip()
        return df

    def _drop_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.drop_duplicates() if self.cfg.drop_duplicates else df

    def _drop_all_nan_cols(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=1, how="all") if self.cfg.drop_all_nan_cols else df

    def _drop_id_like(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.cfg.drop_id_like_columns:
            return df
        n = len(df)
        if n == 0:
            return df
        cols_to_drop = []
        for c in df.columns:
            nunique = df[c].nunique(dropna=True)
            if nunique / max(n, 1) >= self.cfg.id_uniqueness_threshold:
                cols_to_drop.append(c)
        return df.drop(columns=cols_to_drop, errors="ignore")

    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._strip_whitespace(df)
        df = self._drop_duplicates(df)
        df = self._drop_all_nan_cols(df)
        df = self._drop_id_like(df)
        return df


# =========================
# Preprocessor (OOP + separate steps)
# =========================
class PreprocessorBuilder:
    """
    Builds a ColumnTransformer with separate pipelines for numeric and categorical.
    """
    def __init__(self):
        self.numeric_cols: List[str] = []
        self.categorical_cols: List[str] = []
        self.preprocessor: Optional[ColumnTransformer] = None

    def infer_column_types(self, df: pd.DataFrame, target: Optional[str]) -> None:
        feature_df = df.drop(columns=[target], errors="ignore") if target else df
        self.numeric_cols = feature_df.select_dtypes(include=["number"]).columns.tolist()
        self.categorical_cols = [c for c in feature_df.columns if c not in self.numeric_cols]

    def build(self) -> ColumnTransformer:
        num_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ])
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ])
        self.preprocessor = ColumnTransformer(
            transformers=[
                ("num", num_pipe, self.numeric_cols),
                ("cat", cat_pipe, self.categorical_cols),
            ]
        )
        return self.preprocessor


# =========================
# Model Runner
# =========================
class DecisionTreeAuto:
    """
    Wraps a Decision Tree that auto-switches between classifier and regressor
    based on target type/cardinality (unless forced in CleanConfig.force_task).
    """
    def __init__(self, model_cfg: ModelConfig, force_task: Optional[str] = None):
        self.cfg = model_cfg
        self.force_task = force_task
        self.is_classification: bool = True
        self.model: Any = None  # DecisionTreeClassifier | DecisionTreeRegressor

    def _detect_task(self, y: pd.Series) -> str:
        if self.force_task in ("classification", "regression"):
            return self.force_task
        # classification if non-numeric OR few unique values (<=15 threshold)
        if not pd.api.types.is_numeric_dtype(y):
            return "classification"
        nunique = y.nunique(dropna=True)
        return "classification" if nunique <= 15 else "regression"

    def _build(self, task: str) -> Any:
        if task == "classification":
            self.is_classification = True
            return DecisionTreeClassifier(
                random_state=self.cfg.random_state,
                max_depth=self.cfg.dt_max_depth,
                min_samples_split=self.cfg.dt_min_samples_split,
                min_samples_leaf=self.cfg.dt_min_samples_leaf,
            )
        else:
            self.is_classification = False
            return DecisionTreeRegressor(
                random_state=self.cfg.random_state,
                max_depth=self.cfg.dt_max_depth,
                min_samples_split=self.cfg.dt_min_samples_split,
                min_samples_leaf=self.cfg.dt_min_samples_leaf,
            )

    def build_pipeline(self, pre: ColumnTransformer, y: pd.Series) -> Pipeline:
        task = self._detect_task(y)
        self.model = self._build(task)
        return Pipeline(steps=[
            ("preprocessor", pre),
            ("model", self.model),
        ])


# =========================
# Orchestrator
# =========================
class DatasetRunner:
    def __init__(self, read_cfg: CSVReadConfig, clean_cfg: CleanConfig, model_cfg: ModelConfig):
        self.loader = DataLoader(read_cfg)
        self.cleaner = TabularCleaner(clean_cfg)
        self.model_cfg = model_cfg
        self.clean_cfg = clean_cfg

    @staticmethod
    def _guess_target(df: pd.DataFrame) -> Optional[str]:
        # Common names first
        common = ["target", "label", "class", "y", "outcome", "Survived", "quality", "HeartDisease", "Price"]
        for c in common:
            if c in df.columns:
                return c
        # Fallback: last column
        return df.columns[-1] if len(df.columns) > 0 else None

    def _split_xy(self, df: pd.DataFrame, dataset_name: str) -> Tuple[pd.DataFrame, pd.Series, str]:
        t_col = self.clean_cfg.target_map.get(dataset_name)
        if t_col is None:
            t_col = self._guess_target(df)
        if t_col is None or t_col not in df.columns:
            raise ValueError(f"Could not determine target column for {dataset_name}. "
                             f"Provide it in CleanConfig.target_map.")
        X = df.drop(columns=[t_col])
        y = df[t_col]
        return X, y, t_col

    def run_one(self, path: str) -> Dict[str, Any]:
        name = os.path.basename(path)
        df = self.loader.load_csv(path)
        df = self.cleaner.clean(df)

        X, y, t_col = self._split_xy(df, name)

        # Build preprocessing
        pre_builder = PreprocessorBuilder()
        pre_builder.infer_column_types(pd.concat([X, y], axis=1), target=t_col)
        pre = pre_builder.build()

        # Build model pipeline
        dt_auto = DecisionTreeAuto(self.model_cfg, self.clean_cfg.force_task)
        pipe = dt_auto.build_pipeline(pre, y)

        # Fit & evaluate
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.model_cfg.test_size, random_state=self.model_cfg.random_state, stratify=y if dt_auto.is_classification and y.nunique() > 1 else None
        )
        pipe.fit(X_train, y_train)
        preds = pipe.predict(X_test)

        if dt_auto.is_classification:
            metrics = {
                "task": "classification",
                "accuracy": accuracy_score(y_test, preds),
                "precision_macro": precision_score(y_test, preds, average="macro", zero_division=0),
                "recall_macro": recall_score(y_test, preds, average="macro", zero_division=0),
                "f1_macro": f1_score(y_test, preds, average="macro", zero_division=0),
                "n_classes": int(y.nunique()),
            }
        else:
            rmse = mean_squared_error(y_test, preds, squared=False)
            metrics = {
                "task": "regression",
                "r2": r2_score(y_test, preds),
                "mae": mean_absolute_error(y_test, preds),
                "rmse": rmse,
            }

        return {
            "dataset": name,
            "rows": len(df),
            "cols": df.shape[1],
            "target": t_col,
            "numeric_features": len(pre_builder.numeric_cols),
            "categorical_features": len(pre_builder.categorical_cols),
            "metrics": metrics,
        }

    def run_many(self, paths: List[str]) -> List[Dict[str, Any]]:
        results = []
        for p in paths:
            try:
                res = self.run_one(p)
                results.append(res)
                self._pretty_print(res)
            except Exception as e:
                warnings.warn(f"[{os.path.basename(p)}] Skipped due to error: {e}")
        return results

    @staticmethod
    def _pretty_print(res: Dict[str, Any]) -> None:
        print("\n" + "=" * 70)
        print(f"Dataset: {res['dataset']}")
        print(f"Rows x Cols: {res['rows']} x {res['cols']}")
        print(f"Target: {res['target']}")
        print(f"Features → numeric: {res['numeric_features']} | categorical: {res['categorical_features']}")
        print("Metrics:")
        for k, v in res["metrics"].items():
            print(f"  - {k}: {v}")
        print("=" * 70)


# =========================
# Main
# =========================
# =========================
# Main (robust path handling)
# =========================
if __name__ == "__main__":
    import argparse
    from pathlib import Path

    # --- CLI ---
    parser = argparse.ArgumentParser(description="Run preprocessing + Decision Tree on multiple CSVs.")
    parser.add_argument(
        "--data-dir",
        type=str,
        default="",
        help="Directory to search for datasets (overrides auto search).",
    )
    parser.add_argument(
        "--files",
        nargs="*",
        default=[
            "Hp_Preprocessed_data.csv",
            "student-por.csv",
            "StudentsPerformance.csv",
            "car_prices.csv",
            "dropout.csv",
            "heart_disease.csv",
            "study_performance.csv",
            "wholesale.csv",
            "WineQT.csv",
            # add your 10th file name if you have it
        ],
        help="Filenames to locate and run. You can pass explicit paths here too.",
    )
    args = parser.parse_args()

    # --- Where to look ---
    # Priority: explicit --data-dir → then script dir → script_dir/data → CWD → Desktop → Downloads
    script_dir = Path(__file__).resolve().parent
    search_roots = []
    if args.data_dir:
        search_roots.append(Path(args.data_dir).expanduser())
    search_roots += [
        script_dir,
        script_dir / "data",
        Path.cwd(),
        Path.home() / "Desktop",
        Path.home() / "Downloads",
    ]

    def resolve_path(name_or_path: str) -> Optional[Path]:
        p = Path(name_or_path).expanduser()
        if p.is_file():
            return p
        # search by filename in search roots
        fname = Path(name_or_path).name
        for root in search_roots:
            candidate = root / fname
            if candidate.is_file():
                return candidate
        return None

    # Build final list of existing files
    resolved: list[Path] = []
    missing: list[str] = []
    for item in args.files:
        p = resolve_path(item)
        if p is None:
            missing.append(item)
        else:
            resolved.append(p)

    # Friendly summary
    print("\nSearch roots:")
    for r in search_roots:
        print("  -", r)
    print("\nResolved datasets:")
    for r in resolved:
        print("  ✓", r)
    if missing:
        print("\nMissing (not found anywhere):")
        for m in missing:
            print("  ✗", m)
        print("Tip: move the files into one of the search folders above, "
              "or pass --data-dir /absolute/path/to/folder, or pass full paths in --files.")

    # Early exit if nothing to run
    if not resolved:
        raise SystemExit(0)

    # ---- configs (same as before) ----
    read_cfg = CSVReadConfig()
    target_map = {
        # if you know exact targets, set them here; otherwise the runner will guess:
        "WineQT.csv": "quality",
        "heart_disease.csv": "target",      # adjust if your column differs (e.g., "HeartDisease")
        "wholesale.csv": "Channel",         # or "Region" depending on your task
        "car_prices.csv": "price",          # adjust to your column name
        # "StudentsPerformance.csv": "math score",  # uncomment for regression target example
    }
    clean_cfg = CleanConfig(
        drop_duplicates=True,
        drop_all_nan_cols=True,
        strip_whitespace=True,
        drop_id_like_columns=True,
        target_map=target_map,
        force_task=None,  # or "classification"/"regression"
    )
    model_cfg = ModelConfig(
        random_state=42,
        test_size=0.2,
        dt_max_depth=None,
        dt_min_samples_split=2,
        dt_min_samples_leaf=1,
    )

    # ---- run ----
    runner = DatasetRunner(read_cfg, clean_cfg, model_cfg)
    runner.run_many([str(p) for p in resolved])
