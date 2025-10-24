# main.py
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Iterable, Union, Callable
import warnings
import os

# --- Optional deps (graceful degradation) ---
try:
    import pandas as pd
    import numpy as np
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    np = None  # type: ignore

try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
except Exception:  # pragma: no cover
    SimpleImputer = None  # type: ignore
    StandardScaler = None  # type: ignore


# =========================
# Base Interface
# =========================
class BaseAdapter(ABC):
    @abstractmethod
    def fit(self, data: Any) -> "BaseAdapter":
        ...

    @abstractmethod
    def transform(self, data: Any) -> Any:
        ...

    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)


# =========================
# Configs
# =========================
@dataclass
class CSVConfig:
    """How to read CSVs when a path is passed instead of a DataFrame."""
    encoding: Optional[str] = None
    sep: str = ","
    header: Union[int, None] = 0
    index_col: Optional[Union[int, str]] = None
    parse_dates: Optional[Union[List[int], List[str], Dict[str, List[str]]]] = None
    dtype: Optional[Dict[str, Any]] = None
    na_values: Optional[Union[str, List[str], Dict[str, List[str]]]] = None
    extra_read_csv_kwargs: Optional[Dict[str, Any]] = None


@dataclass
class TabularConfig:
    csv: CSVConfig = field(default_factory=CSVConfig)  # <-- FIX: default_factory, not CSVConfig()
    impute_strategy: str = "median"   # 'mean', 'median', 'most_frequent', 'constant'
    scale: bool = True
    drop_duplicates: bool = True
    drop_full_nan_cols: bool = True
    categoricals_as_dummies: bool = True
    dummy_drop_first: bool = False
    # Optional: columns to exclude from processing (e.g., labels/IDs); they will pass through unchanged
    exclude_cols: Optional[List[str]] = None


# =========================
# Adapter: Tabular
# =========================
class TabularAdapter(BaseAdapter):
    def __init__(self, config: Optional[TabularConfig] = None):
        self.config = config or TabularConfig()

        # learned state
        self._imputer: Optional[SimpleImputer] = None
        self._scaler: Optional[StandardScaler] = None
        self._cols_numeric: List[str] = []
        self._cols_categorical: List[str] = []
        self._feature_columns: List[str] = []  # final columns after dummies + keeping excluded columns

        if pd is None or np is None:
            warnings.warn("pandas/numpy not available; TabularAdapter disabled.", RuntimeWarning)
        if SimpleImputer is None or StandardScaler is None:
            warnings.warn("scikit-learn not available; impute/scale features will be limited.", RuntimeWarning)

    # ---- helpers ----
    def _to_dataframe(self, data: Union[str, "pd.DataFrame"]) -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("pandas is required for TabularAdapter")
        if isinstance(data, str):
            if not data.lower().endswith(".csv"):
                raise ValueError("TabularAdapter expected a DataFrame or a .csv path.")
            if not os.path.exists(data):
                raise FileNotFoundError(f"CSV file not found: {data}")
            cfg = self.config.csv
            kwargs = dict(
                sep=cfg.sep,
                header=cfg.header,
                index_col=cfg.index_col,
                parse_dates=cfg.parse_dates,
                dtype=cfg.dtype,
                na_values=cfg.na_values,
            )
            if cfg.encoding is not None:
                kwargs["encoding"] = cfg.encoding
            if cfg.extra_read_csv_kwargs:
                kwargs.update(cfg.extra_read_csv_kwargs)
            return pd.read_csv(data, **kwargs)
        return data

    def _drop_dupes_and_allnan(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if self.config.drop_duplicates:
            df = df.drop_duplicates()
        if self.config.drop_full_nan_cols:
            df = df.dropna(axis=1, how="all")
        return df

    def _split_columns(self, df: "pd.DataFrame") -> None:
        exclude = set(self.config.exclude_cols or [])
        num = df.select_dtypes(include=["number"]).columns.tolist()
        self._cols_numeric = [c for c in num if c not in exclude]
        self._cols_categorical = [c for c in df.columns if c not in exclude and c not in self._cols_numeric]

    def _apply_impute(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if self._cols_numeric:
            if self._imputer is not None:
                df[self._cols_numeric] = self._imputer.transform(df[self._cols_numeric])
        return df

    def _apply_scale(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if self._cols_numeric and self._scaler is not None:
            df[self._cols_numeric] = self._scaler.transform(df[self._cols_numeric])
        return df

    def _apply_dummies(self, df: "pd.DataFrame") -> "pd.DataFrame":
        if self.config.categoricals_as_dummies and self._cols_categorical:
            df = pd.get_dummies(df, columns=self._cols_categorical, drop_first=self.config.dummy_drop_first)
        return df

    # ---- fit/transform ----
    def fit(self, data: Union[str, "pd.DataFrame"]) -> "TabularAdapter":
        if pd is None:
            return self

        df = self._to_dataframe(data).copy()
        df = self._drop_dupes_and_allnan(df)

        # split columns with excludes
        self._split_columns(df)

        # fit imputer on numeric
        if SimpleImputer is not None and self._cols_numeric:
            self._imputer = SimpleImputer(strategy=self.config.impute_strategy)
            self._imputer.fit(df[self._cols_numeric])

        # create an imputed copy for scaling + dummies discovery
        if self._cols_numeric and self._imputer is not None:
            df[self._cols_numeric] = self._imputer.transform(df[self._cols_numeric])

        # fit scaler on numeric
        if StandardScaler is not None and self.config.scale and self._cols_numeric:
            self._scaler = StandardScaler().fit(df[self._cols_numeric])

        # scale for column discovery (keeps names)
        if self.config.scale and self._scaler is not None:
            df[self._cols_numeric] = self._scaler.transform(df[self._cols_numeric])

        # one-hot (to lock in dummy columns)
        df = self._apply_dummies(df)

        # final column order we will produce during transform
        self._feature_columns = df.columns.tolist()
        return self

    def transform(self, data: Union[str, "pd.DataFrame"]) -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("pandas is required for TabularAdapter")

        df = self._to_dataframe(data).copy()
        df = self._drop_dupes_and_allnan(df)

        # ensure columns seen at fit exist (add missing with NaN); also keep unknowns until reindex
        for c in (self._cols_numeric + self._cols_categorical):
            if c not in df.columns:
                df[c] = np.nan

        # impute
        if self._cols_numeric and self._imputer is not None:
            df[self._cols_numeric] = self._imputer.transform(df[self._cols_numeric])

        # scale (all numeric at once to keep shapes aligned)
        if self.config.scale and self._scaler is not None and self._cols_numeric:
            df[self._cols_numeric] = self._scaler.transform(df[self._cols_numeric])

        # dummies
        df = self._apply_dummies(df)

        # now strictly align to training columns (fill missing with 0; drop extras)
        if self._feature_columns:
            df = df.reindex(columns=self._feature_columns, fill_value=0)

        return df


# =========================
# Dispatcher / Registry
# =========================
class AdapterRegistry:
    """
    If you later add other tabular adapters, register predicates here.
    For now, we route CSV paths or DataFrames to TabularAdapter.
    """
    def __init__(self):
        self._rules: List[Tuple[Callable[[Any], bool], Callable[[], BaseAdapter]]] = []

    def register(self, predicate: Callable[[Any], bool], factory: Callable[[], BaseAdapter]) -> None:
        self._rules.append((predicate, factory))

    def resolve(self, data: Any) -> BaseAdapter:
        for predicate, factory in self._rules:
            try:
                if predicate(data):
                    return factory()
            except Exception:
                continue
        raise ValueError("No adapter registered that can handle the provided data.")


class Preprocessor:
    def __init__(self, registry: Optional[AdapterRegistry] = None):
        self.registry = registry or default_registry()
        self._adapter: Optional[BaseAdapter] = None

    def fit(self, data: Any) -> "Preprocessor":
        self._adapter = self.registry.resolve(data).fit(data)
        return self

    def transform(self, data: Any) -> Any:
        if self._adapter is None:
            self._adapter = self.registry.resolve(data)
        return self._adapter.transform(data)

    def fit_transform(self, data: Any) -> Any:
        self._adapter = self.registry.resolve(data)
        return self._adapter.fit_transform(data)


def default_registry() -> AdapterRegistry:
    reg = AdapterRegistry()

    # CSV path or DataFrame (non time-index) => Tabular
    def is_tabular(x: Any) -> bool:
        if isinstance(x, str) and x.lower().endswith(".csv"):
            return True
        if pd is not None and isinstance(x, pd.DataFrame):
            # allow any DataFrame as "tabular" here
            return True
        return False

    reg.register(is_tabular, lambda: TabularAdapter())
    return reg


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # Change this to your local CSV path or pass a pandas DataFrame
    csv_path = "Titanic-Dataset.csv"  # e.g., "/Users/you/Desktop/Titanic-Dataset 2.csv"

    pre = Preprocessor()
    try:
        X = pre.fit_transform(csv_path)
        print("Processed type:", type(X))
        print("Processed shape:", getattr(X, "shape", None))
        # If you want to inspect few columns:
        try:
            print("Columns (first 15):", list(X.columns)[:15])
        except Exception:
            pass
    except Exception as e:
        print("Error:", e)
