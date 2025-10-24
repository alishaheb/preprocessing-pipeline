from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, List, Tuple, Iterable, Union, Callable
import warnings

# --- Optional deps ---
try:
    import pandas as pd
    import numpy as np
except Exception:
    pd = None  # type: ignore
    np = None  # type: ignore

try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    SimpleImputer = StandardScaler = TfidfVectorizer = None  # type: ignore


# =========================
# Base
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
# Configs (incl. CSV)
# =========================
@dataclass
class CSVConfig:
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
    csv: CSVConfig = CSVConfig()
    impute_strategy: str = "median"    # 'mean', 'median', 'most_frequent', 'constant'
    scale: bool = True
    drop_duplicates: bool = True
    drop_full_nan_cols: bool = True
    categoricals_as_dummies: bool = True
    dummy_drop_first: bool = False


@dataclass
class TextConfig:
    lowercase: bool = True
    strip_punct: bool = True
    min_df: int = 2
    max_df: Union[float, int] = 0.95
    ngram_range: Tuple[int, int] = (1, 2)
    max_features: Optional[int] = 20000


@dataclass
class TimeSeriesConfig:
    resample_rule: Optional[str] = None  # e.g., 'D', 'H'
    forward_fill: bool = True
    interpolate: bool = False
    scale: bool = True


# =========================
# Adapters
# =========================
class TabularAdapter(BaseAdapter):
    def __init__(self, config: Optional[TabularConfig] = None):
        self.config = config or TabularConfig()
        self._imputer = None
        self._scaler = None
        self._cols_numeric: List[str] = []
        self._cols_categorical: List[str] = []

        if pd is None or np is None:
            warnings.warn("pandas/numpy not available; TabularAdapter disabled.", RuntimeWarning)
        if SimpleImputer is None or StandardScaler is None:
            warnings.warn("scikit-learn not available; TabularAdapter will be limited.", RuntimeWarning)

    def _to_dataframe(self, data: Union[str, "pd.DataFrame"]) -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("pandas is required for TabularAdapter")
        if isinstance(data, str):
            if not data.lower().endswith(".csv"):
                raise ValueError("TabularAdapter expected a DataFrame or a .csv path.")
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

    def _split_columns(self, df: "pd.DataFrame") -> None:
        self._cols_numeric = df.select_dtypes(include=["number"]).columns.tolist()
        self._cols_categorical = [c for c in df.columns if c not in self._cols_numeric]

    def fit(self, data: Union[str, "pd.DataFrame"]) -> "TabularAdapter":
        if pd is None:
            return self
        df = self._to_dataframe(data).copy()

        if self.config.drop_duplicates:
            df = df.drop_duplicates()

        if self.config.drop_full_nan_cols:
            df = df.dropna(axis=1, how="all")

        self._split_columns(df)

        if SimpleImputer is not None and self._cols_numeric:
            self._imputer = SimpleImputer(strategy=self.config.impute_strategy)
            self._imputer.fit(df[self._cols_numeric])

        if StandardScaler is not None and self.config.scale and self._cols_numeric:
            X_num = df[self._cols_numeric]
            if self._imputer is not None:
                X_num = self._imputer.transform(X_num)
            self._scaler = StandardScaler().fit(X_num)

        return self

    def transform(self, data: Union[str, "pd.DataFrame"]) -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("pandas is required for TabularAdapter")

        df = self._to_dataframe(data).copy()

        if self.config.drop_duplicates:
            df = df.drop_duplicates()

        if self.config.drop_full_nan_cols:
            df = df.loc[:, [c for c in df.columns if not df[c].isna().all()]]

        # Ensure same columns seen in fit (add missing with NaN)
        for c in self._cols_numeric + self._cols_categorical:
            if c not in df.columns:
                df[c] = np.nan

        # Impute numeric
        if self._cols_numeric:
            X_num = df[self._cols_numeric]
            if self._imputer is not None:
                X_num = self._imputer.transform(X_num)
            df[self._cols_numeric] = X_num

        # One-hot encode categoricals
        if self.config.categoricals_as_dummies and self._cols_categorical:
            df = pd.get_dummies(df, columns=self._cols_categorical, drop_first=self.config.dummy_drop_first)

        # Scale numeric
        if self.config.scale and self._cols_numeric and self._scaler is not None:
            for c in self._cols_numeric:
                if c in df.columns:
                    df[c] = self._scaler.transform(df[[c]])

        return df


class TextAdapter(BaseAdapter):
    def __init__(self, config: Optional[TextConfig] = None):
        self.config = config or TextConfig()
        self._vectorizer = None
        if TfidfVectorizer is None:
            warnings.warn("scikit-learn not available; TextAdapter disabled.", RuntimeWarning)

    @staticmethod
    def _basic_clean(texts: Iterable[str], lowercase: bool, strip_punct: bool) -> List[str]:
        import re
        out = []
        for t in texts:
            s = t if isinstance(t, str) else str(t)
            if lowercase:
                s = s.lower()
            if strip_punct:
                s = re.sub(r"[^\w\s]", " ", s)
            s = re.sub(r"\s+", " ", s).strip()
            out.append(s)
        return out

    def fit(self, data: Iterable[str]) -> "TextAdapter":
        texts = list(data)
        texts = self._basic_clean(texts, self.config.lowercase, self.config.strip_punct)
        if TfidfVectorizer is not None:
            self._vectorizer = TfidfVectorizer(
                min_df=self.config.min_df,
                max_df=self.config.max_df,
                ngram_range=self.config.ngram_range,
                max_features=self.config.max_features,
            ).fit(texts)
        return self

    def transform(self, data: Iterable[str]):
        texts = list(data)
        texts = self._basic_clean(texts, self.config.lowercase, self.config.strip_punct)
        if self._vectorizer is None:
            return texts
        return self._vectorizer.transform(texts)


class TimeSeriesAdapter(BaseAdapter):
    def __init__(self, config: Optional[TimeSeriesConfig] = None):
        self.config = config or TimeSeriesConfig()
        self._scaler = None

        if pd is None or np is None:
            warnings.warn("pandas/numpy not available; TimeSeriesAdapter disabled.", RuntimeWarning)
        if StandardScaler is None:
            warnings.warn("scikit-learn not available; scaling disabled for TimeSeriesAdapter.", RuntimeWarning)

    def fit(self, data: "pd.DataFrame") -> "TimeSeriesAdapter":
        if pd is None:
            return self
        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("TimeSeriesAdapter expects a DataFrame with a DatetimeIndex.")

        if self.config.resample_rule:
            df = df.resample(self.config.resample_rule).mean()
        if self.config.interpolate:
            df = df.interpolate()
        if self.config.forward_fill:
            df = df.ffill()

        if self.config.scale and StandardScaler is not None:
            self._scaler = StandardScaler().fit(df.select_dtypes(include="number"))
        return self

    def transform(self, data: "pd.DataFrame") -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("pandas is required for TimeSeriesAdapter")

        df = data.copy()
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("TimeSeriesAdapter expects a DataFrame with a DatetimeIndex.")

        if self.config.resample_rule:
            df = df.resample(self.config.resample_rule).mean()
        if self.config.interpolate:
            df = df.interpolate()
        if self.config.forward_fill:
            df = df.ffill()

        if self.config.scale and self._scaler is not None:
            num_cols = df.select_dtypes(include="number").columns
            df[num_cols] = self._scaler.transform(df[num_cols])
        return df


# =========================
# Dispatcher / Registry
# =========================
class AdapterRegistry:
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


# =========================
# Default registry (NO images)
# =========================
def default_registry() -> AdapterRegistry:
    reg = AdapterRegistry()

    # CSV path or DataFrame (non time-index) => Tabular
    def is_tabular(x: Any) -> bool:
        is_csv = isinstance(x, str) and x.lower().endswith(".csv")
        is_df_no_ts = (pd is not None and isinstance(x, pd.DataFrame)
                       and not isinstance(x.index, pd.DatetimeIndex))
        return is_csv or is_df_no_ts

    reg.register(is_tabular, lambda: TabularAdapter())

    # Time series: DataFrame with DatetimeIndex
    def is_timeseries(x: Any) -> bool:
        return (pd is not None
                and isinstance(x, pd.DataFrame)
                and isinstance(x.index, pd.DatetimeIndex))

    reg.register(is_timeseries, lambda: TimeSeriesAdapter())

    # Text: iterable of strings or pandas Series of object/str
    def is_text(x: Any) -> bool:
        if pd is not None and isinstance(x, pd.Series):
            return x.dtype == "object" or x.dtype == "string"
        if isinstance(x, (list, tuple)):
            return len(x) == 0 or isinstance(x[0], str)
        return False

    reg.register(is_text, lambda: TextAdapter())

    return reg


# =========================
# Example
# =========================
if __name__ == "__main__":
    csv_path = "/mnt/data/Titanic-Dataset 2.csv"  # adjust if needed
    X_tab = Preprocessor().fit_transform(csv_path)
    print("From CSV ->", type(X_tab), getattr(X_tab, "shape", None))

