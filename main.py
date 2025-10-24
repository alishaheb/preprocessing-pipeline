from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, List, Tuple, Iterable, Union, Callable
import warnings

# --- Optional deps (import if available, degrade gracefully) ---
try:
    import pandas as pd
    import numpy as np
except Exception:  # pragma: no cover
    pd = None  # type: ignore
    np = None  # type: ignore

try:
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import StandardScaler
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:  # pragma: no cover
    SimpleImputer = StandardScaler = TfidfVectorizer = None  # type: ignore

try:
    from PIL import Image
except Exception:  # pragma: no cover
    Image = None  # type: ignore


# =========================
# Base Interfaces
# =========================

class BaseAdapter(ABC):
    """
    Interface for all data adapters. Each adapter encapsulates 'fit' and 'transform'
    for a specific data modality. 'data' can be any structure your adapter supports.
    """
    @abstractmethod
    def fit(self, data: Any) -> "BaseAdapter":
        ...

    @abstractmethod
    def transform(self, data: Any) -> Any:
        ...

    def fit_transform(self, data: Any) -> Any:
        return self.fit(data).transform(data)


# =========================
# Config Objects (per modality)
# =========================

@dataclass
class TabularConfig:
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
    ngram_range: Tuple[int, int] = (1, 2)  # unigrams + bigrams
    max_features: Optional[int] = 20000


@dataclass
class ImageConfig:
    target_size: Tuple[int, int] = (224, 224)
    to_rgb: bool = True
    normalize: bool = True  # scale to [0, 1]


@dataclass
class TimeSeriesConfig:
    resample_rule: Optional[str] = None  # e.g., 'D', 'H'; if None, keep as-is
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

    def _split_columns(self, df: "pd.DataFrame") -> None:
        self._cols_numeric = df.select_dtypes(include=["number"]).columns.tolist()
        self._cols_categorical = [c for c in df.columns if c not in self._cols_numeric]

    def fit(self, data: "pd.DataFrame") -> "TabularAdapter":
        if pd is None:
            return self
        df = data.copy()
        if self.config.drop_duplicates:
            df = df.drop_duplicates()

        if self.config.drop_full_nan_cols:
            df = df.dropna(axis=1, how="all")

        self._split_columns(df)

        if SimpleImputer is not None and self._cols_numeric:
            self._imputer = SimpleImputer(strategy=self.config.impute_strategy)
            self._imputer.fit(df[self._cols_numeric])

        if StandardScaler is not None and self.config.scale and self._cols_numeric:
            # We'll fit scaler after imputation to avoid NaNs
            X_num = df[self._cols_numeric]
            if self._imputer is not None:
                X_num = self._imputer.transform(X_num)
            self._scaler = StandardScaler().fit(X_num)

        return self

    def transform(self, data: "pd.DataFrame") -> "pd.DataFrame":
        if pd is None:
            raise RuntimeError("pandas is required for TabularAdapter")

        df = data.copy()

        if self.config.drop_duplicates:
            df = df.drop_duplicates()

        if self.config.drop_full_nan_cols:
            # Align with columns seen at fit if possible
            df = df.loc[:, [c for c in df.columns if not df[c].isna().all()]]

        # Ensure same columns ordering: add missing cols with NaNs
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
            # Some numeric columns may have been expanded by get_dummies; only scale original numeric cols
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
            return texts  # return cleaned strings if sklearn is unavailable
        return self._vectorizer.transform(texts)  # sparse matrix


class ImageAdapter(BaseAdapter):
    def __init__(self, config: Optional[ImageConfig] = None):
        self.config = config or ImageConfig()
        if Image is None:
            warnings.warn("Pillow not available; ImageAdapter disabled.", RuntimeWarning)

    def _load(self, x: Union[str, "Image.Image"]) -> "Image.Image":
        if isinstance(x, str):
            if Image is None:
                raise RuntimeError("Pillow required to load images from paths")
            return Image.open(x)
        return x

    def fit(self, data: Iterable[Union[str, "Image.Image"]]) -> "ImageAdapter":
        # No training state for basic image normalization/resizing
        return self

    def transform(self, data: Iterable[Union[str, "Image.Image"]]):
        if Image is None:
            raise RuntimeError("Pillow is required for ImageAdapter")

        processed = []
        for x in data:
            img = self._load(x)
            if self.config.to_rgb:
                img = img.convert("RGB")
            img = img.resize(self.config.target_size)
            arr = np.array(img, dtype="float32")
            if self.config.normalize:
                arr = arr / 255.0
            processed.append(arr)
        return np.stack(processed, axis=0)  # (N, H, W, C)


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
    """
    Holds rules to pick an adapter based on a predicate on the incoming data.
    You can register new modalities without touching the core pipeline.
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
    """
    High-level entry point:
      pre = Preprocessor().fit(data).transform(data)
    or:
      X = pre.fit_transform(data)
    """
    def __init__(self, registry: Optional[AdapterRegistry] = None):
        self.registry = registry or default_registry()
        self._adapter: Optional[BaseAdapter] = None

    def fit(self, data: Any) -> "Preprocessor":
        self._adapter = self.registry.resolve(data).fit(data)
        return self

    def transform(self, data: Any) -> Any:
        if self._adapter is None:
            # stateless transform if possible
            self._adapter = self.registry.resolve(data)
        return self._adapter.transform(data)

    def fit_transform(self, data: Any) -> Any:
        self._adapter = self.registry.resolve(data)
        return self._adapter.fit_transform(data)


# =========================
# Default registry with sensible predicates
# =========================

def default_registry() -> AdapterRegistry:
    reg = AdapterRegistry()

    # ---- Tabular: pandas DataFrame without DatetimeIndex
    def is_tabular(x: Any) -> bool:
        return (pd is not None
                and isinstance(x, pd.DataFrame)
                and not isinstance(x.index, pd.DatetimeIndex))

    reg.register(is_tabular, lambda: TabularAdapter())

    # ---- Time series: DataFrame with DatetimeIndex
    def is_timeseries(x: Any) -> bool:
        return (pd is not None
                and isinstance(x, pd.DataFrame)
                and isinstance(x.index, pd.DatetimeIndex))

    reg.register(is_timeseries, lambda: TimeSeriesAdapter())

    # ---- Text: iterable of strings (or pandas Series of object/str)
    def is_text(x: Any) -> bool:
        if pd is not None and isinstance(x, pd.Series):
            return x.dtype == "object" or x.dtype == "string"
        if isinstance(x, (list, tuple)):
            return len(x) == 0 or isinstance(x[0], str)
        return False

    reg.register(is_text, lambda: TextAdapter())

    # ---- Images: iterable of file paths or PIL Images
    def is_images(x: Any) -> bool:
        if isinstance(x, (list, tuple)) and len(x) > 0:
            first = x[0]
            if isinstance(first, str):
                return True
            if Image is not None and isinstance(first, Image.Image):
                return True
        return False

    reg.register(is_images, lambda: ImageAdapter())

    return reg


# =========================
# Example usage
# =========================
if __name__ == "__main__":
    # These demos only run if optional deps are installed.
    if pd is not None and np is not None:
        # ---- Tabular
        df = pd.DataFrame({
            "age": [20, 21, None, 23],
            "income": [30_000, None, 45_000, 50_000],
            "city": ["London", "London", "Leeds", None],
        })
        X_tab = Preprocessor().fit_transform(df)
        print("Tabular ->", type(X_tab), X_tab.shape if hasattr(X_tab, "shape") else len(X_tab))

        # ---- Time series
        idx = pd.date_range("2024-01-01", periods=5, freq="D")
        ts = pd.DataFrame({"y": [1.0, None, 3.0, None, 5.0]}, index=idx)
        X_ts = Preprocessor().fit(TimeSeriesAdapter(TimeSeriesConfig(resample_rule="D", forward_fill=True))).transform(ts)  # custom fit path
        print("TimeSeries ->", type(X_ts), X_ts.shape)

    # ---- Text (works without pandas)
    corpus = ["Cats are GREAT!", "I love cats and dogs.", "Dogs are friendly."]
    X_text = Preprocessor().fit_transform(corpus)
    try:
        shape = X_text.shape
    except Exception:
        shape = len(X_text)
    print("Text ->", type(X_text), shape)

