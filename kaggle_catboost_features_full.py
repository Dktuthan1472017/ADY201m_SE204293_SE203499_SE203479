"""Full-data CatBoost experiments for FreshRetailNet-50K.

Paste this file into one Kaggle notebook cell after attaching exactly one
train.parquet and one eval.parquet below /kaggle/input. This standalone script trains CatBoost only.

Methodology is synchronized with the LightGBM experiment:
* observed-sales target;
* operational and weather-enhanced feature sets;
* five predefined configurations per model and feature set;
* fixed-origin recursive 7-day validation as the primary protocol;
* rolling-origin one-day-ahead as the secondary protocol;
* final eval is touched only after fixed-origin validation selection.

Weather during validation/eval is treated as known-ahead/oracle context unless
genuine forecast weather is supplied at the forecast origin.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import catboost
    from catboost import CatBoostRegressor
except ImportError as exc:  # pragma: no cover
    raise ImportError("CatBoost is required. In Kaggle run: !pip install catboost") from exc

TRAIN_START = "2024-03-28"
VALIDATION_TRAIN_END = "2024-06-18"
VALIDATION_START = "2024-06-19"
VALIDATION_END = "2024-06-25"
FINAL_TRAIN_END = "2024-06-25"
TEST_START = "2024-06-26"
TEST_END = "2024-07-02"

TRAIN_PATH: str | None = None
EVAL_PATH: str | None = None
OUTPUT_DIR = "/kaggle/working/freshretail-catboost-features"
MODELS_TO_RUN = ["catboost"]
RANDOM_SEED = 42

SERIES_KEYS = ["store_id", "product_id"]
CATEGORICAL_COLUMNS = [
    "city_id",
    "store_id",
    "management_group_id",
    "first_category_id",
    "second_category_id",
    "third_category_id",
    "product_id",
]
WEATHER_COLUMNS = [
    "precpt",
    "avg_temperature",
    "avg_humidity",
    "avg_wind_level",
]
DYNAMIC_COLUMNS = [
    "lag_1",
    "lag_2",
    "lag_3",
    "lag_7",
    "lag_14",
    "rolling_mean_3",
    "rolling_mean_7",
    "rolling_mean_14",
    "rolling_std_7",
    "past_stockout_count_7",
    "past_stockout_rate_7",
    "past_stockout_count_14",
    "past_stockout_rate_14",
]
OPERATIONAL_FEATURES = CATEGORICAL_COLUMNS + [
    "day_of_week",
    "is_weekend",
    "day_index",
    "holiday_flag",
    "activity_flag",
    "discount",
] + DYNAMIC_COLUMNS
FEATURE_SETS = {
    "operational": OPERATIONAL_FEATURES,
    "weather_enhanced": OPERATIONAL_FEATURES + WEATHER_COLUMNS,
}
REQUIRED_COLUMNS = list(
    dict.fromkeys(
        CATEGORICAL_COLUMNS
        + [
            "dt",
            "sale_amount",
            "stock_hour6_22_cnt",
            "discount",
            "holiday_flag",
            "activity_flag",
        ]
        + WEATHER_COLUMNS
    )
)

# Iteration counts are bounded to keep five full-data candidates practical on Kaggle.
MODEL_CONFIGS: dict[str, dict[str, dict[str, Any]]] = {
    "catboost": {
        "config_1_fast": {
            "iterations": 150,
            "learning_rate": 0.08,
            "depth": 6,
            "l2_leaf_reg": 3.0,
            "random_strength": 1.0,
            "rsm": 0.80,
        },
        "config_2_balanced": {
            "iterations": 250,
            "learning_rate": 0.05,
            "depth": 8,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "rsm": 0.90,
        },
        "config_3_deeper": {
            "iterations": 300,
            "learning_rate": 0.04,
            "depth": 10,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "rsm": 0.90,
        },
        "config_4_regularized": {
            "iterations": 350,
            "learning_rate": 0.035,
            "depth": 8,
            "l2_leaf_reg": 10.0,
            "random_strength": 2.0,
            "rsm": 0.85,
        },
        "config_5_low_lr": {
            "iterations": 500,
            "learning_rate": 0.025,
            "depth": 7,
            "l2_leaf_reg": 5.0,
            "random_strength": 1.0,
            "rsm": 0.90,
        },
    },
}

COMMON_PARAMS: dict[str, dict[str, Any]] = {
    "catboost": {
        "loss_function": "MAE",
        "random_seed": RANDOM_SEED,
        "thread_count": -1,
        "verbose": False,
        "allow_writing_files": False,
        "bootstrap_type": "Bernoulli",
        "subsample": 0.85,
        "task_type": "CPU",
    },
}


def discover_parquet(filename: str, explicit: str | None = None) -> Path:
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Explicit path does not exist: {path}")
        return path
    env_name = f"FRESHRETAIL_{Path(filename).stem.upper()}_PATH"
    if os.environ.get(env_name):
        path = Path(os.environ[env_name]).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"{env_name} points to a missing file: {path}")
        return path
    matches = sorted(
        {path.resolve() for path in Path("/kaggle/input").rglob(filename) if path.is_file()},
        key=str,
    )
    if not matches:
        raise FileNotFoundError(f"No {filename} was found below /kaggle/input")
    if len(matches) != 1:
        choices = "\n".join(f"- {path}" for path in matches)
        raise RuntimeError(f"Expected exactly one {filename}; found {len(matches)}:\n{choices}")
    return matches[0]


def _downcast(frame: pd.DataFrame) -> pd.DataFrame:
    for column in CATEGORICAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], downcast="integer")
    frame["sale_amount"] = frame["sale_amount"].astype(np.float32)
    frame["stock_hour6_22_cnt"] = pd.to_numeric(
        frame["stock_hour6_22_cnt"], downcast="integer"
    )
    for column in ["holiday_flag", "activity_flag"]:
        frame[column] = pd.to_numeric(frame[column], downcast="integer")
    for column in ["discount"] + WEATHER_COLUMNS:
        frame[column] = frame[column].astype(np.float32)
    return frame


def _validate_panel(
    frame: pd.DataFrame,
    expected_start: str,
    expected_end: str,
    expected_days: int,
    label: str,
) -> tuple[pd.DatetimeIndex, pd.DataFrame]:
    if frame[REQUIRED_COLUMNS].isna().any().any():
        missing = frame[REQUIRED_COLUMNS].isna().sum()
        raise ValueError(f"{label} has missing values:\n{missing[missing > 0]}")
    if frame.duplicated(SERIES_KEYS + ["dt"]).any():
        raise ValueError(f"{label} contains duplicate store-product-date keys")
    dates = pd.DatetimeIndex(frame["dt"].drop_duplicates().sort_values())
    expected_dates = pd.date_range(expected_start, expected_end, freq="D")
    if len(dates) != expected_days or not dates.equals(expected_dates):
        raise ValueError(
            f"{label} must cover {expected_start}..{expected_end}; found "
            f"{dates.min().date()}..{dates.max().date()} ({len(dates)} days)"
        )
    counts = frame.groupby(SERIES_KEYS, sort=False, observed=True).size()
    if len(counts) != 50_000 or counts.nunique() != 1 or int(counts.iloc[0]) != expected_days:
        raise ValueError(f"{label} is not a complete 50,000 x {expected_days} panel")
    series = frame.loc[::expected_days, SERIES_KEYS].reset_index(drop=True)
    return dates, series


def load_data(train_path: Path, eval_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print("Loading required train columns ...", flush=True)
    train = pd.read_parquet(train_path, columns=REQUIRED_COLUMNS)
    print("Loading required eval columns ...", flush=True)
    eval_frame = pd.read_parquet(eval_path, columns=REQUIRED_COLUMNS)
    for frame in (train, eval_frame):
        frame["dt"] = pd.to_datetime(frame["dt"], format="%Y-%m-%d")
    train = _downcast(
        train.sort_values(SERIES_KEYS + ["dt"], kind="stable").reset_index(drop=True)
    )
    eval_frame = _downcast(
        eval_frame.sort_values(SERIES_KEYS + ["dt"], kind="stable").reset_index(drop=True)
    )
    train_dates, train_series = _validate_panel(
        train, TRAIN_START, FINAL_TRAIN_END, 90, "train.parquet"
    )
    eval_dates, eval_series = _validate_panel(
        eval_frame, TEST_START, TEST_END, 7, "eval.parquet"
    )
    if not np.array_equal(train_series.to_numpy(), eval_series.to_numpy()):
        raise ValueError("Train and eval store-product panels do not match")

    category_levels: dict[str, list[int]] = {}
    for column in CATEGORICAL_COLUMNS:
        levels = np.sort(train[column].unique())
        unseen = np.setdiff1d(eval_frame[column].unique(), levels)
        if unseen.size:
            raise ValueError(f"Eval contains unseen {column}: {unseen[:10].tolist()}")
        category_levels[column] = [int(value) for value in levels]
        train[column] = pd.Categorical(train[column], categories=levels)
        eval_frame[column] = pd.Categorical(eval_frame[column], categories=levels)
    return train, eval_frame, {
        "train_dates": train_dates,
        "eval_dates": eval_dates,
        "series": train_series,
        "category_levels": category_levels,
    }


def add_calendar_features(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    day_of_week = frame["dt"].dt.dayofweek
    frame["day_of_week"] = day_of_week.astype(np.int8)
    frame["is_weekend"] = (day_of_week >= 5).astype(np.int8)
    frame["day_index"] = (frame["dt"] - origin).dt.days.astype(np.int16)
    return frame


def _shift(values: np.ndarray, lag: int) -> np.ndarray:
    result = np.full(values.shape, np.nan, dtype=np.float32)
    result[:, lag:] = values[:, :-lag]
    return result


def _past_sum(values: np.ndarray, window: int) -> np.ndarray:
    n_series, n_dates = values.shape
    result = np.full((n_series, n_dates), np.nan, dtype=np.float32)
    cumulative = np.empty((n_series, n_dates + 1), dtype=np.float64)
    cumulative[:, 0] = 0.0
    np.cumsum(values, axis=1, dtype=np.float64, out=cumulative[:, 1:])
    result[:, window:] = (
        cumulative[:, window:n_dates] - cumulative[:, : n_dates - window]
    ).astype(np.float32)
    return result


def create_lag_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Create all historical features from t-1 or earlier."""
    n_dates = frame["dt"].nunique()
    n_series = len(frame) // n_dates
    sales = frame["sale_amount"].to_numpy(dtype=np.float32, copy=False).reshape(n_series, n_dates)
    stockout_day = (
        frame["stock_hour6_22_cnt"].to_numpy(copy=False).reshape(n_series, n_dates) > 0
    ).astype(np.float32)
    for lag in (1, 2, 3, 7, 14):
        frame[f"lag_{lag}"] = _shift(sales, lag).reshape(-1)
    for window in (3, 7, 14):
        frame[f"rolling_mean_{window}"] = (
            _past_sum(sales, window) / np.float32(window)
        ).reshape(-1)
    rolling_std = np.full(sales.shape, np.nan, dtype=np.float32)
    windows = np.lib.stride_tricks.sliding_window_view(sales, 7, axis=1)
    rolling_std[:, 7:] = windows[:, :-1].std(axis=2, dtype=np.float32)
    frame["rolling_std_7"] = rolling_std.reshape(-1)
    for window in (7, 14):
        count = _past_sum(stockout_day, window)
        frame[f"past_stockout_count_{window}"] = count.reshape(-1)
        frame[f"past_stockout_rate_{window}"] = (count / np.float32(window)).reshape(-1)
    return frame


def train_model(
    model_family: str,
    training_frame: pd.DataFrame,
    feature_columns: list[str],
    config_name: str,
    config: dict[str, Any],
    training_end: str,
) -> Any:
    valid_rows = training_frame["lag_14"].notna() & (
        training_frame["dt"] <= pd.Timestamp(training_end)
    )
    x_train = training_frame.loc[valid_rows, feature_columns]
    y_train = training_frame.loc[valid_rows, "sale_amount"].to_numpy(
        dtype=np.float32, copy=False
    )
    print(
        f"    fitting {model_family}/{config_name}: {len(x_train):,} rows, "
        f"{len(feature_columns)} features",
        flush=True,
    )
    params = COMMON_PARAMS[model_family] | config
    if model_family != "catboost":  # pragma: no cover
        raise ValueError(f"Unsupported model family: {model_family}")
    model = CatBoostRegressor(**params)
    model.fit(x_train, y_train, cat_features=CATEGORICAL_COLUMNS, verbose=False)
    del x_train, y_train, valid_rows
    gc.collect()
    return model


def _dynamic_features(
    sales_history: np.ndarray,
    stockout_history: np.ndarray,
) -> dict[str, np.ndarray]:
    if sales_history.shape[1] < 14:
        raise ValueError("At least 14 history days are required")
    stockout_day = stockout_history > 0
    return {
        "lag_1": sales_history[:, -1],
        "lag_2": sales_history[:, -2],
        "lag_3": sales_history[:, -3],
        "lag_7": sales_history[:, -7],
        "lag_14": sales_history[:, -14],
        "rolling_mean_3": sales_history[:, -3:].mean(axis=1, dtype=np.float32),
        "rolling_mean_7": sales_history[:, -7:].mean(axis=1, dtype=np.float32),
        "rolling_mean_14": sales_history[:, -14:].mean(axis=1, dtype=np.float32),
        "rolling_std_7": sales_history[:, -7:].std(axis=1, dtype=np.float32),
        "past_stockout_count_7": stockout_day[:, -7:].sum(axis=1).astype(np.float32),
        "past_stockout_rate_7": stockout_day[:, -7:].mean(axis=1).astype(np.float32),
        "past_stockout_count_14": stockout_day[:, -14:].sum(axis=1).astype(np.float32),
        "past_stockout_rate_14": stockout_day[:, -14:].mean(axis=1).astype(np.float32),
    }


def _day_features(
    forecast_frame: pd.DataFrame,
    forecast_date: pd.Timestamp,
    sales_history: np.ndarray,
    stockout_history: np.ndarray,
    feature_columns: list[str],
    n_series: int,
) -> pd.DataFrame:
    base_columns = [column for column in feature_columns if column not in DYNAMIC_COLUMNS]
    rows = forecast_frame.loc[
        forecast_frame["dt"] == forecast_date, base_columns
    ].copy()
    if len(rows) != n_series:
        raise ValueError(f"Expected {n_series:,} rows on {forecast_date.date()}")
    for column, values in _dynamic_features(sales_history, stockout_history).items():
        rows[column] = values
    return rows[feature_columns]


def forecast_fixed_origin(
    model: Any,
    history_sales: np.ndarray,
    history_stockout: np.ndarray,
    forecast_frame: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    feature_columns: list[str],
) -> np.ndarray:
    n_series, history_days = history_sales.shape
    horizon = len(forecast_dates)
    sales_extended = np.empty((n_series, history_days + horizon), dtype=np.float32)
    stockout_extended = np.empty((n_series, history_days + horizon), dtype=np.int8)
    sales_extended[:, :history_days] = history_sales
    stockout_extended[:, :history_days] = history_stockout
    predictions = np.empty((n_series, horizon), dtype=np.float32)
    for step, forecast_date in enumerate(forecast_dates):
        features = _day_features(
            forecast_frame,
            forecast_date,
            sales_extended[:, : history_days + step],
            stockout_extended[:, : history_days + step],
            feature_columns,
            n_series,
        )
        next_prediction = np.maximum(model.predict(features), 0.0).astype(np.float32)
        predictions[:, step] = next_prediction
        sales_extended[:, history_days + step] = next_prediction
        # Future stockout is unknown at the fixed origin and is never read from
        # validation/eval. Zero is the explicit no-observed-stockout assumption.
        stockout_extended[:, history_days + step] = 0
        del features
    return predictions


def forecast_rolling_origin(
    model: Any,
    history_sales: np.ndarray,
    history_stockout: np.ndarray,
    forecast_frame: pd.DataFrame,
    forecast_dates: pd.DatetimeIndex,
    actual_sales: np.ndarray,
    actual_stockout: np.ndarray,
    feature_columns: list[str],
) -> np.ndarray:
    n_series, history_days = history_sales.shape
    horizon = len(forecast_dates)
    sales_extended = np.empty((n_series, history_days + horizon), dtype=np.float32)
    stockout_extended = np.empty((n_series, history_days + horizon), dtype=np.int8)
    sales_extended[:, :history_days] = history_sales
    stockout_extended[:, :history_days] = history_stockout
    predictions = np.empty((n_series, horizon), dtype=np.float32)
    for step, forecast_date in enumerate(forecast_dates):
        features = _day_features(
            forecast_frame,
            forecast_date,
            sales_extended[:, : history_days + step],
            stockout_extended[:, : history_days + step],
            feature_columns,
            n_series,
        )
        predictions[:, step] = np.maximum(model.predict(features), 0.0).astype(np.float32)
        sales_extended[:, history_days + step] = actual_sales[:, step]
        stockout_extended[:, history_days + step] = actual_stockout[:, step]
        del features
    return predictions


def metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float | int]:
    y = actual[eligible].astype(np.float64, copy=False)
    y_hat = np.maximum(prediction[eligible].astype(np.float64, copy=False), 0.0)
    if not y.size:
        return {"n": 0} | {
            key: float("nan")
            for key in [
                "mae",
                "rmse",
                "wape",
                "wpe",
                "underestimation_rate",
                "r2",
                "actual_sum",
                "prediction_sum",
            ]
        }
    error = y_hat - y
    absolute_error = np.abs(error)
    denominator = np.abs(y).sum()
    squared_error = np.square(error).sum()
    centered = np.square(y - y.mean()).sum()
    return {
        "n": int(y.size),
        "mae": float(absolute_error.mean()),
        "rmse": float(math.sqrt(squared_error / y.size)),
        "wape": float(absolute_error.sum() / denominator) if denominator else float("nan"),
        "wpe": float(error.sum() / denominator) if denominator else float("nan"),
        "underestimation_rate": float(np.mean(y_hat < y)),
        "r2": float(1.0 - squared_error / centered) if centered else float("nan"),
        "actual_sum": float(y.sum()),
        "prediction_sum": float(y_hat.sum()),
    }


def horizon_metrics(
    model_family: str,
    feature_set: str,
    config_name: str,
    protocol: str,
    split: str,
    actual: np.ndarray,
    prediction: np.ndarray,
    eligible: np.ndarray,
) -> list[dict[str, Any]]:
    rows = []
    for index in range(actual.shape[1]):
        rows.append(
            {
                "model": model_family,
                "feature_set": feature_set,
                "config": config_name,
                "protocol": protocol,
                "target": "observed_sales",
                "split": split,
                "horizon": index + 1,
            }
            | metrics(actual[:, index], prediction[:, index], eligible[:, index])
        )
    return rows


def cumulative_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float | int]:
    fully_observed = eligible.all(axis=1)
    return metrics(actual.sum(axis=1), prediction.sum(axis=1), fully_observed)


def _panel_arrays(frame: pd.DataFrame, n_dates: int) -> tuple[np.ndarray, np.ndarray]:
    n_series = len(frame) // n_dates
    sales = frame["sale_amount"].to_numpy(dtype=np.float32, copy=False).reshape(n_series, n_dates)
    stockout = frame["stock_hour6_22_cnt"].to_numpy(copy=False).reshape(n_series, n_dates)
    return sales, stockout.astype(np.int8, copy=False)


def _evaluate(
    model_family: str,
    feature_set: str,
    config_name: str,
    split: str,
    protocol_predictions: dict[str, np.ndarray],
    actual: np.ndarray,
    stockout: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = stockout == 0
    overall_rows, horizon_rows, cumulative_rows = [], [], []
    for protocol, prediction in protocol_predictions.items():
        overall_rows.append(
            {
                "model": model_family,
                "feature_set": feature_set,
                "config": config_name,
                "protocol": protocol,
                "target": "observed_sales",
                "split": split,
            }
            | metrics(actual, prediction, eligible)
        )
        horizon_rows.extend(
            horizon_metrics(
                model_family,
                feature_set,
                config_name,
                protocol,
                split,
                actual,
                prediction,
                eligible,
            )
        )
        cumulative_rows.append(
            {
                "model": model_family,
                "feature_set": feature_set,
                "config": config_name,
                "protocol": protocol,
                "target": "observed_sales",
                "split": split,
            }
            | cumulative_metrics(actual, prediction, eligible)
        )
    return overall_rows, horizon_rows, cumulative_rows


def _print_rows(prefix: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"{prefix} {row['protocol']:22s} WAPE={row['wape']:.4f} "
            f"WPE={row['wpe']:.4f} R2={row['r2']:.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FreshRetail CatBoost full experiment")
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--eval-path", default=EVAL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["catboost"],
        default=MODELS_TO_RUN,
    )
    args, _unknown = parser.parse_known_args()
    model_families = list(dict.fromkeys(args.models))
    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = discover_parquet("train.parquet", args.train_path)
    eval_path = discover_parquet("eval.parquet", args.eval_path)
    print(f"Models: {model_families}", flush=True)
    print(f"Train:  {train_path}", flush=True)
    print(f"Eval:   {eval_path}", flush=True)

    train, eval_frame, metadata = load_data(train_path, eval_path)
    origin = pd.Timestamp(TRAIN_START)
    train = add_calendar_features(train, origin)
    eval_frame = add_calendar_features(eval_frame, origin)
    print("Creating leakage-safe historical features ...", flush=True)
    train = create_lag_features(train)

    train_dates: pd.DatetimeIndex = metadata["train_dates"]
    eval_dates: pd.DatetimeIndex = metadata["eval_dates"]
    series: pd.DataFrame = metadata["series"]
    train_sales, train_stockout = _panel_arrays(train, len(train_dates))
    eval_sales, eval_stockout = _panel_arrays(eval_frame, len(eval_dates))
    validation_dates = pd.date_range(VALIDATION_START, VALIDATION_END, freq="D")
    validation_index = int(np.flatnonzero(train_dates == pd.Timestamp(VALIDATION_START))[0])
    validation_actual = train_sales[:, validation_index:]
    validation_stockout = train_stockout[:, validation_index:]
    history_sales = train_sales[:, :validation_index]
    history_stockout = train_stockout[:, :validation_index]
    validation_context = train.loc[
        (train["dt"] >= pd.Timestamp(VALIDATION_START))
        & (train["dt"] <= pd.Timestamp(VALIDATION_END))
    ]

    validation_rows: list[dict[str, Any]] = []
    validation_horizon_rows: list[dict[str, Any]] = []
    validation_cumulative_rows: list[dict[str, Any]] = []

    for model_family in model_families:
        for feature_set, feature_columns in FEATURE_SETS.items():
            print(f"\nVALIDATION: {model_family} / {feature_set}", flush=True)
            for config_name, config in MODEL_CONFIGS[model_family].items():
                model = train_model(
                    model_family,
                    train,
                    feature_columns,
                    config_name,
                    config,
                    VALIDATION_TRAIN_END,
                )
                fixed = forecast_fixed_origin(
                    model,
                    history_sales,
                    history_stockout,
                    validation_context,
                    validation_dates,
                    feature_columns,
                )
                rolling = forecast_rolling_origin(
                    model,
                    history_sales,
                    history_stockout,
                    validation_context,
                    validation_dates,
                    validation_actual,
                    validation_stockout,
                    feature_columns,
                )
                overall, horizons, cumulative = _evaluate(
                    model_family,
                    feature_set,
                    config_name,
                    "validation",
                    {
                        "fixed_origin_7_day": fixed,
                        "rolling_origin_1_day": rolling,
                    },
                    validation_actual,
                    validation_stockout,
                )
                validation_rows.extend(overall)
                validation_horizon_rows.extend(horizons)
                validation_cumulative_rows.extend(cumulative)
                _print_rows(f"  {config_name:22s}", overall)
                del model, fixed, rolling
                gc.collect()

    validation_results = pd.DataFrame(validation_rows)
    validation_results["absolute_wpe"] = validation_results["wpe"].abs()
    validation_results = validation_results.sort_values(
        [
            "protocol",
            "model",
            "feature_set",
            "wape",
            "absolute_wpe",
            "underestimation_rate",
            "mae",
        ],
        kind="stable",
    ).reset_index(drop=True)
    validation_results["model_feature_rank"] = validation_results.groupby(
        ["protocol", "model", "feature_set"], sort=False
    ).cumcount() + 1
    prefix = "catboost"
    validation_results.to_csv(output_dir / f"{prefix}_validation_full_data.csv", index=False)
    pd.DataFrame(validation_horizon_rows).to_csv(
        output_dir / f"{prefix}_validation_by_horizon.csv", index=False
    )
    pd.DataFrame(validation_cumulative_rows).to_csv(
        output_dir / f"{prefix}_validation_cumulative_7_day.csv", index=False
    )

    primary = validation_results.loc[
        validation_results["protocol"] == "fixed_origin_7_day"
    ]
    best = primary.loc[primary["model_feature_rank"] == 1].copy()
    best_configs = {
        f"{row.model}/{row.feature_set}": row.config
        for row in best.itertuples(index=False)
    }
    print("\nLOCKED FIXED-ORIGIN WINNERS:", flush=True)
    print(
        best[["model", "feature_set", "config", "wape", "wpe", "r2"]].to_string(index=False),
        flush=True,
    )

    final_rows: list[dict[str, Any]] = []
    final_horizon_rows: list[dict[str, Any]] = []
    final_cumulative_rows: list[dict[str, Any]] = []
    final_predictions: dict[tuple[str, str, str], np.ndarray] = {}
    for model_family in model_families:
        for feature_set, feature_columns in FEATURE_SETS.items():
            config_name = best_configs[f"{model_family}/{feature_set}"]
            print(f"\nFINAL REFIT: {model_family} / {feature_set} / {config_name}", flush=True)
            model = train_model(
                model_family,
                train,
                feature_columns,
                config_name,
                MODEL_CONFIGS[model_family][config_name],
                FINAL_TRAIN_END,
            )
            fixed = forecast_fixed_origin(
                model,
                train_sales,
                train_stockout,
                eval_frame,
                eval_dates,
                feature_columns,
            )
            rolling = forecast_rolling_origin(
                model,
                train_sales,
                train_stockout,
                eval_frame,
                eval_dates,
                eval_sales,
                eval_stockout,
                feature_columns,
            )
            final_predictions[(model_family, feature_set, "fixed")] = fixed
            final_predictions[(model_family, feature_set, "rolling")] = rolling
            overall, horizons, cumulative = _evaluate(
                model_family,
                feature_set,
                config_name,
                "test",
                {
                    "fixed_origin_7_day": fixed,
                    "rolling_origin_1_day": rolling,
                },
                eval_sales,
                eval_stockout,
            )
            final_rows.extend(overall)
            final_horizon_rows.extend(horizons)
            final_cumulative_rows.extend(cumulative)
            _print_rows("  final", overall)
            del model
            gc.collect()

    final_results = pd.DataFrame(final_rows).sort_values(
        ["protocol", "model", "wape"], kind="stable"
    ).reset_index(drop=True)
    final_results.to_csv(output_dir / f"{prefix}_final_test.csv", index=False)
    final_horizon_frame = pd.DataFrame(final_horizon_rows)
    final_horizon_frame.to_csv(
        output_dir / f"{prefix}_final_test_by_horizon.csv", index=False
    )
    pd.DataFrame(final_cumulative_rows).to_csv(
        output_dir / f"{prefix}_final_test_cumulative_7_day.csv", index=False
    )
    error_growth = final_horizon_frame.pivot_table(
        index=["model", "feature_set", "config", "protocol"],
        columns="horizon",
        values="wape",
    ).reset_index()
    error_growth["wape_growth_h7_minus_h1"] = error_growth[7] - error_growth[1]
    error_growth = error_growth.rename(columns={1: "wape_h1", 7: "wape_h7"})
    error_growth.to_csv(output_dir / f"{prefix}_final_test_error_growth.csv", index=False)

    prediction_data: dict[str, Any] = {
        "store_id": np.repeat(series["store_id"].to_numpy(dtype=np.int32), len(eval_dates)),
        "product_id": np.repeat(series["product_id"].to_numpy(dtype=np.int32), len(eval_dates)),
        "dt": np.tile(eval_dates.to_numpy(), len(series)),
        "actual_sale_amount": eval_sales.reshape(-1),
        "stockout_hours": eval_stockout.reshape(-1),
        "eligible_non_stockout": (eval_stockout == 0).reshape(-1),
    }
    for model_family in model_families:
        for feature_set in FEATURE_SETS:
            prediction_data[f"prediction_{model_family}_{feature_set}_fixed"] = final_predictions[
                (model_family, feature_set, "fixed")
            ].reshape(-1)
            prediction_data[f"prediction_{model_family}_{feature_set}_rolling"] = final_predictions[
                (model_family, feature_set, "rolling")
            ].reshape(-1)
    pd.DataFrame(prediction_data).to_parquet(
        output_dir / f"{prefix}_best_model_predictions.parquet", index=False
    )

    manifest = {
        "experiment": "catboost_observed_sales_feature_comparison",
        "models_run": model_families,
        "full_data": True,
        "train_path": str(train_path),
        "eval_path": str(eval_path),
        "date_ranges": {
            "validation_train": [TRAIN_START, VALIDATION_TRAIN_END],
            "validation": [VALIDATION_START, VALIDATION_END],
            "final_refit": [TRAIN_START, FINAL_TRAIN_END],
            "final_test": [TEST_START, TEST_END],
        },
        "rows": {"train": int(len(train)), "eval": int(len(eval_frame))},
        "series": int(len(series)),
        "target": "observed_sales",
        "feature_sets": FEATURE_SETS,
        "weather_setting": (
            "Weather-enhanced validation/test covariates are known-ahead/oracle features "
            "unless genuine forecast weather is supplied."
        ),
        "configs": {family: MODEL_CONFIGS[family] for family in model_families},
        "common_params": {family: COMMON_PARAMS[family] for family in model_families},
        "best_config_per_model_feature_set": best_configs,
        "primary_protocol": "fixed_origin_7_day",
        "secondary_protocol": "rolling_origin_1_day",
        "selection_rule": "lowest fixed-origin validation WAPE per model and feature set",
        "rolling_result_note": (
            "All rolling validation candidates are reported but not used for selection. "
            "Final rolling results use fixed-origin-selected winners without retuning."
        ),
        "anti_leakage": {
            "historical_windows": "end at t-1",
            "fixed_sales_update": "recursive predictions",
            "fixed_future_stockout_update": "zero/no-observed-stockout assumption",
            "rolling_update": "completed prior-day actual sales and stockout",
            "same_day_sales_feature": False,
            "same_day_stockout_feature": False,
        },
        "stockout_feature_definition": (
            "past_stockout_count_k counts prior days with stock_hour6_22_cnt > 0; "
            "past_stockout_rate_k divides by k"
        ),
        "rolling_std_ddof": 0,
        "evaluation_eligibility": "stock_hour6_22_cnt == 0",
        "cumulative_eligibility": "no stockout on all seven forecast days",
        "validation_candidate_fits": sum(
            len(MODEL_CONFIGS[family]) * len(FEATURE_SETS) for family in model_families
        ),
        "final_refits": len(model_families) * len(FEATURE_SETS),
        "validation_results": validation_results.to_dict(orient="records"),
        "final_test_results": final_results.to_dict(orient="records"),
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "catboost": catboost.__version__,
    }
    (output_dir / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print("\nFINAL TEST - fixed-origin-selected winners under both protocols:", flush=True)
    print(final_results.to_string(index=False), flush=True)
    print(f"\nSaved outputs to: {output_dir}", flush=True)
    print(f"Runtime: {(time.perf_counter() - started) / 60:.1f} minutes", flush=True)


if __name__ == "__main__":
    main()
