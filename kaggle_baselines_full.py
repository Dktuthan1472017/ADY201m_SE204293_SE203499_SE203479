"""Full-data Seasonal Naive and Weighted Moving Average baselines for Kaggle.

This file is intentionally self-contained and can be pasted directly into one
Kaggle notebook cell. It evaluates ten predefined baseline configurations on all
50,000 store-product series, selects one winner per family using fixed-origin
validation, and then evaluates only those two locked winners on eval.parquet
under both fixed-origin and rolling-origin protocols.

Kaggle usage:
    Attach the data, paste this entire file into one cell, and press Run.

No path edits are needed when exactly one train.parquet and one eval.parquet are
mounted below /kaggle/input.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import time
from pathlib import Path

import numpy as np
import pandas as pd


TRAIN_START = "2024-03-28"
VALIDATION_TRAIN_END = "2024-06-18"
VALIDATION_START = "2024-06-19"
VALIDATION_END = "2024-06-25"
FINAL_TRAIN_END = "2024-06-25"
TEST_START = "2024-06-26"
TEST_END = "2024-07-02"
BASELINE_FEATURE_SET = "historical_sales_only"

# Notebook-friendly settings. Edit only when Kaggle has multiple matching files.
TRAIN_PATH: str | None = None
EVAL_PATH: str | None = None
RUN_FINAL_TEST = True
OUTPUT_DIR = "/kaggle/working/freshretail-baselines"

CONFIGS: dict[str, dict[str, dict[str, list[float] | list[int]]]] = {
    "seasonal_naive": {
        "lag_1": {"lags": [1], "weights": [1.0]},
        "lag_7": {"lags": [7], "weights": [1.0]},
        "lag_14": {"lags": [14], "weights": [1.0]},
        "blend_lag_1_7": {"lags": [1, 7], "weights": [0.5, 0.5]},
        "weekly_ensemble": {"lags": [7, 14, 21], "weights": [0.6, 0.3, 0.1]},
    },
    "weighted_moving_average": {
        "wma_3": {"lags": [1, 2, 3], "weights": [0.5, 0.3, 0.2]},
        "wma_7_linear": {
            "lags": [1, 2, 3, 4, 5, 6, 7],
            "weights": [7, 6, 5, 4, 3, 2, 1],
        },
        "wma_7_exp": {
            "lags": [1, 2, 3, 4, 5, 6, 7],
            "weights": [1.0, 0.8, 0.64, 0.512, 0.4096, 0.32768, 0.262144],
        },
        "wma_14_exp": {
            "lags": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
            "weights": [
                1.0, 0.9, 0.81, 0.729, 0.6561, 0.59049, 0.531441,
                0.478297, 0.430467, 0.38742, 0.348678, 0.313811,
                0.28243, 0.254187,
            ],
        },
        "weekly_wma": {"lags": [7, 14, 21], "weights": [0.6, 0.3, 0.1]},
    },
}


def discover_parquet(filename: str, explicit: str | None) -> Path:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    env_name = f"FRESHRETAIL_{Path(filename).stem.upper()}_PATH"
    if os.environ.get(env_name):
        candidates.append(Path(os.environ[env_name]))
    kaggle_input = Path("/kaggle/input")
    if kaggle_input.exists():
        candidates.extend(kaggle_input.rglob(filename))
    existing = sorted({path.resolve() for path in candidates if path.is_file()}, key=str)
    if not existing:
        raise FileNotFoundError(
            f"{filename} was not found. Attach the FreshRetailNet dataset or pass an explicit path."
        )
    if len(existing) > 1:
        choices = "\n".join(f"- {path}" for path in existing)
        raise RuntimeError(f"Multiple {filename} files found; pass an explicit path:\n{choices}")
    return existing[0]


def load_full_panel(
    path: Path,
    expected_dates: int,
) -> tuple[pd.DataFrame, pd.DatetimeIndex, np.ndarray, np.ndarray]:
    columns = ["store_id", "product_id", "dt", "sale_amount", "stock_hour6_22_cnt"]
    frame = pd.read_parquet(path, columns=columns)
    frame["dt"] = pd.to_datetime(frame["dt"], format="%Y-%m-%d")
    frame = frame.sort_values(["store_id", "product_id", "dt"], kind="stable").reset_index(drop=True)

    if frame[columns].isna().any().any():
        raise ValueError("Missing values were found in required baseline columns")
    if frame.duplicated(["store_id", "product_id", "dt"]).any():
        raise ValueError("Duplicate store-product-date keys were found")

    dates = pd.DatetimeIndex(frame["dt"].drop_duplicates().sort_values())
    counts = frame.groupby(["store_id", "product_id"], sort=False, observed=True).size()
    if counts.nunique() != 1 or int(counts.iloc[0]) != len(dates):
        raise ValueError("The daily panel is incomplete or series have unequal lengths")

    n_series = len(counts)
    n_dates = len(dates)
    if len(frame) != n_series * n_dates:
        raise ValueError("Row count does not equal n_series × n_dates")
    if n_series != 50_000 or n_dates != expected_dates:
        raise ValueError(
            f"Expected 50,000 × {expected_dates} full data, found {n_series:,} × {n_dates}"
        )

    series = frame.loc[::n_dates, ["store_id", "product_id"]].reset_index(drop=True)
    target = frame["sale_amount"].to_numpy(dtype=np.float32, copy=True).reshape(n_series, n_dates)
    stockout = frame["stock_hour6_22_cnt"].to_numpy(dtype=np.int8, copy=True).reshape(n_series, n_dates)
    return series, dates, target, stockout


def forecast_rolling(
    target: np.ndarray,
    indices: np.ndarray,
    config: dict[str, list[float] | list[int]],
) -> np.ndarray:
    """One-day-ahead forecasts that may use realized prior target days."""
    lags = np.asarray(config["lags"], dtype=np.int64)
    weights = np.asarray(config["weights"], dtype=np.float64)
    weights /= weights.sum()
    prediction = np.zeros((target.shape[0], indices.size), dtype=np.float32)
    for lag, weight in zip(lags, weights, strict=True):
        prediction += target[:, indices - lag] * np.float32(weight)
    return np.maximum(prediction, 0.0)


def forecast_fixed_origin(
    history: np.ndarray,
    horizon: int,
    config: dict[str, list[float] | list[int]],
) -> np.ndarray:
    """Forecast all horizons from one cutoff, recursively using prior predictions."""
    lags = np.asarray(config["lags"], dtype=np.int64)
    weights = np.asarray(config["weights"], dtype=np.float64)
    weights /= weights.sum()
    if history.shape[1] < int(lags.max()):
        raise ValueError("Insufficient pre-origin history for this configuration")

    origin_length = history.shape[1]
    extended = np.empty((history.shape[0], origin_length + horizon), dtype=np.float32)
    extended[:, :origin_length] = history
    for step in range(horizon):
        target_index = origin_length + step
        next_prediction = np.zeros(history.shape[0], dtype=np.float32)
        for lag, weight in zip(lags, weights, strict=True):
            next_prediction += extended[:, target_index - lag] * np.float32(weight)
        extended[:, target_index] = np.maximum(next_prediction, 0.0)
    return extended[:, origin_length:]


def metrics(actual: np.ndarray, prediction: np.ndarray, eligible: np.ndarray) -> dict[str, float | int]:
    y = actual[eligible].astype(np.float64, copy=False)
    y_hat = np.maximum(prediction[eligible].astype(np.float64, copy=False), 0.0)
    if y.size == 0:
        return {
            "n": 0,
            "mae": float("nan"),
            "rmse": float("nan"),
            "wape": float("nan"),
            "wpe": float("nan"),
            "underestimation_rate": float("nan"),
            "r2": float("nan"),
            "actual_sum": float("nan"),
            "prediction_sum": float("nan"),
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
    family: str,
    config_name: str,
    protocol: str,
    split: str,
    actual: np.ndarray,
    prediction: np.ndarray,
    eligible: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for horizon_index in range(actual.shape[1]):
        rows.append(
            {
                "model": family,
                "family": family,
                "config": config_name,
                "protocol": protocol,
                "target": "observed_sales",
                "feature_set": BASELINE_FEATURE_SET,
                "split": split,
                "horizon": horizon_index + 1,
            }
            | metrics(
                actual[:, horizon_index],
                prediction[:, horizon_index],
                eligible[:, horizon_index],
            )
        )
    return rows


def cumulative_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float | int]:
    """Evaluate seven-day totals only for series uncensored on all seven days."""
    fully_observed_series = eligible.all(axis=1)
    actual_total = actual.sum(axis=1)
    prediction_total = prediction.sum(axis=1)
    return metrics(actual_total, prediction_total, fully_observed_series)


def save_chart(results: pd.DataFrame, output_path: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable; skipping chart.", flush=True)
        return
    chart = results.sort_values("wape", ascending=True)
    labels = chart["protocol"] + " / " + chart["family"] + " / " + chart["config"]
    figure, axis = plt.subplots(figsize=(11, 6))
    axis.barh(labels, chart["wape"] * 100, color="#0F766E")
    axis.set_xlabel("Validation WAPE (%) — lower is better")
    axis.set_title("FreshRetailNet-50K full-data baseline comparison")
    axis.grid(axis="x", alpha=0.2)
    figure.tight_layout()
    figure.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    """Run primary fixed-origin and secondary rolling-origin evaluation."""
    parser = argparse.ArgumentParser(description="Run full-data FreshRetailNet-50K baselines.")
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--eval-path", default=EVAL_PATH)
    parser.add_argument("--run-final-test", action="store_true", dest="run_final_test")
    parser.add_argument("--validation-only", action="store_false", dest="run_final_test")
    parser.set_defaults(run_final_test=RUN_FINAL_TEST)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args, _unknown = parser.parse_known_args()

    started = time.perf_counter()
    train_path = discover_parquet("train.parquet", args.train_path)
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Using full train data: {train_path}", flush=True)

    series, dates, target, stockout = load_full_panel(train_path, expected_dates=90)
    expected_train_dates = pd.date_range(TRAIN_START, FINAL_TRAIN_END, freq="D")
    if not dates.equals(expected_train_dates):
        raise ValueError(
            f"Train must cover {TRAIN_START}..{FINAL_TRAIN_END}; found "
            f"{dates.min().date()}..{dates.max().date()}"
        )
    validation_indices = np.flatnonzero(
        (dates >= pd.Timestamp(VALIDATION_START)) & (dates <= pd.Timestamp(VALIDATION_END))
    )
    actual = target[:, validation_indices]
    eligible = stockout[:, validation_indices] == 0
    fixed_history = target[:, : int(validation_indices[0])]

    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, str, str], np.ndarray] = {}
    validation_horizon_rows: list[dict[str, object]] = []
    validation_cumulative_rows: list[dict[str, object]] = []

    for family, family_configs in CONFIGS.items():
        for config_name, config in family_configs.items():
            protocol_predictions = {
                "fixed_origin_7_day": forecast_fixed_origin(
                    fixed_history, len(validation_indices), config
                ),
                "rolling_origin_1_day": forecast_rolling(target, validation_indices, config),
            }
            for protocol, prediction in protocol_predictions.items():
                predictions[(protocol, family, config_name)] = prediction
                row = {
                    "model": family,
                    "family": family,
                    "config": config_name,
                    "protocol": protocol,
                    "target": "observed_sales",
                    "feature_set": BASELINE_FEATURE_SET,
                    "split": "validation",
                    "validation_start": VALIDATION_START,
                    "validation_end": VALIDATION_END,
                } | metrics(actual, prediction, eligible)
                rows.append(row)
                validation_horizon_rows.extend(
                    horizon_metrics(
                        family, config_name, protocol, "validation", actual, prediction, eligible
                    )
                )
                validation_cumulative_rows.append(
                    {
                        "model": family,
                        "family": family,
                        "config": config_name,
                        "protocol": protocol,
                        "target": "observed_sales",
                        "feature_set": BASELINE_FEATURE_SET,
                        "split": "validation",
                    }
                    | cumulative_metrics(actual, prediction, eligible)
                )
                print(
                    f"{protocol:22s} {family:25s} {config_name:20s} "
                    f"WAPE={row['wape']:.4f} WPE={row['wpe']:.4f}",
                    flush=True,
                )

    results = pd.DataFrame(rows)
    results["absolute_wpe"] = results["wpe"].abs()
    results = results.sort_values(
        ["protocol", "family", "wape", "absolute_wpe", "underestimation_rate", "mae"],
        kind="stable",
    ).reset_index(drop=True)
    results["family_rank"] = results.groupby(["protocol", "family"], sort=False).cumcount() + 1
    results.to_csv(output_dir / "baseline_validation_full_data.csv", index=False)
    results.to_json(output_dir / "baseline_validation_full_data.json", orient="records", indent=2)
    pd.DataFrame(validation_horizon_rows).to_csv(
        output_dir / "baseline_validation_by_horizon.csv", index=False
    )
    pd.DataFrame(validation_cumulative_rows).to_csv(
        output_dir / "baseline_validation_cumulative_7_day.csv", index=False
    )

    primary_validation = results.loc[results["protocol"] == "fixed_origin_7_day"]
    best = primary_validation.loc[primary_validation["family_rank"] == 1]
    winner_names = {row.family: row.config for row in best.itertuples()}
    prediction_frame = pd.DataFrame(
        {
            "store_id": np.repeat(series["store_id"].to_numpy(dtype=np.int32), len(validation_indices)),
            "product_id": np.repeat(series["product_id"].to_numpy(dtype=np.int32), len(validation_indices)),
            "dt": np.tile(dates[validation_indices].to_numpy(), len(series)),
            "actual_sale_amount": actual.reshape(-1),
            "stockout_hours": stockout[:, validation_indices].reshape(-1),
            "eligible_non_stockout": eligible.reshape(-1),
            "seasonal_fixed_prediction": predictions[
                ("fixed_origin_7_day", "seasonal_naive", winner_names["seasonal_naive"])
            ].reshape(-1),
            "seasonal_rolling_prediction": predictions[
                ("rolling_origin_1_day", "seasonal_naive", winner_names["seasonal_naive"])
            ].reshape(-1),
            "wma_fixed_prediction": predictions[
                ("fixed_origin_7_day", "weighted_moving_average", winner_names["weighted_moving_average"])
            ].reshape(-1),
            "wma_rolling_prediction": predictions[
                ("rolling_origin_1_day", "weighted_moving_average", winner_names["weighted_moving_average"])
            ].reshape(-1),
        }
    )
    prediction_frame.to_parquet(
        output_dir / "best_baseline_validation_predictions.parquet", index=False
    )
    save_chart(results, output_dir / "baseline_wape_comparison.png")

    manifest: dict[str, object] = {
        "experiment": "observed_sales_baseline_synchronized",
        "full_data": True,
        "train_path": str(train_path),
        "date_ranges": {
            "validation_train": [TRAIN_START, VALIDATION_TRAIN_END],
            "validation": [VALIDATION_START, VALIDATION_END],
            "final_refit": [TRAIN_START, FINAL_TRAIN_END],
            "final_test": [TEST_START, TEST_END],
        },
        "target": "observed_sales",
        "feature_set": BASELINE_FEATURE_SET,
        "comparison_scope": ["operational", "weather_enhanced"],
        "feature_set_note": (
            "Seasonal Naive and WMA use historical sales only. Their single result is the "
            "common baseline for both LightGBM feature-set comparisons; no duplicate weather run."
        ),
        "rows": int(target.size),
        "series": int(target.shape[0]),
        "dates": int(target.shape[1]),
        "validation_rows": int(actual.size),
        "eligible_non_stockout_rows": int(eligible.sum()),
        "candidate_configs": sum(len(configs) for configs in CONFIGS.values()),
        "protocol_evaluations": len(results),
        "primary_evaluation": "fixed_origin_7_day",
        "secondary_evaluation": "rolling_origin_1_day",
        "selection_rule": "lowest fixed-origin validation WAPE per family",
        "rolling_result_note": (
            "All rolling validation candidates are reported for analysis but are not used "
            "for selection. Final rolling test results use the fixed-origin-selected winners; "
            "there is no rolling-specific retuning."
        ),
        "eval_parquet_used": bool(args.run_final_test),
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "best_by_family": best.to_dict(orient="records"),
    }

    if args.run_final_test:
        eval_path = discover_parquet("eval.parquet", args.eval_path)
        eval_series, eval_dates, eval_target, eval_stockout = load_full_panel(
            eval_path, expected_dates=7
        )
        expected_eval_dates = pd.date_range(TEST_START, TEST_END, freq="D")
        if not eval_dates.equals(expected_eval_dates):
            raise ValueError(
                f"Eval must cover {TEST_START}..{TEST_END}; found "
                f"{eval_dates.min().date()}..{eval_dates.max().date()}"
            )
        if not series.equals(eval_series):
            raise ValueError("Train and eval store-product series do not match")
        expected_start = dates.max() + pd.Timedelta(days=1)
        if eval_dates.min() != expected_start:
            raise ValueError(
                f"Eval must begin on {expected_start.date()}, found {eval_dates.min().date()}"
            )

        combined_target = np.concatenate([target, eval_target], axis=1)
        test_indices = np.arange(target.shape[1], combined_target.shape[1], dtype=np.int64)
        test_eligible = eval_stockout == 0
        test_rows: list[dict[str, object]] = []
        test_horizon_rows: list[dict[str, object]] = []
        test_cumulative_rows: list[dict[str, object]] = []
        test_predictions: dict[tuple[str, str], np.ndarray] = {}

        for row in best.itertuples():
            winner_config = CONFIGS[row.family][row.config]
            protocol_predictions = {
                "fixed_origin_7_day": forecast_fixed_origin(target, len(eval_dates), winner_config),
                "rolling_origin_1_day": forecast_rolling(combined_target, test_indices, winner_config),
            }
            for protocol, winner_prediction in protocol_predictions.items():
                test_predictions[(protocol, row.family)] = winner_prediction
                test_rows.append(
                    {
                        "model": row.family,
                        "family": row.family,
                        "config": row.config,
                        "protocol": protocol,
                        "target": "observed_sales",
                        "feature_set": BASELINE_FEATURE_SET,
                        "split": "test",
                        "test_start": str(eval_dates.min().date()),
                        "test_end": str(eval_dates.max().date()),
                    }
                    | metrics(eval_target, winner_prediction, test_eligible)
                )
                test_horizon_rows.extend(
                    horizon_metrics(
                        row.family,
                        row.config,
                        protocol,
                        "test",
                        eval_target,
                        winner_prediction,
                        test_eligible,
                    )
                )
                test_cumulative_rows.append(
                    {
                        "model": row.family,
                        "family": row.family,
                        "config": row.config,
                        "protocol": protocol,
                        "target": "observed_sales",
                        "feature_set": BASELINE_FEATURE_SET,
                        "split": "test",
                    }
                    | cumulative_metrics(eval_target, winner_prediction, test_eligible)
                )

        test_results = pd.DataFrame(test_rows).sort_values(["protocol", "wape"]).reset_index(drop=True)
        test_results.to_csv(output_dir / "baseline_final_test.csv", index=False)
        test_results.to_json(output_dir / "baseline_final_test.json", orient="records", indent=2)
        test_horizon_frame = pd.DataFrame(test_horizon_rows)
        test_horizon_frame.to_csv(output_dir / "baseline_final_test_by_horizon.csv", index=False)
        pd.DataFrame(test_cumulative_rows).to_csv(
            output_dir / "baseline_final_test_cumulative_7_day.csv", index=False
        )
        error_growth = test_horizon_frame.pivot_table(
            index=["family", "config", "protocol"], columns="horizon", values="wape"
        ).reset_index()
        error_growth["wape_growth_h7_minus_h1"] = error_growth[7] - error_growth[1]
        error_growth = error_growth.rename(columns={1: "wape_h1", 7: "wape_h7"})
        error_growth.to_csv(output_dir / "baseline_final_test_error_growth.csv", index=False)

        test_prediction_frame = pd.DataFrame(
            {
                "store_id": np.repeat(series["store_id"].to_numpy(dtype=np.int32), len(eval_dates)),
                "product_id": np.repeat(series["product_id"].to_numpy(dtype=np.int32), len(eval_dates)),
                "dt": np.tile(eval_dates.to_numpy(), len(series)),
                "actual_sale_amount": eval_target.reshape(-1),
                "stockout_hours": eval_stockout.reshape(-1),
                "eligible_non_stockout": test_eligible.reshape(-1),
                "seasonal_fixed_prediction": test_predictions[
                    ("fixed_origin_7_day", "seasonal_naive")
                ].reshape(-1),
                "seasonal_rolling_prediction": test_predictions[
                    ("rolling_origin_1_day", "seasonal_naive")
                ].reshape(-1),
                "wma_fixed_prediction": test_predictions[
                    ("fixed_origin_7_day", "weighted_moving_average")
                ].reshape(-1),
                "wma_rolling_prediction": test_predictions[
                    ("rolling_origin_1_day", "weighted_moving_average")
                ].reshape(-1),
            }
        )
        test_prediction_frame.to_parquet(
            output_dir / "best_baseline_test_predictions.parquet", index=False
        )
        manifest["eval_path"] = str(eval_path)
        manifest["test_rows"] = int(eval_target.size)
        manifest["eligible_test_rows"] = int(test_eligible.sum())
        manifest["fully_observed_test_series"] = int(test_eligible.all(axis=1).sum())
        manifest["final_test_models"] = test_results.to_dict(orient="records")
        print("\nFINAL TEST - fixed-origin-selected winners, both protocols:", flush=True)
        print(test_results.to_string(index=False), flush=True)

    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\nPRIMARY VALIDATION - fixed-origin 7-day-ahead:", flush=True)
    print(primary_validation.to_string(index=False), flush=True)
    print("\nSECONDARY VALIDATION - rolling-origin one-day-ahead:", flush=True)
    print(
        "All rolling validation candidates are shown for analysis only; they are not used "
        "for configuration selection.",
        flush=True,
    )
    print(results.loc[results["protocol"] == "rolling_origin_1_day"].to_string(index=False), flush=True)
    print(f"\nSaved outputs to: {output_dir}", flush=True)


if __name__ == "__main__":
    main()
