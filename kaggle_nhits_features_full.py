"""Full-data N-HiTS forecasting experiments for FreshRetailNet-50K.

Kaggle usage:
1. Attach exactly one train.parquet and one eval.parquet below /kaggle/input.
2. Enable a GPU accelerator (recommended).
3. Paste this entire file into one notebook cell and run it.

The N-HiTS uses 28 historical days and produces a direct seven-step forecast. It
trains five predefined configurations for operational and weather-enhanced
contexts. Fixed-origin seven-day-ahead evaluation is primary; rolling-origin
one-day-ahead evaluation is secondary. Eval is used only after configuration
selection on fixed-origin validation WAPE.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import os
import platform
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
except ImportError as exc:  # pragma: no cover
    raise ImportError("PyTorch is required for the N-HiTS experiment") from exc


TRAIN_START = "2024-03-28"
VALIDATION_TRAIN_END = "2024-06-18"
VALIDATION_START = "2024-06-19"
VALIDATION_END = "2024-06-25"
FINAL_TRAIN_END = "2024-06-25"
TEST_START = "2024-06-26"
TEST_END = "2024-07-02"

TRAIN_PATH: str | None = None
EVAL_PATH: str | None = None
OUTPUT_DIR = "/kaggle/working/freshretail-nhits-features"
RANDOM_SEED = 42
HISTORY_LENGTH = 28
FORECAST_HORIZON = 7

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

FEATURE_SETS = {
    "operational": [
        "day_index_scaled",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "holiday_flag",
        "activity_flag",
        "discount_z",
    ],
    "weather_enhanced": [
        "day_index_scaled",
        "day_of_week_sin",
        "day_of_week_cos",
        "is_weekend",
        "holiday_flag",
        "activity_flag",
        "discount_z",
        "precpt_z",
        "avg_temperature_z",
        "avg_humidity_z",
        "avg_wind_level_z",
    ],
}

NHITS_CONFIGS: dict[str, dict[str, Any]] = {
    "config_1_fast": {
        "hidden_size": 128,
        "num_blocks": 3,
        "mlp_layers": 2,
        "dropout": 0.0,
        "batch_size": 4096,
        "epochs": 4,
        "learning_rate": 1e-3,
        "weight_decay": 0.0,
    },
    "config_2_balanced": {
        "hidden_size": 192,
        "num_blocks": 3,
        "mlp_layers": 2,
        "dropout": 0.05,
        "batch_size": 2048,
        "epochs": 6,
        "learning_rate": 8e-4,
        "weight_decay": 1e-5,
    },
    "config_3_deeper": {
        "hidden_size": 256,
        "num_blocks": 6,
        "mlp_layers": 3,
        "dropout": 0.10,
        "batch_size": 2048,
        "epochs": 6,
        "learning_rate": 8e-4,
        "weight_decay": 1e-5,
    },
    "config_4_regularized": {
        "hidden_size": 192,
        "num_blocks": 6,
        "mlp_layers": 2,
        "dropout": 0.20,
        "batch_size": 2048,
        "epochs": 8,
        "learning_rate": 6e-4,
        "weight_decay": 1e-4,
    },
    "config_5_low_lr": {
        "hidden_size": 256,
        "num_blocks": 6,
        "mlp_layers": 3,
        "dropout": 0.15,
        "batch_size": 2048,
        "epochs": 10,
        "learning_rate": 3e-4,
        "weight_decay": 5e-5,
    },
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device() -> torch.device:
    """Select CUDA only when the installed PyTorch binary supports this GPU."""
    if not torch.cuda.is_available():
        print("WARNING: CUDA is unavailable; full-data N-HiTS training will be very slow.", flush=True)
        return torch.device("cpu")
    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    supported_arches = set(torch.cuda.get_arch_list())
    if supported_arches and required_arch not in supported_arches:
        gpu_name = torch.cuda.get_device_name(0)
        raise RuntimeError(
            f"GPU {gpu_name} requires {required_arch}, but this PyTorch build supports "
            f"{sorted(supported_arches)}. In Kaggle, switch the accelerator to a T4 and "
            "restart the session. Alternatively install a PyTorch CUDA 11.8 build that "
            "still supports this GPU before running the script."
        )
    # A tiny allocation catches driver/runtime incompatibility before data loading/training.
    probe = torch.ones(1, device="cuda")
    del probe
    return torch.device("cuda")


def make_grad_scaler(enabled: bool) -> Any:
    """Use the current AMP API while retaining compatibility with older PyTorch."""
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        return torch.amp.GradScaler("cuda", enabled=enabled)
    return torch.cuda.amp.GradScaler(enabled=enabled)


def autocast_context(enabled: bool) -> Any:
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        return torch.amp.autocast(device_type="cuda", enabled=enabled)
    return torch.cuda.amp.autocast(enabled=enabled)


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
        {
            path.resolve()
            for path in Path("/kaggle/input").rglob(filename)
            if path.is_file()
        },
        key=str,
    )
    if not matches:
        raise FileNotFoundError(f"No {filename} found below /kaggle/input")
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
    expected = pd.date_range(expected_start, expected_end, freq="D")
    if len(dates) != expected_days or not dates.equals(expected):
        raise ValueError(f"{label} does not match {expected_start}..{expected_end}")
    counts = frame.groupby(SERIES_KEYS, sort=False, observed=True).size()
    if len(counts) != 50_000 or counts.nunique() != 1 or int(counts.iloc[0]) != expected_days:
        raise ValueError(f"{label} is not a complete 50,000 x {expected_days} panel")
    return dates, frame.loc[::expected_days, SERIES_KEYS].reset_index(drop=True)


def load_data(train_path: Path, eval_path: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    print("Loading N-HiTS columns from train.parquet ...", flush=True)
    train = pd.read_parquet(train_path, columns=REQUIRED_COLUMNS)
    print("Loading N-HiTS columns from eval.parquet ...", flush=True)
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
        raise ValueError("Train and eval series/order do not match")

    static_codes = np.empty((len(train_series), len(CATEGORICAL_COLUMNS)), dtype=np.int64)
    cardinalities: list[int] = []
    for index, column in enumerate(CATEGORICAL_COLUMNS):
        train_values = train[column].to_numpy().reshape(len(train_series), 90)
        eval_values = eval_frame[column].to_numpy().reshape(len(eval_series), 7)
        if not np.all(train_values == train_values[:, :1]):
            raise ValueError(f"{column} is not static within each train series")
        if not np.all(eval_values == eval_values[:, :1]):
            raise ValueError(f"{column} is not static within each eval series")
        if not np.array_equal(train_values[:, 0], eval_values[:, 0]):
            raise ValueError(f"Train/eval static values differ for {column}")
        levels = np.sort(train[column].unique())
        unseen = np.setdiff1d(eval_frame[column].unique(), levels)
        if unseen.size:
            raise ValueError(f"Eval contains unseen {column}: {unseen[:10].tolist()}")
        static_values = train_values[:, 0]
        static_codes[:, index] = np.searchsorted(levels, static_values)
        cardinalities.append(int(len(levels)))
    return train, eval_frame, {
        "train_dates": train_dates,
        "eval_dates": eval_dates,
        "series": train_series,
        "static_codes": static_codes,
        "cardinalities": cardinalities,
    }


def add_calendar_features(frame: pd.DataFrame, origin: pd.Timestamp) -> pd.DataFrame:
    day_of_week = frame["dt"].dt.dayofweek.to_numpy(dtype=np.float32)
    frame["day_index_scaled"] = (
        (frame["dt"] - origin).dt.days.to_numpy(dtype=np.float32) / 100.0
    )
    frame["day_of_week_sin"] = np.sin(2.0 * np.pi * day_of_week / 7.0).astype(np.float32)
    frame["day_of_week_cos"] = np.cos(2.0 * np.pi * day_of_week / 7.0).astype(np.float32)
    frame["is_weekend"] = (day_of_week >= 5).astype(np.float32)
    return frame


def fit_context_stats(frame: pd.DataFrame, training_end: str) -> dict[str, dict[str, float]]:
    fit = frame.loc[frame["dt"] <= pd.Timestamp(training_end)]
    stats: dict[str, dict[str, float]] = {}
    for column in ["discount"] + WEATHER_COLUMNS:
        mean = float(fit[column].mean())
        std = max(float(fit[column].std(ddof=0)), 1e-6)
        stats[column] = {"mean": mean, "std": std}
    return stats


def create_context_panel(
    frame: pd.DataFrame,
    n_dates: int,
    feature_set: str,
    stats: dict[str, dict[str, float]],
) -> np.ndarray:
    data: dict[str, np.ndarray] = {
        "day_index_scaled": frame["day_index_scaled"].to_numpy(dtype=np.float32),
        "day_of_week_sin": frame["day_of_week_sin"].to_numpy(dtype=np.float32),
        "day_of_week_cos": frame["day_of_week_cos"].to_numpy(dtype=np.float32),
        "is_weekend": frame["is_weekend"].to_numpy(dtype=np.float32),
        "holiday_flag": frame["holiday_flag"].to_numpy(dtype=np.float32),
        "activity_flag": frame["activity_flag"].to_numpy(dtype=np.float32),
    }
    for column in ["discount"] + WEATHER_COLUMNS:
        data[f"{column}_z"] = (
            (frame[column].to_numpy(dtype=np.float32) - stats[column]["mean"])
            / stats[column]["std"]
        ).astype(np.float32)
    columns = FEATURE_SETS[feature_set]
    matrix = np.column_stack([data[column] for column in columns]).astype(np.float32)
    return matrix.reshape(len(frame) // n_dates, n_dates, len(columns))


def panel_arrays(frame: pd.DataFrame, n_dates: int) -> tuple[np.ndarray, np.ndarray]:
    n_series = len(frame) // n_dates
    sales = frame["sale_amount"].to_numpy(dtype=np.float32).reshape(n_series, n_dates)
    stockout = frame["stock_hour6_22_cnt"].to_numpy().reshape(n_series, n_dates)
    return sales, stockout.astype(np.int8)


class NHiTSBlock(nn.Module):
    def __init__(
        self,
        pool_size: int,
        context_size: int,
        static_size: int,
        hidden_size: int,
        mlp_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.pool_size = pool_size
        pooled_length = math.ceil(HISTORY_LENGTH / pool_size)
        self.backcast_size = pooled_length
        self.forecast_size = math.ceil(FORECAST_HORIZON / pool_size)
        auxiliary_size = 2 * (1 + context_size) + context_size + static_size
        layers: list[nn.Module] = [nn.Linear(pooled_length + auxiliary_size, hidden_size), nn.GELU()]
        for _ in range(mlp_layers - 1):
            layers.extend([nn.Dropout(dropout), nn.Linear(hidden_size, hidden_size), nn.GELU()])
        layers.append(
            nn.Linear(hidden_size, self.backcast_size + self.forecast_size)
        )
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        residual: torch.Tensor,
        history_auxiliary: torch.Tensor,
        future_context: torch.Tensor,
        static: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        pooled = F.avg_pool1d(
            residual.unsqueeze(1), self.pool_size, self.pool_size, ceil_mode=True
        ).squeeze(1)
        auxiliary_mean = history_auxiliary.mean(dim=1)
        auxiliary_std = history_auxiliary.std(dim=1, unbiased=False)
        future_summary = future_context.mean(dim=1)
        hidden = torch.cat(
            [pooled, auxiliary_mean, auxiliary_std, future_summary, static], dim=1
        )
        output = self.network(hidden)
        backcast_coefficients = output[:, : self.backcast_size]
        forecast_coefficients = output[:, self.backcast_size :]
        backcast = F.interpolate(
            backcast_coefficients.unsqueeze(1),
            size=HISTORY_LENGTH,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        forecast = F.interpolate(
            forecast_coefficients.unsqueeze(1),
            size=FORECAST_HORIZON,
            mode="linear",
            align_corners=False,
        ).squeeze(1)
        return backcast, forecast


class NHiTSForecaster(nn.Module):
    def __init__(
        self,
        cardinalities: list[int],
        context_size: int,
        hidden_size: int,
        num_blocks: int,
        mlp_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        embedding_dims = [
            min(16, max(2, int(math.ceil(math.sqrt(size)))))
            for size in cardinalities
        ]
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(size, dim)
                for size, dim in zip(cardinalities, embedding_dims, strict=True)
            ]
        )
        static_size = sum(embedding_dims)
        pool_cycle = [1, 2, 4]
        self.blocks = nn.ModuleList(
            [
                NHiTSBlock(
                    pool_cycle[index % len(pool_cycle)],
                    context_size,
                    static_size,
                    hidden_size,
                    mlp_layers,
                    dropout,
                )
                for index in range(num_blocks)
            ]
        )
        self.step_adjustment = nn.Sequential(
            nn.Linear(1 + context_size + static_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def _static_embedding(self, static_codes: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [embedding(static_codes[:, index]) for index, embedding in enumerate(self.embeddings)],
            dim=1,
        )

    def forward(
        self,
        history_sales: torch.Tensor,
        history_stockout: torch.Tensor,
        history_context: torch.Tensor,
        future_context: torch.Tensor,
        static_codes: torch.Tensor,
    ) -> torch.Tensor:
        static = self._static_embedding(static_codes)
        residual = history_sales
        history_auxiliary = torch.cat(
            [history_stockout.unsqueeze(-1), history_context], dim=2
        )
        base_forecast = torch.zeros(
            history_sales.shape[0], FORECAST_HORIZON,
            dtype=history_sales.dtype, device=history_sales.device,
        )
        for block in self.blocks:
            backcast, block_forecast = block(
                residual, history_auxiliary, future_context, static
            )
            residual = residual - backcast
            base_forecast = base_forecast + block_forecast
        horizon = future_context.shape[1]
        repeated_static = static.unsqueeze(1).expand(-1, horizon, -1)
        adjustment_input = torch.cat(
            [base_forecast[:, :horizon].unsqueeze(-1), future_context, repeated_static],
            dim=2,
        )
        return F.softplus(self.step_adjustment(adjustment_input).squeeze(-1))


def _batch_from_indices(
    flat_indices: np.ndarray,
    origins: np.ndarray,
    sales: np.ndarray,
    stockout: np.ndarray,
    context: np.ndarray,
    static_codes: np.ndarray,
    series_scale: np.ndarray,
    device: torch.device,
) -> tuple[torch.Tensor, ...]:
    n_origins = len(origins)
    series_indices = flat_indices // n_origins
    origin = origins[flat_indices % n_origins]
    history_time = origin[:, None] + np.arange(-HISTORY_LENGTH, 0, dtype=np.int64)
    future_time = origin[:, None] + np.arange(FORECAST_HORIZON, dtype=np.int64)
    scale = series_scale[series_indices, None]
    history_sales = sales[series_indices[:, None], history_time] / scale
    history_stockout = (
        stockout[series_indices[:, None], history_time] > 0
    ).astype(np.float32)
    history_context = context[series_indices[:, None], history_time]
    future_context = context[series_indices[:, None], future_time]
    target = sales[series_indices[:, None], future_time] / scale
    static = static_codes[series_indices]
    arrays = [
        history_sales,
        history_stockout,
        history_context,
        future_context,
        static,
        target,
    ]
    tensors = []
    for index, array in enumerate(arrays):
        dtype = torch.long if index == 4 else torch.float32
        tensors.append(torch.as_tensor(np.ascontiguousarray(array), dtype=dtype, device=device))
    return tuple(tensors)


def train_nhits(
    sales: np.ndarray,
    stockout: np.ndarray,
    context: np.ndarray,
    static_codes: np.ndarray,
    cardinalities: list[int],
    cutoff_days: int,
    config_name: str,
    config: dict[str, Any],
    device: torch.device,
) -> tuple[NHiTSForecaster, np.ndarray, list[float]]:
    seed_everything(RANDOM_SEED)
    origins = np.arange(
        HISTORY_LENGTH,
        cutoff_days - FORECAST_HORIZON + 1,
        dtype=np.int64,
    )
    if not len(origins):
        raise ValueError("Insufficient dates for N-HiTS training windows")
    series_scale = np.maximum(sales[:, :cutoff_days].mean(axis=1), 0.10).astype(np.float32)
    model = NHiTSForecaster(
        cardinalities=cardinalities,
        context_size=context.shape[2],
        hidden_size=config["hidden_size"],
        num_blocks=config["num_blocks"],
        mlp_layers=config["mlp_layers"],
        dropout=config["dropout"],
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    use_amp = device.type == "cuda"
    scaler = make_grad_scaler(use_amp)
    total_samples = sales.shape[0] * len(origins)
    batch_size = int(config["batch_size"])
    rng = np.random.default_rng(RANDOM_SEED)
    losses: list[float] = []
    print(
        f"    fitting {config_name}: {total_samples:,} windows, "
        f"{config['epochs']} epochs, device={device}",
        flush=True,
    )
    for epoch in range(int(config["epochs"])):
        permutation = rng.permutation(total_samples)
        epoch_loss = 0.0
        seen = 0
        model.train()
        for start in range(0, total_samples, batch_size):
            batch_indices = permutation[start : start + batch_size]
            batch = _batch_from_indices(
                batch_indices,
                origins,
                sales,
                stockout,
                context,
                static_codes,
                series_scale,
                device,
            )
            history_sales, history_stockout, history_context, future_context, static, target = batch
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(use_amp):
                prediction = model(
                    history_sales,
                    history_stockout,
                    history_context,
                    future_context,
                    static,
                )
                loss = torch.mean(torch.abs(prediction - target))
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            count = len(batch_indices)
            epoch_loss += float(loss.detach().cpu()) * count
            seen += count
            del (
                batch,
                history_sales,
                history_stockout,
                history_context,
                future_context,
                static,
                target,
                prediction,
                loss,
            )
        mean_loss = epoch_loss / seen
        losses.append(mean_loss)
        print(f"      epoch {epoch + 1}/{config['epochs']} normalized_MAE={mean_loss:.5f}", flush=True)
    return model, series_scale, losses


def forecast_nhits(
    model: NHiTSForecaster,
    history_sales: np.ndarray,
    history_stockout: np.ndarray,
    history_context: np.ndarray,
    future_context: np.ndarray,
    static_codes: np.ndarray,
    series_scale: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    n_series = history_sales.shape[0]
    predictions = np.empty((n_series, future_context.shape[1]), dtype=np.float32)
    model.eval()
    use_amp = device.type == "cuda"
    with torch.no_grad():
        for start in range(0, n_series, batch_size):
            end = min(start + batch_size, n_series)
            scale = series_scale[start:end, None]
            hs = torch.as_tensor(
                history_sales[start:end, -HISTORY_LENGTH:] / scale,
                dtype=torch.float32,
                device=device,
            )
            hso = torch.as_tensor(
                (history_stockout[start:end, -HISTORY_LENGTH:] > 0).astype(np.float32),
                dtype=torch.float32,
                device=device,
            )
            hc = torch.as_tensor(
                history_context[start:end, -HISTORY_LENGTH:],
                dtype=torch.float32,
                device=device,
            )
            fc = torch.as_tensor(
                future_context[start:end], dtype=torch.float32, device=device
            )
            static = torch.as_tensor(
                static_codes[start:end], dtype=torch.long, device=device
            )
            with autocast_context(use_amp):
                normalized = model(hs, hso, hc, fc, static)
            predictions[start:end] = (
                normalized.float().cpu().numpy() * scale
            ).astype(np.float32)
    return np.maximum(predictions, 0.0)


def forecast_fixed_origin_nhits(
    model: NHiTSForecaster,
    history_sales: np.ndarray,
    history_stockout: np.ndarray,
    history_context: np.ndarray,
    future_context: np.ndarray,
    static_codes: np.ndarray,
    series_scale: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    return forecast_nhits(
        model,
        history_sales,
        history_stockout,
        history_context,
        future_context,
        static_codes,
        series_scale,
        batch_size,
        device,
    )


def forecast_rolling_origin_nhits(
    model: NHiTSForecaster,
    history_sales: np.ndarray,
    history_stockout: np.ndarray,
    history_context: np.ndarray,
    future_sales: np.ndarray,
    future_stockout: np.ndarray,
    future_context: np.ndarray,
    static_codes: np.ndarray,
    series_scale: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    combined_sales = np.concatenate([history_sales, future_sales], axis=1)
    combined_stockout = np.concatenate([history_stockout, future_stockout], axis=1)
    combined_context = np.concatenate([history_context, future_context], axis=1)
    origin = history_sales.shape[1]
    predictions = np.empty_like(future_sales, dtype=np.float32)
    for step in range(future_sales.shape[1]):
        predictions[:, step : step + 1] = forecast_nhits(
            model,
            combined_sales[:, : origin + step],
            combined_stockout[:, : origin + step],
            combined_context[:, : origin + step],
            combined_context[:, origin + step : origin + step + 1],
            static_codes,
            series_scale,
            batch_size,
            device,
        )
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
                "mae", "rmse", "wape", "wpe", "underestimation_rate",
                "r2", "actual_sum", "prediction_sum",
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
    feature_set: str,
    config_name: str,
    protocol: str,
    split: str,
    actual: np.ndarray,
    prediction: np.ndarray,
    eligible: np.ndarray,
) -> list[dict[str, Any]]:
    return [
        {
            "model": "nhits",
            "feature_set": feature_set,
            "config": config_name,
            "protocol": protocol,
            "target": "observed_sales",
            "split": split,
            "horizon": index + 1,
        }
        | metrics(actual[:, index], prediction[:, index], eligible[:, index])
        for index in range(actual.shape[1])
    ]


def cumulative_metrics(
    actual: np.ndarray,
    prediction: np.ndarray,
    eligible: np.ndarray,
) -> dict[str, float | int]:
    fully_observed = eligible.all(axis=1)
    return metrics(actual.sum(axis=1), prediction.sum(axis=1), fully_observed)


def evaluate_predictions(
    feature_set: str,
    config_name: str,
    split: str,
    protocol_predictions: dict[str, np.ndarray],
    actual: np.ndarray,
    stockout: np.ndarray,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    eligible = stockout == 0
    overall, horizons, cumulative = [], [], []
    for protocol, prediction in protocol_predictions.items():
        row = {
            "model": "nhits",
            "feature_set": feature_set,
            "config": config_name,
            "protocol": protocol,
            "target": "observed_sales",
            "split": split,
        } | metrics(actual, prediction, eligible)
        overall.append(row)
        horizons.extend(
            horizon_metrics(
                feature_set, config_name, protocol, split, actual, prediction, eligible
            )
        )
        cumulative.append(
            {
                "model": "nhits",
                "feature_set": feature_set,
                "config": config_name,
                "protocol": protocol,
                "target": "observed_sales",
                "split": split,
            }
            | cumulative_metrics(actual, prediction, eligible)
        )
    return overall, horizons, cumulative


def print_rows(prefix: str, rows: list[dict[str, Any]]) -> None:
    for row in rows:
        print(
            f"{prefix} {row['protocol']:22s} WAPE={row['wape']:.4f} "
            f"WPE={row['wpe']:.4f} R2={row['r2']:.4f}",
            flush=True,
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="FreshRetailNet-50K N-HiTS experiment")
    parser.add_argument("--train-path", default=TRAIN_PATH)
    parser.add_argument("--eval-path", default=EVAL_PATH)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    args, _unknown = parser.parse_known_args()

    device = select_device()
    seed_everything(RANDOM_SEED)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("high")
    print(f"PyTorch device: {device}", flush=True)
    if device.type != "cuda":
        print("WARNING: N-HiTS full-data training is much faster with a Kaggle GPU.", flush=True)

    started = time.perf_counter()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = discover_parquet("train.parquet", args.train_path)
    eval_path = discover_parquet("eval.parquet", args.eval_path)
    train, eval_frame, metadata = load_data(train_path, eval_path)
    origin_date = pd.Timestamp(TRAIN_START)
    train = add_calendar_features(train, origin_date)
    eval_frame = add_calendar_features(eval_frame, origin_date)
    train_dates: pd.DatetimeIndex = metadata["train_dates"]
    eval_dates: pd.DatetimeIndex = metadata["eval_dates"]
    series: pd.DataFrame = metadata["series"]
    static_codes: np.ndarray = metadata["static_codes"]
    cardinalities: list[int] = metadata["cardinalities"]
    train_sales, train_stockout = panel_arrays(train, len(train_dates))
    eval_sales, eval_stockout = panel_arrays(eval_frame, len(eval_dates))
    validation_index = int(np.flatnonzero(train_dates == pd.Timestamp(VALIDATION_START))[0])
    validation_actual = train_sales[:, validation_index:]
    validation_stockout = train_stockout[:, validation_index:]

    validation_rows: list[dict[str, Any]] = []
    validation_horizon_rows: list[dict[str, Any]] = []
    validation_cumulative_rows: list[dict[str, Any]] = []
    training_histories: dict[str, dict[str, list[float]]] = {}

    for feature_set in FEATURE_SETS:
        stats = fit_context_stats(train, VALIDATION_TRAIN_END)
        context = create_context_panel(train, len(train_dates), feature_set, stats)
        training_histories[feature_set] = {}
        for config_name, config in NHITS_CONFIGS.items():
            model, series_scale, losses = train_nhits(
                train_sales,
                train_stockout,
                context,
                static_codes,
                cardinalities,
                validation_index,
                config_name,
                config,
                device,
            )
            training_histories[feature_set][config_name] = losses
            fixed = forecast_fixed_origin_nhits(
                model,
                train_sales[:, :validation_index],
                train_stockout[:, :validation_index],
                context[:, :validation_index],
                context[:, validation_index:],
                static_codes,
                series_scale,
                config["batch_size"],
                device,
            )
            rolling = forecast_rolling_origin_nhits(
                model,
                train_sales[:, :validation_index],
                train_stockout[:, :validation_index],
                context[:, :validation_index],
                validation_actual,
                validation_stockout,
                context[:, validation_index:],
                static_codes,
                series_scale,
                config["batch_size"],
                device,
            )
            overall, horizons, cumulative = evaluate_predictions(
                feature_set,
                config_name,
                "validation",
                {"fixed_origin_7_day": fixed, "rolling_origin_1_day": rolling},
                validation_actual,
                validation_stockout,
            )
            validation_rows.extend(overall)
            validation_horizon_rows.extend(horizons)
            validation_cumulative_rows.extend(cumulative)
            print_rows(f"  {feature_set}/{config_name}", overall)
            del model, fixed, rolling, series_scale
            if device.type == "cuda":
                torch.cuda.empty_cache()
            gc.collect()
        del context
        gc.collect()

    validation_results = pd.DataFrame(validation_rows)
    validation_results["absolute_wpe"] = validation_results["wpe"].abs()
    validation_results = validation_results.sort_values(
        ["protocol", "feature_set", "wape", "absolute_wpe", "mae"], kind="stable"
    ).reset_index(drop=True)
    validation_results["feature_set_rank"] = validation_results.groupby(
        ["protocol", "feature_set"], sort=False
    ).cumcount() + 1
    validation_results.to_csv(output_dir / "nhits_validation_full_data.csv", index=False)
    pd.DataFrame(validation_horizon_rows).to_csv(
        output_dir / "nhits_validation_by_horizon.csv", index=False
    )
    pd.DataFrame(validation_cumulative_rows).to_csv(
        output_dir / "nhits_validation_cumulative_7_day.csv", index=False
    )
    primary = validation_results.loc[
        validation_results["protocol"] == "fixed_origin_7_day"
    ]
    best = primary.loc[primary["feature_set_rank"] == 1]
    best_configs = {
        row.feature_set: row.config for row in best.itertuples(index=False)
    }
    print("\nLOCKED FIXED-ORIGIN WINNERS:", flush=True)
    print(best[["feature_set", "config", "wape", "wpe", "r2"]].to_string(index=False), flush=True)

    final_rows: list[dict[str, Any]] = []
    final_horizon_rows: list[dict[str, Any]] = []
    final_cumulative_rows: list[dict[str, Any]] = []
    final_predictions: dict[tuple[str, str], np.ndarray] = {}
    final_training_histories: dict[str, list[float]] = {}
    for feature_set in FEATURE_SETS:
        stats = fit_context_stats(train, FINAL_TRAIN_END)
        train_context = create_context_panel(train, len(train_dates), feature_set, stats)
        eval_context = create_context_panel(eval_frame, len(eval_dates), feature_set, stats)
        config_name = best_configs[feature_set]
        config = NHITS_CONFIGS[config_name]
        print(f"\nFINAL REFIT: {feature_set}/{config_name}", flush=True)
        model, series_scale, losses = train_nhits(
            train_sales,
            train_stockout,
            train_context,
            static_codes,
            cardinalities,
            len(train_dates),
            config_name,
            config,
            device,
        )
        final_training_histories[feature_set] = losses
        fixed = forecast_fixed_origin_nhits(
            model,
            train_sales,
            train_stockout,
            train_context,
            eval_context,
            static_codes,
            series_scale,
            config["batch_size"],
            device,
        )
        rolling = forecast_rolling_origin_nhits(
            model,
            train_sales,
            train_stockout,
            train_context,
            eval_sales,
            eval_stockout,
            eval_context,
            static_codes,
            series_scale,
            config["batch_size"],
            device,
        )
        final_predictions[(feature_set, "fixed")] = fixed
        final_predictions[(feature_set, "rolling")] = rolling
        overall, horizons, cumulative = evaluate_predictions(
            feature_set,
            config_name,
            "test",
            {"fixed_origin_7_day": fixed, "rolling_origin_1_day": rolling},
            eval_sales,
            eval_stockout,
        )
        final_rows.extend(overall)
        final_horizon_rows.extend(horizons)
        final_cumulative_rows.extend(cumulative)
        print_rows("  final", overall)
        del model, series_scale, train_context, eval_context
        if device.type == "cuda":
            torch.cuda.empty_cache()
        gc.collect()

    final_results = pd.DataFrame(final_rows).sort_values(
        ["protocol", "wape"], kind="stable"
    ).reset_index(drop=True)
    final_results.to_csv(output_dir / "nhits_final_test.csv", index=False)
    final_horizon_frame = pd.DataFrame(final_horizon_rows)
    final_horizon_frame.to_csv(output_dir / "nhits_final_test_by_horizon.csv", index=False)
    pd.DataFrame(final_cumulative_rows).to_csv(
        output_dir / "nhits_final_test_cumulative_7_day.csv", index=False
    )
    error_growth = final_horizon_frame.pivot_table(
        index=["model", "feature_set", "config", "protocol"],
        columns="horizon",
        values="wape",
    ).reset_index()
    error_growth["wape_growth_h7_minus_h1"] = error_growth[7] - error_growth[1]
    error_growth = error_growth.rename(columns={1: "wape_h1", 7: "wape_h7"})
    error_growth.to_csv(output_dir / "nhits_final_test_error_growth.csv", index=False)

    prediction_frame = pd.DataFrame(
        {
            "store_id": np.repeat(series["store_id"].to_numpy(dtype=np.int32), len(eval_dates)),
            "product_id": np.repeat(series["product_id"].to_numpy(dtype=np.int32), len(eval_dates)),
            "dt": np.tile(eval_dates.to_numpy(), len(series)),
            "actual_sale_amount": eval_sales.reshape(-1),
            "stockout_hours": eval_stockout.reshape(-1),
            "eligible_non_stockout": (eval_stockout == 0).reshape(-1),
            "prediction_operational_fixed": final_predictions[("operational", "fixed")].reshape(-1),
            "prediction_operational_rolling": final_predictions[("operational", "rolling")].reshape(-1),
            "prediction_weather_fixed": final_predictions[("weather_enhanced", "fixed")].reshape(-1),
            "prediction_weather_rolling": final_predictions[("weather_enhanced", "rolling")].reshape(-1),
        }
    )
    prediction_frame.to_parquet(output_dir / "nhits_best_model_predictions.parquet", index=False)

    manifest = {
        "experiment": "nhits_observed_sales_operational_vs_weather_enhanced",
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
        "implementation": "custom self-contained PyTorch N-HiTS-style implementation",
        "history_length": HISTORY_LENGTH,
        "forecast_horizon": FORECAST_HORIZON,
        "architecture": (
            "N-HiTS-style hierarchical interpolation network with multi-rate pooling, "
            "residual backcast blocks, and direct multi-horizon point forecasts"
        ),
        "feature_sets": FEATURE_SETS,
        "weather_setting": "known-ahead/oracle unless genuine forecasts are supplied",
        "configs": NHITS_CONFIGS,
        "best_config_per_feature_set": best_configs,
        "primary_protocol": "fixed_origin_7_day",
        "secondary_protocol": "rolling_origin_1_day",
        "selection_rule": "lowest fixed-origin validation WAPE per feature set",
        "rolling_result_note": (
            "All rolling validation candidates are reported but not used for selection. "
            "Final rolling uses fixed-origin-selected configurations without retuning."
        ),
        "anti_leakage": {
            "fixed_future_sales": (
                "direct multi-horizon output; no realized future sales are supplied"
            ),
            "rolling_update": "completed prior-day actual sales and stockout",
            "future_context": "scheduled operational context and oracle weather only",
        },
        "series_normalization": "mean observed sales before the forecast cutoff, minimum 0.10",
        "context_normalization": "statistics fitted only on the corresponding training period",
        "evaluation_eligibility": "stock_hour6_22_cnt == 0",
        "cumulative_eligibility": "no stockout on all seven forecast days",
        "training_loss_histories": training_histories,
        "final_training_loss_histories": final_training_histories,
        "validation_results": validation_results.to_dict(orient="records"),
        "final_test_results": final_results.to_dict(orient="records"),
        "runtime_seconds": time.perf_counter() - started,
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "device": str(device),
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
