#!/usr/bin/env python
"""Compare the author's MED implementation with four forecasting baselines.

Baselines
---------
1. Naive: last-observation persistence.
2. ARIMA: univariate ARIMA; order is selected on a chronological validation tail.
3. U-MIDAS (OLS): unrestricted linear regression on flattened target- and
   auxiliary-frequency lag windows created by the author's dataset builder.
4. DeepAR (Gaussian LSTM, one-step): autoregressive LSTM that estimates a
   Gaussian distribution for the next target observation.

MED is loaded from the author's repository. This script intentionally keeps
one-step forecasting and the source repository's train/test split.

Run from the root of the author's repository:
    python compare_models.py --configs configs/2_elec/med.yaml
"""
from __future__ import annotations

import argparse
import copy
import random
import time
import warnings
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_absolute_percentage_error,
    mean_squared_error,
)
from sklearn.preprocessing import MinMaxScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from args_parser import parse_args
from data_processor import DataProcessor
from models.med_model import MEDModel
from utils import CustomDatasetD, create_dataset_two_freq, symmetric_mean_absolute_percentage_error


# ---------------------------------------------------------------------------
# Reproducibility and metrics
# ---------------------------------------------------------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    y_true = np.asarray(y_true, dtype=float).reshape(-1)
    y_pred = np.asarray(y_pred, dtype=float).reshape(-1)
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred have different lengths.")
    mse = float(mean_squared_error(y_true, y_pred))
    return {
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MAPE": float(mean_absolute_percentage_error(y_true, y_pred)),
        "MSE": mse,
        "RMSE": float(np.sqrt(mse)),
        "sMAPE": float(symmetric_mean_absolute_percentage_error(y_true, y_pred)),
    }


def inverse_target_scale(values: np.ndarray, scaler: MinMaxScaler, target_idx: int) -> np.ndarray:
    values = np.asarray(values, dtype=float).reshape(-1)
    dummy = np.zeros((len(values), scaler.n_features_in_), dtype=float)
    dummy[:, target_idx] = values
    return scaler.inverse_transform(dummy)[:, target_idx]


def check_input_data(train_data, test_data) -> None:
    for label, train_frame, test_frame in (
        ("Target", train_data[1], test_data[1]),
        ("Auxiliary", train_data[0], test_data[0]),
    ):
        frame = pd.concat([train_frame, test_frame])
        if frame.isna().any().any():
            count = int(frame.isna().sum().sum())
            raise ValueError(
                "{} data contain {} missing value(s). Clean/impute them before "
                "running the comparison; the original DataProcessor does not "
                "perform missing-value treatment.".format(label, count)
            )
        if not frame.index.is_monotonic_increasing:
            raise ValueError("{} timestamps are not sorted increasingly.".format(label))
        if frame.index.has_duplicates:
            raise ValueError("{} timestamps contain duplicates.".format(label))
        deltas = frame.index.to_series().diff().dropna()
        if len(deltas) and deltas.nunique() > 1:
            warnings.warn(
                "{} timestamps have irregular intervals. The author's window "
                "builder works by row position, not calendar-aware alignment.".format(label)
            )


# ---------------------------------------------------------------------------
# Shared mixed-frequency windows from the author's repository
# ---------------------------------------------------------------------------
def build_windows(
    aux_scaled: np.ndarray,
    target_scaled: np.ndarray,
    args,
    target_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    X_t, X_a, y_t, y_a = create_dataset_two_freq(
        [aux_scaled, target_scaled],
        freq_r=args.t_a1_freq_ratio,
        input_steps=args.t_enc_time_steps,
        t_output_steps=args.t_output_steps,
        a_output_steps=args.a_output_steps,
        target_idx=target_idx,
    )
    X_t = np.asarray(X_t, dtype=np.float32)
    X_a = np.asarray(X_a, dtype=np.float32)
    y_t = np.asarray(y_t, dtype=np.float32)
    y_a = np.asarray(y_a, dtype=np.float32)
    if y_t.ndim == 1:
        y_t = y_t.reshape(-1, 1)
    if y_a.ndim == 1:
        y_a = y_a.reshape(-1, 1)
    if X_t.ndim != 3 or X_a.ndim != 3:
        raise ValueError("The author's dataset builder did not return 3-D input arrays.")
    if y_t.shape[1] != 1:
        raise ValueError("This script currently supports one-step target forecasting only; set t_output_steps: 1.")
    return X_t, X_a, y_t, y_a


def check_window_lengths(X_t: np.ndarray, X_a: np.ndarray, args) -> None:
    if X_t.shape[1] != int(args.t_enc_time_steps):
        raise ValueError(
            "Target input window length is {}, expected {}.".format(
                X_t.shape[1], args.t_enc_time_steps
            )
        )
    if X_a.shape[1] != int(args.a_enc_time_steps):
        raise ValueError(
            "Auxiliary input window length is {}, but a_enc_time_steps={} in YAML. "
            "Check t_a1_freq_ratio and encoder lengths.".format(
                X_a.shape[1], args.a_enc_time_steps
            )
        )

def make_chronological_validation_windows(
    train_data,
    args,
    val_ratio: float = 0.2,
):
    """
    Create train/validation windows using a chronological raw-data split.

    Scalers are fitted only on the first part of the training period.
    The validation windows retain the previous encoder context.
    """
    df_a, df_t = train_data

    input_steps = int(args.t_enc_time_steps)
    split_idx = int(len(df_t) * (1.0 - val_ratio))

    if split_idx <= input_steps:
        raise ValueError("Not enough data for chronological validation.")

    # 1. Split target data by time
    prefix_t = df_t.iloc[:split_idx]
    val_t = df_t.iloc[split_idx:]

    # 2. Auxiliary data before validation starts
    val_start = val_t.index[0]
    prefix_a = df_a.loc[df_a.index < val_start]

    # 3. Keep previous target observations as validation encoder context
    context_t = pd.concat([
        prefix_t.tail(input_steps),
        val_t,
    ])

    # Auxiliary data beginning at the same timestamp as context_t
    context_a = df_a.loc[context_t.index[0]:]

    # 4. Check frequency ratio
    ratio = 1.0 / float(args.t_a1_freq_ratio)
    step = round(ratio)

    if not np.isclose(ratio, step):
        raise ValueError("Frequency ratio must be an integer.")

    if len(prefix_a) != len(prefix_t) * step:
        raise ValueError("Training prefix auxiliary/target lengths do not match.")

    if len(context_a) != len(context_t) * step:
        raise ValueError("Validation auxiliary/target lengths do not match.")

    # 5. Fit scalers ONLY on the training prefix
    scaler_a = MinMaxScaler()
    scaler_t = MinMaxScaler()

    prefix_a_scaled = scaler_a.fit_transform(prefix_a)
    prefix_t_scaled = scaler_t.fit_transform(prefix_t)

    # Validation data are transformed using the already-fitted scalers
    context_a_scaled = scaler_a.transform(context_a)
    context_t_scaled = scaler_t.transform(context_t)

    target_idx = df_t.columns.get_loc(args.key_var)

    # 6. Create training windows
    X_t_train, X_a_train, y_t_train, y_a_train = build_windows(
        prefix_a_scaled,
        prefix_t_scaled,
        args,
        target_idx,
    )

    # 7. Create validation windows
    X_t_val, X_a_val, y_t_val, y_a_val = build_windows(
        context_a_scaled,
        context_t_scaled,
        args,
        target_idx,
    )

    check_window_lengths(X_t_train, X_a_train, args)
    check_window_lengths(X_t_val, X_a_val, args)

    if len(X_t_val) != len(val_t):
        raise ValueError(
            "Validation windows do not align with validation observations."
        )

    return {
        "X_t_train": X_t_train,
        "X_a_train": X_a_train,
        "y_t_train": y_t_train,
        "y_a_train": y_a_train,
        "X_t_val": X_t_val,
        "X_a_val": X_a_val,
        "y_t_val": y_t_val,
        "y_a_val": y_a_val,
        "scaler_a": scaler_a,
        "scaler_t": scaler_t,
    }

def flatten_mixed_windows(X_t: np.ndarray, X_a: np.ndarray) -> np.ndarray:
    """Create unrestricted lag regressors: all target and auxiliary lags."""
    return np.concatenate(
        [X_t.reshape(len(X_t), -1), X_a.reshape(len(X_a), -1)], axis=1
    ).astype(np.float64)

def check_mixed_frequency_alignment(
    train_data,
    test_data,
    args
):
    # Số mẫu auxiliary tương ứng với một mẫu target
    ratio = 1.0 / float(args.t_a1_freq_ratio)
    step = round(ratio)

    if not np.isclose(ratio, step):
        raise ValueError(
            "Frequency ratio does not correspond to an integer step."
        )

    for name, (data_a, data_t) in [
        ("Train", train_data),
        ("Test", test_data)
    ]:
        # Kiểm tra số lượng mẫu
        if len(data_a) != len(data_t) * step:
            raise ValueError(
                f"{name}: Auxiliary/Target length mismatch. "
                f"Expected {len(data_t) * step}, "
                f"got {len(data_a)}."
            )

        # Lấy timestamp auxiliary cách nhau 6 giờ
        aligned_aux_index = data_a.index[::step]

        # So sánh với timestamp target
        if not aligned_aux_index.equals(data_t.index):
            raise ValueError(
                f"{name}: Auxiliary and Target timestamps "
                "are not aligned."
            )

    print("Train/Test frequency alignment: PASSED")
# ---------------------------------------------------------------------------
# 1. Naive baseline
# ---------------------------------------------------------------------------
def naive_forecast(train_series: np.ndarray, test_series: np.ndarray) -> np.ndarray:
    """One-step rolling persistence: forecast y[t] with observed y[t-1]."""
    train_series = np.asarray(train_series, dtype=float).reshape(-1)
    test_series = np.asarray(test_series, dtype=float).reshape(-1)
    if len(train_series) == 0:
        raise ValueError("Naive model requires at least one training observation.")
    return np.concatenate([train_series[-1:], test_series[:-1]])


# ---------------------------------------------------------------------------
# 2. ARIMA baseline
# ---------------------------------------------------------------------------
ARIMA_CANDIDATES: Sequence[Tuple[int, int, int]] = (
    (1, 0, 0),
    (2, 0, 0),
    (1, 0, 1),
    (2, 0, 1),
    (1, 1, 0),
    (1, 1, 1),
)


def _arima_rolling_forecast(
    train_values: np.ndarray,
    future_values: np.ndarray,
    order: Tuple[int, int, int],
) -> Tuple[np.ndarray, int]:
    """Fit ARIMA and produce rolling one-step forecasts using prior observations."""
    try:
        from statsmodels.tsa.arima.model import ARIMA
    except ImportError as exc:
        raise ImportError(
            "ARIMA requires statsmodels. Install it with: pip install statsmodels"
        ) from exc

    history = np.asarray(train_values, dtype=float).reshape(-1)
    future = np.asarray(future_values, dtype=float).reshape(-1)
    if len(history) < max(12, order[0] + order[2] + order[1] + 3):
        raise ValueError("Training sequence is too short for ARIMA order {}.".format(order))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fitted = ARIMA(
            history,
            order=order,
            enforce_stationarity=False,
            enforce_invertibility=False,
        ).fit()
        forecasts: List[float] = []
        for observed in future:
            forecast = fitted.forecast(steps=1)
            forecasts.append(float(np.asarray(forecast).reshape(-1)[0]))
            # Update state with the just-observed value; do not refit parameters.
            fitted = fitted.append(endog=[float(observed)], refit=False)
    return np.asarray(forecasts, dtype=float), int(len(fitted.params))


def run_arima(
    train_series: np.ndarray,
    test_series: np.ndarray,
    validation_count: int,
) -> Tuple[np.ndarray, Tuple[int, int, int], float, int]:
    """Select ARIMA order on the chronological tail, then refit on all train data."""
    started = time.perf_counter()
    train_series = np.asarray(train_series, dtype=float).reshape(-1)
    test_series = np.asarray(test_series, dtype=float).reshape(-1)
    validation_count = min(max(1, int(validation_count)), max(1, len(train_series) // 3))
    split = len(train_series) - validation_count
    if split < 15:
        raise ValueError("Not enough observations to create an ARIMA train/validation split.")

    fit_part = train_series[:split]
    validation_part = train_series[split:]
    best_order = None
    best_rmse = float("inf")
    candidate_errors = []
    for order in ARIMA_CANDIDATES:
        try:
            val_pred, _val_param_count = _arima_rolling_forecast(fit_part, validation_part, order)
            rmse = float(np.sqrt(mean_squared_error(validation_part, val_pred)))
            candidate_errors.append((order, rmse))
            if np.isfinite(rmse) and rmse < best_rmse:
                best_rmse = rmse
                best_order = order
        except Exception as exc:  # a failed candidate should not stop other orders
            warnings.warn("Skip ARIMA{}: {}".format(order, exc))

    if best_order is None:
        raise RuntimeError("All ARIMA candidates failed. Check statsmodels and the target series.")

    # Refit the chosen specification on the complete training period.
    prediction, parameter_count = _arima_rolling_forecast(train_series, test_series, best_order)
    elapsed = time.perf_counter() - started
    return prediction, best_order, elapsed, parameter_count


# ---------------------------------------------------------------------------
# 3. U-MIDAS (unrestricted linear regression)
# ---------------------------------------------------------------------------
def run_umidas(
    X_t_train: np.ndarray,
    X_a_train: np.ndarray,
    y_train: np.ndarray,
    X_t_test: np.ndarray,
    X_a_test: np.ndarray,
) -> Tuple[np.ndarray, float, int, int]:
    """OLS over unrestricted lags of target and auxiliary frequency variables.

    The target and auxiliary windows are flattened separately, then concatenated.
    Every lag receives its own coefficient; no MIDAS polynomial restriction is
    imposed. This is the standard unrestricted-lag interpretation of U-MIDAS.
    """
    started = time.perf_counter()
    train_features = flatten_mixed_windows(X_t_train, X_a_train)
    test_features = flatten_mixed_windows(X_t_test, X_a_test)
    target = np.asarray(y_train, dtype=float).reshape(-1)
    model = LinearRegression()
    model.fit(train_features, target)
    prediction = model.predict(test_features)
    elapsed = time.perf_counter() - started
    parameter_count = int(model.coef_.size + 1)  # coefficients + intercept
    return prediction, elapsed, parameter_count, train_features.shape[1]


# ---------------------------------------------------------------------------
# 4. DeepAR (Gaussian autoregressive LSTM, one-step adaptation)
# ---------------------------------------------------------------------------
class DeepARGaussian(nn.Module):
    """LSTM that parameterizes a Gaussian distribution for the next value."""
    def __init__(self, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.distribution_head = nn.Linear(hidden_size, 2)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        _, (hidden, _) = self.lstm(x)
        raw = self.distribution_head(self.dropout(hidden[-1]))
        mean = raw[:, 0:1]
        sigma = torch.nn.functional.softplus(raw[:, 1:2]) + 1e-4
        return mean, sigma


def gaussian_nll(mean: torch.Tensor, sigma: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    standardized = (target - mean) / sigma
    return (torch.log(sigma) + 0.5 * standardized.pow(2)).mean()


def select_deepar_epoch(
    model: DeepARGaussian,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    args,
    device: torch.device,
) -> Tuple[int, float]:
    """Select the best training epoch using chronological validation."""

    train_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
    )

    val_loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_val, dtype=torch.float32),
            torch.as_tensor(y_val, dtype=torch.float32).reshape(-1, 1),
        ),
        batch_size=int(args.batch_size),
        shuffle=False,
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
    )

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    patience = 10

    started = time.perf_counter()

    for epoch in range(1, int(args.epoch) + 1):
        model.train()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)

            mean, sigma = model(batch_x)
            loss = gaussian_nll(mean, sigma, batch_y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Validation
        model.eval()

        val_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.to(device)

                mean, sigma = model(batch_x)
                val_loss = gaussian_nll(mean, sigma, batch_y)

                val_sum += float(val_loss.item()) * len(batch_y)
                val_count += len(batch_y)

        val_loss_value = val_sum / max(1, val_count)

        if val_loss_value < best_val:
            best_val = val_loss_value
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

            if bad_epochs >= patience:
                break

    elapsed = time.perf_counter() - started

    if best_epoch == 0:
        raise RuntimeError("DeepAR failed to select a valid training epoch.")

    return best_epoch, elapsed


def fit_deepar_full_train(
    model: DeepARGaussian,
    X_train: np.ndarray,
    y_train: np.ndarray,
    args,
    device: torch.device,
    epochs: int,
) -> Tuple[DeepARGaussian, float]:
    """Train a fresh DeepAR model on all training windows."""

    loader = DataLoader(
        TensorDataset(
            torch.as_tensor(X_train, dtype=torch.float32),
            torch.as_tensor(y_train, dtype=torch.float32).reshape(-1, 1),
        ),
        batch_size=int(args.batch_size),
        shuffle=True,
    )

    model = model.to(device)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(args.learning_rate),
    )

    started = time.perf_counter()

    for _epoch in range(epochs):
        model.train()

        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            optimizer.zero_grad(set_to_none=True)

            mean, sigma = model(batch_x)
            loss = gaussian_nll(mean, sigma, batch_y)

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    elapsed = time.perf_counter() - started

    return model, elapsed

def predict_deepar(
    model: DeepARGaussian,
    X: np.ndarray,
    args,
    device: torch.device,
) -> np.ndarray:
    loader = DataLoader(
        torch.as_tensor(X, dtype=torch.float32),
        batch_size=int(args.batch_size),
        shuffle=False,
    )
    model.eval()
    predictions = []
    with torch.no_grad():
        for batch_x in loader:
            mean, _sigma = model(batch_x.to(device))
            predictions.append(mean.detach().cpu().numpy().reshape(-1))
    return np.concatenate(predictions) if predictions else np.asarray([], dtype=float)


# ---------------------------------------------------------------------------
# MED: retain the author's model, use a chronological validation tail
# ---------------------------------------------------------------------------
def fit_med_same_validation(
    med_wrapper: MEDModel,
    train_data: Tuple[pd.DataFrame, pd.DataFrame],
    args,
    device: torch.device,
) -> Tuple[float, int]:
    """
    Select the best epoch using chronological validation.

    Scalers are fitted only on the training prefix.
    The returned model is used only for epoch selection.
    """

    df_a, df_t = train_data

    # Create chronological train/validation windows
    windows = make_chronological_validation_windows(
        train_data,
        args,
        val_ratio=0.2,
    )

    # Use scalers fitted on the training prefix
    med_wrapper.scaler_a = windows["scaler_a"]
    med_wrapper.scaler_t = windows["scaler_t"]

    med_wrapper.key_idx = df_t.columns.get_loc(med_wrapper.key_var)

    X_t_train = windows["X_t_train"]
    X_a_train = windows["X_a_train"]
    y_t_train = windows["y_t_train"]
    y_a_train = windows["y_a_train"]

    X_t_val = windows["X_t_val"]
    X_a_val = windows["X_a_val"]
    y_t_val = windows["y_t_val"]
    y_a_val = windows["y_a_val"]

    def make_loader(
        X_t,
        X_a,
        y_t,
        y_a,
        shuffle: bool,
    ) -> DataLoader:

        dataset = CustomDatasetD(
            X_t,
            X_a,
            X_t[:, -med_wrapper.t_dec_steps:],
            X_a[:, -med_wrapper.a_dec_steps:],
            y_t,
            y_a,
        )

        return DataLoader(
            dataset,
            batch_size=med_wrapper.batch_size,
            shuffle=shuffle,
        )

    train_loader = make_loader(
        X_t_train,
        X_a_train,
        y_t_train,
        y_a_train,
        shuffle=True,
    )

    val_loader = make_loader(
        X_t_val,
        X_a_val,
        y_t_val,
        y_a_val,
        shuffle=False,
    )

    model = med_wrapper.model.to(device)
    optimizer = med_wrapper.optimizer
    criterion = med_wrapper.criterion
    alpha = med_wrapper.loss_alpha

    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0
    patience = 10

    started = time.perf_counter()

    for epoch in range(1, int(args.epoch) + 1):

        # Training
        model.train()

        for batch in train_loader:
            t_src = batch["X_enc_t"].to(device)
            a_src = batch["X_enc_a"].to(device)

            t_dec = batch["X_dec_t"].to(device)
            a_dec = batch["X_dec_a"].to(device)

            t_y = batch["y_train_t"].to(device)
            a_y = batch["y_train_a"].to(device)

            optimizer.zero_grad(set_to_none=True)

            t_out, a_out = model(t_src, a_src, t_dec, a_dec)

            t_out = t_out[:, -1]
            a_out = a_out[:, -1]

            loss_t = criterion(t_out, t_y.view_as(t_out))
            loss_a = criterion(a_out, a_y.view_as(a_out))

            loss = loss_t + alpha * loss_a

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                med_wrapper.grad_clip,
            )

            optimizer.step()

        # Validation: select checkpoint based on target forecasting loss
        model.eval()

        val_sum = 0.0
        val_count = 0

        with torch.no_grad():
            for batch in val_loader:
                t_src = batch["X_enc_t"].to(device)
                a_src = batch["X_enc_a"].to(device)

                t_dec = batch["X_dec_t"].to(device)
                a_dec = batch["X_dec_a"].to(device)

                t_y = batch["y_train_t"].to(device)

                t_out, _ = model(t_src, a_src, t_dec, a_dec)
                t_out = t_out[:, -1]

                val_loss = criterion(t_out, t_y.view_as(t_out))

                val_sum += float(val_loss.item()) * len(t_y)
                val_count += len(t_y)

        val_loss_value = val_sum / max(1, val_count)

        if val_loss_value < best_val:
            best_val = val_loss_value
            best_epoch = epoch
            bad_epochs = 0
        else:
            bad_epochs += 1

            if bad_epochs >= patience:
                break

    elapsed = time.perf_counter() - started

    if best_epoch == 0:
        raise RuntimeError("MED failed to select a valid training epoch.")

    return elapsed, best_epoch

def fit_med_full_train(
    med_wrapper: MEDModel,
    train_data: Tuple[pd.DataFrame, pd.DataFrame],
    args,
    device: torch.device,
    epochs: int,
) -> float:
    """Train a fresh MED model on all training windows."""

    df_a, df_t = train_data

    med_wrapper.key_idx = df_t.columns.get_loc(med_wrapper.key_var)

    # Fit scalers on the COMPLETE training period
    scaled_a = med_wrapper.scaler_a.fit_transform(df_a)
    scaled_t = med_wrapper.scaler_t.fit_transform(df_t)

    # Create all training windows
    X_t, X_a, y_t, y_a = create_dataset_two_freq(
        [scaled_a, scaled_t],
        freq_r=med_wrapper.freq_ratio,
        input_steps=med_wrapper.t_enc_steps,
        t_output_steps=med_wrapper.t_out_steps,
        a_output_steps=med_wrapper.a_out_steps,
        target_idx=med_wrapper.key_idx,
    )

    X_t = np.asarray(X_t, dtype=np.float32)
    X_a = np.asarray(X_a, dtype=np.float32)
    y_t = np.asarray(y_t, dtype=np.float32)
    y_a = np.asarray(y_a, dtype=np.float32)

    dataset = CustomDatasetD(
        X_t,
        X_a,
        X_t[:, -med_wrapper.t_dec_steps:],
        X_a[:, -med_wrapper.a_dec_steps:],
        y_t,
        y_a,
    )

    loader = DataLoader(
        dataset,
        batch_size=med_wrapper.batch_size,
        shuffle=True,
    )

    model = med_wrapper.model.to(device)
    optimizer = med_wrapper.optimizer
    criterion = med_wrapper.criterion
    alpha = med_wrapper.loss_alpha

    started = time.perf_counter()

    # Train for the epoch count selected by validation
    for _epoch in range(epochs):

        model.train()

        for batch in loader:
            t_src = batch["X_enc_t"].to(device)
            a_src = batch["X_enc_a"].to(device)

            t_dec = batch["X_dec_t"].to(device)
            a_dec = batch["X_dec_a"].to(device)

            t_y = batch["y_train_t"].to(device)
            a_y = batch["y_train_a"].to(device)

            optimizer.zero_grad(set_to_none=True)

            t_out, a_out = model(t_src, a_src, t_dec, a_dec)

            t_out = t_out[:, -1]
            a_out = a_out[:, -1]

            loss_t = criterion(t_out, t_y.view_as(t_out))
            loss_a = criterion(a_out, a_y.view_as(a_out))

            loss = loss_t + alpha * loss_a

            loss.backward()

            nn.utils.clip_grad_norm_(
                model.parameters(),
                med_wrapper.grad_clip,
            )

            optimizer.step()

    return time.perf_counter() - started

def count_parameters(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters() if param.requires_grad))


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------
def run_experiment(args, output_dir: Path, skip_med: bool = False) -> None:
    set_seed(int(args.seed))
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    train_data, test_data = DataProcessor(args).preprocess()
    check_mixed_frequency_alignment(
    train_data,
    test_data,
    args
    )
    train_a, train_t = train_data
    test_a, test_t = test_data
    check_input_data(train_data, test_data)

    target_idx = train_t.columns.get_loc(args.key_var)
    train_target_raw = train_t[args.key_var].to_numpy(dtype=float)
    actual = test_t[args.key_var].to_numpy(dtype=float)
    timestamps = test_t.index

    # Fit scalers on training data only; these are also used by the MED wrapper.
    scaler_a = MinMaxScaler()
    scaler_t = MinMaxScaler()
    aux_train_scaled = scaler_a.fit_transform(train_a)
    target_train_scaled = scaler_t.fit_transform(train_t)

    X_t_train, X_a_train, y_train, _y_a_train = build_windows(
        aux_train_scaled, target_train_scaled, args, target_idx
    )
    check_window_lengths(X_t_train, X_a_train, args)
    
    # Preserve the last source encoder context when creating test windows.
    aux_context = pd.concat([train_a.tail(int(args.a_enc_time_steps)), test_a])
    target_context = pd.concat([train_t.tail(int(args.t_enc_time_steps)), test_t])
    X_t_context = scaler_t.transform(target_context)
    X_a_context = scaler_a.transform(aux_context)
    X_t_test, X_a_test, _y_test_scaled, _y_a_test = build_windows(
        X_a_context, X_t_context, args, target_idx
    )
    check_window_lengths(X_t_test, X_a_test, args)
    # Save the exact scaled windows used in the experiment
    np.savez_compressed(
        output_dir / "med_shap_data.npz",
        X_t_train=X_t_train,
        X_a_train=X_a_train,
        X_t_test=X_t_test,
        X_a_test=X_a_test,
    )

    print("Saved MED SHAP input windows.")    
    if len(X_t_test) != len(actual):
        raise ValueError(
            "Test windows ({}) do not align with test observations ({}). "
            "Check the frequency ratio and encoder windows.".format(len(X_t_test), len(actual))
        )

    # Extract target history for DeepAR
    deepar_train_x = X_t_train[:, :, target_idx:target_idx + 1]
    deepar_test_x = X_t_test[:, :, target_idx:target_idx + 1]

    records = []
    predictions: Dict[str, np.ndarray] = {}

    # 1) Naive
    print("\n[1/5] Naive")
    naive_pred = naive_forecast(train_target_raw, actual)
    predictions["naive"] = naive_pred
    records.append({
        "Model": "Naive",
        **regression_metrics(actual, naive_pred),
        "Parameters": 0,
        "TrainTimeSec": 0.0,
        "Details": "Last observed target value",
    })

    # 2) ARIMA (target-only univariate statistical baseline)
    print("\n[2/5] ARIMA")
    validation_count = max(1, int(round(len(y_train) * 0.2)))
    arima_pred, arima_order, arima_time, arima_params = run_arima(
        train_target_raw, actual, validation_count
    )
    predictions["arima"] = arima_pred
    records.append({
        "Model": "ARIMA",
        **regression_metrics(actual, arima_pred),
        "Parameters": arima_params,
        "TrainTimeSec": arima_time,
        "Details": "Selected order {}; validation RMSE".format(arima_order),
    })

    # 3) U-MIDAS (target + auxiliary lagged regressors)
    print("\n[3/5] U-MIDAS (OLS)")
    umidas_pred, umidas_time, umidas_params, umidas_features = run_umidas(
        X_t_train, X_a_train, y_train,
        X_t_test, X_a_test,
    )
    umidas_pred_raw = inverse_target_scale(umidas_pred, scaler_t, target_idx)
    predictions["umidas"] = umidas_pred_raw
    records.append({
        "Model": "U-MIDAS (OLS)",
        **regression_metrics(actual, umidas_pred_raw),
        "Parameters": umidas_params,
        "TrainTimeSec": umidas_time,
        "Details": "Unrestricted OLS; {} lag features".format(umidas_features),
    })

    # 4) DeepAR: chronological validation + full training refit
    print("\n[4/5] DeepAR (Gaussian LSTM)")

    # Create validation windows using scalers fitted only on the prefix
    validation_data = make_chronological_validation_windows(
        train_data,
        args,
        val_ratio=0.2,
    )

    deepar_val_train_x = validation_data["X_t_train"][
        :, :, target_idx:target_idx + 1
    ]
    deepar_val_train_y = validation_data["y_t_train"]

    deepar_val_x = validation_data["X_t_val"][
        :, :, target_idx:target_idx + 1
    ]
    deepar_val_y = validation_data["y_t_val"]

    # Stage 1: Select the best epoch
    set_seed(int(args.seed))

    deepar_validation_model = DeepARGaussian(
        hidden_size=int(args.t_enc_hd),
        dropout=float(args.t_enc_dropout),
    )

    best_epoch, validation_time = select_deepar_epoch(
        deepar_validation_model,
        deepar_val_train_x,
        deepar_val_train_y,
        deepar_val_x,
        deepar_val_y,
        args,
        device,
    )

    # Stage 2: Initialize a fresh model and train on ALL training windows
    set_seed(int(args.seed))

    deepar = DeepARGaussian(
        hidden_size=int(args.t_enc_hd),
        dropout=float(args.t_enc_dropout),
    )

    deepar, final_train_time = fit_deepar_full_train(
        deepar,
        deepar_train_x,
        y_train,
        args,
        device,
        epochs=best_epoch,
    )

    deepar_time = validation_time + final_train_time

    # Test prediction
    deepar_mean_scaled = predict_deepar(
        deepar,
        deepar_test_x,
        args,
        device,
    )

    deepar_pred = inverse_target_scale(
        deepar_mean_scaled,
        scaler_t,
        target_idx,
    )

    predictions["deepar"] = deepar_pred

    records.append({
        "Model": "DeepAR (Gaussian LSTM)",
        **regression_metrics(actual, deepar_pred),
        "Parameters": count_parameters(deepar),
        "TrainTimeSec": deepar_time,
        "Details": (
            "One-step Gaussian LSTM; chronological validation; "
            "refit on full train; best epoch {}".format(best_epoch)
        ),
    })

    # 5) MED: chronological validation + full training refit
    if not skip_med:
        print("\n[5/5] MED (author implementation)")

        # Stage 1: Select the best epoch
        set_seed(int(args.seed))

        med_validation_wrapper = MEDModel(args)
        med_validation_wrapper.device = device

        med_validation_time, med_best_epoch = fit_med_same_validation(
            med_validation_wrapper,
            train_data,
            args,
            device,
        )

        # Stage 2: Create a completely fresh MED model
        set_seed(int(args.seed))

        med_wrapper = MEDModel(args)
        med_wrapper.device = device

        med_params = count_parameters(med_wrapper.model)

        med_final_time = fit_med_full_train(
            med_wrapper,
            train_data,
            args,
            device,
            epochs=med_best_epoch,
        )

        # Save the final trained MED model
        model_path = output_dir / "med_model_state_dict.pth"

        torch.save(
            med_wrapper.model.state_dict(),
            model_path,
        )

        print(f"Saved MED model: {model_path}")

        med_time = med_validation_time + med_final_time

        # Predict on test data
        med_pred = np.asarray(
            med_wrapper.test(test_data, train_data),
            dtype=float,
        ).reshape(-1)

        if len(med_pred) != len(actual):
            raise ValueError(
                "MED predictions do not align with the test observations."
            )

        predictions["med"] = med_pred

        records.append({
            "Model": "MED (author implementation)",
            **regression_metrics(actual, med_pred),
            "Parameters": med_params,
            "TrainTimeSec": med_time,
            "Details": (
                "Author's dual-frequency MED with MFA; "
                "chronological validation; refit on full train; "
                "best epoch {}".format(med_best_epoch)
            ),
        })

    metrics_df = pd.DataFrame(records).sort_values("RMSE", ascending=True)
    metrics_path = output_dir / "comparison_metrics.csv"
    metrics_df.to_csv(metrics_path, index=False, encoding="utf-8-sig")

    pred_df = pd.DataFrame({"timestamp": timestamps, "actual": actual})
    for name, values in predictions.items():
        if len(values) != len(actual):
            raise ValueError("Prediction length mismatch for {}.".format(name))
        pred_df[name] = values
    predictions_path = output_dir / "comparison_predictions.csv"
    pred_df.to_csv(predictions_path, index=False, encoding="utf-8-sig")

    print("\n=== Model comparison (lower error is better) ===")
    print(metrics_df.to_string(index=False, float_format=lambda value: "{:.6f}".format(value)))
    print("\nSaved: {}".format(metrics_path))
    print("Saved: {}".format(predictions_path))


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MED with paper-inspired forecasting baselines")
    parser.add_argument("--configs", default="configs/0_simu/med.yaml")
    parser.add_argument("--output_dir", default="results/comparison")
    parser.add_argument("--skip_med", action="store_true", help="Run the four baselines without MED")
    parser.add_argument("--verbose", default=10, type=int)
    args = parse_args(parser)
    if not hasattr(args, "verbose"):
        args.verbose = 10
    run_experiment(args, Path(args.output_dir), skip_med=bool(args.skip_med))


if __name__ == "__main__":
    main()
