
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import torch
from torch import nn
from sklearn.preprocessing import MinMaxScaler

from args_parser import parse_args
from data_processor import DataProcessor
from models.dual_encoder_decoder import MED


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42
N_BACKGROUND = 50

OUTPUT_DIR = Path("results/comparison")

MODEL_PATH = OUTPUT_DIR / "med_model_state_dict.pth"
DATA_PATH = OUTPUT_DIR / "med_shap_data.npz"
SHAP_PATH = OUTPUT_DIR / "med_shap_values.npz"
PREDICTION_PATH = OUTPUT_DIR / "comparison_predictions.csv"


# ============================================================
# 2. MED PREDICTION WRAPPER
# ============================================================

class MEDTargetPredictor(nn.Module):

    def __init__(self, med_model, t_dec_steps, a_dec_steps):
        super().__init__()

        self.med = med_model
        self.t_dec_steps = int(t_dec_steps)
        self.a_dec_steps = int(a_dec_steps)

    def forward(self, t_src, a_src):

        t_trg = t_src[:, -self.t_dec_steps:, :]
        a_trg = a_src[:, -self.a_dec_steps:, :]

        t_out, _ = self.med(
            t_src,
            a_src,
            t_trg,
            a_trg
        )

        return t_out[:, -1, :]


# ============================================================
# 3. LOAD SAVED EXPERIMENT
# ============================================================

def load_experiment(args, device):

    for path in [
        MODEL_PATH,
        DATA_PATH,
        SHAP_PATH,
        PREDICTION_PATH
    ]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    print("\n[1/5] Loading saved experiment...")

    saved_data = np.load(DATA_PATH)
    saved_shap = np.load(SHAP_PATH)

    X_t_train = saved_data["X_t_train"].astype(np.float32)
    X_a_train = saved_data["X_a_train"].astype(np.float32)

    X_t = saved_shap["X_t"].astype(np.float32)
    X_a = saved_shap["X_a"].astype(np.float32)

    shap_t = saved_shap["shap_t"]
    shap_a = saved_shap["shap_a"]

    test_indices = saved_shap["test_indices"]

    print("Explained samples:", len(X_t))
    print("Target SHAP:", shap_t.shape)
    print("Auxiliary SHAP:", shap_a.shape)

    # --------------------------------------------------------
    # Load MED checkpoint
    # --------------------------------------------------------

    print("\n[2/5] Loading MED checkpoint...")

    med = MED(args).to(device)

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device
        )

    med.load_state_dict(state_dict)
    med.eval()

    predictor = MEDTargetPredictor(
        med,
        args.t_dec_time_steps,
        args.a_dec_time_steps
    ).to(device)

    predictor.eval()

    # --------------------------------------------------------
    # Read original feature names and fit scalers
    # --------------------------------------------------------

    print("\n[3/5] Reading data and scalers...")

    train_data, _ = DataProcessor(args).preprocess()

    train_a, train_t = train_data

    scaler_t = MinMaxScaler()
    scaler_a = MinMaxScaler()

    scaler_t.fit(train_t)
    scaler_a.fit(train_a)

    target_names = list(train_t.columns)
    aux_names = list(train_a.columns)

    target_idx = train_t.columns.get_loc(args.key_var)

    predictions = pd.read_csv(
        PREDICTION_PATH,
        parse_dates=["timestamp"]
    )

    if len(test_indices) != len(X_t):
        raise ValueError("SHAP sample indices and inputs do not match.")

    if np.any(test_indices >= len(predictions)):
        raise ValueError("A SHAP test index exceeds prediction file length.")

    return {
        "predictor": predictor,
        "X_t_train": X_t_train,
        "X_a_train": X_a_train,
        "X_t": X_t,
        "X_a": X_a,
        "shap_t": shap_t,
        "shap_a": shap_a,
        "test_indices": test_indices,
        "target_names": target_names,
        "aux_names": aux_names,
        "scaler_t": scaler_t,
        "scaler_a": scaler_a,
        "target_idx": target_idx,
        "predictions": predictions,
        "train_t": train_t,
        "train_a": train_a
    }


# ============================================================
# 4. SELECT A LOCAL SAMPLE
# ============================================================

def select_local_sample(data):

    predictions = data["predictions"]
    test_indices = data["test_indices"]

    # Map 100 explained samples to original test rows
    selected_predictions = predictions.iloc[
        test_indices
    ].reset_index(drop=True)

    # Forecast error = prediction - actual
    errors = (
        selected_predictions["med"].to_numpy()
        - selected_predictions["actual"].to_numpy()
    )

    # Select the largest absolute error
    local_position = int(np.argmax(np.abs(errors)))

    original_test_index = int(test_indices[local_position])

    row = predictions.iloc[original_test_index]

    print("\nSelected local case:")
    print("Position in explained samples:", local_position)
    print("Original test index:", original_test_index)
    print("Timestamp:", row["timestamp"])
    print("Actual:", row["actual"])
    print("MED prediction:", row["med"])
    print("Prediction error:", row["med"] - row["actual"])

    return local_position, original_test_index, row


# ============================================================
# 5. CONVERT SHAP AND FEATURES TO ORIGINAL SCALE
# ============================================================

def inverse_target_value(value, scaler, target_idx):

    return (
        float(value) - scaler.min_[target_idx]
    ) / scaler.scale_[target_idx]


def get_feature_labels(
    target_names,
    aux_names,
    t_steps,
    a_steps,
    t_interval,
    a_interval
):

    target_labels = []

    for i in range(t_steps):

        lag = (t_steps - i) * t_interval

        target_labels.append(
            f"{target_names[0]} (t-{lag:g}h)"
        )

    aux_labels = []

    for i in range(a_steps):

        lag = (a_steps - i) * a_interval

        for name in aux_names:

            aux_labels.append(
                f"{name} (t-{lag:g}h)"
            )

    return target_labels, aux_labels


def prepare_local_explanation(
    data,
    local_position,
    device
):

    shap_t = data["shap_t"][local_position]
    shap_a = data["shap_a"][local_position]

    X_t_scaled = data["X_t"][local_position]
    X_a_scaled = data["X_a"][local_position]

    scaler_t = data["scaler_t"]
    scaler_a = data["scaler_a"]

    target_idx = data["target_idx"]

    # --------------------------------------------------------
    # Convert input features to their original units
    # --------------------------------------------------------

    X_t_raw = scaler_t.inverse_transform(
        X_t_scaled.reshape(-1, X_t_scaled.shape[-1])
    )

    X_t_raw = X_t_raw.reshape(X_t_scaled.shape)

    X_a_raw = scaler_a.inverse_transform(
        X_a_scaled.reshape(-1, X_a_scaled.shape[-1])
    )

    X_a_raw = X_a_raw.reshape(X_a_scaled.shape)

    feature_values = np.concatenate([
        X_t_raw.reshape(-1),
        X_a_raw.reshape(-1)
    ])

    # --------------------------------------------------------
    # Create feature labels
    # --------------------------------------------------------

    train_t = data["train_t"]
    train_a = data["train_a"]

    t_interval = (
        train_t.index[1] - train_t.index[0]
    ).total_seconds() / 3600

    a_interval = (
        train_a.index[1] - train_a.index[0]
    ).total_seconds() / 3600

    target_labels, aux_labels = get_feature_labels(
        data["target_names"],
        data["aux_names"],
        X_t_scaled.shape[0],
        X_a_scaled.shape[0],
        t_interval,
        a_interval
    )

    feature_names = target_labels + aux_labels

    # --------------------------------------------------------
    # Reconstruct the same background samples
    # --------------------------------------------------------

    rng = np.random.default_rng(SEED)

    n_background = min(
        N_BACKGROUND,
        len(data["X_t_train"])
    )

    bg_idx = rng.choice(
        len(data["X_t_train"]),
        size=n_background,
        replace=False
    )

    background_t = torch.tensor(
        data["X_t_train"][bg_idx],
        dtype=torch.float32,
        device=device
    )

    background_a = torch.tensor(
        data["X_a_train"][bg_idx],
        dtype=torch.float32,
        device=device
    )

    predictor = data["predictor"]

    with torch.no_grad():

        background_predictions = predictor(
            background_t,
            background_a
        )

    # Expected output on background, in scaled units
    base_scaled = float(
        background_predictions.mean().cpu().item()
    )

    # Convert base value to original target units
    base_raw = inverse_target_value(
        base_scaled,
        scaler_t,
        target_idx
    )

    # --------------------------------------------------------
    # Calculate model prediction for the selected sample
    # --------------------------------------------------------

    local_t = torch.tensor(
        X_t_scaled[None],
        dtype=torch.float32,
        device=device
    )

    local_a = torch.tensor(
        X_a_scaled[None],
        dtype=torch.float32,
        device=device
    )

    # --------------------------------------------------------
    # Recalculate local SHAP using more interpolation samples
    # --------------------------------------------------------

    LOCAL_NSAMPLES = 5000

    explainer = shap.GradientExplainer(
        predictor,
        [background_t, background_a]
    )

    raw_shap = explainer.shap_values(
        [local_t, local_a],
        nsamples=LOCAL_NSAMPLES,
        rseed=SEED
    )

    if isinstance(raw_shap, (list, tuple)):

        if (
            len(raw_shap) == 1
            and isinstance(raw_shap[0], (list, tuple))
        ):
            raw_shap = raw_shap[0]

        if len(raw_shap) != 2:
            raise ValueError("Expected two SHAP input arrays.")

        raw_t, raw_a = raw_shap

    else:
        raise ValueError("Expected two SHAP input arrays.")


    def normalize_local_shap(values, expected_shape):

        values = np.asarray(values)

        if (
            values.ndim == len(expected_shape) + 1
            and values.shape[-1] == 1
        ):
            values = values[..., 0]

        if values.shape != expected_shape:
            raise ValueError(
                f"Expected {expected_shape}, got {values.shape}"
            )

        return values[0]


    # SHAP values from the saved 100-sample explanation
    old_shap_t = data["shap_t"][local_position]
    old_shap_a = data["shap_a"][local_position]

    # Newly recalculated SHAP values
    new_shap_t = normalize_local_shap(
        raw_t,
        tuple(local_t.shape)
    )

    new_shap_a = normalize_local_shap(
        raw_a,
        tuple(local_a.shape)
    )

    # Compare old and new SHAP values
    print("\n=== CHECK SHAP RECALCULATION ===")

    print(
        "Target max difference:",
        np.max(np.abs(new_shap_t - old_shap_t))
    )

    print(
        "Auxiliary max difference:",
        np.max(np.abs(new_shap_a - old_shap_a))
    )

    print(
        "Target mean difference:",
        np.mean(np.abs(new_shap_t - old_shap_t))
    )

    print(
        "Auxiliary mean difference:",
        np.mean(np.abs(new_shap_a - old_shap_a))
    )


    # --------------------------------------------------------
    # 1. USE NEW SHAP VALUES
    # --------------------------------------------------------

    shap_t = new_shap_t
    shap_a = new_shap_a

    # Combine NEW SHAP values
    shap_scaled = np.concatenate([
        shap_t.reshape(-1),
        shap_a.reshape(-1)
    ])

    # Convert SHAP values to original target units
    output_scale = scaler_t.scale_[target_idx]
    shap_raw = shap_scaled / output_scale

    print("\n=== SHAP UNIT CHECK ===")
    print("Target output scale:", output_scale)
    print("New SHAP sum (scaled):", shap_scaled.sum())
    print("New SHAP sum (original units):", shap_raw.sum())

    # --------------------------------------------------------
    # 2. VERIFY OLD AND NEW SHAP
    # --------------------------------------------------------

    print("\n=== CHECK SHAP RECALCULATION ===")

    print(
        "Target max difference:",
        np.max(np.abs(new_shap_t - old_shap_t))
    )

    print(
        "Auxiliary max difference:",
        np.max(np.abs(new_shap_a - old_shap_a))
    )

    print(
        "Target mean difference:",
        np.mean(np.abs(new_shap_t - old_shap_t))
    )

    print(
        "Auxiliary mean difference:",
        np.mean(np.abs(new_shap_a - old_shap_a))
    )

    print("New target SHAP sum:", shap_t.sum())
    print("Old target SHAP sum:", old_shap_t.sum())

    print("New auxiliary SHAP sum:", shap_a.sum())
    print("Old auxiliary SHAP sum:", old_shap_a.sum())

    # --------------------------------------------------------
    # 3. PREDICT SELECTED SAMPLE
    # --------------------------------------------------------

    with torch.no_grad():
        prediction_scaled = float(
            predictor(local_t, local_a).cpu().item()
        )

    prediction_raw = inverse_target_value(
        prediction_scaled,
        scaler_t,
        target_idx
    )

    # --------------------------------------------------------
    # 4. ADDITIVITY CHECK - SCALED SPACE
    # --------------------------------------------------------

    scaled_difference = prediction_scaled - base_scaled
    scaled_shap_sum = shap_scaled.sum()

    scaled_residual = scaled_difference - scaled_shap_sum

    relative_residual = (
        abs(scaled_residual)
        / (abs(scaled_difference) + 1e-12)
    ) * 100

    print(
        "Relative additivity residual:",
        relative_residual,
        "%"
)

    print("\n=== ADDITIVITY CHECK IN SCALED SPACE ===")

    print("Prediction scaled:", prediction_scaled)
    print("Base value scaled:", base_scaled)
    print("Prediction - Base:", scaled_difference)
    print("Sum of SHAP:", scaled_shap_sum)
    print("Residual scaled:", scaled_residual)

    # --------------------------------------------------------
    # 5. ADDITIVITY CHECK - ORIGINAL TARGET UNITS
    # --------------------------------------------------------

    shap_sum_prediction = base_raw + shap_raw.sum()

    residual = prediction_raw - shap_sum_prediction

    print("\n=== ADDITIVITY CHECK IN ORIGINAL UNITS ===")

    print("Base value:", base_raw)
    print("MED prediction:", prediction_raw)
    print("Base + sum(SHAP):", shap_sum_prediction)
    print("Additivity residual:", residual)


    return {
        "values": shap_raw,
        "base_value": base_raw,
        "prediction": prediction_raw,
        "shap_sum_prediction": shap_sum_prediction,
        "additivity_residual": residual,
        "feature_values": feature_values,
        "feature_names": feature_names,
        "X_t_raw": X_t_raw,
        "X_a_raw": X_a_raw
    }


# ============================================================
# 6. CREATE WATERFALL PLOT
# ============================================================

def create_waterfall(explanation, row):

    print("\n[4/5] Creating local waterfall plot...")

    shap_explanation = shap.Explanation(
        values=explanation["values"],
        base_values=explanation["base_value"],
        data=explanation["feature_values"],
        feature_names=explanation["feature_names"]
    )

    plt.figure(figsize=(12, 8))

    shap.plots.waterfall(
        shap_explanation,
        max_display=20,
        show=False
    )

    plt.title(
        "MED Local Explanation - {}".format(
            row["timestamp"]
        )
    )

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_local_waterfall.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 7. EXPORT LOCAL FEATURE CONTRIBUTIONS
# ============================================================

def export_local_results(explanation, row, original_test_index):

    values = explanation["values"]
    names = explanation["feature_names"]
    feature_values = explanation["feature_values"]

    result = pd.DataFrame({
        "Feature": names,
        "InputValue": feature_values,
        "SHAP": values,
        "AbsSHAP": np.abs(values)
    })

    result["Effect"] = np.where(
        result["SHAP"] > 0,
        "Increases prediction",
        np.where(
            result["SHAP"] < 0,
            "Decreases prediction",
            "No contribution"
        )
    )

    result = result.sort_values(
        "AbsSHAP",
        ascending=False
    )

    result.to_csv(
        OUTPUT_DIR / "shap_local_contributions.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # --------------------------------------------------------
    # Save case information
    # --------------------------------------------------------

    case = pd.DataFrame([{
        "Timestamp": row["timestamp"],
        "Actual": row["actual"],
        "MED_Prediction_CSV": row["med"],
        "MED_Prediction_Reconstructed": explanation["prediction"],
        "BaseValue": explanation["base_value"],
        "SHAP_Sum_Prediction": explanation["shap_sum_prediction"],
        "Additivity_Residual": explanation["additivity_residual"],
        "Forecast_Error": row["med"] - row["actual"],
        "Original_Test_Index": original_test_index
    }])

    case.to_csv(
        OUTPUT_DIR / "shap_local_case.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nTop local contributions:")
    print(result.head(15).to_string(index=False))

    # --------------------------------------------------------
    # Bar chart: strongest positive and negative contributions
    # --------------------------------------------------------

    top = result.head(15).sort_values("SHAP")

    plt.figure(figsize=(11, 7))

    plt.barh(
        top["Feature"],
        top["SHAP"]
    )

    plt.axvline(0, linewidth=0.8)

    plt.xlabel("SHAP contribution (original target units)")
    plt.ylabel("Input feature")
    plt.title("MED - Local Feature Contributions")

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_local_contributions.png",
        dpi=200
    )

    plt.close()

    print("\nLocal case:")
    print(case.to_string(index=False))


# ============================================================
# 8. MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device("cpu")

    print("Device:", device)
    print("SHAP version:", shap.__version__)

    parser = argparse.ArgumentParser(
        description="Local SHAP explanation for MED"
    )

    parser.add_argument(
        "--configs",
        default="configs/2_elec/med.yaml"
    )

    args = parse_args(parser)

    data = load_experiment(args, device)

    print("\n[4/5] Selecting local case...")

    (
        local_position,
        original_test_index,
        row
    ) = select_local_sample(data)

    explanation = prepare_local_explanation(
        data,
        local_position,
        device
    )

    print("\nBase value:", explanation["base_value"])
    print("MED reconstructed prediction:", explanation["prediction"])
    print("Base + sum(SHAP):", explanation["shap_sum_prediction"])
    print("Additivity residual:", explanation["additivity_residual"])

    create_waterfall(explanation, row)

    export_local_results(
        explanation,
        row,
        original_test_index
    )

    print("\n[5/5] Local SHAP completed.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
