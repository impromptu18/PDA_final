
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

from args_parser import parse_args
from data_processor import DataProcessor
from models.dual_encoder_decoder import MED


# ============================================================
# 1. CONFIGURATION
# ============================================================

SEED = 42

# Số mẫu dùng làm dữ liệu nền cho SHAP
N_BACKGROUND = 100

# Số mẫu test được giải thích ở lần chạy đầu
N_EXPLAIN = 200

# Số mẫu nội suy trong Expected Gradients
NSAMPLES = 300

OUTPUT_DIR = Path("results/comparison")
MODEL_PATH = OUTPUT_DIR / "med_model_state_dict.pth"
DATA_PATH = OUTPUT_DIR / "med_shap_data.npz"


# ============================================================
# 2. MODEL WRAPPER
# ============================================================

class MEDTargetPredictor(nn.Module):
    """
    Wrapper chỉ trả về dự báo target của MED.

    Hai decoder inputs được tạo giống quy trình
    trong MEDModel và compare_models.py.
    """

    def __init__(self, med_model, t_dec_steps, a_dec_steps):
        super().__init__()

        self.med = med_model
        self.t_dec_steps = int(t_dec_steps)
        self.a_dec_steps = int(a_dec_steps)

    def forward(self, t_src, a_src):

        # Decoder inputs lấy từ các bước cuối của encoder inputs
        t_trg = t_src[:, -self.t_dec_steps:, :]
        a_trg = a_src[:, -self.a_dec_steps:, :]

        # MED prediction
        t_out, _ = self.med(
            t_src,
            a_src,
            t_trg,
            a_trg
        )

        # Chỉ lấy dự báo target ở bước cuối
        return t_out[:, -1, :]


# ============================================================
# 3. LOAD DATA AND MODEL
# ============================================================

def load_experiment(args, device):

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy model checkpoint: {MODEL_PATH}"
        )

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Không tìm thấy SHAP data: {DATA_PATH}"
        )

    print("\n[1/5] Loading saved data...")

    saved = np.load(DATA_PATH)

    X_t_train = saved["X_t_train"].astype(np.float32)
    X_a_train = saved["X_a_train"].astype(np.float32)

    X_t_test = saved["X_t_test"].astype(np.float32)
    X_a_test = saved["X_a_test"].astype(np.float32)

    print("X_t_train:", X_t_train.shape)
    print("X_a_train:", X_a_train.shape)
    print("X_t_test:", X_t_test.shape)
    print("X_a_test:", X_a_test.shape)

    # Recreate the same MED architecture
    print("\n[2/5] Loading trained MED...")

    med = MED(args).to(device)

    try:
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch
        state_dict = torch.load(
            MODEL_PATH,
            map_location=device
        )

    med.load_state_dict(state_dict, strict=True)
    med.eval()

    predictor = MEDTargetPredictor(
        med,
        args.t_dec_time_steps,
        args.a_dec_time_steps
    ).to(device)

    predictor.eval()

    print("MED checkpoint loaded successfully.")

    # Get original feature names from DataProcessor
    print("\n[3/5] Reading feature names...")

    train_data, _ = DataProcessor(args).preprocess()

    train_a, train_t = train_data

    target_names = list(train_t.columns)
    aux_names = list(train_a.columns)

    if len(target_names) != X_t_test.shape[-1]:
        raise ValueError(
            "Target feature names do not match X_t dimensions."
        )

    if len(aux_names) != X_a_test.shape[-1]:
        raise ValueError(
            "Auxiliary feature names do not match X_a dimensions."
        )

    return (
        predictor,
        X_t_train,
        X_a_train,
        X_t_test,
        X_a_test,
        train_t,
        train_a,
        target_names,
        aux_names
    )


# ============================================================
# 4. TIME LAG LABELS
# ============================================================

def interval_hours(index):

    if len(index) < 2:
        raise ValueError("Không đủ timestamp để xác định tần suất.")

    delta = index[1] - index[0]

    return delta.total_seconds() / 3600.0


def make_feature_labels(
    target_names,
    aux_names,
    t_length,
    a_length,
    t_interval,
    a_interval
):

    target_labels = []

    for i in range(t_length):

        lag = (t_length - i) * t_interval

        target_labels.append(
            "{} (t-{:.0f}h)".format(
                target_names[0],
                lag
            )
        )

    aux_labels = []

    for i in range(a_length):

        lag = (a_length - i) * a_interval

        for name in aux_names:

            aux_labels.append(
                "{} (t-{:.0f}h)".format(
                    name,
                    lag
                )
            )

    return target_labels, aux_labels


# ============================================================
# 5. NORMALIZE SHAP OUTPUT
# ============================================================

def remove_output_dimension(values, expected_shape):

    values = np.asarray(values)

    # SHAP versions may append a singleton output dimension
    if values.ndim == len(expected_shape) + 1:
        if values.shape[-1] == 1:
            values = values[..., 0]

    if values.shape != expected_shape:
        raise ValueError(
            "SHAP shape mismatch. "
            f"Expected {expected_shape}, got {values.shape}"
        )

    return values


def calculate_shap(
    predictor,
    X_t_train,
    X_a_train,
    X_t_test,
    X_a_test
):

    print("\n[4/5] Calculating SHAP values...")

    rng = np.random.default_rng(SEED)

    # Background samples are selected from training windows
    n_background = min(N_BACKGROUND, len(X_t_train))

    bg_idx = rng.choice(
        len(X_t_train),
        size=n_background,
        replace=False
    )

    background_t = torch.tensor(
        X_t_train[bg_idx],
        dtype=torch.float32
    )

    background_a = torch.tensor(
        X_a_train[bg_idx],
        dtype=torch.float32
    )

    # Select evenly spaced test samples
    n_explain = min(N_EXPLAIN, len(X_t_test))

    test_idx = np.linspace(
        0,
        len(X_t_test) - 1,
        n_explain,
        dtype=int
    )

    explain_t = torch.tensor(
        X_t_test[test_idx],
        dtype=torch.float32
    )

    explain_a = torch.tensor(
        X_a_test[test_idx],
        dtype=torch.float32
    )

    print("Background samples:", n_background)
    print("Explained test samples:", n_explain)
    print("Expected Gradients samples:", NSAMPLES)

    # GradientExplainer supports multiple tensor inputs
    explainer = shap.GradientExplainer(
        predictor,
        [background_t, background_a]
    )

    raw_shap = explainer.shap_values(
        [explain_t, explain_a],
        nsamples=NSAMPLES,
        rseed=SEED
    )

    # Expected output: one SHAP array for each input tensor
    if isinstance(raw_shap, (list, tuple)):

        if (
            len(raw_shap) == 1
            and isinstance(raw_shap[0], (list, tuple))
        ):
            raw_shap = raw_shap[0]

        if len(raw_shap) != 2:
            raise ValueError(
                "Không nhận được hai nhóm SHAP cho hai đầu vào MED."
            )

        shap_t_raw, shap_a_raw = raw_shap

    else:
        raise ValueError(
            "SHAP không trả về hai nhóm giá trị tương ứng với X_t và X_a. "
            "Hãy kiểm tra phiên bản SHAP."
        )

    shap_t = remove_output_dimension(
        shap_t_raw,
        tuple(explain_t.shape)
    )

    shap_a = remove_output_dimension(
        shap_a_raw,
        tuple(explain_a.shape)
    )

    explain_t_np = explain_t.numpy()
    explain_a_np = explain_a.numpy()

    # Save raw SHAP arrays and explained inputs
    np.savez_compressed(
        OUTPUT_DIR / "med_shap_values.npz",
        shap_t=shap_t,
        shap_a=shap_a,
        X_t=explain_t_np,
        X_a=explain_a_np,
        test_indices=test_idx
    )

    print("SHAP target:", shap_t.shape)
    print("SHAP auxiliary:", shap_a.shape)

    return (
        shap_t,
        shap_a,
        explain_t_np,
        explain_a_np
    )


# ============================================================
# 6. GLOBAL FEATURE IMPORTANCE
# ============================================================

def analyze_features(
    shap_t,
    shap_a,
    target_names,
    aux_names,
    target_labels,
    aux_labels
):

    print("\n[5/5] Creating SHAP analysis...")

    # Average absolute attribution by variable
    target_importance = np.abs(shap_t).mean(axis=(0, 1))

    aux_importance = np.abs(shap_a).mean(axis=(0, 1))

    rows = []

    for name, score in zip(target_names, target_importance):

        rows.append({
            "Frequency": "Target",
            "Feature": name,
            "MeanAbsSHAP": float(score)
        })

    for name, score in zip(aux_names, aux_importance):

        rows.append({
            "Frequency": "Auxiliary",
            "Feature": name,
            "MeanAbsSHAP": float(score)
        })

    feature_df = pd.DataFrame(rows)

    feature_df = feature_df.sort_values(
        "MeanAbsSHAP",
        ascending=False
    )

    feature_df.to_csv(
        OUTPUT_DIR / "shap_feature_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nFeature importance:")
    print(feature_df.to_string(index=False))

    # Plot top features
    top = feature_df.head(15).sort_values(
        "MeanAbsSHAP",
        ascending=True
    )

    plt.figure(figsize=(10, 6))

    plt.barh(
        top["Feature"],
        top["MeanAbsSHAP"]
    )

    plt.xlabel("Mean absolute SHAP value")
    plt.ylabel("Feature")
    plt.title("MED - Global Feature Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_feature_importance.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Contribution by frequency
    # --------------------------------------------------------

    target_mass = float(
        np.abs(shap_t).sum(axis=(1, 2)).mean()
    )

    aux_mass = float(
        np.abs(shap_a).sum(axis=(1, 2)).mean()
    )

    total_mass = target_mass + aux_mass

    stream_df = pd.DataFrame([
        {
            "Frequency": "Target",
            "MeanAbsContribution": target_mass,
            "SharePercent": 100 * target_mass / total_mass
        },
        {
            "Frequency": "Auxiliary",
            "MeanAbsContribution": aux_mass,
            "SharePercent": 100 * aux_mass / total_mass
        }
    ])

    stream_df.to_csv(
        OUTPUT_DIR / "shap_stream_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    print("\nContribution by frequency:")
    print(stream_df.to_string(index=False))

    plt.figure(figsize=(7, 5))

    plt.bar(
        stream_df["Frequency"],
        stream_df["SharePercent"]
    )

    plt.ylabel("Share of absolute SHAP contribution (%)")
    plt.title("MED - Contribution by Frequency")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_frequency_contribution.png",
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Beeswarm plot
    # --------------------------------------------------------

    X_all = np.concatenate(
        [
            # target inputs will be attached later in main
        ],
        axis=0
    ) if False else None

    return feature_df, stream_df


# ============================================================
# 7. TEMPORAL IMPORTANCE
# ============================================================

def analyze_time_importance(
    shap_t,
    shap_a,
    target_labels,
    aux_names,
    t_length,
    a_length,
    t_interval,
    a_interval
):

    # Mean absolute contribution at each time step
    time_t = np.abs(shap_t).mean(axis=(0, 2))
    time_a = np.abs(shap_a).mean(axis=(0, 2))

    target_time_labels = [
        "t-{:.0f}h".format((t_length - i) * t_interval)
        for i in range(t_length)
    ]

    aux_time_labels = [
        "t-{:.0f}h".format((a_length - i) * a_interval)
        for i in range(a_length)
    ]

    target_df = pd.DataFrame({
        "TimeLag": target_time_labels,
        "MeanAbsSHAP": time_t
    })

    aux_df = pd.DataFrame({
        "TimeLag": aux_time_labels,
        "MeanAbsSHAP": time_a
    })

    target_df.to_csv(
        OUTPUT_DIR / "shap_target_time_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    aux_df.to_csv(
        OUTPUT_DIR / "shap_aux_time_importance.csv",
        index=False,
        encoding="utf-8-sig"
    )

    # Target time plot
    plt.figure(figsize=(8, 5))

    plt.bar(
        target_time_labels,
        time_t
    )

    plt.xlabel("Historical target time")
    plt.ylabel("Mean absolute SHAP")
    plt.title("MED - Target Temporal Importance")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_target_time_importance.png",
        dpi=200
    )

    plt.close()

    # Auxiliary time plot
    plt.figure(figsize=(11, 5))

    plt.bar(
        aux_time_labels,
        time_a
    )

    plt.xlabel("Historical auxiliary time")
    plt.ylabel("Mean absolute SHAP")
    plt.title("MED - Auxiliary Temporal Importance")
    plt.xticks(rotation=45)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_aux_time_importance.png",
        dpi=200
    )

    plt.close()

    return target_df, aux_df


# ============================================================
# 8. SHAP SUMMARY PLOT
# ============================================================

def create_summary_plot(
    shap_t,
    shap_a,
    X_t,
    X_a,
    target_labels,
    aux_labels
):

    n = len(X_t)

    values_all = np.concatenate(
        [
            shap_t.reshape(n, -1),
            shap_a.reshape(n, -1)
        ],
        axis=1
    )

    features_all = np.concatenate(
        [
            X_t.reshape(n, -1),
            X_a.reshape(n, -1)
        ],
        axis=1
    )

    feature_labels = target_labels + aux_labels

    plt.figure(figsize=(12, 9))

    shap.summary_plot(
        values_all,
        features_all,
        feature_names=feature_labels,
        max_display=20,
        show=False
    )

    plt.title("MED - SHAP Summary Plot")
    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "shap_summary.png",
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# 9. MAIN
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # Run on CPU, matching the completed experiment
    device = torch.device("cpu")

    print("Device:", device)
    print("SHAP version:", shap.__version__)

    parser = argparse.ArgumentParser(
        description="Explain trained MED model using SHAP"
    )

    parser.add_argument(
        "--configs",
        default="configs/2_elec/med.yaml"
    )

    args = parse_args(parser)

    (
        predictor,
        X_t_train,
        X_a_train,
        X_t_test,
        X_a_test,
        train_t,
        train_a,
        target_names,
        aux_names
    ) = load_experiment(args, device)

    t_interval = interval_hours(train_t.index)
    a_interval = interval_hours(train_a.index)

    target_labels, aux_labels = make_feature_labels(
        target_names,
        aux_names,
        X_t_test.shape[1],
        X_a_test.shape[1],
        t_interval,
        a_interval
    )

    shap_t, shap_a, X_t, X_a = calculate_shap(
        predictor,
        X_t_train,
        X_a_train,
        X_t_test,
        X_a_test
    )

    analyze_features(
        shap_t,
        shap_a,
        target_names,
        aux_names,
        target_labels,
        aux_labels
    )

    analyze_time_importance(
        shap_t,
        shap_a,
        target_labels,
        aux_names,
        X_t_test.shape[1],
        X_a_test.shape[1],
        t_interval,
        a_interval
    )

    create_summary_plot(
        shap_t,
        shap_a,
        X_t,
        X_a,
        target_labels,
        aux_labels
    )

    print("\nSHAP analysis completed.")
    print("Results saved in:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
