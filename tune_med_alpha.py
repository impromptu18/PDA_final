
from __future__ import annotations

import argparse
import copy
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from args_parser import parse_args
from data_processor import DataProcessor
from models.med_model import MEDModel
from utils import CustomDatasetD

from compare_models import make_chronological_validation_windows


# ============================================================
# 1. CONFIGURATION
# ============================================================

ALPHA_VALUES = [0.1, 0.3, 0.5]

PATIENCE = 10

OUTPUT_DIR = Path("results/tuning_alpha")


# ============================================================
# 2. REPRODUCIBILITY
# ============================================================

def set_seed(seed):

    import random
    import os

    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# 3. CREATE DATA LOADERS
# ============================================================

def create_loader(
    X_t,
    X_a,
    y_t,
    y_a,
    args,
    shuffle
):

    dataset = CustomDatasetD(
        X_t,
        X_a,

        X_t[:, -int(args.t_dec_time_steps):],

        X_a[:, -int(args.a_dec_time_steps):],

        y_t,
        y_a
    )

    return DataLoader(
        dataset,
        batch_size=int(args.batch_size),
        shuffle=shuffle
    )


# ============================================================
# 4. RUN ONE ALPHA EXPERIMENT
# ============================================================

def run_alpha_experiment(
    base_args,
    windows,
    alpha,
    device
):

    print("\n" + "=" * 65)
    print(f"EXPERIMENT: loss_alpha = {alpha}")
    print("=" * 65)

    # --------------------------------------------------------
    # Use a fresh configuration and model for every alpha
    # --------------------------------------------------------

    args = copy.deepcopy(base_args)
    args.loss_alpha = float(alpha)

    # Reset seed so all configurations start from comparable
    # initial weights and data-shuffling conditions.
    set_seed(int(args.seed))

    wrapper = MEDModel(args)
    wrapper.device = device

    wrapper.scaler_a = windows["scaler_a"]
    wrapper.scaler_t = windows["scaler_t"]

    train_t = windows["X_t_train"]
    train_a = windows["X_a_train"]

    train_y_t = windows["y_t_train"]
    train_y_a = windows["y_a_train"]

    val_t = windows["X_t_val"]
    val_a = windows["X_a_val"]

    val_y_t = windows["y_t_val"]
    val_y_a = windows["y_a_val"]

    wrapper.key_idx = wrapper.args_key_idx if hasattr(
        wrapper, "args_key_idx"
    ) else 0

    # Correct target dimension
    wrapper.key_idx = windows["target_idx"]

    train_loader = create_loader(
        train_t,
        train_a,
        train_y_t,
        train_y_a,
        args,
        shuffle=True
    )

    val_loader = create_loader(
        val_t,
        val_a,
        val_y_t,
        val_y_a,
        args,
        shuffle=False
    )

    model = wrapper.model.to(device)
    optimizer = wrapper.optimizer
    criterion = wrapper.criterion

    best_val_total = float("inf")
    best_val_target = float("inf")
    best_val_aux = float("inf")

    best_epoch = 0
    bad_epochs = 0

    started = time.perf_counter()

    # --------------------------------------------------------
    # Training + chronological validation
    # --------------------------------------------------------

    for epoch in range(1, int(args.epoch) + 1):

        model.train()

        for batch in train_loader:

            t_src = batch["X_enc_t"].to(device)
            a_src = batch["X_enc_a"].to(device)

            t_dec = batch["X_dec_t"].to(device)
            a_dec = batch["X_dec_a"].to(device)

            t_y = batch["y_train_t"].to(device)
            a_y = batch["y_train_a"].to(device)

            optimizer.zero_grad(set_to_none=True)

            t_out, a_out = model(
                t_src,
                a_src,
                t_dec,
                a_dec
            )

            t_out = t_out[:, -1]
            a_out = a_out[:, -1]

            loss_t = criterion(
                t_out,
                t_y.view_as(t_out)
            )

            loss_a = criterion(
                a_out,
                a_y.view_as(a_out)
            )

            loss = loss_t + alpha * loss_a

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                wrapper.grad_clip
            )

            optimizer.step()

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        model.eval()

        val_target_sum = 0.0
        val_aux_sum = 0.0
        val_count = 0

        with torch.no_grad():

            for batch in val_loader:

                t_src = batch["X_enc_t"].to(device)
                a_src = batch["X_enc_a"].to(device)

                t_dec = batch["X_dec_t"].to(device)
                a_dec = batch["X_dec_a"].to(device)

                t_y = batch["y_train_t"].to(device)
                a_y = batch["y_train_a"].to(device)

                t_out, a_out = model(
                    t_src,
                    a_src,
                    t_dec,
                    a_dec
                )

                t_out = t_out[:, -1]
                a_out = a_out[:, -1]

                loss_t = criterion(
                    t_out,
                    t_y.view_as(t_out)
                )

                loss_a = criterion(
                    a_out,
                    a_y.view_as(a_out)
                )

                val_target_sum += loss_t.item()
                val_aux_sum += loss_a.item()

                val_count += 1

        # Average validation losses across batches
        val_target = val_target_sum / val_count
        val_aux = val_aux_sum / val_count

        # Combined validation loss
        val_total = val_target + alpha * val_aux

        if epoch % 10 == 0:
            print(
                f"Alpha={alpha} | "
                f"Epoch={epoch:03d} | "
                f"Val Target MSE={val_target:.6f} | "
                f"Val Auxiliary MSE={val_aux:.6f} | "
                f"Val Total Loss={val_total:.6f}"
            )

        # ----------------------------------------------------
        # Early stopping based on TOTAL validation loss
        # ----------------------------------------------------

        if val_total < best_val_total:

            best_val_total = val_total

            best_val_target = val_target
            best_val_aux = val_aux

            best_epoch = epoch
            bad_epochs = 0

        else:

            bad_epochs += 1

        if bad_epochs >= PATIENCE:

            print(
                f"Early stopping at epoch {epoch}"
            )

            break


    elapsed = time.perf_counter() - started

    
    # --------------------------------------------------------
    # Validation metrics in original target units
    # --------------------------------------------------------

    target_idx = windows["target_idx"]

    target_scale = wrapper.scaler_t.scale_[target_idx]

    val_rmse_scaled = np.sqrt(best_val_target)

    val_rmse_original = val_rmse_scaled / target_scale

    print("\nRESULT:")
    print("Alpha:", alpha)
    print("Best epoch:", best_epoch)
    print("Best validation target MSE:", best_val_target)
    print("Validation target RMSE:", val_rmse_original)
    print("Validation auxiliary MSE:", best_val_aux)
    print("Training time:", elapsed)

    return {
        "LossAlpha": alpha,
        "BestEpoch": best_epoch,
        "ValidationTargetMSE": best_val_target,
        "ValidationTargetRMSE": val_rmse_original,
        "ValidationAuxiliaryMSE": best_val_aux,
        "ValidationTotalLoss": best_val_total,
        "TrainTimeSec": elapsed
    }


# ============================================================
# 5. MAIN EXPERIMENT
# ============================================================

def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)

    parser = argparse.ArgumentParser(
        description="MED loss_alpha tuning"
    )

    parser.add_argument(
        "--configs",
        default="configs/2_elec/med.yaml"
    )

    args = parse_args(parser)

    # --------------------------------------------------------
    # Load training data only
    # --------------------------------------------------------

    print("\n[1/3] Loading Electricity training data...")

    train_data, _ = DataProcessor(args).preprocess()

    # --------------------------------------------------------
    # Create the same chronological validation split
    # --------------------------------------------------------

    print("\n[2/3] Creating chronological validation windows...")

    windows = make_chronological_validation_windows(
        train_data,
        args,
        val_ratio=0.2
    )

    train_t = train_data[1]

    windows["target_idx"] = train_t.columns.get_loc(
        args.key_var
    )

    print("Train windows:", len(windows["X_t_train"]))
    print("Validation windows:", len(windows["X_t_val"]))

    # --------------------------------------------------------
    # Run experiments
    # --------------------------------------------------------

    print("\n[3/3] Running alpha experiments...")

    results = []

    for alpha in ALPHA_VALUES:

        result = run_alpha_experiment(
            args,
            windows,
            alpha,
            device
        )

        results.append(result)

        # Save after each trial, so completed trials are preserved
        results_df = pd.DataFrame(results)

        results_df.to_csv(
            OUTPUT_DIR / "alpha_tuning_results.csv",
            index=False,
            encoding="utf-8-sig"
        )

    # --------------------------------------------------------
    # Rank experiments by validation target MSE
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_df = results_df.sort_values(
        "ValidationTargetMSE",
        ascending=True
    ).reset_index(drop=True)

    results_df.to_csv(
        OUTPUT_DIR / "alpha_tuning_results.csv",
        index=False,
        encoding="utf-8-sig"
    )

    best = results_df.iloc[0]

    print("\n" + "=" * 65)
    print("ALPHA TUNING RESULTS")
    print("=" * 65)

    print(
        results_df.to_string(
            index=False,
            float_format=lambda x: f"{x:.6f}"
        )
    )

    print("\nBEST CONFIGURATION:")
    print("loss_alpha:", best["LossAlpha"])
    print("Best epoch:", best["BestEpoch"])
    print(
        "Validation Target RMSE:",
        best["ValidationTargetRMSE"]
    )

    print(
        "\nSaved:",
        OUTPUT_DIR / "alpha_tuning_results.csv"
    )


if __name__ == "__main__":
    main()
