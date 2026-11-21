
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path


# =====================================================
# 1. READ PREDICTION DATA
# =====================================================

BASE_DIR = Path(__file__).resolve().parent

RESULT_DIR = BASE_DIR / "results" / "comparison"

PREDICTION_FILE = RESULT_DIR / "comparison_predictions.csv"
METRICS_FILE = RESULT_DIR / "comparison_metrics.csv"

df = pd.read_csv(PREDICTION_FILE)
metrics = pd.read_csv(METRICS_FILE)

# Convert timestamp to datetime
df["timestamp"] = pd.to_datetime(df["timestamp"])

# Sort by time
df = df.sort_values("timestamp").reset_index(drop=True)

print("Prediction data:")
print(df.head())

print("\nData shape:", df.shape)


# =====================================================
# 2. CONFIGURATION
# =====================================================

model_columns = [
    "naive",
    "arima",
    "umidas",
    "deepar",
    "med",
]

model_names = {
    "naive": "Naive",
    "arima": "ARIMA",
    "umidas": "U-MIDAS",
    "deepar": "DeepAR",
    "med": "MED",
}

# Get RMSE from the evaluation table
rmse_values = dict(zip(metrics["Model"], metrics["RMSE"]))

metric_names = {
    "naive": "Naive",
    "arima": "ARIMA",
    "umidas": "U-MIDAS (OLS)",
    "deepar": "DeepAR (Gaussian LSTM)",
    "med": "MED (author implementation)",
}

# Check columns
required_columns = ["timestamp", "actual"] + model_columns

missing_columns = [
    col for col in required_columns if col not in df.columns
]

if missing_columns:
    raise ValueError(
        "Missing columns in prediction CSV: {}".format(missing_columns)
    )

if df[required_columns].isna().any().any():
    raise ValueError("Prediction data contain missing values.")


# Format the time axis
def format_time_axis(ax):
    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(
        mdates.ConciseDateFormatter(locator)
    )


# =====================================================
# 3. ACTUAL VS MED
# =====================================================

fig, ax = plt.subplots(figsize=(14, 5))

ax.plot(
    df["timestamp"],
    df["actual"],
    label="Actual",
    linewidth=1.8,
)

ax.plot(
    df["timestamp"],
    df["med"],
    label="MED",
    linewidth=1.3,
)

ax.set_title("Actual vs MED Prediction", fontsize=14)
ax.set_xlabel("Time")
ax.set_ylabel("Electricity Load")

ax.legend()
ax.grid(True, alpha=0.3)

format_time_axis(ax)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "actual_vs_med.png",
    dpi=300,
)

plt.show()
plt.close(fig)


# =====================================================
# 4. ACTUAL VS ALL MODELS
# =====================================================

fig, ax = plt.subplots(figsize=(16, 6))

# Actual values
ax.plot(
    df["timestamp"],
    df["actual"],
    label="Actual",
    linewidth=2.2,
)

# Predictions of all models
for col in model_columns:

    display_name = model_names[col]
    metric_name = metric_names[col]

    rmse = rmse_values[metric_name]

    ax.plot(
        df["timestamp"],
        df[col],
        label="{} (RMSE={:,.0f})".format(
            display_name,
            rmse,
        ),
        linewidth=1,
        alpha=0.85,
    )

ax.set_title(
    "Actual vs Predicted Values - Model Comparison",
    fontsize=14,
)

ax.set_xlabel("Time")
ax.set_ylabel("Electricity Load")

ax.legend(ncol=2)
ax.grid(True, alpha=0.3)

format_time_axis(ax)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "actual_vs_all_models.png",
    dpi=300,
)

plt.show()
plt.close(fig)


# =====================================================
# 5. ABSOLUTE ERROR COMPARISON
# =====================================================

fig, ax = plt.subplots(figsize=(16, 6))

for col in model_columns:

    absolute_error = (
        df["actual"] - df[col]
    ).abs()

    ax.plot(
        df["timestamp"],
        absolute_error,
        label=model_names[col],
        linewidth=1,
        alpha=0.8,
    )

ax.set_title(
    "Absolute Forecasting Error Over Time",
    fontsize=14,
)

ax.set_xlabel("Time")
ax.set_ylabel("Absolute Error")

ax.legend(ncol=3)
ax.grid(True, alpha=0.3)

format_time_axis(ax)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "absolute_errors.png",
    dpi=300,
)

plt.show()
plt.close(fig)


print("\nAll plots have been saved to:")
print(RESULT_DIR)
