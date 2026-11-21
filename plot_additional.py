import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Đọc dữ liệu dự báo
result_dir = Path("results/comparison")
pred = pd.read_csv(result_dir / "comparison_predictions.csv")

pred["timestamp"] = pd.to_datetime(pred["timestamp"])
pred = pred.sort_values("timestamp").reset_index(drop=True)


# =====================================================
# BIỂU ĐỒ 1: SO SÁNH DỰ BÁO TRONG 2 TUẦN
# =====================================================

# Dữ liệu 6 giờ/lần: 2 tuần = 56 quan sát
zoom = pred.head(56)

plt.figure(figsize=(12, 5))

plt.plot(zoom["timestamp"], zoom["actual"],
         label="Actual", linewidth=2)

plt.plot(zoom["timestamp"], zoom["med"],
         label="MED")

plt.plot(zoom["timestamp"], zoom["umidas"],
         label="U-MIDAS")

plt.plot(zoom["timestamp"], zoom["deepar"],
         label="DeepAR")

plt.title("Actual vs Predicted Values (First 2 Weeks)")
plt.xlabel("Time")
plt.ylabel("Electricity Consumption")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(result_dir / "zoom_2_weeks.png", dpi=300)
plt.show()


# =====================================================
# BIỂU ĐỒ 2: ROLLING MAE
# =====================================================

# Cửa sổ 28 quan sát = 7 ngày
window = 28

models = {
    "med": "MED",
    "umidas": "U-MIDAS",
    "deepar": "DeepAR"
}

plt.figure(figsize=(12, 5))

for model, label in models.items():

    # Tính sai số tuyệt đối
    absolute_error = (pred["actual"] - pred[model]).abs()

    # Tính MAE trượt
    rolling_mae = absolute_error.rolling(window=window).mean()

    plt.plot(
        pred["timestamp"],
        rolling_mae,
        label=label
    )

plt.title("Rolling MAE Comparison (Window = 28 Observations)")
plt.xlabel("Time")
plt.ylabel("Rolling MAE")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.4)
plt.xticks(rotation=30)
plt.tight_layout()

plt.savefig(result_dir / "rolling_mae.png", dpi=300)
plt.show()