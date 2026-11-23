from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt

# Đọc kết quả dự đoán
file_path = Path("results/comparison/comparison_predictions.csv")
df = pd.read_csv(file_path, parse_dates=["timestamp"])
df = df.sort_values("timestamp").reset_index(drop=True)

# Kiểm tra dữ liệu
print("Số dòng:", len(df))
print("Thời điểm đầu:", df["timestamp"].min())
print("Thời điểm cuối:", df["timestamp"].max())
print("Có timestamp trùng:", df["timestamp"].duplicated().any())
print("Có giá trị thiếu:", df.isna().any().any())

# --------------------------------------------------
# 1. Actual và MED trên toàn bộ tập test
# --------------------------------------------------
plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["actual"], label="Actual", linewidth=1.5)
plt.plot(df["timestamp"], df["med"], label="MED", linewidth=1.2)
plt.title("Electricity: Actual vs MED on Test Set")
plt.xlabel("Timestamp")
plt.ylabel("Electricity consumption")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/comparison/actual_vs_med.png", dpi=150)
plt.show()

# --------------------------------------------------
# 2. Phóng to 14 ngày đầu của tập test
# --------------------------------------------------
start = df["timestamp"].min()
end = start + pd.Timedelta(days=14)
zoom = df[df["timestamp"] < end]

plt.figure(figsize=(14, 5))
plt.plot(zoom["timestamp"], zoom["actual"], label="Actual", marker=".", linewidth=1.5)
plt.plot(zoom["timestamp"], zoom["med"], label="MED", marker=".", linewidth=1.2)
plt.title("Electricity: Actual vs MED (First 14 Test Days)")
plt.xlabel("Timestamp")
plt.ylabel("Electricity consumption")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/comparison/actual_vs_med_zoom.png", dpi=150)
plt.show()

# --------------------------------------------------
# 3. MAE trượt 7 ngày (28 điểm, vì mỗi điểm cách nhau 6 giờ)
# --------------------------------------------------
df["med_abs_error"] = (df["actual"] - df["med"]).abs()
df["med_rolling_mae"] = df["med_abs_error"].rolling(
    window=28,
    min_periods=28
).mean()

plt.figure(figsize=(14, 5))
plt.plot(df["timestamp"], df["med_rolling_mae"], label="MED rolling MAE (7 days)")
plt.title("MED: Rolling MAE over Test Set")
plt.xlabel("Timestamp")
plt.ylabel("Rolling MAE")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("results/comparison/med_rolling_mae.png", dpi=150)
plt.show()

print("Đã lưu biểu đồ trong results/comparison/")