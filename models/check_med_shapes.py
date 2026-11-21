
from pathlib import Path
import pandas as pd

file_path = Path("data/3_Wea/wea.xlsx")

x = pd.read_excel(file_path, sheet_name="x", index_col=0)
y = pd.read_excel(file_path, sheet_name="y", index_col=0)

print("Sheet x:", x.shape)
print("Sheet y:", y.shape)

print("\nSố cột của x:", len(x.columns))
print("Số cột của y:", len(y.columns))

print("\n5 dòng đầu của x:")
print(x.head())

print("\n5 dòng đầu của y:")
print(y.head())
