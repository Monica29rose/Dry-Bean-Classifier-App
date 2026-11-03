"""Convert the UCI Dry Bean workbook to CSV (the repo and the training script use the CSV).
Usage: python prepare_data.py Dry_Bean_Dataset.xlsx data/Dry_Bean_Dataset.csv
(needs openpyxl only for this one-off conversion; it is NOT needed by the deployed app)."""
import sys
import pandas as pd

src, dst = sys.argv[1], sys.argv[2]
df = pd.read_excel(src, sheet_name=0)          # sheet 0 = "Dry_Beans_Dataset"; sheet 1 is the citation
assert df.shape == (13611, 17), df.shape        # 13,611 beans x (16 features + Class)
df.to_csv(dst, index=False)
print("saved", dst, df.shape)
