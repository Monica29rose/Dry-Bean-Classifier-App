"""EDA for the Dry Bean data: prints the facts used in README section (b) and saves two figures.
Run from the project root:  python analysis/eda.py"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
(ROOT / "figures").mkdir(exist_ok=True)
df = pd.read_csv(ROOT / "data" / "Dry_Bean_Dataset.csv")
X, y = df.drop(columns="Class"), df["Class"]
print(f"rows = {len(df)}, features = {X.shape[1]}, classes = {y.nunique()}")
print("missing values:", int(df.isna().sum().sum()), "| duplicate rows:", int(df.duplicated().sum()))
dist = y.value_counts()
print("\nclass distribution (count, %):")
for k, v in dist.items():
    print(f"  {k:10s} {v:5d}  {v / len(df):6.2%}")
print(f"imbalance ratio (largest/smallest) = {dist.max() / dist.min():.1f}")
corr = X.corr()
pairs = [(a, b, corr.loc[a, b]) for i, a in enumerate(corr.columns) for b in corr.columns[i + 1:]
         if abs(corr.loc[a, b]) > 0.95]
print(f"\nfeature pairs with |r| > 0.95: {len(pairs)}")
for a, b, r in sorted(pairs, key=lambda t: -abs(t[2]))[:6]:
    print(f"  {a:16s} {b:16s} r = {r:.4f}")
print("\nmost skewed features:", X.skew().abs().sort_values(ascending=False).head(3).round(2).to_dict())

fig, ax = plt.subplots(figsize=(7, 4))
ax.bar(dist.index, dist.values, color="tab:brown")
for i, v in enumerate(dist.values):
    ax.text(i, v + 40, str(v), ha="center", fontsize=9)
ax.set_xlabel("Bean variety (Class)")
ax.set_ylabel("Number of beans")
ax.set_title("Dry Bean dataset: class distribution")
ax.grid(axis="y", alpha=0.3)
plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "class_distribution.png", dpi=150)
plt.close(fig)

fig, ax = plt.subplots(figsize=(8, 6.5))
im = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
ax.set_xticks(range(len(corr)))
ax.set_xticklabels(corr.columns, rotation=60, ha="right", fontsize=8)
ax.set_yticks(range(len(corr)))
ax.set_yticklabels(corr.columns, fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.046, label="Pearson r")
ax.set_title("Feature correlation matrix (many near-duplicate size features)")
fig.tight_layout()
fig.savefig(ROOT / "figures" / "correlation_matrix.png", dpi=150)
plt.close(fig)
print("\nsaved figures/class_distribution.png and figures/correlation_matrix.png")
