"""Evidence for the README observations. Needs model/ and test_data.csv from train_models.py.
Run from the project root:  python analysis/analyze_results.py"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from scipy.stats import binomtest
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, matthews_corrcoef

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
meta = json.loads((ROOT / "model" / "metadata.json").read_text())
td = pd.read_csv(ROOT / "test_data.csv")
X, y = td[meta["feature_columns"]], td[meta["target"]].astype(str).values
res = pd.read_csv(ROOT / "model" / "results_test.csv").set_index("ML Model Name")
classes = meta["classes"]
pred, pipes = {}, {}
for name, fn in meta["models"].items():
    pipes[name] = joblib.load(ROOT / "model" / fn)
    pred[name] = pipes[name].predict(X)

print(f"test rows = {len(y)}")
base = pd.Series(y).value_counts(normalize=True).max()
print(f"majority-class baseline: accuracy = {base:.4f}, MCC = 0 (any model must beat this)\n")

print("1) GENERALISATION GAP (train accuracy vs test accuracy; CV MCC from tuning)")
gap = res[["CV MCC (train)", "Train Acc", "Accuracy", "MCC", "Fit time (s)"]].copy()
gap["gap (train-test)"] = (gap["Train Acc"] - gap["Accuracy"]).round(4)
print(gap.to_string(), "\n")

print("2) PER-CLASS F1 ON THE TEST SET (rows = bean variety)")
f1 = pd.DataFrame({n: f1_score(y, p, labels=classes, average=None, zero_division=0)
                   for n, p in pred.items()}, index=classes).round(3)
print(f1.to_string(), "\n")

print("3) MOST FREQUENT CONFUSIONS (true -> predicted : count)")
for n, p in pred.items():
    cm = confusion_matrix(y, p, labels=classes)
    off = [(cm[i, j], classes[i], classes[j]) for i in range(len(classes))
           for j in range(len(classes)) if i != j]
    top = sorted(off, reverse=True)[:3]
    print(f"  {n:24s} " + "; ".join(f"{a}->{b}: {c}" for c, a, b in top))
print()

winner = res["MCC"].idxmax()
print(f"4) WINNER BY MCC = {winner}. Significantly better? (exact McNemar test, same test rows)")
rng = np.random.default_rng(0)
for n in pred:
    if n == winner:
        continue
    w_ok, o_ok = pred[winner] == y, pred[n] == y
    b, c = int((w_ok & ~o_ok).sum()), int((~w_ok & o_ok).sum())
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    verdict = "significant (p<0.05)" if p < 0.05 else "NOT significant"
    print(f"  vs {n:24s} winner-only-correct={b:4d} other-only-correct={c:4d} p={p:.4g} -> {verdict}")
print("\n   95% bootstrap confidence intervals (1000 resamples of the test set):")
for n, p in pred.items():
    accs, mccs = [], []
    for _ in range(1000):
        idx = rng.integers(0, len(y), len(y))
        accs.append(accuracy_score(y[idx], p[idx]))
        mccs.append(matthews_corrcoef(y[idx], p[idx]))
    print(f"  {n:24s} acc [{np.percentile(accs, 2.5):.4f}, {np.percentile(accs, 97.5):.4f}]"
          f"  MCC [{np.percentile(mccs, 2.5):.4f}, {np.percentile(mccs, 97.5):.4f}]")

dt = pipes["Decision Tree"].named_steps["clf"]
print(f"\n5) DECISION TREE: depth = {dt.get_depth()}, leaves = {dt.get_n_leaves()}")
print("6) TUNED HYPER-PARAMETERS:")
for n, bp in meta["best_params"].items():
    print(f"  {n:24s} {bp}")
