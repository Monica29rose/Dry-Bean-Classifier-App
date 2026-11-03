"""Train the five required classifiers on ONE dataset and save everything the app needs.

Run (on the BITS Virtual Lab):   python train_models.py --data data/mydata.csv --target <column>

Produces (relative to --outdir, default = current folder):
    model/<name>.joblib   one FULL pipeline per model (preprocessing + classifier)
    model/metadata.json   feature names, classes, metric rules, best params, library versions
    model/results_test.csv   the comparison table (copy it into the README)
    test_data.csv         stratified hold-out set (features + target) for the Streamlit app

Design decisions (justify these in the README):
  * The test set is split off FIRST and never used for fitting or tuning -> unbiased metrics.
  * Preprocessing lives INSIDE each pipeline and is fitted on training folds only -> no leakage,
    and the app can feed a raw uploaded CSV straight into the saved model (no retraining).
  * Hyper-parameters are tuned by 5-fold stratified CV on the training data, scored with MCC
    (one number that stays honest under class imbalance, defined for binary and multi-class).
  * The same random seed is used everywhere -> reproducible.
"""
import os
_MSG = "The `probability` parameter was deprecated"
os.environ["PYTHONWARNINGS"] = f"ignore:{_MSG}:FutureWarning"
import warnings
warnings.filterwarnings("ignore", message=_MSG, category=FutureWarning)

import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from eval_utils import METRIC_NAMES, averaging_rule, compute_metrics

# ------------------------------- CONFIG (edit these) --------------------------------------
SEED = 42
TEST_SIZE = 0.20
CV_FOLDS = 5
MAX_TRAIN_ROWS = 20000     # stratified cap so SVM/KNN stay fast; None = use all training rows
TEST_MAX_ROWS = 5000       # keeps test_data.csv small enough for the free Streamlit tier
N_JOBS = -1
# -------------------------------------------------------------------------------------------


def load_dataset(path, target, drop):
    df = pd.read_csv(path)
    if target not in df.columns:
        raise SystemExit(f"Target column '{target}' not found. Columns: {list(df.columns)}")
    df = df.drop(columns=[c for c in drop if c in df.columns])
    n_before = len(df)
    df = df.drop_duplicates()
    print(f"duplicate rows removed: {n_before - len(df)} (mention in README pre-processing)")
    df = df.dropna(subset=[target]).reset_index(drop=True)
    X = df.drop(columns=[target])
    y = df[target].astype(str)          # labels as strings -> identical handling everywhere
    print("=== DATASET SUMMARY (paste into README section b) ===")
    print(f"rows = {len(df)}   features = {X.shape[1]}   target = '{target}'")
    print(f"requirement check -> rows >= 500: {len(df) >= 500} | features >= 12: {X.shape[1] >= 12}")
    print("class distribution:\n", y.value_counts().to_string())
    miss = X.isna().mean()
    print("columns with missing values:", {c: f"{v:.1%}" for c, v in miss[miss > 0].items()} or "none")
    return X, y


def stratified_cap(X, y, n):
    if n and len(X) > n:
        X, _, y, _ = train_test_split(X, y, train_size=n, stratify=y, random_state=SEED)
    return X, y


def make_preprocessor(num_cols, cat_cols):
    parts = []
    if num_cols:
        parts.append(("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                                       ("scale", StandardScaler())]), num_cols))
    if cat_cols:
        parts.append(("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                                       ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                                sparse_output=False))]), cat_cols))
    return ColumnTransformer(parts)


def model_zoo():
    """name -> (estimator, parameter grid). Grid keys use the pipeline step name 'clf'."""
    return {
        "Logistic Regression": (LogisticRegression(max_iter=3000, random_state=SEED),
                                {"clf__C": [0.01, 0.1, 1, 10, 100, 1000]}),
        "Decision Tree": (DecisionTreeClassifier(random_state=SEED),
                          {"clf__max_depth": [3, 5, 8, 12, None],
                           "clf__min_samples_leaf": [1, 5, 20],
                           "clf__criterion": ["gini", "entropy"]}),
        "KNN": (KNeighborsClassifier(),
                {"clf__n_neighbors": [3, 5, 9, 15, 25, 41, 61],
                 "clf__weights": ["uniform", "distance"]}),
        "Naive Bayes": (GaussianNB(),
                        {"clf__var_smoothing": list(np.logspace(-14, -5, 10))}),
        # probability=False during the search (much faster); refitted with probability=True below
        "Support Vector Machine": (SVC(kernel="rbf", probability=False, random_state=SEED),
                                   {"clf__C": [0.5, 1, 5, 10, 50, 100],
                                    "clf__gamma": ["scale", 0.01, 0.1]}),
    }


def fit_one(est, grid, prep, X_tr, y_tr):
    pipe = Pipeline([("prep", clone(prep)), ("clf", est)])
    cv = StratifiedKFold(CV_FOLDS, shuffle=True, random_state=SEED)
    search = GridSearchCV(pipe, grid, scoring=make_scorer(matthews_corrcoef), cv=cv,
                          n_jobs=N_JOBS)
    t0 = time.perf_counter()
    search.fit(X_tr, y_tr)
    best = search.best_estimator_
    if hasattr(est, "probability") and not est.probability:   # SVC: need predict_proba for AUC
        best = clone(pipe).set_params(**search.best_params_, clf__probability=True)
        best.fit(X_tr, y_tr)
    return best, search, time.perf_counter() - t0


def edge_warnings(name, grid, best_params):
    """A tuned value on the border of its grid means the optimum may lie outside the grid."""
    fmt = lambda v: "None" if v is None else f"{v:g}"
    for key, vals in grid.items():
        best = best_params[key]
        nums = [v for v in vals if isinstance(v, (int, float)) and not isinstance(v, bool)]
        at_edge = len(nums) >= 3 and ((best is None and None in vals) or
                                      (best in (min(nums), max(nums))))
        if at_edge:
            print(f"WARNING [{name}] {key.replace('clf__', '')} = {fmt(best)} is at a grid edge "
                  f"({fmt(min(nums))}..{fmt(max(nums))}).")
            print("   -> widen the grid, or show in model/cv_results.csv that the score is flat there.")


def to_markdown(df):
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in df.values]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--target", required=True)
    ap.add_argument("--drop", nargs="*", default=[], help="id / leakage columns to remove")
    ap.add_argument("--positive", default=None,
                    help="binary only: class of interest for Precision/Recall/F1/AUC "
                         "(default = alphabetically second class)")
    ap.add_argument("--outdir", default=".")
    args = ap.parse_args()
    out = Path(args.outdir)
    model_dir = out / "model"
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_dataset(args.data, args.target, args.drop)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=TEST_SIZE, stratify=y,
                                              random_state=SEED)
    X_te, y_te = stratified_cap(X_te, y_te, TEST_MAX_ROWS)
    X_tr, y_tr = stratified_cap(X_tr, y_tr, MAX_TRAIN_ROWS)
    test_df = X_te.copy()
    test_df[args.target] = y_te.values
    test_df.to_csv(out / "test_data.csv", index=False)
    print(f"\ntrain rows = {len(X_tr)}   test rows (saved to test_data.csv) = {len(X_te)}")

    num_cols = X.select_dtypes(include="number").columns.tolist()
    cat_cols = [c for c in X.columns if c not in num_cols]
    prep = make_preprocessor(num_cols, cat_cols)

    classes = sorted(y.unique())                    # same order as the models' classes_
    positive = None
    if len(classes) == 2:
        positive = args.positive if args.positive is not None else classes[1]
        if positive not in classes:
            raise SystemExit(f"--positive must be one of {classes}")
        print(f"binary problem: positive class = '{positive}' (justify this choice in the README)")
    rows, best_params, files, cv_rows = [], {}, {}, []
    for name, (est, grid) in model_zoo().items():
        print(f"\n--- {name}: grid search ({CV_FOLDS}-fold CV, scoring = MCC) ---")
        best, search, secs = fit_one(est, grid, prep, X_tr, y_tr)
        classes = [str(c) for c in best.classes_]
        pred, proba = best.predict(X_te), best.predict_proba(X_te)
        m = compute_metrics(y_te.values, pred, proba, classes, positive)
        train_acc = float((best.predict(X_tr) == y_tr.values).mean())
        rows.append({"ML Model Name": name, **{k: round(m[k], 4) for k in METRIC_NAMES},
                     "CV MCC (train)": round(search.best_score_, 4),
                     "Train Acc": round(train_acc, 4), "Fit time (s)": round(secs, 1)})
        edge_warnings(name, grid, search.best_params_)
        cvr = pd.DataFrame(search.cv_results_)
        cv_rows.append(pd.DataFrame({
            "model": name,
            "params": [str({k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                            for k, v in p.items()}) for p in cvr["params"]],
            "mean_cv_MCC": cvr["mean_test_score"].round(4),
            "std_cv_MCC": cvr["std_test_score"].round(4), "rank": cvr["rank_test_score"]}))
        best_params[name] = {k.replace("clf__", ""): (v.item() if hasattr(v, "item") else v)
                             for k, v in search.best_params_.items()}
        files[name] = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(best, model_dir / files[name], compress=3)
        size_mb = (model_dir / files[name]).stat().st_size / 1e6
        print(f"best params: {best_params[name]} | test MCC = {m['MCC']:.4f} | file = {size_mb:.1f} MB")
        if size_mb > 50:
            print("WARNING: file > 50 MB - GitHub limit is 100 MB; reduce MAX_TRAIN_ROWS / K grid.")

    res = pd.DataFrame(rows)
    res.to_csv(model_dir / "results_test.csv", index=False)
    pd.concat(cv_rows).to_csv(model_dir / "cv_results.csv", index=False)
    print("\n=== COMPARISON TABLE ON THE HOLD-OUT TEST SET (paste into README section d) ===")
    print(to_markdown(res[["ML Model Name"] + METRIC_NAMES]))
    print("\n=== EXTRA EVIDENCE FOR YOUR OBSERVATIONS ===")
    print(to_markdown(res[["ML Model Name", "CV MCC (train)", "Train Acc", "Accuracy", "Fit time (s)"]]))

    meta = {
        "target": args.target, "classes": classes, "averaging": averaging_rule(classes),
        "positive_class": positive,
        "feature_columns": list(X.columns), "numeric_columns": num_cols,
        "categorical_columns": cat_cols, "models": files, "best_params": best_params,
        "n_rows_total": int(len(X)), "n_train": int(len(X_tr)), "n_test": int(len(X_te)),
        "class_distribution": y.value_counts().to_dict(), "seed": SEED,
        "versions": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                     "numpy": np.__version__, "pandas": pd.__version__, "joblib": joblib.__version__},
    }
    (model_dir / "metadata.json").write_text(
        json.dumps(meta, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("\nSaved models + metadata to", model_dir)
    print("Paste these lines into requirements.txt (they pin the libraries that wrote the models):")
    for pkg, ver in (("scikit-learn", sklearn.__version__), ("numpy", np.__version__),
                     ("pandas", pd.__version__), ("joblib", joblib.__version__)):
        print(f"{pkg}=={ver}")


if __name__ == "__main__":
    main()
