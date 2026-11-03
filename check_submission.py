"""Self-audit of the project folder against the assignment brief.  Run:  python check_submission.py
Prints PASS / WARN / FAIL per requirement (with the brief's step number). Fix every FAIL before
submitting; read every WARN. It cannot check the live Streamlit link, the GitHub visibility or
the BITS-Lab screenshot - those are in the manual checklist."""
import ast, json, re, subprocess, sys
from pathlib import Path

import joblib
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
MODELS = ["Logistic Regression", "Decision Tree", "KNN", "Naive Bayes", "Support Vector Machine"]
METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
PYPI = {"sklearn": "scikit-learn", "cv2": "opencv-python", "PIL": "pillow", "yaml": "pyyaml"}
counts = {"PASS": 0, "WARN": 0, "FAIL": 0}


def report(level, step, msg):
    counts[level] += 1
    print(f"[{level}] {step}: {msg}")


def check(cond, step, ok, bad, level="FAIL"):
    report("PASS" if cond else level, step, ok if cond else bad)
    return cond


# ---------------- Step 3: repository contents ----------------
app = next((ROOT / n for n in ("app.py", "streamlit_app.py") if (ROOT / n).exists()), None)
check(app is not None, "Step 3", "app.py / streamlit_app.py present", "app.py (or streamlit_app.py) missing")
for f in ("requirements.txt", "README.md", "test_data.csv"):
    check((ROOT / f).exists(), "Step 3", f"{f} present", f"{f} missing")
mdir = ROOT / "model"
mfiles = sorted(list(mdir.glob("*.joblib")) + list(mdir.glob("*.pkl"))) if mdir.is_dir() else []
check(len(mfiles) >= 5, "Step 3", f"model/ holds {len(mfiles)} saved model files (>= 5)",
      f"model/ must hold 5 saved models, found {len(mfiles)}")
big = [f.name for f in mfiles if f.stat().st_size > 100e6]
check(not big, "Step 3", "every model file < 100 MB (GitHub limit)", f"files over 100 MB: {big}")
check(not [f for f in mfiles if 50e6 < f.stat().st_size <= 100e6], "Step 3",
      "no model file > 50 MB", "a model file is > 50 MB - risky for GitHub/Streamlit", "WARN")
check(any((ROOT / n).exists() for n in ("train_models.py",)) or list(ROOT.glob("*.ipynb")), "Step 3",
      "training source code (.py/.ipynb) present", "no training source code (.py or .ipynb) in the repo")

# ---------------- Step 4: requirements.txt covers every import ----------------
req_txt = (ROOT / "requirements.txt").read_text() if (ROOT / "requirements.txt").exists() else ""
req = {re.split(r"[<>=!~\[ ;]", l.strip(), 1)[0].lower().replace("_", "-")
       for l in req_txt.splitlines() if l.strip() and not l.strip().startswith("#")}
local = {p.stem for p in ROOT.glob("*.py")}
need = set()
for py in ROOT.glob("*.py"):
    if py.name == "check_submission.py":
        continue
    for node in ast.walk(ast.parse(py.read_text())):
        mods = [a.name for a in node.names] if isinstance(node, ast.Import) else \
               [node.module] if isinstance(node, ast.ImportFrom) and node.module and not node.level else []
        for m in mods:
            top = m.split(".")[0]
            if top not in sys.stdlib_module_names and top not in local:
                need.add(PYPI.get(top, top).lower())
missing = sorted(need - req)
check(not missing, "Step 4", "requirements.txt lists every imported third-party package",
      f"imported but NOT in requirements.txt: {missing}")
check("X.Y.Z" not in req_txt, "Step 4", "no placeholder versions left", "requirements.txt still has X.Y.Z")
pins = {l.split("==")[0].strip().lower(): l.split("==")[1].strip() for l in req_txt.splitlines()
        if "==" in l and not l.strip().startswith("#")}
check("scikit-learn" in pins, "Step 4", "scikit-learn is pinned (models must load with same version)",
      "pin scikit-learn==<version printed by train_models.py>", "WARN")
meta_p = mdir / "metadata.json"
meta = json.loads(meta_p.read_text()) if meta_p.exists() else None
check(meta is not None, "Step 3", "model/metadata.json present", "model/metadata.json missing")
if meta:
    tv = meta["versions"]["scikit-learn"]
    if "scikit-learn" in pins:
        check(pins["scikit-learn"] == tv, "Step 4", f"pinned scikit-learn == training version ({tv})",
              f"requirements pins {pins['scikit-learn']} but models were trained with {tv}")
    # ---------------- Step 1: dataset constraints ----------------
    check(meta["n_rows_total"] >= 500, "Step 1", f"dataset has {meta['n_rows_total']} rows (>= 500)",
          f"only {meta['n_rows_total']} rows (< 500)")
    nf = len(meta["feature_columns"])
    check(nf >= 12, "Step 1", f"dataset has {nf} features (>= 12)", f"only {nf} features (< 12)")
    check(len(meta["models"]) == 5 and set(meta["models"]) == set(MODELS), "Step 2",
          "the five required models are present", f"models in metadata: {list(meta['models'])}")

# ---------------- test data + saved models actually work ----------------
res = None
if meta and (ROOT / "test_data.csv").exists():
    td = pd.read_csv(ROOT / "test_data.csv")
    miss = [c for c in meta["feature_columns"] + [meta["target"]] if c not in td.columns]
    check(not miss, "Step 6", "test_data.csv has all feature columns + the target",
          f"test_data.csv is missing {miss}")
    check((ROOT / "test_data.csv").stat().st_size < 25e6, "Step 6", "test_data.csv < 25 MB",
          "test_data.csv is large for the free tier", "WARN")
    if not miss:
        from eval_utils import compute_metrics
        X, y = td[meta["feature_columns"]], td[meta["target"]].astype(str).values
        res = {}
        for name, fn in meta["models"].items():
            try:
                pipe = joblib.load(mdir / fn)
                classes = [str(c) for c in pipe.classes_]
                m = compute_metrics(y, pipe.predict(X), pipe.predict_proba(X), classes,
                                    meta.get("positive_class"))
                res[name] = {k: round(v, 4) for k, v in m.items()}
                check(all(v == v for v in m.values()), "Step 2", f"{name}: loads, scores test_data.csv, "
                      "all 6 metrics defined", f"{name}: a metric is NaN (AUC needs every class in the test file)")
            except Exception as e:
                report("FAIL", "Step 2", f"{name}: cannot load/score ({type(e).__name__}: {e})")

# ---------------- Step 6: app features (static) ----------------
if app:
    src = app.read_text()
    for label, pat in [("a) CSV upload", r"file_uploader"), ("b) model dropdown", r"selectbox"),
                       ("c) evaluation metrics", r"\.metric\(|st\.table|st\.dataframe"),
                       ("d) confusion matrix / report", r"confusion|classification_report|report_frame")]:
        check(re.search(pat, src, re.I) is not None, "Step 6", f"app feature {label} found",
              f"app feature {label} NOT found in {app.name}")
    check(re.search(r"compute_metrics|accuracy_score", src) is not None, "Step 6",
          "app computes metrics from the saved model's predictions", "app does not compute metrics")
    check(".fit(" not in src and "GridSearchCV" not in src, "Step 6",
          "app never calls fit() - no retraining", "app.py trains a model; it must only load saved ones")
    check("joblib.load" in src or "pickle.load" in src, "Step 6", "app loads saved models",
          "app does not load saved models")
    check("Classifier Evaluation Dashboard" not in src, "Anti-plagiarism",
          "APP_TITLE customised", "APP_TITLE is still the template default - customise the UI", "WARN")

# ---------------- Step 5: README structure + numbers ----------------
rd_p = ROOT / "README.md"
if rd_p.exists():
    rd = rd_p.read_text()
    check("<<" not in rd, "Step 5", "no unfilled <<placeholders>>", "README still contains <<placeholders>>")
    for tag, pat in [("a. Problem statement", r"problem statement"), ("b. Dataset description", r"dataset description"),
                     ("c. GitHub repository link", r"github repository link"), ("d. Models used", r"models used")]:
        check(re.search(pat, rd, re.I) is not None, "Step 5", f"section '{tag}' present", f"section '{tag}' missing")
    check(re.search(r"https://github\.com/\S+", rd) is not None, "Step 5", "GitHub link in README",
          "no https://github.com/... link in README")
    check(re.search(r"https://\S*streamlit\.app", rd) is not None, "Step 5", "Streamlit app link in README",
          "no https://....streamlit.app link in README", "WARN")
    head = [h for h in METRICS if h not in rd]
    check(not head, "Step 5", "comparison table has all 6 metric columns", f"README lacks columns {head}")
    check("Overall, Winner for your dataset?" in rd, "Step 5", "observation table has the 'Overall, Winner' row",
          "add the row 'Overall, Winner for your dataset?'")
    for name in MODELS:
        rows = [l for l in rd.splitlines() if l.strip().startswith("|") and name in l]
        check(len(rows) >= 2, "Step 5", f"'{name}' appears in both tables (metrics + observation)",
              f"'{name}' must appear in the comparison table AND the observation table")
        if res and name in res and rows:
            nums_ok = all(f"{res[name][k]:.4f}".rstrip("0").rstrip(".") in rows[0] or f"{res[name][k]:.4f}" in rows[0]
                          for k in METRICS)
            check(nums_ok, "Step 5", f"{name}: README metrics match what the saved model scores on test_data.csv",
                  f"{name}: README numbers differ from the app/test_data.csv results {res[name]}")

# ---------------- commit history (code-level plagiarism check) ----------------
if (ROOT / ".git").exists():
    try:
        n = int(subprocess.run(["git", "rev-list", "--count", "HEAD"], cwd=ROOT, capture_output=True,
                               text=True, check=True).stdout)
        check(n >= 5, "Anti-plagiarism", f"{n} commits in history", f"only {n} commit(s) - commit in stages", "WARN")
    except Exception:
        report("WARN", "Anti-plagiarism", "could not read git history")
else:
    report("WARN", "Anti-plagiarism", "no .git folder here - run this inside your cloned repo too")

print(f"\nSUMMARY: {counts['PASS']} pass, {counts['WARN']} warn, {counts['FAIL']} fail")
sys.exit(1 if counts["FAIL"] else 0)
