"""Evaluation helpers shared by train_models.py AND app.py.

Why a shared module: the numbers in the README comparison table and the numbers shown in the
Streamlit app are then produced by exactly the same code, so they can never disagree.

Metric conventions (state these in the README):
  * binary problem      -> Precision / Recall / F1 are for the POSITIVE class (the class of
                           interest, chosen with --positive; default = classes[1]); AUC uses its probability.
  * multi-class problem -> Precision / Recall / F1 are MACRO averages (every class counts equally);
                           AUC is one-vs-rest, macro averaged, computed from predict_proba.
  * Accuracy and MCC are defined for both cases (MCC = multi-class generalisation of Matthews).
"""
import numpy as np
import pandas as pd
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, matthews_corrcoef, precision_score, recall_score,
                             roc_auc_score)

METRIC_NAMES = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def averaging_rule(classes):
    """'binary' for two classes, otherwise 'macro'."""
    return "binary" if len(classes) == 2 else "macro"


def compute_metrics(y_true, y_pred, y_proba, classes, positive_class=None):
    """Return {'Accuracy','AUC','Precision','Recall','F1','MCC'} as floats.

    y_proba columns must follow the order of `classes` (= model.classes_).
    positive_class (binary only): the class of interest; default = classes[1].
    AUC is NaN when it is undefined (e.g. the test file contains only one class).
    """
    classes = list(classes)
    avg = averaging_rule(classes)
    kw = {"average": avg, "zero_division": 0}
    y_true = np.asarray(y_true)
    pos = None
    if avg == "binary":
        pos = positive_class if positive_class is not None else classes[1]
        kw["pos_label"] = pos
    try:
        if avg == "binary":
            auc = roc_auc_score(y_true == pos, y_proba[:, classes.index(pos)])
        else:
            auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="macro",
                                labels=classes)
    except ValueError:
        auc = float("nan")
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": auc,
        "Precision": precision_score(y_true, y_pred, **kw),
        "Recall": recall_score(y_true, y_pred, **kw),
        "F1": f1_score(y_true, y_pred, **kw),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


def confusion_array(y_true, y_pred, classes):
    """Confusion matrix (rows = true, columns = predicted) in the order of `classes`."""
    return confusion_matrix(y_true, y_pred, labels=list(classes))


def report_frame(y_true, y_pred, classes):
    """Per-class precision/recall/F1/support as a DataFrame (classification report)."""
    rep = classification_report(y_true, y_pred, labels=list(classes), output_dict=True,
                                zero_division=0)
    return pd.DataFrame(rep).T.round(4)
