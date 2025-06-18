import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc, roc_curve
import torch

def run_model(model, features):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(features, dtype=torch.float32).unsqueeze(0)
        outputs = model(inputs).squeeze(0)
        return torch.sigmoid(outputs).numpy()

def compute_md_metrics(predictions, labels, verbose=True):
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        if verbose:
            print("[WARNING] Only one class present in labels. Skipping AUC-based metrics.")
        return {
            "roc_auc": None,
            "pr_auc": None,
            "precision_at_95": None,
            "fpr_at_95": None,
            "iou": None
        }

    # ROC AUC
    roc_auc = roc_auc_score(labels, predictions)

    # Precision-Recall curve
    precision, recall, thresholds_pr = precision_recall_curve(labels, predictions)
    pr_auc = auc(recall, precision)

    # FPR at 80% recall
    target_recall = 0.80
    threshold_at_80 = None

    for idx in reversed(range(1, len(recall))):
        if recall[idx] >= target_recall:
            # threshold_at_80 = thresholds_pr[min(idx, len(thresholds_pr) - 1)]
            # threshold_at_80 = thresholds_pr[min(idx - 1, len(thresholds_pr) - 1)]
            threshold_at_80 = thresholds_pr[idx - 1]  # highest threshold for recall ≥ 80%
            break

    if threshold_at_80 is not None:
        preds = (predictions > threshold_at_80).astype(np.uint8)

        TP = np.sum((preds == 1) & (labels == 1))
        FP = np.sum((preds == 1) & (labels == 0))
        FN = np.sum((preds == 0) & (labels == 1))
        TN = np.sum((preds == 0) & (labels == 0))

        precision_at_95 = TP / (TP + FP + 1e-8)
        fpr_at_95 = FP / (FP + np.sum(labels == 0) + 1e-8)
        iou_95 = TP / (TP + FP + FN + 1e-8)
    else:
        print("[WARN] Could not find recall ≥ 0.95 in evaluation.")

    if verbose:
        print(f"[METRICS] ROC AUC: {roc_auc:.4f}")
        print(f"[METRICS] FPR @ 95% Recall: {fpr_at_95:.4f}")
        print(f"[METRICS] Precision @ 95% Recall: {precision_at_95:.4f}")
        print(f"[METRICS] IoU @ 95% Recall: {iou_95:.4f}")
        print(f"[METRICS] Computed Threshold @ 80% Recall: {threshold_at_80:.4f}")

    return {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "precision_at_95": precision_at_95,
        "fpr_at_95": fpr_at_95,
        "iou": iou_95,
        "threshold_at_80": threshold_at_80
    }

def evaluate_model_md_metrics(model, points, features, labels, mismatch_type, predictions=None, verbose=True):
    """
    If `predictions` is provided, it is used directly.
    Otherwise, inference is run on `features`.
    """
    if predictions is None:
        predictions = run_model(model, features)
    return compute_md_metrics(predictions, labels, verbose=verbose)
