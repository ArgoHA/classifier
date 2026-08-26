"""Metrics, per-class breakdown, confusion matrix, and threshold selection.

Pulled out of `Trainer` so that `bench` can compute identical metrics without importing the
training stack (and its wandb import), and so the reporting artifacts have one owner.

The averaging mode is decided by the *configured* class count, never by how many classes
happen to appear in the batch of labels being scored. Deriving it from `set(gt_labels)`, as
this used to, meant a val split that happened to miss a class silently switched macro
averaging to binary - two different numbers reported under one name.
"""

from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from loguru import logger
from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

METRIC_ORDER = ("accuracy", "f1", "precision", "recall")


class Validator:
    def __init__(
        self,
        n_labels: int,
        label_to_name: Dict[int, str],
        thresholds: Optional[Sequence[float]] = None,
    ) -> None:
        self.n_labels = int(n_labels)
        self.label_to_name = {int(k): str(v) for k, v in label_to_name.items()}
        self.thresholds = (
            list(thresholds) if thresholds is not None else list(np.arange(0.05, 1.0, 0.05))
        )

    @property
    def average(self) -> str:
        """Binary tasks report the positive class; everything else reports macro."""
        return "binary" if self.n_labels == 2 else "macro"

    def get_metrics(
        self, gt_labels: List[int], preds: List[int], per_class: bool = False
    ) -> Tuple[Dict[str, float], Optional[Dict[str, Dict[str, float]]]]:
        average = self.average
        labels = list(range(self.n_labels))
        # `labels=` pins the macro average to the configured class set. Without it sklearn
        # averages over whichever classes happen to appear, so a split missing one reports
        # a mean over n-1 classes under the same name - not comparable run to run.
        kwargs = {"average": average, "zero_division": 0}
        if average == "macro":
            kwargs["labels"] = labels
            missing = sorted(set(labels) - set(int(g) for g in gt_labels))
            if missing:
                logger.warning(
                    "classes absent from the ground truth of this split: "
                    f"{[self.label_to_name.get(m, m) for m in missing]}; they count as 0 in "
                    "the macro average"
                )

        metrics = {
            "accuracy": accuracy_score(gt_labels, preds),
            "f1": f1_score(gt_labels, preds, **kwargs),
            "precision": precision_score(gt_labels, preds, **kwargs),
            "recall": recall_score(gt_labels, preds, **kwargs),
        }

        if not per_class or self.n_labels <= 2:
            return metrics, None

        f1s = f1_score(gt_labels, preds, average=None, labels=labels, zero_division=0)
        precisions = precision_score(gt_labels, preds, average=None, labels=labels, zero_division=0)
        recalls = recall_score(gt_labels, preds, average=None, labels=labels, zero_division=0)

        gt_arr, pred_arr = np.asarray(gt_labels), np.asarray(preds)
        per_class_metrics = {}
        for i, cl in enumerate(labels):
            mask = gt_arr == cl
            acc = float((pred_arr[mask] == cl).mean()) if mask.any() else 0.0
            per_class_metrics[self.label_to_name.get(cl, str(cl))] = {
                "accuracy": acc,
                "f1": float(f1s[i]),
                "precision": float(precisions[i]),
                "recall": float(recalls[i]),
            }
        return metrics, per_class_metrics

    def save_confusion_matrix(
        self, gt_labels: List[int], preds: List[int], path_to_save: Path, mode: str
    ) -> np.ndarray:
        labels = list(range(self.n_labels))
        matrix = confusion_matrix(gt_labels, preds, labels=labels)
        names = [self.label_to_name.get(i, str(i)) for i in labels]

        size = max(6, min(20, self.n_labels))
        plt.figure(figsize=(size, size * 0.8))
        plt.imshow(matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(f"Confusion matrix ({mode})")
        plt.colorbar()
        ticks = np.arange(len(names))
        plt.xticks(ticks, names, rotation=45, ha="right")
        plt.yticks(ticks, names)

        thresh = matrix.max() / 2.0 if matrix.max() else 0.5
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                plt.text(
                    j,
                    i,
                    format(matrix[i, j], "d"),
                    horizontalalignment="center",
                    color="white" if matrix[i, j] > thresh else "black",
                )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        path_to_save.mkdir(parents=True, exist_ok=True)
        plt.savefig(path_to_save / f"{mode}_confusion_matrix.png")
        plt.close()
        return matrix

    def threshold_sweep(
        self, probs: np.ndarray, gt_labels: List[int], path_to_save: Path, mode: str
    ) -> Optional[Dict[str, float]]:
        """F1-optimal decision threshold for a binary task, as a logged artifact.

        Only meaningful when there are two classes: with one score to threshold, moving it
        trades precision against recall. Hand-tuning that constant in an inference script
        means the choice is invisible and unversioned; here it lands in a plot and a csv
        next to the weights that produced it.
        """
        if self.n_labels != 2:
            return None

        positive = np.asarray(probs)[:, 1]
        gt_arr = np.asarray(gt_labels)
        rows = []
        for threshold in self.thresholds:
            preds = (positive >= threshold).astype(int)
            rows.append(
                (
                    threshold,
                    precision_score(gt_arr, preds, zero_division=0),
                    recall_score(gt_arr, preds, zero_division=0),
                    f1_score(gt_arr, preds, zero_division=0),
                )
            )

        thresholds, precisions, recalls, f1s = map(np.array, zip(*rows))
        path_to_save.mkdir(parents=True, exist_ok=True)

        plt.figure()
        plt.plot(thresholds, precisions, label="Precision", marker="o")
        plt.plot(thresholds, recalls, label="Recall", marker="o")
        plt.plot(thresholds, f1s, label="F1", marker="o")
        plt.xlabel("Threshold")
        plt.ylabel("Score")
        plt.title(f"Precision / recall / F1 vs threshold ({mode})")
        plt.legend()
        plt.grid(True)
        plt.savefig(path_to_save / f"{mode}_threshold_sweep.png")
        plt.close()

        best_idx = int(np.argmax(f1s))
        best = {
            "best_threshold": float(thresholds[best_idx]),
            "best_f1": float(f1s[best_idx]),
            "best_precision": float(precisions[best_idx]),
            "best_recall": float(recalls[best_idx]),
        }
        logger.info(
            f"{mode}: F1-optimal threshold {best['best_threshold']:.2f} "
            f"(F1 {best['best_f1']:.4f}, default 0.50 F1 "
            f"{f1_score(gt_arr, (positive >= 0.5).astype(int), zero_division=0):.4f})"
        )
        return best

    def save_extended_metrics(
        self,
        path_to_save: Path,
        per_class: Dict[str, Dict[str, Dict[str, float]]],
        threshold: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        """One long-format csv: split, class, metric, value - easy to diff across runs."""
        rows = []
        for split, classes in (per_class or {}).items():
            if not classes:
                continue
            for cls_name, metrics in classes.items():
                for metric in METRIC_ORDER:
                    rows.append(
                        {
                            "id": split,
                            "class": cls_name,
                            "metric": metric,
                            "value": round(float(metrics.get(metric, float("nan"))), 4),
                        }
                    )
        for split, best in (threshold or {}).items():
            if not best:
                continue
            for metric, value in best.items():
                rows.append(
                    {"id": split, "class": "__all__", "metric": metric, "value": round(value, 4)}
                )

        if rows:
            path_to_save.mkdir(parents=True, exist_ok=True)
            pd.DataFrame(rows, columns=["id", "class", "metric", "value"]).to_csv(
                path_to_save / "extended_metrics.csv", index=False
            )

    @staticmethod
    def postprocess(probs: torch.Tensor, gt_labels: torch.Tensor) -> Tuple[List[int], List[int]]:
        return torch.argmax(probs, dim=1).tolist(), gt_labels.tolist()
